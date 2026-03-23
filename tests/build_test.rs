//! Headless test for incremental tensegrity construction.
//!
//! Run with: cargo test --release --test build_test -- --nocapture

use glam::Vec3;
use chopstix::build::approach::ApproachManager;
use chopstix::build::brick::BrickTemplate;
use chopstix::build::executor::{BuildExecutor, BuildNode};
use chopstix::build::face::FaceRegistry;
use chopstix::build::placement;
use chopstix::build::Spin;
use chopstix::gpu::growable::GrowablePhysics;
use chopstix::gpu::physics::PhysicsConfig;

fn create_headless_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("No suitable GPU adapter found");

    pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Headless Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        },
    ))
    .expect("Failed to create device")
}

fn compute_spread(positions: &[[f32; 4]]) -> f32 {
    if positions.is_empty() { return 0.0; }
    let n = positions.len() as f32;
    let cx: f32 = positions.iter().map(|p| p[0]).sum::<f32>() / n;
    let cy: f32 = positions.iter().map(|p| p[1]).sum::<f32>() / n;
    let cz: f32 = positions.iter().map(|p| p[2]).sum::<f32>() / n;
    positions.iter()
        .map(|p| {
            let dx = p[0] - cx;
            let dy = p[1] - cy;
            let dz = p[2] - cz;
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(0.0f32, f32::max)
}

fn has_nan(positions: &[[f32; 4]]) -> bool {
    positions.iter().any(|p| p[0].is_nan() || p[1].is_nan() || p[2].is_nan())
}

#[test]
fn brick_template_geometry() {
    let template = BrickTemplate::single_twist_left(2.0, 7.0);

    assert_eq!(template.joints.len(), 6, "Should have 6 joints");
    assert_eq!(template.pushes.len(), 3, "Should have 3 push struts");
    assert_eq!(template.pulls.len(), 3, "Should have 3 pull cables");
    assert_eq!(template.faces.len(), 2, "Should have 2 faces");

    // Check that push lengths are reasonable
    for &(a, o, ideal) in &template.pushes {
        let actual = (template.joints[a] - template.joints[o]).length();
        println!("Push {}->{}: ideal={:.3}, actual={:.3}, diff={:.6}",
            a, o, ideal, actual, (ideal - actual).abs());
        assert!((ideal - actual).abs() < 1e-5,
            "Push ideal should match computed distance");
        assert!(ideal > 5.0 && ideal < 10.0,
            "Push length {:.2} should be reasonable", ideal);
    }

    // Check that pull lengths are reasonable
    for &(a, o, ideal) in &template.pulls {
        let actual = (template.joints[a] - template.joints[o]).length();
        println!("Pull {}->{}: ideal={:.3}, actual={:.3}, diff={:.6}",
            a, o, ideal, actual, (ideal - actual).abs());
        assert!((ideal - actual).abs() < 1e-5,
            "Pull ideal should match computed distance");
        assert!(ideal > 5.0 && ideal < 10.0,
            "Pull length {:.2} should be reasonable", ideal);
    }

    // Check face definitions
    let attach = template.faces.iter().find(|f| f.is_attach).unwrap();
    assert_eq!(attach.corners, [0, 1, 2], "Attach face should use bottom joints");
    let forward = template.faces.iter().find(|f| f.is_forward).unwrap();
    assert_eq!(forward.corners, [3, 4, 5], "Forward face should use top joints");

    println!("SingleTwistLeft template geometry OK");
}

#[test]
fn seed_brick_placement() {
    let (device, queue) = create_headless_device();
    let config = PhysicsConfig::default();
    let mut physics = GrowablePhysics::new(&device, &config);
    let mut approach = ApproachManager::new();
    let mut faces = FaceRegistry::new();
    let template = BrickTemplate::single_twist_left(2.0, 7.0);

    let (face_ids, positions) = placement::place_seed_brick(
        &mut physics, &queue, &mut approach, &mut faces,
        &template, Vec3::ZERO, config.pull_k_at_1m,
    );

    println!("Seed brick: {} joints active, {} elastic, {} push",
        physics.active_joints, physics.active_elastic, physics.active_push);
    println!("Created {} faces", face_ids.len());
    println!("Positions: {} entries", positions.len());

    assert_eq!(face_ids.len(), 2, "Seed brick should create 2 faces");
    // 6 brick joints + 2 face centroids = 8
    assert_eq!(physics.active_joints, 8, "Should have 8 active joints (6 brick + 2 centroids)");
    assert_eq!(physics.active_push, 3, "Should have 3 push struts");
    // 3 brick pulls + 3 face-A radials + 3 face-B radials = 9
    assert_eq!(physics.active_elastic, 9, "Should have 9 elastic intervals");

    // Verify no NaN in positions
    assert!(!has_nan(&positions), "Positions should not contain NaN");

    // Run a few physics frames and verify stability
    physics.update_counts(&queue);
    for frame in 0..10 {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Physics"),
        });
        physics.dispatch(&mut encoder, 80);
        physics.copy_positions_to_staging(&mut encoder);
        queue.submit(std::iter::once(encoder.finish()));

        let pos = physics.read_positions(&device);
        assert!(!has_nan(&pos), "NaN at frame {}", frame);
        let spread = compute_spread(&pos);
        println!("Frame {}: spread={:.2}, {} positions", frame, spread, pos.len());
        assert!(spread < 100.0, "Spread exploded at frame {}: {:.2}", frame, spread);
        assert!(spread > 0.1, "Spread collapsed at frame {}: {:.2}", frame, spread);
    }

    println!("Seed brick placement and physics OK");
}

#[test]
fn column_build_incremental() {
    let (device, queue) = create_headless_device();
    let config = PhysicsConfig::default();
    let mut physics = GrowablePhysics::new(&device, &config);
    let pull_k = config.pull_k_at_1m;
    let face_radius = 2.0;
    let brick_height = 7.0;

    let program = BuildNode::Column { count: 3, spin: Spin::Left };
    let mut builder = BuildExecutor::new(
        &mut physics, &queue, program, pull_k, face_radius, brick_height,
    );

    println!("Initial: {} joints, {} elastic, {} push",
        physics.active_joints, physics.active_elastic, physics.active_push);

    let max_ticks = 500;
    let mut last_joints = physics.active_joints;

    for tick in 0..max_ticks {
        let changed = builder.tick(&mut physics, &queue);

        // Run physics
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Physics"),
        });
        physics.dispatch(&mut encoder, 80);
        physics.copy_positions_to_staging(&mut encoder);
        queue.submit(std::iter::once(encoder.finish()));

        let positions = physics.read_positions(&device);
        builder.update_positions(positions.clone());

        if changed || physics.active_joints != last_joints {
            println!("Tick {}: {} joints, {} elastic, {} push, stage={}",
                tick, physics.active_joints, physics.active_elastic, physics.active_push,
                builder.stage_name());
            last_joints = physics.active_joints;
        }

        // Sanity checks
        assert!(!has_nan(&positions), "NaN at tick {}", tick);
        let spread = compute_spread(&positions);
        if spread > 500.0 {
            panic!("EXPLODED at tick {}: spread={:.2}, {} joints", tick, spread, positions.len());
        }

        if builder.stage_name() != "building" {
            println!("Build finished at tick {}: stage={}", tick, builder.stage_name());
            println!("Final: {} joints, {} elastic, {} push",
                physics.active_joints, physics.active_elastic, physics.active_push);
            let final_spread = compute_spread(&positions);
            println!("Final spread: {:.2}", final_spread);
            break;
        }
    }
}
