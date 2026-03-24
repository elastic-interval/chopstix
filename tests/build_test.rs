//! Headless test for incremental tensegrity construction.
//!
//! Run with: cargo test --release --test build_test -- --nocapture

use glam::Vec3;
use chopstix::build::approach::ApproachManager;
use chopstix::build::brick::BrickTemplate;
use chopstix::build::executor::BuildExecutor;
use chopstix::build::fabric_library::FabricName;
use chopstix::build::face::FaceRegistry;
use chopstix::build::placement;
use chopstix::build::Spin;
use chopstix::gpu::growable::GrowablePhysics;
use chopstix::gpu::physics::PhysicsConfig;

/// Helper to run a build program to completion and verify stability.
fn run_build_to_completion(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    program: chopstix::build::executor::BuildProgram,
    max_ticks: usize,
    explode_threshold: f32,
) -> (u32, u32, u32, f32) {
    let config = PhysicsConfig::default();
    let mut physics = GrowablePhysics::new(device, &config);
    let pull_k = config.pull_k_at_1m;

    let mut builder = BuildExecutor::new(
        &mut physics, queue, program, pull_k,
    );

    println!("Initial: {} joints, {} elastic, {} push",
        physics.active_joints, physics.active_elastic, physics.active_push);

    let mut last_joints = physics.active_joints;

    for tick in 0..max_ticks {
        let changed = builder.tick(&mut physics, queue);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Physics"),
        });
        physics.dispatch(&mut encoder, 80);
        physics.copy_positions_to_staging(&mut encoder);
        queue.submit(std::iter::once(encoder.finish()));

        let positions = physics.read_positions(device);
        builder.update_positions(positions.clone());

        if changed || physics.active_joints != last_joints {
            println!("Tick {}: {} joints, {} elastic, {} push, stage={}",
                tick, physics.active_joints, physics.active_elastic, physics.active_push,
                builder.stage_name());
            last_joints = physics.active_joints;
        }

        assert!(!has_nan(&positions), "NaN at tick {}", tick);
        let spread = compute_spread(&positions);
        assert!(spread < explode_threshold,
            "EXPLODED at tick {}: spread={:.2}, {} joints", tick, spread, positions.len());

        if builder.stage_name() != "building" {
            println!("Build finished at tick {}: stage={}", tick, builder.stage_name());
            let final_spread = compute_spread(&positions);
            println!("Final: {} joints, {} elastic, {} push, spread={:.2}",
                physics.active_joints, physics.active_elastic, physics.active_push, final_spread);
            return (physics.active_joints, physics.active_elastic, physics.active_push, final_spread);
        }
    }

    panic!("Build did not complete within {} ticks", max_ticks);
}

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

    let program = FabricName::Column3.program();
    let mut builder = BuildExecutor::new(
        &mut physics, &queue, program, pull_k,
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

#[test]
fn omni_brick_geometry() {
    let template = BrickTemplate::omni_symmetrical();

    assert_eq!(template.joints.len(), 12, "Omni should have 12 joints");
    assert_eq!(template.pushes.len(), 6, "Omni should have 6 push struts");
    assert_eq!(template.pulls.len(), 0, "Omni should have 0 pulls");
    assert_eq!(template.faces.len(), 8, "Omni should have 8 faces");

    // Verify push lengths are consistent (all should be equal for symmetrical)
    let push_len = template.pushes[0].2;
    for (i, push) in template.pushes.iter().enumerate() {
        println!("Push {}: {} → {}, len={:.4}", i, push.0, push.1, push.2);
        assert!((push.2 - push_len).abs() < 0.01,
            "Push {} length {:.4} != {:.4}", i, push.2, push_len);
    }
    println!("Push length: {:.4}", push_len);

    // Verify named faces exist
    for name in &["OmniTop", "OmniBot", "OmniBotX", "OmniBotY", "OmniBotZ"] {
        assert!(template.face_index_by_name(name).is_some(),
            "Missing face '{}'", name);
    }

    // Verify face edge lengths are consistent within each face
    for (i, face) in template.faces.iter().enumerate() {
        let c = face.corners;
        let e01 = (template.joints[c[0]] - template.joints[c[1]]).length();
        let e12 = (template.joints[c[1]] - template.joints[c[2]]).length();
        let e20 = (template.joints[c[2]] - template.joints[c[0]]).length();
        println!("Face {} ({}): edges {:.4}, {:.4}, {:.4}, spin={:?}",
            i, face.name.unwrap_or("?"), e01, e12, e20, face.spin);
        // Equilateral check (within tolerance for equilibrium positions)
        assert!((e01 - e12).abs() < 0.1, "Face {} not equilateral", i);
        assert!((e12 - e20).abs() < 0.1, "Face {} not equilateral", i);
    }

    println!("Omni brick geometry OK");
}

#[test]
fn baked_single_face_compatibility() {
    // Verify baked single twist face edges match omni face edges
    let single = BrickTemplate::single_twist_left_baked();
    let omni = BrickTemplate::omni_symmetrical();

    let single_bottom = &single.faces[0]; // attach face
    let c = single_bottom.corners;
    let single_edge = (single.joints[c[0]] - single.joints[c[1]]).length();
    println!("Single twist bottom face edge: {:.4}", single_edge);

    let omni_botx = omni.face_index_by_name("OmniBotX").unwrap();
    let c = omni.faces[omni_botx].corners;
    let omni_edge = (omni.joints[c[0]] - omni.joints[c[1]]).length();
    println!("Omni BotX face edge: {:.4}", omni_edge);

    let ratio = single_edge / omni_edge;
    println!("Edge ratio (single/omni): {:.4}", ratio);
    assert!((ratio - 1.0).abs() < 0.05,
        "Face size mismatch: single={:.4}, omni={:.4}, ratio={:.4}",
        single_edge, omni_edge, ratio);

    println!("Baked face compatibility OK");
}

#[test]
fn open_claw_build() {
    let (device, queue) = create_headless_device();

    let (joints, elastic, push, spread) = run_build_to_completion(
        &device, &queue,
        FabricName::OpenClaw.program(),
        2000,   // max ticks (3 legs × 4 bricks × 60 frames = 720 + overhead)
        500.0,  // explode threshold
    );

    // Open Claw with 4-brick legs:
    // Hub: 12 joints + 8 face centroids = 20
    // Each leg: 4 bricks × (3 new joints + centroid) = ~16 per leg
    // Plus join centroids, circumference cables, etc.
    println!("Open Claw 4: {} joints, {} elastic, {} push, spread={:.2}",
        joints, elastic, push, spread);

    assert!(joints > 30, "Too few joints: {}", joints);
    assert!(push > 10, "Too few push struts: {}", push);
    assert!(elastic > 20, "Too few elastic intervals: {}", elastic);
    assert!(spread > 1.0, "Structure collapsed: spread={:.2}", spread);
    assert!(spread < 50.0, "Structure too spread: spread={:.2}", spread);
}

/// Detailed diagnostic test modeled on tensegrity-lab's plan_runner_test.
/// Tracks construction step-by-step and validates structural properties.
#[test]
fn triped_construction_diagnostic() {
    let (device, queue) = create_headless_device();
    let config = PhysicsConfig::default();
    let mut physics = GrowablePhysics::new(&device, &config);
    let pull_k = config.pull_k_at_1m;

    // Triped: omni hub + 3 legs of 8 bricks (matching tensegrity-lab's Triped)
    let program = FabricName::Triped.program();
    let mut builder = BuildExecutor::new(&mut physics, &queue, program, pull_k);

    // === Step 1: Verify seed brick ===
    println!("\n=== SEED BRICK (OmniSymmetrical) ===");
    println!("Joints: {}", physics.active_joints);
    println!("Elastic: {}", physics.active_elastic);
    println!("Push: {}", physics.active_push);

    // Omni seed: 12 brick joints + 8 face centroids = 20
    assert_eq!(physics.active_joints, 20,
        "Seed should have 20 joints (12 brick + 8 face centroids)");
    // 6 push struts
    assert_eq!(physics.active_push, 6,
        "Seed should have 6 push struts");
    // 8 faces × 3 radials = 24 elastic
    assert_eq!(physics.active_elastic, 24,
        "Seed should have 24 elastic (8 faces × 3 radials)");

    // Read initial positions and check centroid is near origin
    physics.update_counts(&queue);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Readback"),
    });
    physics.copy_positions_to_staging(&mut encoder);
    queue.submit(std::iter::once(encoder.finish()));
    let positions = physics.read_positions(&device);
    assert!(!has_nan(&positions), "NaN in seed positions");

    let centroid = compute_centroid(&positions);
    println!("Seed centroid: ({:.4}, {:.4}, {:.4})", centroid[0], centroid[1], centroid[2]);
    assert!(centroid[0].abs() < 0.1 && centroid[2].abs() < 0.1,
        "Seed centroid should be near X=0, Z=0");

    // Verify seed has 3-fold symmetry: check Y range
    let min_y = positions.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
    let max_y = positions.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);
    println!("Seed Y range: {:.4} to {:.4}", min_y, max_y);

    // === Step 2: Run build and track each round ===
    println!("\n=== INCREMENTAL BUILD ===");
    let max_ticks = 2000;
    let mut round = 0;
    let mut prev_joints = physics.active_joints;
    let mut prev_push = physics.active_push;
    let mut prev_elastic = physics.active_elastic;

    for tick in 0..max_ticks {
        let changed = builder.tick(&mut physics, &queue);

        if changed {
            round += 1;
            let new_joints = physics.active_joints - prev_joints;
            let new_push = physics.active_push - prev_push;
            let new_elastic = physics.active_elastic - prev_elastic;

            // Read positions to see where new joints were placed
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Diag"),
            });
            physics.copy_positions_to_staging(&mut enc);
            queue.submit(std::iter::once(enc.finish()));
            let cur_pos = physics.read_positions(&device);

            // Show position of new brick joints (the first 3 new joints of each brick)
            if round <= 3 {
                println!("Round {} (tick {}): +{} joints, +{} push, +{} elastic",
                    round, tick, new_joints, new_push, new_elastic);
                for i in prev_joints..physics.active_joints {
                    let p = cur_pos[i as usize];
                    let r = (p[0]*p[0] + p[2]*p[2]).sqrt();
                    println!("  joint {}: ({:+.3}, {:+.3}, {:+.3}) r={:.3}",
                        i, p[0], p[1], p[2], r);
                }
            } else {
                println!("Round {} (tick {}): +{} J, +{} P, +{} E | total: {} J",
                    round, tick, new_joints, new_push, new_elastic,
                    physics.active_joints);
            }

            // Each column round adds 3 bricks (one per leg, parallel) = 9 push struts.
            // Round 1 may also include prisms (OmniTop prism = +1 push).
            if round <= 8 {
                assert!(new_push >= 9,
                    "Round {}: expected at least 9 new push struts, got {}", round, new_push);
            }

            prev_joints = physics.active_joints;
            prev_push = physics.active_push;
            prev_elastic = physics.active_elastic;
        }

        // Run physics
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Physics"),
        });
        physics.dispatch(&mut encoder, 80);
        physics.copy_positions_to_staging(&mut encoder);
        queue.submit(std::iter::once(encoder.finish()));

        let positions = physics.read_positions(&device);
        builder.update_positions(positions.clone());

        assert!(!has_nan(&positions), "NaN at tick {}", tick);
        let spread = compute_spread(&positions);
        assert!(spread < 200.0,
            "EXPLODED at tick {}: spread={:.2}, {} joints", tick, spread, positions.len());

        if builder.stage_name() != "building" {
            println!("\nBuild finished at tick {}: stage={}", tick, builder.stage_name());
            break;
        }
    }

    // === Step 3: Verify final structure ===
    println!("\n=== FINAL STRUCTURE ===");
    println!("Joints: {}", physics.active_joints);
    println!("Elastic: {}", physics.active_elastic);
    println!("Push: {}", physics.active_push);

    // 3-fold symmetry check: total push minus hub pushes should be divisible by 3
    // Hub pushes = 6 (omni struts) + 1 (OmniTop prism) = 7
    let hub_pushes = 7u32;
    let leg_pushes = physics.active_push - hub_pushes;
    assert!(leg_pushes % 3 == 0,
        "Leg push struts ({}) should be divisible by 3 for 3-fold symmetry", leg_pushes);
    println!("Leg pushes: {} = {} per leg (hub: {})", leg_pushes, leg_pushes / 3, hub_pushes);

    // Read final positions
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Final readback"),
    });
    physics.copy_positions_to_staging(&mut encoder);
    queue.submit(std::iter::once(encoder.finish()));
    let positions = physics.read_positions(&device);

    let spread = compute_spread(&positions);
    let centroid = compute_centroid(&positions);
    let min_y = positions.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
    let max_y = positions.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);
    let height = max_y - min_y;

    println!("Spread: {:.4}", spread);
    println!("Centroid: ({:.4}, {:.4}, {:.4})", centroid[0], centroid[1], centroid[2]);
    println!("Height (Y): {:.4} (min={:.4}, max={:.4})", height, min_y, max_y);

    // The structure should extend vertically (legs grow outward/downward from hub)
    assert!(height > 1.0,
        "Structure too flat: height={:.4}", height);

    // Verify 3-fold symmetry by checking XZ distribution
    // Group joints by angle around Y axis, should cluster in 3 groups
    let mut angles: Vec<f32> = positions.iter()
        .filter(|p| (p[0]*p[0] + p[2]*p[2]).sqrt() > 0.5) // skip near-center joints
        .map(|p| p[2].atan2(p[0]))
        .collect();
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if angles.len() > 6 {
        // Check angular gaps — should see 3 clusters ~120° apart
        println!("\nAngular distribution ({} off-center joints):", angles.len());
        let mut gap_count = 0;
        for i in 1..angles.len() {
            let gap = angles[i] - angles[i-1];
            if gap > 0.5 { // > ~30 degrees
                gap_count += 1;
            }
        }
        println!("Large angular gaps (>30°): {}", gap_count);
    }

    println!("\n=== TRIPED DIAGNOSTIC COMPLETE ===");
}

fn compute_centroid(positions: &[[f32; 4]]) -> [f32; 3] {
    if positions.is_empty() { return [0.0; 3]; }
    let n = positions.len() as f32;
    [
        positions.iter().map(|p| p[0]).sum::<f32>() / n,
        positions.iter().map(|p| p[1]).sum::<f32>() / n,
        positions.iter().map(|p| p[2]).sum::<f32>() / n,
    ]
}
