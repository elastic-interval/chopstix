//! Headless stress test for Klein bottle tensegrity.
//!
//! Run with: cargo test --release --test klein_stress -- --nocapture

use chopstix::gpu::physics::{PhysicsCompute, PhysicsConfig};
use chopstix::klein;

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

fn is_exploded(positions: &[[f32; 4]]) -> bool {
    positions.iter().any(|p| {
        p[3] > 0.5
            || p[0].is_nan()
            || p[1].is_nan()
            || p[2].is_nan()
    })
}

fn nuked_count(positions: &[[f32; 4]]) -> usize {
    positions.iter().filter(|p| p[3] > 0.5).count()
}

fn compute_spread(positions: &[[f32; 4]]) -> f32 {
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

#[test]
fn klein_generates_and_settles() {
    let (device, queue) = create_headless_device();

    let config = PhysicsConfig::default();
    let mut buffers = klein::generate_klein(10, 31, 0, config.pull_k_at_1m);

    println!("Klein 10x31: {} joints, {} push struts, {} cables",
        buffers.num_joints(), buffers.num_push(), buffers.num_elastic());

    assert_eq!(buffers.num_joints(), 155);
    assert!(buffers.num_push() > 0, "Should have push struts");
    assert!(buffers.num_elastic() > 0, "Should have cables");
    assert!(buffers.use_spring_push, "Klein should use spring-push mode");
    assert_eq!(buffers.num_rigid(), 0, "Klein should have no rigid intervals");

    // Run approach-based settling
    println!("Settling with approach...");
    buffers.positions = PhysicsCompute::settle_with_approach(
        &device, &queue, &mut buffers, &config,
    );
    buffers.velocities = vec![[0.0f32; 4]; buffers.positions.len()];

    let spread = compute_spread(&buffers.positions);
    println!("Post-settle spread: {:.2}", spread);
    assert!(spread > 1.0, "Should have expanded from random unit sphere, got {:.2}", spread);
    assert!(!is_exploded(&buffers.positions), "Should not have exploded during settling");
}

#[test]
fn klein_physics_stable() {
    let (device, queue) = create_headless_device();

    let config = PhysicsConfig::default();
    let mut buffers = klein::generate_klein(10, 31, 0, config.pull_k_at_1m);

    // Settle
    buffers.positions = PhysicsCompute::settle_with_approach(
        &device, &queue, &mut buffers, &config,
    );
    buffers.velocities = vec![[0.0f32; 4]; buffers.positions.len()];

    let initial_spread = compute_spread(&buffers.positions);
    println!("Initial spread after settling: {:.2}", initial_spread);

    // Run physics with gravity for 300 frames
    let physics = PhysicsCompute::new(&device, &queue, &buffers, &config);
    let max_frames = 300;

    for frame in 0..max_frames {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Physics"),
        });
        physics.dispatch(&mut encoder, config.iterations_per_frame);
        physics.copy_positions_to_staging(&mut encoder);
        queue.submit(std::iter::once(encoder.finish()));

        let positions = physics.read_positions(&device);

        if is_exploded(&positions) {
            let nuked = nuked_count(&positions);
            panic!("EXPLODED at frame {} ({}/{} nuked)", frame, nuked, positions.len());
        }

        if frame == max_frames - 1 {
            let final_spread = compute_spread(&positions);
            let collapse_threshold = initial_spread * 0.3;
            println!("Final spread: {:.2} (initial: {:.2})", final_spread, initial_spread);
            assert!(final_spread > collapse_threshold,
                "Structure collapsed: spread {:.2} < {:.2} threshold", final_spread, collapse_threshold);
        }
    }

    println!("Klein bottle survived {} frames of physics", max_frames);
}
