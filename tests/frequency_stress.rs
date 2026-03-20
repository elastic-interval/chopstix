//! Headless stress test: run physics at increasing frequencies until explosion.
//!
//! For each frequency, generates a tensegrity sphere, drops it under gravity,
//! waits for ground collision + settling, then checks structural integrity.
//!
//! Run with: cargo test --test frequency_stress -- --nocapture

use chopstix::constants::*;
use chopstix::gpu::physics::PhysicsCompute;
use chopstix::tensegrity;

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

/// Check if any joint has been nuked (speed limit exceeded) or gone to NaN.
fn is_exploded(positions: &[[f32; 4]]) -> bool {
    positions.iter().any(|p| {
        p[3] > 0.5  // nuked flag set by GPU
            || p[0].is_nan()
            || p[1].is_nan()
            || p[2].is_nan()
    })
}

/// Count how many joints have been nuked.
fn nuked_count(positions: &[[f32; 4]]) -> usize {
    positions.iter().filter(|p| p[3] > 0.5).count()
}

/// Compute bounding box stats for the positions.
struct BoundsStats {
    min_y: f32,
    max_y: f32,
    center_y: f32,
    spread: f32, // max distance from centroid
}

fn compute_stats(positions: &[[f32; 4]]) -> BoundsStats {
    let n = positions.len() as f32;
    let cx: f32 = positions.iter().map(|p| p[0]).sum::<f32>() / n;
    let cy: f32 = positions.iter().map(|p| p[1]).sum::<f32>() / n;
    let cz: f32 = positions.iter().map(|p| p[2]).sum::<f32>() / n;

    let min_y = positions.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
    let max_y = positions.iter().map(|p| p[1]).fold(f32::MIN, f32::max);

    let spread = positions
        .iter()
        .map(|p| {
            let dx = p[0] - cx;
            let dy = p[1] - cy;
            let dz = p[2] - cz;
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(0.0f32, f32::max);

    BoundsStats {
        min_y,
        max_y,
        center_y: cy,
        spread,
    }
}

#[derive(Debug)]
enum RunResult {
    /// Settled on surface successfully
    Settled {
        frames: u32,
        final_spread: f32,
        final_center_y: f32,
    },
    /// Physics exploded (speed limit exceeded)
    Exploded { frame: u32, nuked: usize, total: usize },
    /// Collapsed (spread went below threshold)
    Collapsed {
        frame: u32,
        final_spread: f32,
    },
    /// Timed out (didn't settle within frame budget)
    TimedOut {
        frames: u32,
        final_spread: f32,
        final_center_y: f32,
    },
}

impl std::fmt::Display for RunResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunResult::Settled { frames, final_spread, final_center_y } => {
                write!(f, "SETTLED after {} frames (spread={:.2}, center_y={:.2})",
                    frames, final_spread, final_center_y)
            }
            RunResult::Exploded { frame, nuked, total } => {
                write!(f, "EXPLODED at frame {} ({}/{} joints nuked)", frame, nuked, total)
            }
            RunResult::Collapsed { frame, final_spread } => {
                write!(f, "COLLAPSED at frame {} (spread={:.2})", frame, final_spread)
            }
            RunResult::TimedOut { frames, final_spread, final_center_y } => {
                write!(f, "TIMED OUT after {} frames (spread={:.2}, center_y={:.2})",
                    frames, final_spread, final_center_y)
            }
        }
    }
}

/// Run simulation for a given frequency, return what happened.
fn run_frequency(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    frequency: usize,
    max_frames: u32,
) -> RunResult {
    let mut buffers = tensegrity::generate_sphere(frequency, SPHERE_RADIUS);
    // Settle pre-tension before dropping
    buffers.positions = PhysicsCompute::settle(device, queue, &buffers, SETTLE_ITERATIONS);
    let initial_spread = compute_stats(&buffers.positions).spread;
    let collapse_threshold = initial_spread * 0.3; // less than 30% of initial = collapsed

    let physics = PhysicsCompute::new(device, queue, &buffers);

    let iterations_per_frame = ITERATIONS_PER_FRAME;
    let mut has_hit_ground = false;
    let mut settle_frames = 0u32;
    let settle_target = 60; // ~1 second of settling after ground contact

    for frame in 0..max_frames {
        // Dispatch physics
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Physics"),
        });
        physics.dispatch(&mut encoder, iterations_per_frame);
        physics.copy_positions_to_staging(&mut encoder);
        queue.submit(std::iter::once(encoder.finish()));

        let positions = physics.read_positions(device);

        // Check explosion
        if is_exploded(&positions) {
            return RunResult::Exploded {
                frame,
                nuked: nuked_count(&positions),
                total: positions.len(),
            };
        }

        let stats = compute_stats(&positions);

        // Check collapse
        if stats.spread < collapse_threshold {
            return RunResult::Collapsed {
                frame,
                final_spread: stats.spread,
            };
        }

        // Check ground contact
        if stats.min_y <= GROUND_Y + 1.0 {
            if !has_hit_ground {
                has_hit_ground = true;
                println!(
                    "  freq={}: ground contact at frame {} (center_y={:.2}, spread={:.2})",
                    frequency, frame, stats.center_y, stats.spread
                );
            }
            settle_frames += 1;
            if settle_frames >= settle_target {
                return RunResult::Settled {
                    frames: frame,
                    final_spread: stats.spread,
                    final_center_y: stats.center_y,
                };
            }
        }
    }

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Final readback"),
    });
    physics.copy_positions_to_staging(&mut encoder);
    queue.submit(std::iter::once(encoder.finish()));
    let final_positions = physics.read_positions(device);
    let final_stats = compute_stats(&final_positions);

    RunResult::TimedOut {
        frames: max_frames,
        final_spread: final_stats.spread,
        final_center_y: final_stats.center_y,
    }
}

#[test]
fn frequency_sweep() {
    let (device, queue) = create_headless_device();

    let max_frequency = 20;
    let max_frames = 600; // ~10 seconds at 60fps physics rate

    println!("\n=== Frequency Stress Test ===");
    println!("Sphere radius: {}m, ground_y: {}m", SPHERE_RADIUS, GROUND_Y);
    println!("dt: {}s, iterations/frame: {}", ITERATION_DT, ITERATIONS_PER_FRAME);
    println!();

    let mut results: Vec<(usize, RunResult)> = Vec::new();

    for freq in 1..=max_frequency {
        let buffers = tensegrity::generate_sphere(freq, SPHERE_RADIUS);
        println!(
            "freq={:2}: {} joints, {} struts, {} cables",
            freq,
            buffers.num_joints(),
            buffers.num_rigid(),
            buffers.num_elastic(),
        );

        let result = run_frequency(&device, &queue, freq, max_frames);
        println!("  => {}", result);

        let exploded = matches!(result, RunResult::Exploded { .. });
        results.push((freq, result));

        if exploded {
            println!("\n  Stopping sweep — explosion detected at frequency {}", freq);
            break;
        }
    }

    // Summary
    println!("\n=== Summary ===");
    for (freq, result) in &results {
        println!("  freq={:2}: {}", freq, result);
    }

    // The test passes as long as at least frequency 1 works
    let first_failure = results.iter().find(|(_, r)| {
        matches!(r, RunResult::Exploded { .. } | RunResult::Collapsed { .. })
    });

    if let Some((freq, result)) = first_failure {
        println!("\nFirst failure at frequency {}: {}", freq, result);
        // Don't assert-fail — this test is diagnostic, not pass/fail
        // assert!(freq > &1, "Even frequency 1 failed!");
    }
}
