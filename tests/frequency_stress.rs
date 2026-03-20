//! Headless stress test: sweep parameter combinations × frequency until explosion.
//!
//! Run with: cargo test --release --test frequency_stress -- --nocapture

use std::time::Instant;

use chopstix::constants::*;
use chopstix::gpu::physics::{PhysicsCompute, PhysicsConfig};
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

struct BoundsStats {
    min_y: f32,
    center_y: f32,
    spread: f32,
}

fn compute_stats(positions: &[[f32; 4]]) -> BoundsStats {
    let n = positions.len() as f32;
    let cx: f32 = positions.iter().map(|p| p[0]).sum::<f32>() / n;
    let cy: f32 = positions.iter().map(|p| p[1]).sum::<f32>() / n;
    let cz: f32 = positions.iter().map(|p| p[2]).sum::<f32>() / n;

    let min_y = positions.iter().map(|p| p[1]).fold(f32::MAX, f32::min);

    let spread = positions
        .iter()
        .map(|p| {
            let dx = p[0] - cx;
            let dy = p[1] - cy;
            let dz = p[2] - cz;
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(0.0f32, f32::max);

    BoundsStats { min_y, center_y: cy, spread }
}

#[derive(Debug)]
enum RunResult {
    Settled { frames: u32, final_spread: f32, final_center_y: f32 },
    Exploded { frame: u32, nuked: usize, total: usize },
    Collapsed { frame: u32, final_spread: f32 },
    TimedOut { frames: u32, final_spread: f32, final_center_y: f32 },
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

fn result_tag(r: &RunResult) -> &'static str {
    match r {
        RunResult::Settled { .. } => "OK",
        RunResult::Exploded { .. } => "BOOM",
        RunResult::Collapsed { .. } => "FLAT",
        RunResult::TimedOut { .. } => "TIME",
    }
}

/// Run simulation for a given frequency + config, return what happened.
fn run_frequency(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    frequency: usize,
    config: &PhysicsConfig,
    max_frames: u32,
) -> RunResult {
    let mut buffers = tensegrity::generate_sphere_with_k(frequency, SPHERE_RADIUS, config.pull_k_at_1m);
    buffers.positions = PhysicsCompute::settle(device, queue, &buffers, config);
    let initial_spread = compute_stats(&buffers.positions).spread;
    let collapse_threshold = initial_spread * 0.3;

    let physics = PhysicsCompute::new(device, queue, &buffers, config);

    let mut has_hit_ground = false;
    let mut settle_frames = 0u32;
    let settle_target = 60;

    for frame in 0..max_frames {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Physics"),
        });
        physics.dispatch(&mut encoder, config.iterations_per_frame);
        physics.copy_positions_to_staging(&mut encoder);
        queue.submit(std::iter::once(encoder.finish()));

        let positions = physics.read_positions(device);

        if is_exploded(&positions) {
            return RunResult::Exploded {
                frame,
                nuked: nuked_count(&positions),
                total: positions.len(),
            };
        }

        let stats = compute_stats(&positions);

        if stats.spread < collapse_threshold {
            return RunResult::Collapsed {
                frame,
                final_spread: stats.spread,
            };
        }

        if stats.min_y <= GROUND_Y + 1.0 {
            if !has_hit_ground {
                has_hit_ground = true;
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

struct Scenario {
    name: &'static str,
    config: PhysicsConfig,
    max_frequency: usize,
}

#[test]
fn frequency_sweep() {
    let (device, queue) = create_headless_device();

    let max_frames = 600;

    let scenarios = vec![
        Scenario {
            name: "Baseline (current defaults)",
            config: PhysicsConfig::default(),
            max_frequency: 20,
        },
        Scenario {
            name: "Higher iterations (dt=0.1ms, iter=200)",
            config: PhysicsConfig {
                dt: 0.1e-3,
                iterations_per_frame: 200,
                ..PhysicsConfig::default()
            },
            max_frequency: 20,
        },
        Scenario {
            name: "Match tensegrity-lab (dt=60us, iter=333)",
            config: PhysicsConfig {
                dt: 60e-6,
                iterations_per_frame: 333,
                ..PhysicsConfig::default()
            },
            max_frequency: 20,
        },
        Scenario {
            name: "Stiffer cables (K=1e8, dt=60us, iter=333)",
            config: PhysicsConfig {
                dt: 60e-6,
                iterations_per_frame: 333,
                pull_k_at_1m: 1e8,
                ..PhysicsConfig::default()
            },
            max_frequency: 20,
        },
    ];

    // Collect results: scenario_index → vec of (freq, result, elapsed)
    let mut all_results: Vec<Vec<(usize, RunResult, f64)>> = Vec::new();

    for (si, scenario) in scenarios.iter().enumerate() {
        println!("\n{}", "=".repeat(60));
        println!("Scenario {}: {}", si + 1, scenario.name);
        println!("  dt={:.6}s, iter={}, sim_time/frame={:.3}ms, K={:.0e}, force_scale={:.0}",
            scenario.config.dt,
            scenario.config.iterations_per_frame,
            scenario.config.dt * scenario.config.iterations_per_frame as f32 * 1000.0,
            scenario.config.pull_k_at_1m,
            scenario.config.force_scale,
        );
        println!();

        let mut results: Vec<(usize, RunResult, f64)> = Vec::new();

        for freq in 1..=scenario.max_frequency {
            let buffers = tensegrity::generate_sphere_with_k(freq, SPHERE_RADIUS, scenario.config.pull_k_at_1m);
            print!("  freq={:2} ({:4}j, {:3}s, {:4}c) ... ",
                freq, buffers.num_joints(), buffers.num_rigid(), buffers.num_elastic());

            let t0 = Instant::now();
            let result = run_frequency(&device, &queue, freq, &scenario.config, max_frames);
            let elapsed = t0.elapsed().as_secs_f64();

            println!("{} ({:.2}s) — {}", result_tag(&result), elapsed, result);

            let exploded = matches!(result, RunResult::Exploded { .. });
            results.push((freq, result, elapsed));

            if exploded {
                println!("  Stopping sweep — explosion at frequency {}", freq);
                break;
            }
        }
        all_results.push(results);
    }

    // Summary table
    println!("\n\n{}", "=".repeat(80));
    println!("COMPARISON TABLE");
    println!("{}", "=".repeat(80));

    // Header
    print!("{:>6}", "freq");
    for scenario in &scenarios {
        print!(" | {:^20}", scenario.name.chars().take(20).collect::<String>());
    }
    println!();
    print!("{:->6}", "");
    for _ in &scenarios {
        print!("-+-{:->20}", "");
    }
    println!();

    // Find max frequency tested across all scenarios
    let max_freq_tested = all_results.iter()
        .flat_map(|r| r.iter().map(|(f, _, _)| *f))
        .max()
        .unwrap_or(0);

    for freq in 1..=max_freq_tested {
        print!("{:>6}", freq);
        for results in &all_results {
            if let Some((_, result, elapsed)) = results.iter().find(|(f, _, _)| *f == freq) {
                print!(" | {:>4} {:>6.2}s {:>7}", result_tag(result), elapsed,
                    match result {
                        RunResult::Settled { final_spread, .. } => format!("r={:.1}", final_spread),
                        RunResult::Exploded { frame, .. } => format!("f={}", frame),
                        RunResult::Collapsed { frame, .. } => format!("f={}", frame),
                        RunResult::TimedOut { .. } => "timeout".to_string(),
                    }
                );
            } else {
                print!(" | {:>20}", "—");
            }
        }
        println!();
    }

    // Max stable frequency per scenario
    println!();
    print!("{:>6}", "max");
    for results in &all_results {
        let max_stable = results.iter()
            .filter(|(_, r, _)| matches!(r, RunResult::Settled { .. } | RunResult::TimedOut { .. }))
            .map(|(f, _, _)| *f)
            .max()
            .unwrap_or(0);
        print!(" | {:^20}", format!("freq {}", max_stable));
    }
    println!();
}
