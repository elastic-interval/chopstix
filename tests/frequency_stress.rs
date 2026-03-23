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

/// Check if any position has gone to NaN.
fn has_nan(positions: &[[f32; 4]]) -> bool {
    positions.iter().any(|p| p[0].is_nan() || p[1].is_nan() || p[2].is_nan())
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
    Exploded { frame: u32 },
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
            RunResult::Exploded { frame } => {
                write!(f, "EXPLODED at frame {} (speed limit exceeded)", frame)
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

        if physics.read_frozen(device) || has_nan(&positions) {
            return RunResult::Exploded { frame };
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
    name: String,
    max_frequency: usize,
    /// If true, config is scaled per-frequency via scaled_for_frequency()
    scale_with_freq: bool,
    base_config: PhysicsConfig,
}

impl Scenario {
    fn config_for_freq(&self, freq: usize) -> PhysicsConfig {
        if self.scale_with_freq {
            self.base_config.clone().scaled_for_frequency(freq)
        } else {
            self.base_config.clone()
        }
    }
}

#[test]
fn frequency_sweep() {
    let (device, queue) = create_headless_device();

    let max_frames = 300;

    let scenarios = vec![
        Scenario {
            name: "Scaled for frequency".into(),
            base_config: PhysicsConfig::default(),
            max_frequency: 30,
            scale_with_freq: true,
        },
    ];

    // Collect results: scenario_index → vec of (freq, result, elapsed, dt_used, iter_used)
    let mut all_results: Vec<Vec<(usize, RunResult, f64, f32, u32)>> = Vec::new();

    for (si, scenario) in scenarios.iter().enumerate() {
        println!("\n{}", "=".repeat(70));
        println!("Scenario {}: {}", si + 1, scenario.name);
        println!("  base dt={:.6}s, base iter={}, scaled per frequency",
            scenario.base_config.dt, scenario.base_config.iterations_per_frame);
        println!("  K={:.0e}, force_scale={:.0}",
            scenario.base_config.pull_k_at_1m, scenario.base_config.force_scale);
        println!();

        let mut results: Vec<(usize, RunResult, f64, f32, u32)> = Vec::new();

        // Test a representative sample: 1,3,5,10,15,20,25,30
        let freqs: Vec<usize> = vec![1, 3, 5, 10, 15, 20, 25, 30];
        for freq in freqs {
            let config = scenario.config_for_freq(freq);
            let buffers = tensegrity::generate_sphere_with_k(freq, SPHERE_RADIUS, config.pull_k_at_1m);
            print!("  freq={:2} ({:5}j) dt={:.0}us iter={:4} ... ",
                freq, buffers.num_joints(), config.dt * 1e6, config.iterations_per_frame);

            let t0 = Instant::now();
            let result = run_frequency(&device, &queue, freq, &config, max_frames);
            let elapsed = t0.elapsed().as_secs_f64();

            println!("{} ({:.2}s) — {}", result_tag(&result), elapsed, result);

            let exploded = matches!(result, RunResult::Exploded { .. });
            results.push((freq, result, elapsed, config.dt, config.iterations_per_frame));

            if exploded {
                println!("  Stopping sweep — explosion at frequency {}", freq);
                break;
            }
        }
        all_results.push(results);
    }

    // Summary table
    println!("\n\n{}", "=".repeat(90));
    println!("COMPARISON TABLE");
    println!("{}", "=".repeat(90));

    // Header
    print!("{:>6}", "freq");
    for scenario in &scenarios {
        print!(" | {:^25}", &scenario.name[..scenario.name.len().min(25)]);
    }
    println!();
    print!("{:->6}", "");
    for _ in &scenarios {
        print!("-+-{:->25}", "");
    }
    println!();

    // Find max frequency tested across all scenarios
    let max_freq_tested = all_results.iter()
        .flat_map(|r| r.iter().map(|(f, _, _, _, _)| *f))
        .max()
        .unwrap_or(0);

    for freq in 1..=max_freq_tested {
        print!("{:>6}", freq);
        for results in &all_results {
            if let Some((_, result, elapsed, _, _)) = results.iter().find(|(f, _, _, _, _)| *f == freq) {
                print!(" | {:>4} {:>6.2}s {:>12}", result_tag(result), elapsed,
                    match result {
                        RunResult::Settled { final_spread, .. } => format!("r={:.1}", final_spread),
                        RunResult::Exploded { frame, .. } => format!("f={}", frame),
                        RunResult::Collapsed { frame, .. } => format!("f={}", frame),
                        RunResult::TimedOut { .. } => "timeout".to_string(),
                    }
                );
            } else {
                print!(" | {:>25}", "—");
            }
        }
        println!();
    }

    // Max stable frequency per scenario
    println!();
    print!("{:>6}", "max");
    for results in &all_results {
        let max_stable = results.iter()
            .filter(|(_, r, _, _, _)| matches!(r, RunResult::Settled { .. } | RunResult::TimedOut { .. }))
            .map(|(f, _, _, _, _)| *f)
            .max()
            .unwrap_or(0);
        print!(" | {:^25}", format!("freq {}", max_stable));
    }
    println!();
}
