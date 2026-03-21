use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, Modifiers, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::camera::Camera;
use crate::constants::*;
use crate::gpu::hud::Hud;
use crate::gpu::physics::{PhysicsCompute, PhysicsConfig};
use crate::gpu::renderer::Renderer;
use crate::gpu::Gpu;
use crate::tensegrity::{self, TensegritySphereBuffers};

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Orbit,
    Stiffness,
    Pretension,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Orbit => "ORBIT",
            Mode::Stiffness => "STIFFNESS",
            Mode::Pretension => "PRETENSION",
        }
    }
}

struct AppState {
    window: Arc<Window>,
    gpu: Gpu,
    physics: PhysicsCompute,
    renderer: Renderer,
    hud: Hud,
    camera: Camera,
    buffers: TensegritySphereBuffers,
    frequency: usize,
    iterations: u32,
    paused: bool,
    last_frame: Instant,
    frame_count: u32,
    fps_timer: Instant,
    last_fps: f32,
    cursor_pos: (f64, f64),
    physics_frame: u32,
    modifiers: Modifiers,
    pull_k_at_1m: f32,
    pretension: f32,
    show_hud: bool,
    mode: Mode,
}

pub struct App {
    state: Option<AppState>,
    frequency: usize,
}

impl App {
    pub fn new(frequency: usize) -> Self {
        Self {
            state: None,
            frequency,
        }
    }

    fn rebuild_sphere(state: &mut AppState) {
        let config = PhysicsConfig {
            pull_k_at_1m: state.pull_k_at_1m,
            ..PhysicsConfig::default()
        }.scaled_for_frequency(state.frequency);
        let mut buffers = tensegrity::generate_sphere_with_k(state.frequency, SPHERE_RADIUS, state.pull_k_at_1m);
        // Settle: run physics with high drag, no gravity to find pre-stress equilibrium
        buffers.positions = PhysicsCompute::settle(
            &state.gpu.device, &state.gpu.queue, &buffers, &config,
        );
        state.iterations = config.iterations_per_frame;
        log::info!("Physics: dt={:.6}s, iterations={}, sim_time/frame={:.3}ms",
            config.dt, state.iterations, config.dt * state.iterations as f32 * 1000.0);
        state.physics = PhysicsCompute::new(&state.gpu.device, &state.gpu.queue, &buffers, &config);
        state.renderer.update_topology(&buffers, state.frequency);
        state.camera.set_distance(SPHERE_RADIUS * 2.8);
        state.buffers = buffers;
        state.renderer.update_instances(&state.gpu.device, &state.buffers.positions);
        update_title(state);
    }

    /// Update cable stiffness without resettling — keeps current positions/velocities.
    fn update_stiffness(state: &mut AppState) {
        let config = PhysicsConfig {
            pull_k_at_1m: state.pull_k_at_1m,
            ..PhysicsConfig::default()
        }.scaled_for_frequency(state.frequency);

        // Read back current positions from GPU
        let mut encoder = state.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Stiffness readback"),
        });
        state.physics.copy_positions_to_staging(&mut encoder);
        state.gpu.queue.submit(std::iter::once(encoder.finish()));
        let current_positions = state.physics.read_positions(&state.gpu.device);

        // Recompute K values from existing ideal lengths — no sphere regeneration
        for (k, ideal) in state.buffers.elastic_k.iter_mut().zip(state.buffers.elastic_ideal.iter()) {
            *k = state.pull_k_at_1m / ideal;
        }
        state.buffers.positions = current_positions;

        state.iterations = config.iterations_per_frame;
        state.physics = PhysicsCompute::new(&state.gpu.device, &state.gpu.queue, &state.buffers, &config);
        update_title(state);
    }

    /// Adjust pretension by scaling all cable ideal lengths — approach span strategy.
    /// factor < 1.0 tightens (shorter rest length), > 1.0 loosens.
    fn adjust_pretension(state: &mut AppState, factor: f32) {
        state.pretension = (state.pretension * factor).clamp(0.5, 1.0);

        let config = PhysicsConfig {
            pull_k_at_1m: state.pull_k_at_1m,
            ..PhysicsConfig::default()
        }.scaled_for_frequency(state.frequency);

        // Read back current positions
        let mut encoder = state.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Pretension readback"),
        });
        state.physics.copy_positions_to_staging(&mut encoder);
        state.gpu.queue.submit(std::iter::once(encoder.finish()));
        let current_positions = state.physics.read_positions(&state.gpu.device);

        // Scale all cable ideal lengths and recompute K
        for i in 0..state.buffers.elastic_ideal.len() {
            state.buffers.elastic_ideal[i] *= factor;
            state.buffers.elastic_k[i] = state.pull_k_at_1m / state.buffers.elastic_ideal[i];
        }
        state.buffers.positions = current_positions;

        state.iterations = config.iterations_per_frame;
        state.physics = PhysicsCompute::new(&state.gpu.device, &state.gpu.queue, &state.buffers, &config);
        update_title(state);
    }
}

fn update_title(state: &AppState) {
    state.window.set_title(&format!(
        "Chopstix | freq={} | joints={} | struts={} | cables={} | K={:.0e} | {:.0} FPS",
        state.frequency,
        state.buffers.num_joints(),
        state.buffers.num_rigid(),
        state.buffers.num_elastic(),
        state.pull_k_at_1m,
        state.last_fps,
    ));
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("Chopstix")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280, 800)),
                )
                .expect("Failed to create window"),
        );

        let gpu = Gpu::new(window.clone());
        let config = PhysicsConfig::default().scaled_for_frequency(self.frequency);
        let mut buffers = tensegrity::generate_sphere(self.frequency, SPHERE_RADIUS);
        buffers.positions = PhysicsCompute::settle(
            &gpu.device, &gpu.queue, &buffers, &config,
        );
        let iterations = config.iterations_per_frame;
        log::info!("Physics: dt={:.6}s, iterations={}, sim_time/frame={:.3}ms",
            config.dt, iterations, config.dt * iterations as f32 * 1000.0);
        let physics = PhysicsCompute::new(&gpu.device, &gpu.queue, &buffers, &config);
        let size = window.inner_size();
        let renderer = Renderer::new(&gpu, &buffers, self.frequency);
        let hud = Hud::new(&gpu.device, &gpu.queue, gpu.surface_config.format);
        let camera = Camera::new(size.width as f32, size.height as f32, SPHERE_RADIUS * 2.8);

        let mut state = AppState {
            window,
            gpu,
            physics,
            renderer,
            hud,
            camera,
            buffers,
            frequency: self.frequency,
            iterations,
            paused: true,
            last_frame: Instant::now(),
            frame_count: 0,
            fps_timer: Instant::now(),
            last_fps: 0.0,
            cursor_pos: (0.0, 0.0),
            physics_frame: 0,
            modifiers: Modifiers::default(),
            pull_k_at_1m: PULL_K_AT_1M,
            pretension: 0.95,
            show_hud: true,
            mode: Mode::Orbit,
        };
        state.renderer.update_instances(&state.gpu.device, &state.buffers.positions);
        log::info!("Initial instances populated, {} positions", state.buffers.positions.len());
        if let Some(pos) = state.buffers.positions.first() {
            log::info!("First joint position: [{:.2}, {:.2}, {:.2}]", pos[0], pos[1], pos[2]);
        }
        update_title(&state);
        self.state = Some(state);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else { return };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                state.gpu.resize(size.width, size.height);
                state.camera.set_size(size.width as f32, size.height as f32);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                // Global keys (all modes)
                match &logical_key {
                    Key::Named(NamedKey::Escape) => {
                        if state.mode != Mode::Orbit {
                            state.camera.mouse_released();
                            state.mode = Mode::Orbit;
                        } else {
                            event_loop.exit();
                        }
                        return;
                    }
                    Key::Named(NamedKey::Space) => {
                        state.paused = !state.paused;
                        return;
                    }
                    Key::Character(c) if c.as_str() == "h" => {
                        state.show_hud = !state.show_hud;
                        return;
                    }
                    _ => {}
                }

                // Mode-switching keys
                match &logical_key {
                    Key::Character(c) if c.as_str() == "s" => {
                        state.camera.mouse_released();
                        state.mode = if state.mode == Mode::Stiffness { Mode::Orbit } else { Mode::Stiffness };
                        return;
                    }
                    Key::Character(c) if c.as_str() == "p" => {
                        state.camera.mouse_released();
                        state.mode = if state.mode == Mode::Pretension { Mode::Orbit } else { Mode::Pretension };
                        return;
                    }
                    _ => {}
                }

                // Mode-specific keys
                match state.mode {
                    Mode::Orbit => match &logical_key {
                        Key::Character(c) if c.as_str() == "=" || c.as_str() == "+" => {
                            state.frequency += 1;
                            App::rebuild_sphere(state);
                        }
                        Key::Character(c) if c.as_str() == "-" => {
                            if state.frequency > 1 {
                                state.frequency -= 1;
                                App::rebuild_sphere(state);
                            }
                        }
                        _ => {}
                    },
                    Mode::Stiffness | Mode::Pretension => {}
                }
            }
            WindowEvent::MouseInput {
                state: button_state,
                button: MouseButton::Left,
                ..
            } => {
                if state.mode == Mode::Orbit {
                    match button_state {
                        ElementState::Pressed => {
                            state.camera.mouse_pressed(state.cursor_pos.0, state.cursor_pos.1);
                        }
                        ElementState::Released => {
                            state.camera.mouse_released();
                        }
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                state.cursor_pos = (position.x, position.y);
                if state.mode == Mode::Orbit {
                    state.camera.mouse_moved(position.x, position.y);
                }
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                state.modifiers = modifiers;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 50.0,
                };
                match state.mode {
                    Mode::Orbit => {
                        state.camera.scroll(scroll);
                    }
                    Mode::Stiffness => {
                        let factor = 1.2_f32.powf(scroll);
                        state.pull_k_at_1m = (state.pull_k_at_1m * factor).clamp(1e3, 1e10);
                        App::update_stiffness(state);
                    }
                    Mode::Pretension => {
                        // Scroll up (negative on macOS natural) = tighten
                        let factor = 1.03_f32.powf(scroll);
                        App::adjust_pretension(state, factor);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                // FPS tracking
                state.frame_count += 1;
                let fps_elapsed = state.fps_timer.elapsed().as_secs_f32();
                if fps_elapsed >= 0.5 {
                    state.last_fps = state.frame_count as f32 / fps_elapsed;
                    state.frame_count = 0;
                    state.fps_timer = Instant::now();
                    update_title(state);
                }
                state.last_frame = Instant::now();

                // Update HUD
                if state.show_hud {
                    let config = PhysicsConfig { pull_k_at_1m: state.pull_k_at_1m, ..PhysicsConfig::default() }
                        .scaled_for_frequency(state.frequency);

                    // Title: mode name + primary value
                    let paused_tag = if state.paused { "  PAUSED" } else { "" };
                    let title_value = match state.mode {
                        Mode::Orbit => format!(
                            "freq {}  {:.0} FPS{}", state.frequency, state.last_fps, paused_tag,
                        ),
                        Mode::Stiffness => format!(
                            "K = {:.2e} N/m{}", state.pull_k_at_1m, paused_tag,
                        ),
                        Mode::Pretension => format!(
                            "{:.1}%{}", (1.0 - state.pretension) * 100.0, paused_tag,
                        ),
                    };
                    state.hud.set_title(state.mode.name(), &title_value);

                    // Legend: mode-specific keys
                    match state.mode {
                        Mode::Orbit => {
                            state.hud.set_legend(&[
                                ("+/-", &format!("frequency  {}", state.frequency)),
                                ("scroll", "zoom"),
                                ("drag", "orbit"),
                                ("S", "stiffness mode"),
                                ("P", "pretension mode"),
                                ("Space", if state.paused { "resume" } else { "pause" }),
                                ("H", "hide HUD"),
                            ]);
                        }
                        Mode::Stiffness => {
                            state.hud.set_legend(&[
                                ("scroll", "adjust stiffness"),
                                ("S", "back to orbit"),
                                ("Esc", "back to orbit"),
                                ("Space", if state.paused { "resume" } else { "pause" }),
                            ]);
                            // Log-scale slider: K range 1e3..1e10
                            let t = (state.pull_k_at_1m.log10() - 3.0) / 7.0;
                            state.hud.set_slider(t, &format!("{:.1e}", state.pull_k_at_1m));
                        }
                        Mode::Pretension => {
                            state.hud.set_legend(&[
                                ("scroll", "adjust pretension"),
                                ("P", "back to orbit"),
                                ("Esc", "back to orbit"),
                                ("Space", if state.paused { "resume" } else { "pause" }),
                            ]);
                            // Linear slider: pretension 0.5..1.0 (50%..0%)
                            let t = (1.0 - state.pretension) / 0.5; // 0% → 0.0, 50% → 1.0
                            state.hud.set_slider(t, &format!("{:.1}%", (1.0 - state.pretension) * 100.0));
                        }
                    }

                    // Slider: hide in orbit mode
                    if state.mode == Mode::Orbit {
                        state.hud.hide_slider();
                    }

                    // Info: stats (bottom-right)
                    state.hud.set_info(&format!(
                        "{} joints  {} struts  {} cables\ndt {:.0}us  {} iter/frame",
                        state.buffers.num_joints(), state.buffers.num_rigid(), state.buffers.num_elastic(),
                        config.dt * 1e6, config.iterations_per_frame,
                    ));

                    let size = state.window.inner_size();
                    state.hud.prepare(&state.gpu.device, &state.gpu.queue, size.width, size.height);
                }

                // Physics
                if !state.paused {
                    let mut encoder = state.gpu.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("Physics Encoder"),
                        },
                    );
                    state.physics.dispatch(&mut encoder, state.iterations);

                    // Only do the blocking readback every N frames
                    let do_readback = state.physics_frame % READBACK_INTERVAL == 0;
                    if do_readback {
                        state.physics.copy_positions_to_staging(&mut encoder);
                    }
                    state.gpu.queue.submit(std::iter::once(encoder.finish()));

                    if do_readback {
                        let positions = state.physics.read_positions(&state.gpu.device);
                        state.renderer.update_instances(&state.gpu.device, &positions);

                        // Track centroid with gentle drift
                        if !positions.is_empty() {
                            let n = positions.len() as f32;
                            let mut cx = 0.0f32;
                            let mut cy = 0.0f32;
                            let mut cz = 0.0f32;
                            for p in &positions {
                                cx += p[0];
                                cy += p[1];
                                cz += p[2];
                            }
                            state.camera.track_target(glam::Vec3::new(cx / n, cy / n, cz / n));
                        }
                    }
                    state.physics_frame += 1;
                }

                // Update MVP
                let mvp = state.camera.mvp_matrix();
                state.renderer.update_mvp(&state.gpu.queue, &mvp);

                // Render
                let output = match state.gpu.surface.get_current_texture() {
                    Ok(output) => output,
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = state.window.inner_size();
                        state.gpu.resize(size.width, size.height);
                        return;
                    }
                    Err(e) => {
                        log::error!("Surface error: {e:?}");
                        return;
                    }
                };

                let view = output
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let mut encoder = state.gpu.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("Render Encoder"),
                    },
                );

                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.02,
                                    g: 0.02,
                                    b: 0.04,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &state.gpu.depth_texture_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    state.renderer.render(&mut render_pass);
                    if state.show_hud {
                        state.hud.render(&mut render_pass);
                    }
                }

                state.gpu.queue.submit(std::iter::once(encoder.finish()));
                output.present();

                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}
