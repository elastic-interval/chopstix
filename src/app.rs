use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::camera::Camera;
use crate::constants::*;
use crate::gpu::physics::{PhysicsCompute, PhysicsConfig};
use crate::gpu::renderer::Renderer;
use crate::gpu::Gpu;
use crate::tensegrity::{self, TensegritySphereBuffers};

struct AppState {
    window: Arc<Window>,
    gpu: Gpu,
    physics: PhysicsCompute,
    renderer: Renderer,
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
        let config = PhysicsConfig::default();
        let mut buffers = tensegrity::generate_sphere(state.frequency, SPHERE_RADIUS);
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
}

fn update_title(state: &AppState) {
    state.window.set_title(&format!(
        "Chopstix | freq={} | joints={} | struts={} | cables={} | {:.0} FPS",
        state.frequency,
        state.buffers.num_joints(),
        state.buffers.num_rigid(),
        state.buffers.num_elastic(),
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
        let config = PhysicsConfig::default();
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
        let camera = Camera::new(size.width as f32, size.height as f32, SPHERE_RADIUS * 2.8);

        let mut state = AppState {
            window,
            gpu,
            physics,
            renderer,
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
            } => match logical_key {
                Key::Named(NamedKey::Escape) => event_loop.exit(),
                Key::Named(NamedKey::Space) => {
                    state.paused = !state.paused;
                    log::info!("Physics {}", if state.paused { "paused" } else { "resumed" });
                }
                Key::Character(ref c) if c.as_str() == "=" || c.as_str() == "+" => {
                    state.frequency += 1;
                    log::info!("Frequency → {}", state.frequency);
                    App::rebuild_sphere(state);
                }
                Key::Character(ref c) if c.as_str() == "-" => {
                    if state.frequency > 1 {
                        state.frequency -= 1;
                        log::info!("Frequency → {}", state.frequency);
                        App::rebuild_sphere(state);
                    }
                }
                _ => {}
            },
            WindowEvent::MouseInput {
                state: button_state,
                button: MouseButton::Left,
                ..
            } => match button_state {
                ElementState::Pressed => {
                    state.camera.mouse_pressed(state.cursor_pos.0, state.cursor_pos.1);
                }
                ElementState::Released => {
                    state.camera.mouse_released();
                }
            },
            WindowEvent::CursorMoved { position, .. } => {
                state.cursor_pos = (position.x, position.y);
                state.camera.mouse_moved(position.x, position.y);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 50.0,
                };
                state.camera.scroll(scroll);
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

                        // Track centroid
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
