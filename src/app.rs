use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, Modifiers, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::camera::Camera;
use crate::constants::*;
use crate::gpu::hud::{Hud, FREQ_CHOICES, FREQ_BUTTON_WIDTH, FREQ_BAR_HEIGHT, FREQ_BAR_TOP,
    KLEIN_CHOICES, KLEIN_COLS, KLEIN_BUTTON_WIDTH, KLEIN_ROW_HEIGHT, KLEIN_GRID_TOP,
    MOBIUS_CHOICES, MOBIUS_BUTTON_WIDTH, MOBIUS_ROW_HEIGHT, MOBIUS_BAR_TOP,
    SURFACE_BUTTON_WIDTH, SURFACE_BAR_TOP, SURFACE_BAR_HEIGHT};
use crate::gpu::physics::{PhysicsCompute, PhysicsConfig, SURFACE_NAMES};
use crate::gpu::renderer::Renderer;
use crate::gpu::Gpu;
use crate::tensegrity::{self, TensegritySphereBuffers};
use crate::klein;
use crate::mobius;
use crate::twitcher::Twitcher;
use crate::ShapeConfig;


struct AppState {
    window: Arc<Window>,
    gpu: Gpu,
    physics: PhysicsCompute,
    renderer: Renderer,
    hud: Hud,
    camera: Camera,
    buffers: TensegritySphereBuffers,
    shape: ShapeConfig,
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
    surface_character: u32,
    show_hud: bool,
    /// Which slider is being dragged (0=stiffness, 1=pretension), or None
    dragging_slider: Option<usize>,
    /// Deferred shape rebuild — counts down frames before executing (0 = not pending)
    pending_rebuild: u32,
    /// Muscle twitching animation (active for Möbius)
    twitcher: Option<Twitcher>,
}

pub struct App {
    state: Option<AppState>,
    shape: ShapeConfig,
}

impl App {
    pub fn new(shape: ShapeConfig) -> Self {
        Self {
            state: None,
            shape,
        }
    }

    fn generate_shape(shape: &ShapeConfig, pull_k_at_1m: f32) -> TensegritySphereBuffers {
        match shape {
            ShapeConfig::Sphere { frequency } => {
                tensegrity::generate_sphere_with_k(*frequency, SPHERE_RADIUS, pull_k_at_1m)
            }
            ShapeConfig::Klein { width, height, shift } => {
                klein::generate_klein(*width, *height, *shift, pull_k_at_1m)
            }
            ShapeConfig::Mobius { segments } => {
                mobius::generate_mobius(*segments, pull_k_at_1m)
            }
        }
    }

    fn frequency_for_shape(shape: &ShapeConfig) -> usize {
        match shape {
            ShapeConfig::Sphere { frequency } => *frequency,
            ShapeConfig::Klein { width, height, .. } => {
                let joints = width * height / 2;
                ((joints as f32 / 10.0).sqrt() as usize).max(1)
            }
            ShapeConfig::Mobius { segments } => {
                // Möbius has fewer intervals per joint than a sphere, so intervals
                // are longer and need thinner rendering. Scale as if 3× the joints.
                let joints = (segments * 2 + 1) * 3;
                ((joints as f32 / 10.0).sqrt() as usize).max(1)
            }
        }
    }

    /// Build a PhysicsConfig appropriate for the current shape and surface.
    fn physics_config(shape: &ShapeConfig, pull_k_at_1m: f32, frequency: usize, surface_character: u32) -> PhysicsConfig {
        let mut config = PhysicsConfig {
            pull_k_at_1m,
            surface_character,
            ..PhysicsConfig::default()
        };
        if matches!(shape, ShapeConfig::Klein { .. } | ShapeConfig::Mobius { .. }) {
            config.gravity = 0.0;
            config.ground_y = -1e6;
        }
        config.scaled_for_frequency(frequency)
    }

    /// Whether the current shape uses approach-based settling
    fn uses_approach(shape: &ShapeConfig) -> bool {
        matches!(shape, ShapeConfig::Klein { .. } | ShapeConfig::Mobius { .. })
    }

    /// Create a twitcher appropriate for the shape, or None.
    fn create_twitcher(shape: &ShapeConfig, buffers: &TensegritySphereBuffers) -> Option<Twitcher> {
        match shape {
            ShapeConfig::Mobius { segments } => {
                let joint_count = segments * 2 + 1;
                Some(Twitcher::for_mobius(buffers.elastic_ideal.clone(), joint_count))
            }
            _ => None,
        }
    }

    /// Whether the ground grid should be visible
    fn show_ground(shape: &ShapeConfig, surface_character: u32) -> bool {
        // No ground for zero-gravity shapes or absent surface
        if matches!(shape, ShapeConfig::Klein { .. } | ShapeConfig::Mobius { .. }) {
            return false;
        }
        surface_character != 0 // 0 = Absent
    }

    /// Schedule a shape rebuild — clears display immediately, defers heavy work to next frame.
    fn schedule_rebuild(state: &mut AppState) {
        state.pending_rebuild = 2; // render one blank frame, then rebuild on the next
        state.paused = true;
        // Clear the display so the old shape disappears instantly
        state.renderer.update_instances(&state.gpu.device, &[], App::show_ground(&state.shape, state.surface_character));
    }

    fn rebuild_shape(state: &mut AppState) {
        let config = App::physics_config(&state.shape, state.pull_k_at_1m, state.frequency, state.surface_character);
        let mut buffers = App::generate_shape(&state.shape, state.pull_k_at_1m);

        // Settle: approach for random-start topologies, regular for spheres
        if App::uses_approach(&state.shape) {
            buffers.positions = PhysicsCompute::settle_with_approach(
                &state.gpu.device, &state.gpu.queue, &mut buffers, &config,
            );
            buffers.velocities = vec![[0.0f32; 4]; buffers.positions.len()];
        } else {
            buffers.positions = PhysicsCompute::settle(
                &state.gpu.device, &state.gpu.queue, &buffers, &config,
            );
        }

        state.iterations = config.iterations_per_frame;
        log::info!("Physics: dt={:.6}s, iterations={}, sim_time/frame={:.3}ms",
            config.dt, state.iterations, config.dt * state.iterations as f32 * 1000.0);
        state.physics = PhysicsCompute::new(&state.gpu.device, &state.gpu.queue, &buffers, &config);
        state.renderer.update_topology(&buffers, state.frequency);

        // Camera distance based on bounding radius
        let bounding_radius = compute_bounding_radius(&buffers.positions);
        state.camera.set_distance(bounding_radius * 2.8);

        state.twitcher = App::create_twitcher(&state.shape, &buffers);
        state.buffers = buffers;
        state.renderer.update_instances(&state.gpu.device, &state.buffers.positions, App::show_ground(&state.shape, state.surface_character));
        update_title(state);
    }

    /// Update cable stiffness without resettling — keeps current positions/velocities.
    fn update_stiffness(state: &mut AppState) {
        let config = App::physics_config(&state.shape, state.pull_k_at_1m, state.frequency, state.surface_character);

        // Read back current positions from GPU
        let mut encoder = state.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Stiffness readback"),
        });
        state.physics.copy_positions_to_staging(&mut encoder);
        state.gpu.queue.submit(std::iter::once(encoder.finish()));
        let current_positions = state.physics.read_positions(&state.gpu.device);

        // Recompute K values from existing ideal lengths — no regeneration
        for (k, ideal) in state.buffers.elastic_k.iter_mut().zip(state.buffers.elastic_ideal.iter()) {
            *k = state.pull_k_at_1m / ideal;
        }
        for (k, ideal) in state.buffers.push_k.iter_mut().zip(state.buffers.push_ideal.iter()) {
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

        let config = App::physics_config(&state.shape, state.pull_k_at_1m, state.frequency, state.surface_character);

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
        // Also scale push ideal lengths for Klein
        for i in 0..state.buffers.push_ideal.len() {
            state.buffers.push_ideal[i] *= factor;
            state.buffers.push_k[i] = state.pull_k_at_1m / state.buffers.push_ideal[i];
        }
        state.buffers.positions = current_positions;

        state.iterations = config.iterations_per_frame;
        state.physics = PhysicsCompute::new(&state.gpu.device, &state.gpu.queue, &state.buffers, &config);
        update_title(state);
    }

    /// Returns which frequency button the cursor is over, if any.
    fn freq_bar_hover_index(state: &AppState) -> Option<usize> {
        let bar_left = state.hud.freq_bar_left() as f64;
        let (cx, cy) = state.cursor_pos;
        if cy < FREQ_BAR_TOP as f64 || cy > (FREQ_BAR_TOP + FREQ_BAR_HEIGHT) as f64 {
            return None;
        }
        if cx < bar_left {
            return None;
        }
        let idx = ((cx - bar_left) / FREQ_BUTTON_WIDTH as f64) as usize;
        if idx < FREQ_CHOICES.len() {
            Some(idx)
        } else {
            None
        }
    }

    /// Returns the frequency at the cursor position, if clicking on the freq bar.
    fn freq_bar_hit(state: &AppState) -> Option<usize> {
        App::freq_bar_hover_index(state).map(|i| FREQ_CHOICES[i])
    }

    /// Returns the Klein button index the cursor is over, if any.
    fn klein_hover_index(state: &AppState) -> Option<usize> {
        let bar_left = state.hud.klein_bar_left() as f64;
        let (cx, cy) = state.cursor_pos;
        if cy < KLEIN_GRID_TOP as f64 || cy > (KLEIN_GRID_TOP + KLEIN_ROW_HEIGHT) as f64 {
            return None;
        }
        if cx < bar_left { return None; }
        let idx = ((cx - bar_left) / KLEIN_BUTTON_WIDTH as f64) as usize;
        if idx < KLEIN_COLS { Some(idx) } else { None }
    }

    fn klein_hit(state: &AppState) -> Option<(usize, usize)> {
        App::klein_hover_index(state).map(|i| KLEIN_CHOICES[i])
    }

    /// Returns the Möbius button index the cursor is over, if any.
    fn mobius_hover_index(state: &AppState) -> Option<usize> {
        let bar_left = state.hud.mobius_bar_left() as f64;
        let (cx, cy) = state.cursor_pos;
        if cy < MOBIUS_BAR_TOP as f64 || cy > (MOBIUS_BAR_TOP + MOBIUS_ROW_HEIGHT) as f64 {
            return None;
        }
        if cx < bar_left { return None; }
        let idx = ((cx - bar_left) / MOBIUS_BUTTON_WIDTH as f64) as usize;
        if idx < MOBIUS_CHOICES.len() { Some(idx) } else { None }
    }

    fn mobius_hit(state: &AppState) -> Option<usize> {
        App::mobius_hover_index(state).map(|i| MOBIUS_CHOICES[i])
    }

    /// Returns the surface button index the cursor is over, if any.
    fn surface_hover_index(state: &AppState) -> Option<usize> {
        let bar_left = state.hud.surface_bar_left() as f64;
        let (cx, cy) = state.cursor_pos;
        if cy < SURFACE_BAR_TOP as f64 || cy > (SURFACE_BAR_TOP + SURFACE_BAR_HEIGHT) as f64 {
            return None;
        }
        if cx < bar_left { return None; }
        let idx = ((cx - bar_left) / SURFACE_BUTTON_WIDTH as f64) as usize;
        if idx < 5 { Some(idx) } else { None }
    }

    /// Apply slider position from current cursor Y.
    fn apply_slider_drag(state: &mut AppState) {
        let Some(col) = state.dragging_slider else { return };
        let t = state.hud.slider_y_to_t(state.cursor_pos.1);
        match col {
            0 => {
                // Stiffness: log scale 1e3..1e10
                state.pull_k_at_1m = 10.0_f32.powf(3.0 + t * 7.0).clamp(1e3, 1e10);
                App::update_stiffness(state);
            }
            1 => {
                // Pretension: t=0 → 1.0 (0%), t=1 → 0.5 (50%)
                let new_pretension = (1.0 - t * 0.5).clamp(0.5, 1.0);
                let factor = new_pretension / state.pretension;
                App::adjust_pretension(state, factor);
            }
            _ => {}
        }
    }
}

fn compute_bounding_radius(positions: &[[f32; 4]]) -> f32 {
    if positions.is_empty() {
        return SPHERE_RADIUS;
    }
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

fn update_title(state: &AppState) {
    let shape_tag = match &state.shape {
        ShapeConfig::Sphere { frequency } => format!("sphere freq={}", frequency),
        ShapeConfig::Klein { width, height, .. } => format!("klein {}x{}", width, height),
        ShapeConfig::Mobius { segments } => format!("mobius seg={}", segments),
    };
    let surface_tag = SURFACE_NAMES[state.surface_character as usize];
    let num_struts = state.buffers.num_rigid() + state.buffers.num_push();
    state.window.set_title(&format!(
        "Chopstix | {} | {} | joints={} | struts={} | cables={} | K={:.0e} | {:.0} FPS",
        shape_tag,
        surface_tag,
        state.buffers.num_joints(),
        num_struts,
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
        let frequency = App::frequency_for_shape(&self.shape);
        let initial_surface = match &self.shape {
            ShapeConfig::Sphere { .. } => 1, // Bouncy
            _ => 0, // Absent
        };
        let config = App::physics_config(&self.shape, PULL_K_AT_1M, frequency, initial_surface);
        let mut buffers = App::generate_shape(&self.shape, PULL_K_AT_1M);

        // Settle
        if App::uses_approach(&self.shape) {
            buffers.positions = PhysicsCompute::settle_with_approach(
                &gpu.device, &gpu.queue, &mut buffers, &config,
            );
            buffers.velocities = vec![[0.0f32; 4]; buffers.positions.len()];
        } else {
            buffers.positions = PhysicsCompute::settle(
                &gpu.device, &gpu.queue, &buffers, &config,
            );
        }

        let iterations = config.iterations_per_frame;
        log::info!("Physics: dt={:.6}s, iterations={}, sim_time/frame={:.3}ms",
            config.dt, iterations, config.dt * iterations as f32 * 1000.0);
        let physics = PhysicsCompute::new(&gpu.device, &gpu.queue, &buffers, &config);
        let size = window.inner_size();
        let renderer = Renderer::new(&gpu, &buffers, frequency);
        let hud = Hud::new(&gpu.device, &gpu.queue, gpu.surface_config.format);

        let bounding_radius = compute_bounding_radius(&buffers.positions);
        let camera = Camera::new(size.width as f32, size.height as f32, bounding_radius * 2.8);

        let mut state = AppState {
            window,
            gpu,
            physics,
            renderer,
            hud,
            camera,
            buffers,
            shape: self.shape.clone(),
            frequency,
            iterations,
            paused: false,
            last_frame: Instant::now(),
            frame_count: 0,
            fps_timer: Instant::now(),
            last_fps: 0.0,
            cursor_pos: (0.0, 0.0),
            physics_frame: 0,
            modifiers: Modifiers::default(),
            pull_k_at_1m: PULL_K_AT_1M,
            pretension: 0.95,
            surface_character: initial_surface,
            show_hud: true,
            dragging_slider: None,
            pending_rebuild: 0,
            twitcher: None,
        };
        state.renderer.update_instances(&state.gpu.device, &state.buffers.positions, App::show_ground(&state.shape, state.surface_character));
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
                match &logical_key {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Named(NamedKey::Space) => {
                        state.paused = !state.paused;
                    }
                    Key::Character(c) if c.as_str() == "h" => {
                        state.show_hud = !state.show_hud;
                    }
                    Key::Named(NamedKey::Enter) => {
                        // Regenerate current shape (new random seed for Klein/Möbius)
                        App::schedule_rebuild(state);
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseInput {
                state: button_state,
                button: MouseButton::Left,
                ..
            } => match button_state {
                ElementState::Pressed => {
                    // Sphere frequency bar — click always switches to sphere
                    if let Some(freq) = App::freq_bar_hit(state) {
                        let already = matches!(state.shape, ShapeConfig::Sphere { frequency: f } if f == freq);
                        if !already {
                            state.shape = ShapeConfig::Sphere { frequency: freq };
                            state.frequency = App::frequency_for_shape(&state.shape);
                            App::schedule_rebuild(state);
                            return;
                        }
                    }
                    // Klein grid — click always switches to klein
                    if let Some((w, h)) = App::klein_hit(state) {
                        let already = matches!(state.shape, ShapeConfig::Klein { width, height, .. } if width == w && height == h);
                        if !already {
                            state.shape = ShapeConfig::Klein { width: w, height: h, shift: 0 };
                            state.frequency = App::frequency_for_shape(&state.shape);
                            App::schedule_rebuild(state);
                            return;
                        }
                    }
                    // Möbius bar — click always switches to möbius
                    if let Some(seg) = App::mobius_hit(state) {
                        let already = matches!(state.shape, ShapeConfig::Mobius { segments } if segments == seg);
                        if !already {
                            state.shape = ShapeConfig::Mobius { segments: seg };
                            state.frequency = App::frequency_for_shape(&state.shape);
                            App::schedule_rebuild(state);
                            return;
                        }
                    }
                    // Surface character bar
                    if let Some(idx) = App::surface_hover_index(state) {
                        let new_char = idx as u32;
                        if new_char != state.surface_character {
                            state.surface_character = new_char;
                            App::update_stiffness(state); // rebuilds physics with new surface
                            return;
                        }
                    }
                    if let Some(col) = state.hud.slider_hit(state.cursor_pos.0) {
                        state.dragging_slider = Some(col);
                        App::apply_slider_drag(state);
                    } else {
                        state.camera.mouse_pressed(state.cursor_pos.0, state.cursor_pos.1);
                    }
                }
                ElementState::Released => {
                    state.dragging_slider = None;
                    state.camera.mouse_released();
                }
            },
            WindowEvent::CursorMoved { position, .. } => {
                state.cursor_pos = (position.x, position.y);
                if state.dragging_slider.is_some() {
                    App::apply_slider_drag(state);
                } else {
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
                // Scroll near the slider adjusts the parameter; elsewhere zooms
                match state.hud.slider_hit(state.cursor_pos.0) {
                    Some(0) => {
                        // Stiffness slider
                        let factor = 1.2_f32.powf(scroll);
                        state.pull_k_at_1m = (state.pull_k_at_1m * factor).clamp(1e3, 1e10);
                        App::update_stiffness(state);
                    }
                    Some(1) => {
                        // Pretension slider
                        let factor = 1.03_f32.powf(scroll);
                        App::adjust_pretension(state, factor);
                    }
                    _ => {
                        state.camera.scroll(scroll);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                // Execute deferred rebuild: count down frames, rebuild when it hits 1
                if state.pending_rebuild > 0 {
                    state.pending_rebuild -= 1;
                    if state.pending_rebuild == 0 {
                        App::rebuild_shape(state);
                        state.paused = false;
                    }
                }

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
                    let config = App::physics_config(&state.shape, state.pull_k_at_1m, state.frequency, state.surface_character);
                    let paused_tag = if state.paused { "  PAUSED" } else { "" };

                    // Title
                    state.hud.set_title("CHOPSTIX", &format!(
                        "{:.0} FPS{}", state.last_fps, paused_tag,
                    ));

                    // Both shape selection bars always visible
                    let sphere_freq = match &state.shape {
                        ShapeConfig::Sphere { frequency } => *frequency,
                        _ => 0, // nothing highlighted
                    };
                    let hover_idx = App::freq_bar_hover_index(state);
                    state.hud.set_freq_bar(sphere_freq, hover_idx);

                    // Centered shape description
                    let shape_title = match &state.shape {
                        ShapeConfig::Sphere { frequency } => format!("Geodesic Sphere  freq {}", frequency),
                        ShapeConfig::Klein { width, height, .. } => format!("Klein Bottle  {}x{}", width, height),
                        ShapeConfig::Mobius { segments } => format!("Mobius Band  {} segments", segments),
                    };
                    state.hud.set_shape_title(&shape_title);

                    let (klein_w, klein_h) = match &state.shape {
                        ShapeConfig::Klein { width, height, .. } => (*width, *height),
                        _ => (0, 0), // nothing highlighted
                    };
                    let klein_hover = App::klein_hover_index(state);
                    state.hud.set_klein_bar(klein_w, klein_h, klein_hover);

                    // Möbius bar
                    let mobius_seg = match &state.shape {
                        ShapeConfig::Mobius { segments } => *segments,
                        _ => 0,
                    };
                    let mobius_hover = App::mobius_hover_index(state);
                    state.hud.set_mobius_bar(mobius_seg, mobius_hover);

                    // Surface character bar
                    let surface_hover = App::surface_hover_index(state);
                    state.hud.set_surface_bar(state.surface_character, surface_hover);

                    // Legend
                    state.hud.set_legend(&[
                        ("Space", if state.paused { "resume" } else { "pause" }),
                        ("H", "hide HUD"),
                    ]);

                    // Both sliders always visible, highlight on hover
                    let hover_col = state.hud.slider_hit(state.cursor_pos.0);
                    let k_t = (state.pull_k_at_1m.log10() - 3.0) / 7.0;
                    state.hud.set_stiffness_slider(k_t, &format!("K {:.0e}", state.pull_k_at_1m),
                        hover_col == Some(0) || state.dragging_slider == Some(0));
                    let p_t = (1.0 - state.pretension) / 0.5;
                    state.hud.set_pretension_slider(p_t, &format!("P {:.0}%", (1.0 - state.pretension) * 100.0),
                        hover_col == Some(1) || state.dragging_slider == Some(1));

                    // Info: stats (bottom-right)
                    let num_struts = state.buffers.num_rigid() + state.buffers.num_push();
                    let mode_tag = if state.buffers.use_spring_push { "spring" } else { "SHAKE" };
                    state.hud.set_info(&format!(
                        "{} joints  {} struts  {} cables  [{}]\ndt {:.0}us  {} iter/frame",
                        state.buffers.num_joints(), num_struts, state.buffers.num_elastic(),
                        mode_tag,
                        config.dt * 1e6, config.iterations_per_frame,
                    ));

                    let size = state.window.inner_size();
                    state.hud.prepare(&state.gpu.device, &state.gpu.queue, size.width, size.height);
                }

                // Physics
                if !state.paused {
                    // Update muscle twitching before physics dispatch
                    if let Some(ref mut twitcher) = state.twitcher {
                        if twitcher.tick() {
                            let (ideals, ks) = twitcher.current_ideals(state.pull_k_at_1m);
                            state.physics.write_elastic_ideals(&state.gpu.queue, &ideals, &ks);
                        }
                    }

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
                        state.renderer.update_instances(&state.gpu.device, &positions, App::show_ground(&state.shape, state.surface_character));

                        // Check if physics violated the speed limit
                        if state.physics.read_frozen(&state.gpu.device) {
                            state.paused = true;
                            log::warn!("Speed limit exceeded — simulation frozen");
                        }

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
