use wgpu::util::DeviceExt;

use crate::constants::*;
use crate::tensegrity::TensegritySphereBuffers;

/// Runtime-configurable physics parameters.
/// Constants in `constants.rs` become the defaults.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct PhysicsConfig {
    pub dt: f32,
    pub iterations_per_frame: u32,
    pub pull_k_at_1m: f32,
    pub force_scale: f32,
    pub drag: f32,
    pub speed_limit: f32,
    pub settle_iterations: u32,
    pub settle_drag: f32,
    pub ambient_mass: f32,
    pub gravity: f32,
    pub ground_y: f32,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            dt: ITERATION_DT,
            iterations_per_frame: ITERATIONS_PER_FRAME,
            pull_k_at_1m: PULL_K_AT_1M,
            force_scale: FORCE_SCALE,
            drag: DRAG,
            speed_limit: SPEED_LIMIT,
            settle_iterations: SETTLE_ITERATIONS,
            settle_drag: 100.0,
            ambient_mass: JOINT_AMBIENT_MASS,
            gravity: GRAVITY,
            ground_y: GROUND_Y,
        }
    }
}

impl PhysicsConfig {
    /// Scale dt and iterations for a given geodesic frequency.
    ///
    /// Higher frequency → shorter cables → stiffer system → smaller dt needed.
    /// Baseline is stable through ~freq 30, so scaling only kicks in above
    /// a reference frequency (20). Below that, no penalty.
    /// Above it, dt scales as 1/sqrt(freq/ref) with iterations scaled up to match.
    const FREQ_REF: f32 = 20.0;

    pub fn scaled_for_frequency(mut self, frequency: usize) -> Self {
        let f = frequency as f32;
        if f > Self::FREQ_REF {
            let scale = (f / Self::FREQ_REF).sqrt();
            self.dt /= scale;
            self.iterations_per_frame = (self.iterations_per_frame as f32 * scale).ceil() as u32;
        }
        self
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PhysicsParams {
    dt: f32,
    gravity: f32,
    drag: f32,
    viscosity: f32,
    num_joints: u32,
    num_elastic: u32,
    num_rigid: u32,
    ambient_mass: f32,
    force_scale: f32,
    ground_y: f32,
    restitution: f32,
    speed_limit: f32,
    num_push: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

pub struct PhysicsCompute {
    joint_bind_group: wgpu::BindGroup,
    interval_bind_group: wgpu::BindGroup,
    params_bind_group: wgpu::BindGroup,
    push_bind_group: wgpu::BindGroup,
    half_kick_pipeline: wgpu::ComputePipeline,
    elastic_forces_pipeline: wgpu::ComputePipeline,
    rigid_mass_pipeline: wgpu::ComputePipeline,
    second_half_kick_pipeline: wgpu::ComputePipeline,
    shake_pipeline: wgpu::ComputePipeline,
    rattle_pipeline: wgpu::ComputePipeline,
    ground_collision_pipeline: wgpu::ComputePipeline,
    push_forces_pipeline: wgpu::ComputePipeline,
    position_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    num_joints: u32,
    num_elastic: u32,
    num_rigid: u32,
    num_push: u32,
    use_spring_push: bool,
}

impl PhysicsCompute {
    /// Run a settling phase: high drag, no gravity, no ground.
    /// Returns settled positions ready for the real simulation.
    pub fn settle(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffers: &TensegritySphereBuffers,
        config: &PhysicsConfig,
    ) -> Vec<[f32; 4]> {
        let num_joints = buffers.num_joints();
        let settling_params = PhysicsParams {
            dt: config.dt,
            gravity: 0.0,           // no gravity during settling
            drag: config.settle_drag,
            viscosity: 0.0,
            num_joints,
            num_elastic: buffers.num_elastic(),
            num_rigid: buffers.num_rigid(),
            ambient_mass: config.ambient_mass,
            force_scale: config.force_scale,
            ground_y: -1e6,         // ground far away — irrelevant
            restitution: 0.0,
            speed_limit: f32::MAX,  // no speed limit during settling
            num_push: buffers.num_push(),
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let physics = Self::with_params(device, buffers, settling_params);

        // Batch settling into chunks — with single compute pass this can be larger
        let chunk = 200;
        let mut remaining = config.settle_iterations;
        while remaining > 0 {
            let n = remaining.min(chunk);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Settle"),
            });
            physics.dispatch(&mut encoder, n);
            queue.submit(std::iter::once(encoder.finish()));
            device.poll(wgpu::PollType::Wait).unwrap();
            remaining -= n;
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Settle readback"),
        });
        physics.copy_positions_to_staging(&mut encoder);
        queue.submit(std::iter::once(encoder.finish()));
        physics.read_positions(device)
    }

    /// Approach-based settling for Klein bottles (and other random-start topologies).
    /// Ideal lengths interpolate from actual → target over a small number of steps
    /// to avoid violent forces from random initial placement.
    ///
    /// 20 approach steps × 2000 iterations each = 40k total approach iterations,
    /// then 5000 final iterations at target lengths. Each step rebuilds the
    /// PhysicsCompute with updated ideal lengths.
    pub fn settle_with_approach(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffers: &mut TensegritySphereBuffers,
        config: &PhysicsConfig,
    ) -> Vec<[f32; 4]> {
        // Save target ideal lengths
        let target_elastic_ideal = buffers.elastic_ideal.clone();
        let target_push_ideal = buffers.push_ideal.clone();

        // Compute initial actual lengths for all intervals
        let initial_elastic_actual: Vec<f32> = (0..buffers.elastic_alpha.len())
            .map(|i| {
                let a = buffers.elastic_alpha[i] as usize;
                let o = buffers.elastic_omega[i] as usize;
                let pa = buffers.positions[a];
                let po = buffers.positions[o];
                let dx = po[0] - pa[0];
                let dy = po[1] - pa[1];
                let dz = po[2] - pa[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .collect();
        let initial_push_actual: Vec<f32> = (0..buffers.push_alpha.len())
            .map(|i| {
                let a = buffers.push_alpha[i] as usize;
                let o = buffers.push_omega[i] as usize;
                let pa = buffers.positions[a];
                let po = buffers.positions[o];
                let dx = po[0] - pa[0];
                let dy = po[1] - pa[1];
                let dz = po[2] - pa[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .collect();

        let num_steps: u32 = 20;
        let iters_per_step: u32 = 2000;
        let gpu_chunk: u32 = 200; // max iterations per GPU submission

        for step in 0..num_steps {
            let progress = (step as f32 + 1.0) / num_steps as f32;

            // Interpolate ideal lengths: initial_actual → target
            for i in 0..buffers.elastic_ideal.len() {
                buffers.elastic_ideal[i] = initial_elastic_actual[i]
                    + (target_elastic_ideal[i] - initial_elastic_actual[i]) * progress;
                buffers.elastic_k[i] = config.pull_k_at_1m / buffers.elastic_ideal[i];
            }
            for i in 0..buffers.push_ideal.len() {
                buffers.push_ideal[i] = initial_push_actual[i]
                    + (target_push_ideal[i] - initial_push_actual[i]) * progress;
                buffers.push_k[i] = config.pull_k_at_1m / buffers.push_ideal[i];
            }

            let settling_params = PhysicsParams {
                dt: config.dt,
                gravity: 0.0,
                drag: config.settle_drag,
                viscosity: 0.0,
                num_joints: buffers.num_joints(),
                num_elastic: buffers.num_elastic(),
                num_rigid: buffers.num_rigid(),
                ambient_mass: config.ambient_mass,
                force_scale: config.force_scale,
                ground_y: -1e6,
                restitution: 0.0,
                speed_limit: f32::MAX,
                num_push: buffers.num_push(),
                _pad1: 0,
                _pad2: 0,
                _pad3: 0,
            };
            let physics = Self::with_params(device, buffers, settling_params);

            // Run this step's iterations in GPU chunks
            let mut remaining = iters_per_step;
            while remaining > 0 {
                let n = remaining.min(gpu_chunk);
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Approach settle"),
                });
                physics.dispatch(&mut encoder, n);
                queue.submit(std::iter::once(encoder.finish()));
                device.poll(wgpu::PollType::Wait).unwrap();
                remaining -= n;
            }

            // Read back positions for the next step's buffer rebuild
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Approach readback"),
            });
            physics.copy_positions_to_staging(&mut encoder);
            queue.submit(std::iter::once(encoder.finish()));
            buffers.positions = physics.read_positions(device);
            buffers.velocities = vec![[0.0f32; 4]; buffers.positions.len()];

            log::info!("Approach settling: {:.0}%", progress * 100.0);
        }

        // Restore final target ideal lengths
        buffers.elastic_ideal = target_elastic_ideal;
        buffers.push_ideal = target_push_ideal;
        for i in 0..buffers.elastic_k.len() {
            buffers.elastic_k[i] = config.pull_k_at_1m / buffers.elastic_ideal[i];
        }
        for i in 0..buffers.push_k.len() {
            buffers.push_k[i] = config.pull_k_at_1m / buffers.push_ideal[i];
        }

        // Final settling with target lengths
        let settling_params = PhysicsParams {
            dt: config.dt,
            gravity: 0.0,
            drag: config.settle_drag,
            viscosity: 0.0,
            num_joints: buffers.num_joints(),
            num_elastic: buffers.num_elastic(),
            num_rigid: buffers.num_rigid(),
            ambient_mass: config.ambient_mass,
            force_scale: config.force_scale,
            ground_y: -1e6,
            restitution: 0.0,
            speed_limit: f32::MAX,
            num_push: buffers.num_push(),
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let physics = Self::with_params(device, buffers, settling_params);
        let final_settle: u32 = 5000;
        let mut remaining = final_settle;
        while remaining > 0 {
            let n = remaining.min(gpu_chunk);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Final settle"),
            });
            physics.dispatch(&mut encoder, n);
            queue.submit(std::iter::once(encoder.finish()));
            device.poll(wgpu::PollType::Wait).unwrap();
            remaining -= n;
        }
        log::info!("Approach settling complete");

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Settle readback"),
        });
        physics.copy_positions_to_staging(&mut encoder);
        queue.submit(std::iter::once(encoder.finish()));
        physics.read_positions(device)
    }

    pub fn new(device: &wgpu::Device, _queue: &wgpu::Queue, buffers: &TensegritySphereBuffers, config: &PhysicsConfig) -> Self {
        let params = PhysicsParams {
            dt: config.dt,
            gravity: config.gravity,
            drag: config.drag,
            viscosity: VISCOSITY,
            num_joints: buffers.num_joints(),
            num_elastic: buffers.num_elastic(),
            num_rigid: buffers.num_rigid(),
            ambient_mass: config.ambient_mass,
            force_scale: config.force_scale,
            ground_y: config.ground_y,
            restitution: RESTITUTION,
            speed_limit: config.speed_limit,
            num_push: buffers.num_push(),
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        Self::with_params(device, buffers, params)
    }

    fn with_params(
        device: &wgpu::Device,
        buffers: &TensegritySphereBuffers,
        params: PhysicsParams,
    ) -> Self {
        let num_joints = params.num_joints;
        let num_elastic = params.num_elastic;
        let num_rigid = params.num_rigid;
        let num_push = params.num_push;
        let use_spring_push = buffers.use_spring_push;

        // Create GPU buffers
        let position_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Positions"),
            contents: bytemuck::cast_slice(&buffers.positions),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let velocity_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Velocities"),
            contents: bytemuck::cast_slice(&buffers.velocities),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Force accumulators (atomic i32)
        let zero_i32s = vec![0i32; num_joints as usize];
        let force_x_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Force X"),
            contents: bytemuck::cast_slice(&zero_i32s),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let force_y_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Force Y"),
            contents: bytemuck::cast_slice(&zero_i32s),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let force_z_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Force Z"),
            contents: bytemuck::cast_slice(&zero_i32s),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Mass accumulator (atomic i32, initialized to ambient_mass * MASS_SCALE)
        let ambient_i32 = (params.ambient_mass * 1e4) as i32;
        let mass_init = vec![ambient_i32; num_joints as usize];
        let mass_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Masses"),
            contents: bytemuck::cast_slice(&mass_init),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Elastic interval buffers
        let elastic_alpha_buf = create_storage_buffer(device, "Elastic Alpha", bytemuck::cast_slice(&buffers.elastic_alpha));
        let elastic_omega_buf = create_storage_buffer(device, "Elastic Omega", bytemuck::cast_slice(&buffers.elastic_omega));
        let elastic_ideal_buf = create_storage_buffer(device, "Elastic Ideal", bytemuck::cast_slice(&buffers.elastic_ideal));
        let elastic_k_buf = create_storage_buffer(device, "Elastic K", bytemuck::cast_slice(&buffers.elastic_k));

        // Rigid interval buffers
        let rigid_alpha_buf = create_storage_buffer(device, "Rigid Alpha", bytemuck::cast_slice(&buffers.rigid_alpha));
        let rigid_omega_buf = create_storage_buffer(device, "Rigid Omega", bytemuck::cast_slice(&buffers.rigid_omega));
        let rigid_length_buf = create_storage_buffer(device, "Rigid Length", bytemuck::cast_slice(&buffers.rigid_length));
        let rigid_half_mass_buf = create_storage_buffer(device, "Rigid Half Mass", bytemuck::cast_slice(&buffers.rigid_half_mass));

        // Push interval buffers (spring-based push for Klein etc.)
        let push_alpha_buf = create_storage_buffer(device, "Push Alpha", bytemuck::cast_slice(&buffers.push_alpha));
        let push_omega_buf = create_storage_buffer(device, "Push Omega", bytemuck::cast_slice(&buffers.push_omega));
        let push_ideal_buf = create_storage_buffer(device, "Push Ideal", bytemuck::cast_slice(&buffers.push_ideal));
        let push_k_buf = create_storage_buffer(device, "Push K", bytemuck::cast_slice(&buffers.push_k));
        let push_half_mass_buf = create_storage_buffer(device, "Push Half Mass", bytemuck::cast_slice(&buffers.push_half_mass));

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Physics Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Staging buffer for position readback
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size: (num_joints as u64) * 16, // vec4<f32>
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group layouts
        let joint_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Joint BGL"),
            entries: &[
                storage_entry(0, false), // positions
                storage_entry(1, false), // velocities
                storage_entry(2, false), // force_x
                storage_entry(3, false), // force_y
                storage_entry(4, false), // force_z
                storage_entry(5, false), // masses
            ],
        });

        let interval_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Interval BGL"),
            entries: &[
                storage_entry(0, true), // elastic_alpha
                storage_entry(1, true), // elastic_omega
                storage_entry(2, true), // elastic_ideal
                storage_entry(3, true), // elastic_k
                storage_entry(4, true), // rigid_alpha
                storage_entry(5, true), // rigid_omega
                storage_entry(6, true), // rigid_length
                storage_entry(7, true), // rigid_half_mass
            ],
        });

        let params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Params BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let push_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Push BGL"),
            entries: &[
                storage_entry(0, true), // push_alpha
                storage_entry(1, true), // push_omega
                storage_entry(2, true), // push_ideal
                storage_entry(3, true), // push_k
                storage_entry(4, true), // push_half_mass
            ],
        });

        // Bind groups
        let joint_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Joint BG"),
            layout: &joint_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: position_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: velocity_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: force_x_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: force_y_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: force_z_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: mass_buffer.as_entire_binding() },
            ],
        });

        let interval_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Interval BG"),
            layout: &interval_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: elastic_alpha_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: elastic_omega_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: elastic_ideal_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: elastic_k_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: rigid_alpha_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: rigid_omega_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: rigid_length_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: rigid_half_mass_buf.as_entire_binding() },
            ],
        });

        let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Params BG"),
            layout: &params_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
            ],
        });

        let push_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Push BG"),
            layout: &push_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: push_alpha_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: push_omega_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: push_ideal_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: push_k_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: push_half_mass_buf.as_entire_binding() },
            ],
        });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Physics Pipeline Layout"),
            bind_group_layouts: &[&joint_bgl, &interval_bgl, &params_bgl, &push_bgl],
            push_constant_ranges: &[],
        });

        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Physics Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("physics.wgsl").into()),
        });

        let make_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let half_kick_pipeline = make_pipeline("half_kick_and_drift");
        let elastic_forces_pipeline = make_pipeline("elastic_forces");
        let rigid_mass_pipeline = make_pipeline("rigid_mass");
        let second_half_kick_pipeline = make_pipeline("second_half_kick");
        let shake_pipeline = make_pipeline("shake_constraints");
        let rattle_pipeline = make_pipeline("rattle_constraints");
        let ground_collision_pipeline = make_pipeline("ground_collision");
        let push_forces_pipeline = make_pipeline("push_forces");

        Self {
            joint_bind_group,
            interval_bind_group,
            params_bind_group,
            push_bind_group,
            half_kick_pipeline,
            elastic_forces_pipeline,
            rigid_mass_pipeline,
            second_half_kick_pipeline,
            shake_pipeline,
            rattle_pipeline,
            ground_collision_pipeline,
            push_forces_pipeline,
            position_buffer,
            staging_buffer,
            num_joints,
            num_elastic,
            num_rigid,
            num_push,
            use_spring_push,
        }
    }

    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, iterations: u32) {
        let joint_groups = (self.num_joints + 63) / 64;
        let elastic_groups = if self.num_elastic > 0 { (self.num_elastic + 63) / 64 } else { 0 };
        let rigid_groups = if self.num_rigid > 0 { (self.num_rigid + 63) / 64 } else { 0 };
        let push_groups = if self.num_push > 0 { (self.num_push + 63) / 64 } else { 0 };

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Physics"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.joint_bind_group, &[]);
        pass.set_bind_group(1, &self.interval_bind_group, &[]);
        pass.set_bind_group(2, &self.params_bind_group, &[]);
        pass.set_bind_group(3, &self.push_bind_group, &[]);

        if self.use_spring_push {
            // Spring-push mode (Klein bottles etc.): no SHAKE/RATTLE needed
            // 1. half_kick_and_drift
            // 2. elastic_forces (cables)
            // 3. push_forces (struts as springs)
            // 4. second_half_kick (includes force reset)
            // 5. ground_collision
            for _ in 0..iterations {
                pass.set_pipeline(&self.half_kick_pipeline);
                pass.dispatch_workgroups(joint_groups, 1, 1);

                if elastic_groups > 0 {
                    pass.set_pipeline(&self.elastic_forces_pipeline);
                    pass.dispatch_workgroups(elastic_groups, 1, 1);
                }

                if push_groups > 0 {
                    pass.set_pipeline(&self.push_forces_pipeline);
                    pass.dispatch_workgroups(push_groups, 1, 1);
                }

                pass.set_pipeline(&self.second_half_kick_pipeline);
                pass.dispatch_workgroups(joint_groups, 1, 1);

                pass.set_pipeline(&self.ground_collision_pipeline);
                pass.dispatch_workgroups(joint_groups, 1, 1);
            }
        } else {
            // SHAKE/RATTLE mode (geodesic spheres): rigid constraints
            for _ in 0..iterations {
                // 1: Half kick + drift
                pass.set_pipeline(&self.half_kick_pipeline);
                pass.dispatch_workgroups(joint_groups, 1, 1);

                // 2: SHAKE — correct positions to maintain rigid strut lengths
                if rigid_groups > 0 {
                    pass.set_pipeline(&self.shake_pipeline);
                    pass.dispatch_workgroups(rigid_groups, 1, 1);
                }

                // 3: Elastic forces
                if elastic_groups > 0 {
                    pass.set_pipeline(&self.elastic_forces_pipeline);
                    pass.dispatch_workgroups(elastic_groups, 1, 1);
                }

                // 4: Rigid mass
                if rigid_groups > 0 {
                    pass.set_pipeline(&self.rigid_mass_pipeline);
                    pass.dispatch_workgroups(rigid_groups, 1, 1);
                }

                // 5: Second half kick (includes force reset for next iteration)
                pass.set_pipeline(&self.second_half_kick_pipeline);
                pass.dispatch_workgroups(joint_groups, 1, 1);

                // 6: RATTLE — project out velocity along rigid strut axes
                if rigid_groups > 0 {
                    pass.set_pipeline(&self.rattle_pipeline);
                    pass.dispatch_workgroups(rigid_groups, 1, 1);
                }

                // 7: Ground collision
                pass.set_pipeline(&self.ground_collision_pipeline);
                pass.dispatch_workgroups(joint_groups, 1, 1);

                // 8: SHAKE again — ground collision moved joints, fix constraint violations
                if rigid_groups > 0 {
                    pass.set_pipeline(&self.shake_pipeline);
                    pass.dispatch_workgroups(rigid_groups, 1, 1);
                }
            }
        }
        // pass drops here, ending the single compute pass
    }

    pub fn copy_positions_to_staging(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(
            &self.position_buffer,
            0,
            &self.staging_buffer,
            0,
            (self.num_joints as u64) * 16,
        );
    }

    pub fn read_positions(&self, device: &wgpu::Device) -> Vec<[f32; 4]> {
        let buffer_slice = self.staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device.poll(wgpu::PollType::Wait).unwrap();
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let positions: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.staging_buffer.unmap();
        positions
    }

}

/// Create a storage buffer from a slice, using a single zero element if the slice is empty.
/// Some GPU backends don't support zero-size storage buffers in bind groups.
fn create_storage_buffer(device: &wgpu::Device, label: &str, data: &[u8]) -> wgpu::Buffer {
    let contents = if data.is_empty() { &[0u8; 4] } else { data };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
