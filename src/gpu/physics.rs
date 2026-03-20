use wgpu::util::DeviceExt;

use crate::constants::*;
use crate::tensegrity::TensegritySphereBuffers;

/// Runtime-configurable physics parameters.
/// Constants in `constants.rs` become the defaults.
#[derive(Clone, Debug)]
pub struct PhysicsConfig {
    pub dt: f32,
    pub iterations_per_frame: u32,
    pub pull_k_at_1m: f32,
    pub force_scale: f32,
    pub drag: f32,
    pub speed_limit: f32,
    pub settle_iterations: u32,
    pub settle_drag: f32,
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
        }
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
}

pub struct PhysicsCompute {
    joint_bind_group: wgpu::BindGroup,
    interval_bind_group: wgpu::BindGroup,
    params_bind_group: wgpu::BindGroup,
    half_kick_pipeline: wgpu::ComputePipeline,
    elastic_forces_pipeline: wgpu::ComputePipeline,
    rigid_mass_pipeline: wgpu::ComputePipeline,
    second_half_kick_pipeline: wgpu::ComputePipeline,
    shake_pipeline: wgpu::ComputePipeline,
    rattle_pipeline: wgpu::ComputePipeline,
    ground_collision_pipeline: wgpu::ComputePipeline,
    position_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    num_joints: u32,
    num_elastic: u32,
    num_rigid: u32,
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
            ambient_mass: JOINT_AMBIENT_MASS,
            force_scale: config.force_scale,
            ground_y: -1e6,         // ground far away — irrelevant
            restitution: 0.0,
            speed_limit: f32::MAX,  // no speed limit during settling
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

    pub fn new(device: &wgpu::Device, _queue: &wgpu::Queue, buffers: &TensegritySphereBuffers, config: &PhysicsConfig) -> Self {
        let params = PhysicsParams {
            dt: config.dt,
            gravity: GRAVITY,
            drag: config.drag,
            viscosity: VISCOSITY,
            num_joints: buffers.num_joints(),
            num_elastic: buffers.num_elastic(),
            num_rigid: buffers.num_rigid(),
            ambient_mass: JOINT_AMBIENT_MASS,
            force_scale: config.force_scale,
            ground_y: GROUND_Y,
            restitution: RESTITUTION,
            speed_limit: config.speed_limit,
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
        let ambient_i32 = (JOINT_AMBIENT_MASS * 1e4) as i32;
        let mass_init = vec![ambient_i32; num_joints as usize];
        let mass_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Masses"),
            contents: bytemuck::cast_slice(&mass_init),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Elastic interval buffers
        let elastic_alpha_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Elastic Alpha"),
            contents: bytemuck::cast_slice(&buffers.elastic_alpha),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let elastic_omega_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Elastic Omega"),
            contents: bytemuck::cast_slice(&buffers.elastic_omega),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let elastic_ideal_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Elastic Ideal"),
            contents: bytemuck::cast_slice(&buffers.elastic_ideal),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let elastic_k_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Elastic K"),
            contents: bytemuck::cast_slice(&buffers.elastic_k),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Rigid interval buffers
        let rigid_alpha_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Rigid Alpha"),
            contents: bytemuck::cast_slice(&buffers.rigid_alpha),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let rigid_omega_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Rigid Omega"),
            contents: bytemuck::cast_slice(&buffers.rigid_omega),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let rigid_length_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Rigid Length"),
            contents: bytemuck::cast_slice(&buffers.rigid_length),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let rigid_half_mass_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Rigid Half Mass"),
            contents: bytemuck::cast_slice(&buffers.rigid_half_mass),
            usage: wgpu::BufferUsages::STORAGE,
        });

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

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Physics Pipeline Layout"),
            bind_group_layouts: &[&joint_bgl, &interval_bgl, &params_bgl],
            push_constant_ranges: &[],
        });

        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Physics Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("physics.wgsl").into()),
        });

        // Create 6 compute pipelines
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

        Self {
            joint_bind_group,
            interval_bind_group,
            params_bind_group,
            half_kick_pipeline,
            elastic_forces_pipeline,
            rigid_mass_pipeline,
            second_half_kick_pipeline,
            shake_pipeline,
            rattle_pipeline,
            ground_collision_pipeline,
            position_buffer,
            staging_buffer,
            num_joints,
            num_elastic,
            num_rigid,
        }
    }

    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, iterations: u32) {
        let joint_groups = (self.num_joints + 63) / 64;
        let elastic_groups = if self.num_elastic > 0 { (self.num_elastic + 63) / 64 } else { 0 };
        let rigid_groups = if self.num_rigid > 0 { (self.num_rigid + 63) / 64 } else { 0 };

        // Single compute pass for all iterations — wgpu guarantees sequential
        // execution with implicit storage barriers between dispatches.
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Physics"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.joint_bind_group, &[]);
        pass.set_bind_group(1, &self.interval_bind_group, &[]);
        pass.set_bind_group(2, &self.params_bind_group, &[]);

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
