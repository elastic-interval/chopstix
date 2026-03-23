use wgpu::util::DeviceExt;

use crate::gpu::physics::{PhysicsConfig, SURFACE_ABSENT};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PhysicsParams {
    dt: f32,
    gravity: f32,
    drag: f32,
    _reserved0: f32,
    num_joints: u32,
    num_elastic: u32,
    num_rigid: u32,
    ambient_mass: f32,
    force_scale: f32,
    ground_y: f32,
    _reserved1: f32,
    speed_limit: f32,
    num_push: u32,
    surface_character: u32,
    _pad2: u32,
    _pad3: u32,
}

#[allow(dead_code)]
pub struct GrowablePhysics {
    // Bind groups
    joint_bind_group: wgpu::BindGroup,
    interval_bind_group: wgpu::BindGroup,
    params_bind_group: wgpu::BindGroup,
    push_bind_group: wgpu::BindGroup,

    // Pipelines
    half_kick_pipeline: wgpu::ComputePipeline,
    elastic_forces_pipeline: wgpu::ComputePipeline,
    second_half_kick_pipeline: wgpu::ComputePipeline,
    ground_collision_pipeline: wgpu::ComputePipeline,
    push_forces_pipeline: wgpu::ComputePipeline,

    // Buffers that need CPU writes
    position_buffer: wgpu::Buffer,
    velocity_buffer: wgpu::Buffer,
    force_x_buffer: wgpu::Buffer,
    force_y_buffer: wgpu::Buffer,
    force_z_buffer: wgpu::Buffer,
    mass_buffer: wgpu::Buffer,
    frozen_buffer: wgpu::Buffer,

    elastic_alpha_buffer: wgpu::Buffer,
    elastic_omega_buffer: wgpu::Buffer,
    elastic_ideal_buffer: wgpu::Buffer,
    elastic_k_buffer: wgpu::Buffer,

    push_alpha_buffer: wgpu::Buffer,
    push_omega_buffer: wgpu::Buffer,
    push_ideal_buffer: wgpu::Buffer,
    push_k_buffer: wgpu::Buffer,
    push_half_mass_buffer: wgpu::Buffer,

    params_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    frozen_staging_buffer: wgpu::Buffer,

    // Active counts (how many elements are in use)
    pub active_joints: u32,
    pub active_elastic: u32,
    pub active_push: u32,

    // Capacity (buffer size in elements)
    capacity_joints: u32,
    capacity_elastic: u32,
    capacity_push: u32,

    // Config snapshot for param updates
    config: PhysicsConfig,

    // Bind group layouts (needed for rebuild after realloc)
    joint_bgl: wgpu::BindGroupLayout,
    interval_bgl: wgpu::BindGroupLayout,
    params_bgl: wgpu::BindGroupLayout,
    push_bgl: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,

    // CPU-side topology tracking (for renderer)
    pub cpu_elastic_alpha: Vec<u32>,
    pub cpu_elastic_omega: Vec<u32>,
    pub cpu_push_alpha: Vec<u32>,
    pub cpu_push_omega: Vec<u32>,
}

impl GrowablePhysics {
    pub fn new(device: &wgpu::Device, config: &PhysicsConfig) -> Self {
        // Start with generous capacity
        let cap_joints = 512u32;
        let cap_elastic = 1024u32;
        let cap_push = 512u32;

        let writable_storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let readback_storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;

        // Joint buffers at capacity, all writable, zero-initialized
        let position_buffer = create_zero_buffer(device, "G Positions", cap_joints as u64 * 16, readback_storage);
        let velocity_buffer = create_zero_buffer(device, "G Velocities", cap_joints as u64 * 16, writable_storage);
        let force_x_buffer = create_zero_buffer(device, "G Force X", cap_joints as u64 * 4, writable_storage);
        let force_y_buffer = create_zero_buffer(device, "G Force Y", cap_joints as u64 * 4, writable_storage);
        let force_z_buffer = create_zero_buffer(device, "G Force Z", cap_joints as u64 * 4, writable_storage);

        // Mass buffer initialized to ambient_mass
        let ambient_i32 = (config.ambient_mass * 1e4) as i32;
        let mass_init = vec![ambient_i32; cap_joints as usize];
        let mass_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("G Masses"),
            contents: bytemuck::cast_slice(&mass_init),
            usage: writable_storage,
        });

        let frozen_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("G Frozen"),
            contents: bytemuck::bytes_of(&0u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Elastic interval buffers at capacity
        let elastic_alpha_buffer = create_zero_buffer(device, "G Elastic Alpha", cap_elastic as u64 * 4, writable_storage);
        let elastic_omega_buffer = create_zero_buffer(device, "G Elastic Omega", cap_elastic as u64 * 4, writable_storage);
        let elastic_ideal_buffer = create_zero_buffer(device, "G Elastic Ideal", cap_elastic as u64 * 4, writable_storage);
        let elastic_k_buffer = create_zero_buffer(device, "G Elastic K", cap_elastic as u64 * 4, writable_storage);

        // Rigid interval buffers (empty, DSL builds don't use SHAKE/RATTLE)
        let rigid_dummy = create_zero_buffer(device, "G Rigid Dummy", 4, wgpu::BufferUsages::STORAGE);

        // Push interval buffers at capacity
        let push_alpha_buffer = create_zero_buffer(device, "G Push Alpha", cap_push as u64 * 4, writable_storage);
        let push_omega_buffer = create_zero_buffer(device, "G Push Omega", cap_push as u64 * 4, writable_storage);
        let push_ideal_buffer = create_zero_buffer(device, "G Push Ideal", cap_push as u64 * 4, writable_storage);
        let push_k_buffer = create_zero_buffer(device, "G Push K", cap_push as u64 * 4, writable_storage);
        let push_half_mass_buffer = create_zero_buffer(device, "G Push HalfMass", cap_push as u64 * 4, writable_storage);

        // Params buffer with COPY_DST for count updates
        let params = PhysicsParams {
            dt: config.dt,
            gravity: 0.0,  // no gravity during building
            drag: 10.0,    // high drag during building for stability
            _reserved0: 0.0,
            num_joints: 0,
            num_elastic: 0,
            num_rigid: 0,
            ambient_mass: config.ambient_mass,
            force_scale: config.force_scale,
            ground_y: -1e6,
            _reserved1: 0.0,
            speed_limit: f32::MAX,  // no speed limit during building
            num_push: 0,
            surface_character: SURFACE_ABSENT,
            _pad2: 0,
            _pad3: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("G Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Staging buffers
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G Staging"),
            size: (cap_joints as u64) * 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frozen_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("G Frozen Staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group layouts
        let joint_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("G Joint BGL"),
            entries: &[
                storage_entry(0, false),
                storage_entry(1, false),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, false),
            ],
        });

        let interval_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("G Interval BGL"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),  // rigid_alpha (dummy)
                storage_entry(5, true),  // rigid_omega (dummy)
                storage_entry(6, true),  // rigid_length (dummy)
                storage_entry(7, true),  // rigid_half_mass (dummy)
            ],
        });

        let params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("G Params BGL"),
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
            label: Some("G Push BGL"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
            ],
        });

        // Bind groups
        let joint_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("G Joint BG"),
            layout: &joint_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: position_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: velocity_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: force_x_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: force_y_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: force_z_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: mass_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: frozen_buffer.as_entire_binding() },
            ],
        });

        let interval_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("G Interval BG"),
            layout: &interval_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: elastic_alpha_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: elastic_omega_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: elastic_ideal_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: elastic_k_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: rigid_dummy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: rigid_dummy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: rigid_dummy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: rigid_dummy.as_entire_binding() },
            ],
        });

        let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("G Params BG"),
            layout: &params_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
            ],
        });

        let push_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("G Push BG"),
            layout: &push_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: push_alpha_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: push_omega_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: push_ideal_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: push_k_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: push_half_mass_buffer.as_entire_binding() },
            ],
        });

        // Pipeline layout & pipelines
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("G Physics Pipeline Layout"),
            bind_group_layouts: &[&joint_bgl, &interval_bgl, &params_bgl, &push_bgl],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("G Physics Shader"),
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

        Self {
            joint_bind_group,
            interval_bind_group,
            params_bind_group,
            push_bind_group,
            half_kick_pipeline: make_pipeline("half_kick_and_drift"),
            elastic_forces_pipeline: make_pipeline("elastic_forces"),
            second_half_kick_pipeline: make_pipeline("second_half_kick"),
            ground_collision_pipeline: make_pipeline("ground_collision"),
            push_forces_pipeline: make_pipeline("push_forces"),
            position_buffer,
            velocity_buffer,
            force_x_buffer,
            force_y_buffer,
            force_z_buffer,
            mass_buffer,
            frozen_buffer,
            elastic_alpha_buffer,
            elastic_omega_buffer,
            elastic_ideal_buffer,
            elastic_k_buffer,
            push_alpha_buffer,
            push_omega_buffer,
            push_ideal_buffer,
            push_k_buffer,
            push_half_mass_buffer,
            params_buffer,
            staging_buffer,
            frozen_staging_buffer,
            active_joints: 0,
            active_elastic: 0,
            active_push: 0,
            capacity_joints: cap_joints,
            capacity_elastic: cap_elastic,
            capacity_push: cap_push,
            config: config.clone(),
            joint_bgl,
            interval_bgl,
            params_bgl,
            push_bgl,
            pipeline_layout,
            cpu_elastic_alpha: Vec::new(),
            cpu_elastic_omega: Vec::new(),
            cpu_push_alpha: Vec::new(),
            cpu_push_omega: Vec::new(),
        }
    }

    /// Append joints. Returns the starting global index of the first new joint.
    pub fn append_joints(
        &mut self,
        queue: &wgpu::Queue,
        positions: &[[f32; 4]],
        velocities: &[[f32; 4]],
    ) -> u32 {
        let start = self.active_joints;
        let count = positions.len() as u32;
        assert!(
            start + count <= self.capacity_joints,
            "Joint capacity exceeded: {} + {} > {}",
            start, count, self.capacity_joints
        );

        let byte_offset = (start as u64) * 16;
        queue.write_buffer(&self.position_buffer, byte_offset, bytemuck::cast_slice(positions));
        queue.write_buffer(&self.velocity_buffer, byte_offset, bytemuck::cast_slice(velocities));

        // Initialize mass for new joints
        let ambient_i32 = (self.config.ambient_mass * 1e4) as i32;
        let mass_init: Vec<i32> = vec![ambient_i32; count as usize];
        queue.write_buffer(&self.mass_buffer, (start as u64) * 4, bytemuck::cast_slice(&mass_init));

        // Zero forces for new joints
        let zeros: Vec<i32> = vec![0i32; count as usize];
        queue.write_buffer(&self.force_x_buffer, (start as u64) * 4, bytemuck::cast_slice(&zeros));
        queue.write_buffer(&self.force_y_buffer, (start as u64) * 4, bytemuck::cast_slice(&zeros));
        queue.write_buffer(&self.force_z_buffer, (start as u64) * 4, bytemuck::cast_slice(&zeros));

        self.active_joints += count;
        start
    }

    /// Append elastic intervals. Returns the starting index.
    pub fn append_elastic(
        &mut self,
        queue: &wgpu::Queue,
        alpha: &[u32],
        omega: &[u32],
        ideal: &[f32],
        k: &[f32],
    ) -> u32 {
        let start = self.active_elastic;
        let count = alpha.len() as u32;
        assert!(
            start + count <= self.capacity_elastic,
            "Elastic capacity exceeded: {} + {} > {}",
            start, count, self.capacity_elastic
        );

        let offset = start as u64 * 4;
        queue.write_buffer(&self.elastic_alpha_buffer, offset, bytemuck::cast_slice(alpha));
        queue.write_buffer(&self.elastic_omega_buffer, offset, bytemuck::cast_slice(omega));
        queue.write_buffer(&self.elastic_ideal_buffer, offset, bytemuck::cast_slice(ideal));
        queue.write_buffer(&self.elastic_k_buffer, offset, bytemuck::cast_slice(k));

        self.cpu_elastic_alpha.extend_from_slice(alpha);
        self.cpu_elastic_omega.extend_from_slice(omega);

        self.active_elastic += count;
        start
    }

    /// Append push intervals. Returns the starting index.
    pub fn append_push(
        &mut self,
        queue: &wgpu::Queue,
        alpha: &[u32],
        omega: &[u32],
        ideal: &[f32],
        k: &[f32],
        half_mass: &[f32],
    ) -> u32 {
        let start = self.active_push;
        let count = alpha.len() as u32;
        assert!(
            start + count <= self.capacity_push,
            "Push capacity exceeded: {} + {} > {}",
            start, count, self.capacity_push
        );

        let offset = start as u64 * 4;
        queue.write_buffer(&self.push_alpha_buffer, offset, bytemuck::cast_slice(alpha));
        queue.write_buffer(&self.push_omega_buffer, offset, bytemuck::cast_slice(omega));
        queue.write_buffer(&self.push_ideal_buffer, offset, bytemuck::cast_slice(ideal));
        queue.write_buffer(&self.push_k_buffer, offset, bytemuck::cast_slice(k));
        queue.write_buffer(&self.push_half_mass_buffer, offset, bytemuck::cast_slice(half_mass));

        self.cpu_push_alpha.extend_from_slice(alpha);
        self.cpu_push_omega.extend_from_slice(omega);

        self.active_push += count;
        start
    }

    /// Write updated active counts to the GPU params buffer.
    pub fn update_counts(&self, queue: &wgpu::Queue) {
        let params = PhysicsParams {
            dt: self.config.dt,
            gravity: 0.0,
            drag: 10.0,
            _reserved0: 0.0,
            num_joints: self.active_joints,
            num_elastic: self.active_elastic,
            num_rigid: 0,
            ambient_mass: self.config.ambient_mass,
            force_scale: self.config.force_scale,
            ground_y: -1e6,
            _reserved1: 0.0,
            speed_limit: f32::MAX,
            num_push: self.active_push,
            surface_character: SURFACE_ABSENT,
            _pad2: 0,
            _pad3: 0,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    /// Update params with gravity and surface for post-build phase.
    pub fn enable_gravity(&self, queue: &wgpu::Queue, gravity: f32, ground_y: f32, surface_character: u32) {
        let params = PhysicsParams {
            dt: self.config.dt,
            gravity,
            drag: self.config.drag,
            _reserved0: 0.0,
            num_joints: self.active_joints,
            num_elastic: self.active_elastic,
            num_rigid: 0,
            ambient_mass: self.config.ambient_mass,
            force_scale: self.config.force_scale,
            ground_y,
            _reserved1: 0.0,
            speed_limit: self.config.speed_limit,
            num_push: self.active_push,
            surface_character,
            _pad2: 0,
            _pad3: 0,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    /// Write a single elastic ideal + K value at a specific index.
    pub fn write_elastic_ideal_at(&self, queue: &wgpu::Queue, index: usize, ideal: f32, k: f32) {
        let offset = (index as u64) * 4;
        queue.write_buffer(&self.elastic_ideal_buffer, offset, bytemuck::bytes_of(&ideal));
        queue.write_buffer(&self.elastic_k_buffer, offset, bytemuck::bytes_of(&k));
    }

    /// Write a single push ideal + K value at a specific index.
    pub fn write_push_ideal_at(&self, queue: &wgpu::Queue, index: usize, ideal: f32, k: f32) {
        let offset = (index as u64) * 4;
        queue.write_buffer(&self.push_ideal_buffer, offset, bytemuck::bytes_of(&ideal));
        queue.write_buffer(&self.push_k_buffer, offset, bytemuck::bytes_of(&k));
    }

    /// Dispatch physics compute pass (spring-push mode only, no SHAKE/RATTLE).
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, iterations: u32) {
        if self.active_joints == 0 {
            return;
        }
        let joint_groups = (self.active_joints + 63) / 64;
        let elastic_groups = if self.active_elastic > 0 { (self.active_elastic + 63) / 64 } else { 0 };
        let push_groups = if self.active_push > 0 { (self.active_push + 63) / 64 } else { 0 };

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("G Physics"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.joint_bind_group, &[]);
        pass.set_bind_group(1, &self.interval_bind_group, &[]);
        pass.set_bind_group(2, &self.params_bind_group, &[]);
        pass.set_bind_group(3, &self.push_bind_group, &[]);

        // Spring-push mode only
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
    }

    pub fn copy_positions_to_staging(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.active_joints == 0 {
            return;
        }
        encoder.copy_buffer_to_buffer(
            &self.position_buffer,
            0,
            &self.staging_buffer,
            0,
            (self.active_joints as u64) * 16,
        );
        encoder.copy_buffer_to_buffer(
            &self.frozen_buffer,
            0,
            &self.frozen_staging_buffer,
            0,
            4,
        );
    }

    pub fn read_positions(&self, device: &wgpu::Device) -> Vec<[f32; 4]> {
        if self.active_joints == 0 {
            return Vec::new();
        }
        let buffer_slice = self.staging_buffer.slice(..(self.active_joints as u64 * 16));
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

    pub fn read_frozen(&self, device: &wgpu::Device) -> bool {
        let buffer_slice = self.frozen_staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device.poll(wgpu::PollType::Wait).unwrap();
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let value: u32 = *bytemuck::from_bytes(&data);
        drop(data);
        self.frozen_staging_buffer.unmap();
        value != 0
    }

    /// Synchronous position readback (blocks until complete).
    /// Used during brick placement to get current positions.
    pub fn read_positions_sync(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Vec<[f32; 4]> {
        if self.active_joints == 0 {
            return Vec::new();
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("G Sync Readback"),
        });
        self.copy_positions_to_staging(&mut encoder);
        queue.submit(std::iter::once(encoder.finish()));
        self.read_positions(device)
    }
}

fn create_zero_buffer(device: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    let size = size.max(4); // minimum 4 bytes
    let zeros = vec![0u8; size as usize];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: &zeros,
        usage,
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
