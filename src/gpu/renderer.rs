use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::constants::GROUND_Y;
use crate::gpu::Gpu;
use crate::tensegrity::TensegritySphereBuffers;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CylinderInstance {
    pub start: [f32; 3],
    pub radius_factor: f32,
    pub end: [f32; 3],
    pub material_type: u32,
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CylinderVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

pub struct Renderer {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    instance_buffer: Option<wgpu::Buffer>,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
    num_indices: u32,
    num_instances: u32,
    // Store interval topology for instance building
    elastic_alpha: Vec<u32>,
    elastic_omega: Vec<u32>,
    rigid_alpha: Vec<u32>,
    rigid_omega: Vec<u32>,
    push_alpha: Vec<u32>,
    push_omega: Vec<u32>,
    radius_scale: f32,
}

impl Renderer {
    pub fn new(gpu: &Gpu, buffers: &TensegritySphereBuffers, frequency: usize) -> Self {
        let (vertex_buffer, index_buffer, num_indices) = create_cylinder(&gpu.device);

        // Uniform buffer for MVP matrix
        let uniform_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&glam::Mat4::IDENTITY.to_cols_array()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Uniform BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let uniform_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform BG"),
            layout: &uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Render Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("render.wgsl").into()),
            });

        let pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[&uniform_bgl],
                    push_constant_ranges: &[],
                });

        let render_pipeline =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    cache: None,
                    label: Some("Render Pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        compilation_options: Default::default(),
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[
                            // Vertex buffer
                            wgpu::VertexBufferLayout {
                                array_stride: size_of::<CylinderVertex>() as wgpu::BufferAddress,
                                step_mode: wgpu::VertexStepMode::Vertex,
                                attributes: &[
                                    wgpu::VertexAttribute {
                                        offset: 0,
                                        shader_location: 0,
                                        format: wgpu::VertexFormat::Float32x3,
                                    },
                                    wgpu::VertexAttribute {
                                        offset: size_of::<[f32; 3]>() as wgpu::BufferAddress,
                                        shader_location: 1,
                                        format: wgpu::VertexFormat::Float32x3,
                                    },
                                    wgpu::VertexAttribute {
                                        offset: size_of::<[f32; 6]>() as wgpu::BufferAddress,
                                        shader_location: 2,
                                        format: wgpu::VertexFormat::Float32x2,
                                    },
                                ],
                            },
                            // Instance buffer
                            wgpu::VertexBufferLayout {
                                array_stride: size_of::<CylinderInstance>() as wgpu::BufferAddress,
                                step_mode: wgpu::VertexStepMode::Instance,
                                attributes: &[
                                    wgpu::VertexAttribute {
                                        offset: 0,
                                        shader_location: 3,
                                        format: wgpu::VertexFormat::Float32x3,
                                    },
                                    wgpu::VertexAttribute {
                                        offset: size_of::<[f32; 3]>() as wgpu::BufferAddress,
                                        shader_location: 4,
                                        format: wgpu::VertexFormat::Float32,
                                    },
                                    wgpu::VertexAttribute {
                                        offset: size_of::<[f32; 4]>() as wgpu::BufferAddress,
                                        shader_location: 5,
                                        format: wgpu::VertexFormat::Float32x3,
                                    },
                                    wgpu::VertexAttribute {
                                        offset: size_of::<[f32; 7]>() as wgpu::BufferAddress,
                                        shader_location: 6,
                                        format: wgpu::VertexFormat::Uint32,
                                    },
                                    wgpu::VertexAttribute {
                                        offset: (size_of::<[f32; 7]>() + size_of::<u32>())
                                            as wgpu::BufferAddress,
                                        shader_location: 7,
                                        format: wgpu::VertexFormat::Float32x4,
                                    },
                                ],
                            },
                        ],
                    },
                    fragment: Some(wgpu::FragmentState {
                        compilation_options: Default::default(),
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: gpu.surface_config.format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: Some(wgpu::Face::Back),
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: 1,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview: None,
                });

        Self {
            vertex_buffer,
            index_buffer,
            instance_buffer: None,
            uniform_buffer,
            uniform_bind_group,
            render_pipeline,
            num_indices,
            num_instances: 0,
            elastic_alpha: buffers.elastic_alpha.clone(),
            elastic_omega: buffers.elastic_omega.clone(),
            rigid_alpha: buffers.rigid_alpha.clone(),
            rigid_omega: buffers.rigid_omega.clone(),
            push_alpha: buffers.push_alpha.clone(),
            push_omega: buffers.push_omega.clone(),
            radius_scale: 3.0 / frequency as f32,
        }
    }

    pub fn update_mvp(&self, queue: &wgpu::Queue, mvp: &glam::Mat4) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&mvp.to_cols_array()));
    }

    pub fn update_instances(&mut self, device: &wgpu::Device, positions: &[[f32; 4]], show_ground: bool) {
        if positions.is_empty() {
            self.num_instances = 0;
            self.instance_buffer = None;
            return;
        }
        let total_struts = self.rigid_alpha.len() + self.push_alpha.len();
        let mut instances = Vec::with_capacity(total_struts + self.elastic_alpha.len());

        // Rigid struts (SHAKE/RATTLE mode) - silver, thick
        for i in 0..self.rigid_alpha.len() {
            let a = self.rigid_alpha[i] as usize;
            let o = self.rigid_omega[i] as usize;
            instances.push(CylinderInstance {
                start: [positions[a][0], positions[a][1], positions[a][2]],
                radius_factor: 3.0 * self.radius_scale,
                end: [positions[o][0], positions[o][1], positions[o][2]],
                material_type: 0, // Push
                color: [0.75, 0.75, 0.78, 1.0], // Silver
            });
        }

        // Spring-push struts (Klein etc.) - silver, thick
        for i in 0..self.push_alpha.len() {
            let a = self.push_alpha[i] as usize;
            let o = self.push_omega[i] as usize;
            instances.push(CylinderInstance {
                start: [positions[a][0], positions[a][1], positions[a][2]],
                radius_factor: 3.0 * self.radius_scale,
                end: [positions[o][0], positions[o][1], positions[o][2]],
                material_type: 0, // Push
                color: [0.75, 0.75, 0.78, 1.0], // Silver
            });
        }

        // Cables (elastic) - blue, thin
        for i in 0..self.elastic_alpha.len() {
            let a = self.elastic_alpha[i] as usize;
            let o = self.elastic_omega[i] as usize;
            instances.push(CylinderInstance {
                start: [positions[a][0], positions[a][1], positions[a][2]],
                radius_factor: 1.0 * self.radius_scale,
                end: [positions[o][0], positions[o][1], positions[o][2]],
                material_type: 1, // Pull
                color: [0.2, 0.4, 0.9, 1.0], // Blue
            });
        }

        // Ground grid (only when surface is active)
        if !show_ground {
            self.num_instances = instances.len() as u32;
            if self.num_instances > 0 {
                self.instance_buffer = Some(device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("Instance Buffer"),
                        contents: bytemuck::cast_slice(&instances),
                        usage: wgpu::BufferUsages::VERTEX,
                    },
                ));
            }
            return;
        }
        // Triangular ground lattice: 3 families of parallel lines at 0°, 60°, 120°.
        // Each family includes a line through the origin. Lines are spaced
        // perpendicular to their direction by grid_spacing.
        let grid_extent = 50.0f32;
        let grid_spacing = 5.0f32;
        let grid_color = [0.25, 0.3, 0.25, 1.0];

        // For each direction: (dir_x, dir_z) is the line direction,
        // (perp_x, perp_z) is the perpendicular offset direction.
        let directions: [(f32, f32, f32, f32); 3] = [
            // 0°: lines along X, offset along Z
            (1.0, 0.0, 0.0, 1.0),
            // 60°: lines along (cos60, sin60), offset along (-sin60, cos60)
            (0.5, 0.866025, -0.866025, 0.5),
            // 120°: lines along (cos120, sin120), offset along (-sin120, cos120)
            (-0.5, 0.866025, -0.866025, -0.5),
        ];

        for (dx, dz, px, pz) in directions {
            let n_lines = (grid_extent / grid_spacing) as i32;
            for i in -n_lines..=n_lines {
                let offset = i as f32 * grid_spacing;
                let cx = px * offset;
                let cz = pz * offset;
                instances.push(CylinderInstance {
                    start: [cx - dx * grid_extent, GROUND_Y, cz - dz * grid_extent],
                    radius_factor: 0.4,
                    end: [cx + dx * grid_extent, GROUND_Y, cz + dz * grid_extent],
                    material_type: 0,
                    color: grid_color,
                });
            }
        }

        self.num_instances = instances.len() as u32;
        if self.num_instances > 0 {
            self.instance_buffer = Some(device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Instance Buffer"),
                    contents: bytemuck::cast_slice(&instances),
                    usage: wgpu::BufferUsages::VERTEX,
                },
            ));
        }
    }

    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
    ) {
        if self.num_instances == 0 || self.instance_buffer.is_none() {
            return;
        }
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.as_ref().unwrap().slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.num_indices, 0, 0..self.num_instances);
    }

    pub fn update_topology(&mut self, buffers: &TensegritySphereBuffers, frequency: usize) {
        self.elastic_alpha = buffers.elastic_alpha.clone();
        self.elastic_omega = buffers.elastic_omega.clone();
        self.rigid_alpha = buffers.rigid_alpha.clone();
        self.rigid_omega = buffers.rigid_omega.clone();
        self.push_alpha = buffers.push_alpha.clone();
        self.push_omega = buffers.push_omega.clone();
        self.radius_scale = 3.0 / frequency as f32;
    }
}

fn create_cylinder(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    use std::f32::consts::PI;

    const HALF_HEIGHT: f32 = 0.5;
    const SEGMENTS: u32 = 12;

    let mut vertices = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let mut ring_vertices = Vec::with_capacity(SEGMENTS as usize);
    for i in 0..SEGMENTS {
        let angle = (i as f32) / (SEGMENTS as f32) * 2.0 * PI;
        let x = angle.cos();
        let z = angle.sin();
        let normal = [angle.cos(), 0.0, angle.sin()];
        ring_vertices.push((x, z, normal));
    }

    // Side vertices
    for i in 0..SEGMENTS {
        let (x, z, normal) = ring_vertices[i as usize];
        vertices.push(CylinderVertex {
            position: [x, HALF_HEIGHT, z],
            normal,
            uv: [i as f32 / SEGMENTS as f32, 0.0],
        });
        vertices.push(CylinderVertex {
            position: [x, -HALF_HEIGHT, z],
            normal,
            uv: [i as f32 / SEGMENTS as f32, 1.0],
        });
    }

    // Side indices
    for i in 0..SEGMENTS {
        let top_current = i * 2;
        let bottom_current = i * 2 + 1;
        let top_next = ((i + 1) % SEGMENTS) * 2;
        let bottom_next = ((i + 1) % SEGMENTS) * 2 + 1;
        indices.push(top_current);
        indices.push(top_next);
        indices.push(bottom_current);
        indices.push(bottom_current);
        indices.push(top_next);
        indices.push(bottom_next);
    }

    // Top cap
    let top_center_idx = vertices.len() as u32;
    vertices.push(CylinderVertex {
        position: [0.0, HALF_HEIGHT, 0.0],
        normal: [0.0, 1.0, 0.0],
        uv: [0.5, 0.5],
    });
    let top_ring_start = vertices.len() as u32;
    for i in 0..SEGMENTS {
        let (x, z, _) = ring_vertices[i as usize];
        vertices.push(CylinderVertex {
            position: [x, HALF_HEIGHT, z],
            normal: [0.0, 1.0, 0.0],
            uv: [0.5 + 0.5 * x, 0.5 + 0.5 * z],
        });
    }
    for i in 0..SEGMENTS {
        indices.push(top_center_idx);
        indices.push(top_ring_start + ((i + 1) % SEGMENTS));
        indices.push(top_ring_start + i);
    }

    // Bottom cap
    let bottom_center_idx = vertices.len() as u32;
    vertices.push(CylinderVertex {
        position: [0.0, -HALF_HEIGHT, 0.0],
        normal: [0.0, -1.0, 0.0],
        uv: [0.5, 0.5],
    });
    let bottom_ring_start = vertices.len() as u32;
    for i in 0..SEGMENTS {
        let (x, z, _) = ring_vertices[i as usize];
        vertices.push(CylinderVertex {
            position: [x, -HALF_HEIGHT, z],
            normal: [0.0, -1.0, 0.0],
            uv: [0.5 + 0.5 * x, 0.5 + 0.5 * z],
        });
    }
    for i in 0..SEGMENTS {
        indices.push(bottom_center_idx);
        indices.push(bottom_ring_start + i);
        indices.push(bottom_ring_start + ((i + 1) % SEGMENTS));
    }

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cylinder Vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cylinder Indices"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    (vertex_buffer, index_buffer, indices.len() as u32)
}
