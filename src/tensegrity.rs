use glam::{Quat, Vec3};

use crate::constants::*;
use crate::sphere::{SphereScaffold, Vertex};

pub struct TensegritySphereBuffers {
    pub positions: Vec<[f32; 4]>,
    pub velocities: Vec<[f32; 4]>,
    pub elastic_alpha: Vec<u32>,
    pub elastic_omega: Vec<u32>,
    pub elastic_ideal: Vec<f32>,
    pub elastic_k: Vec<f32>,
    pub rigid_alpha: Vec<u32>,
    pub rigid_omega: Vec<u32>,
    pub rigid_length: Vec<f32>,
    pub rigid_half_mass: Vec<f32>,
}

impl TensegritySphereBuffers {
    pub fn num_joints(&self) -> u32 {
        self.positions.len() as u32
    }

    pub fn num_elastic(&self) -> u32 {
        self.elastic_alpha.len() as u32
    }

    pub fn num_rigid(&self) -> u32 {
        self.rigid_alpha.len() as u32
    }
}

enum Cell {
    PushPlaceholder {
        alpha_vertex: usize,
        omega_vertex: usize,
    },
    PushInterval {
        alpha_vertex: usize,
        omega_vertex: usize,
        alpha_joint: usize,
        omega_joint: usize,
        length: f32,
    },
}

struct Spoke {
    far_vertex: usize,
    near_joint: usize,
    length: f32,
}

pub fn generate_sphere(frequency: usize, radius: f32) -> TensegritySphereBuffers {
    use Cell::*;

    let mut scaffold = SphereScaffold::new(frequency);
    scaffold.generate();
    scaffold.set_radius(radius);

    let mut positions: Vec<[f32; 4]> = Vec::new();
    let mut rigid_alpha: Vec<u32> = Vec::new();
    let mut rigid_omega: Vec<u32> = Vec::new();
    let mut rigid_length: Vec<f32> = Vec::new();
    let mut rigid_half_mass: Vec<f32> = Vec::new();
    let mut elastic_alpha: Vec<u32> = Vec::new();
    let mut elastic_omega: Vec<u32> = Vec::new();
    let mut elastic_ideal: Vec<f32> = Vec::new();
    let mut elastic_k: Vec<f32> = Vec::new();

    let mut create_joint = |pos: Vec3| -> usize {
        let idx = positions.len();
        positions.push([pos.x, pos.y, pos.z, 0.0]);
        idx
    };

    let locations = scaffold.locations();

    let vertex_cells: Vec<Vec<Cell>> = scaffold
        .vertex
        .iter()
        .map(
            |Vertex {
                 index: vertex_here,
                 adjacent,
                 ..
             }| {
                adjacent
                    .iter()
                    .map(|adjacent_vertex| {
                        if *adjacent_vertex > *vertex_here {
                            let (alpha_base, omega_base) =
                                (locations[*vertex_here], locations[*adjacent_vertex]);
                            let axis = alpha_base.lerp(omega_base, 0.5).normalize();
                            let quaternion = Quat::from_axis_angle(axis, TWIST_ANGLE);
                            let alpha_joint = create_joint(quaternion * alpha_base);
                            let omega_joint = create_joint(quaternion * omega_base);
                            let length = (omega_base - alpha_base).length();

                            rigid_alpha.push(alpha_joint as u32);
                            rigid_omega.push(omega_joint as u32);
                            rigid_length.push(length);
                            rigid_half_mass.push(PUSH_LINEAR_DENSITY * length / 2.0);

                            PushInterval {
                                alpha_vertex: *vertex_here,
                                omega_vertex: *adjacent_vertex,
                                alpha_joint,
                                omega_joint,
                                length,
                            }
                        } else {
                            PushPlaceholder {
                                alpha_vertex: *vertex_here,
                                omega_vertex: *adjacent_vertex,
                            }
                        }
                    })
                    .collect()
            },
        )
        .collect();

    let vertex_spokes: Vec<Vec<Spoke>> = vertex_cells
        .iter()
        .map(|cells| {
            cells
                .iter()
                .map(|cell| match cell {
                    PushPlaceholder {
                        alpha_vertex,
                        omega_vertex,
                    } => {
                        let (sought_omega, sought_alpha) = (alpha_vertex, omega_vertex);
                        for omega_vertex_adjacent in &vertex_cells[*omega_vertex] {
                            if let PushInterval {
                                alpha_vertex,
                                omega_vertex,
                                omega_joint,
                                length,
                                ..
                            } = omega_vertex_adjacent
                            {
                                if *sought_alpha == *alpha_vertex && *omega_vertex == *sought_omega {
                                    return Spoke {
                                        far_vertex: *alpha_vertex,
                                        near_joint: *omega_joint,
                                        length: *length,
                                    };
                                }
                            }
                        }
                        panic!("Adjacent not found!");
                    }
                    PushInterval {
                        omega_vertex,
                        alpha_joint,
                        length,
                        ..
                    } => Spoke {
                        far_vertex: *omega_vertex,
                        near_joint: *alpha_joint,
                        length: *length,
                    },
                })
                .collect()
        })
        .collect();

    let actual_dist = |a: usize, b: usize| -> f32 {
        let pa = Vec3::new(positions[a][0], positions[a][1], positions[a][2]);
        let pb = Vec3::new(positions[b][0], positions[b][1], positions[b][2]);
        (pb - pa).length()
    };

    let mut slack_count = 0u32;
    let pretension = 0.95; // cables want to be 5% shorter than initial placement
    for (hub, spokes) in vertex_spokes.iter().enumerate() {
        for (spoke_index, spoke) in spokes.iter().enumerate() {
            let next_spoke = &spokes[(spoke_index + 1) % spokes.len()];
            // Circumference cable — ideal is shorter than actual for pre-tension
            let actual = actual_dist(spoke.near_joint, next_spoke.near_joint);
            let circ_ideal = actual * pretension;
            if actual < spoke.length / 3.0 {
                slack_count += 1;
            }
            elastic_alpha.push(spoke.near_joint as u32);
            elastic_omega.push(next_spoke.near_joint as u32);
            elastic_ideal.push(circ_ideal);
            elastic_k.push(PULL_K_AT_1M / circ_ideal);

            // Diagonal cable
            let next_near = spokes[(spoke_index + 1) % spokes.len()].near_joint;
            let next_far = {
                let far_vertex = &vertex_spokes[spoke.far_vertex];
                let hub_position = far_vertex.iter().position(|v| v.far_vertex == hub).unwrap();
                far_vertex[(hub_position + 1) % far_vertex.len()].near_joint
            };
            if next_far > next_near {
                let actual = actual_dist(next_near, next_far);
                let diag_ideal = actual * pretension;
                elastic_alpha.push(next_near as u32);
                elastic_omega.push(next_far as u32);
                elastic_ideal.push(diag_ideal);
                elastic_k.push(PULL_K_AT_1M / diag_ideal);
            }
        }
    }
    if slack_count > 0 {
        log::warn!("{} circumference cables would have been slack with old ideal lengths", slack_count);
    }

    // Shrink initial positions so cables start near their ideal (at rest)
    // and struts start slightly compressed (pushing outward) — this is the
    // tensegrity self-stress equilibrium, avoiding a violent initial transient.
    for pos in &mut positions {
        pos[0] *= pretension;
        pos[1] *= pretension;
        pos[2] *= pretension;
    }

    let num_joints = positions.len();
    let velocities = vec![[0.0f32; 4]; num_joints];

    log::info!(
        "Generated sphere freq={}: {} joints, {} struts, {} cables",
        frequency,
        num_joints,
        rigid_alpha.len(),
        elastic_alpha.len()
    );

    TensegritySphereBuffers {
        positions,
        velocities,
        elastic_alpha,
        elastic_omega,
        elastic_ideal,
        elastic_k,
        rigid_alpha,
        rigid_omega,
        rigid_length,
        rigid_half_mass,
    }
}
