use glam::Vec3;

use super::approach::ApproachManager;
use super::brick::BrickTemplate;
use super::face::{FaceId, FaceRegistry};
use super::placement;
use super::Spin;
use crate::gpu::growable::GrowablePhysics;

/// Describes a tensegrity construction program.
#[derive(Clone, Debug)]
pub enum BuildNode {
    /// Build a column of N bricks, alternating spin.
    Column { count: usize, spin: Spin },
    /// Open end — no further growth from this face.
    Open,
}

#[derive(Debug, PartialEq)]
pub enum BuildStage {
    /// Placing bricks, waiting for approaches to settle between placements.
    Building,
    /// All bricks placed, enabling pretension.
    Pretensing,
    /// Gravity enabled, letting the structure fall.
    Falling,
    /// Construction complete.
    Complete,
}

struct Bud {
    face_id: FaceId,
    remaining: BuildNode,
}

#[allow(dead_code)]
pub struct BuildExecutor {
    pub faces: FaceRegistry,
    pub approach: ApproachManager,
    buds: Vec<Bud>,
    pub stage: BuildStage,
    /// Current positions cache (updated from GPU readback)
    pub positions: Vec<[f32; 4]>,
    pub pull_k_at_1m: f32,
    face_radius: f32,
    brick_height: f32,
    /// Topology tracking for renderer
    pub elastic_alpha: Vec<u32>,
    pub elastic_omega: Vec<u32>,
    pub push_alpha: Vec<u32>,
    pub push_omega: Vec<u32>,
}

impl BuildExecutor {
    /// Create a new build executor with the given program.
    pub fn new(
        physics: &mut GrowablePhysics,
        queue: &wgpu::Queue,
        program: BuildNode,
        pull_k_at_1m: f32,
        face_radius: f32,
        brick_height: f32,
    ) -> Self {
        let mut faces = FaceRegistry::new();
        let mut approach = ApproachManager::new();

        // Place seed brick at origin
        let seed_spin = match &program {
            BuildNode::Column { spin, .. } => *spin,
            BuildNode::Open => Spin::Left,
        };
        let template = BrickTemplate::for_spin(seed_spin, face_radius, brick_height);
        let (face_ids, positions) = placement::place_seed_brick(
            physics,
            queue,
            &mut approach,
            &mut faces,
            &template,
            Vec3::ZERO,
            pull_k_at_1m,
        );

        // Collect initial topology from physics state
        let elastic_alpha = Vec::new();
        let elastic_omega = Vec::new();
        let push_alpha = Vec::new();
        let push_omega = Vec::new();

        // The forward face becomes a bud (the face that grows the column)
        let mut buds = Vec::new();
        let forward_face = template
            .faces
            .iter()
            .enumerate()
            .find(|(_, f)| f.is_forward)
            .map(|(i, _)| i);

        if let Some(forward_idx) = forward_face {
            let remaining = match program {
                BuildNode::Column { count, spin } => {
                    if count > 1 {
                        BuildNode::Column {
                            count: count - 1,
                            spin: spin.opposite(),
                        }
                    } else {
                        BuildNode::Open
                    }
                }
                BuildNode::Open => BuildNode::Open,
            };
            if !matches!(remaining, BuildNode::Open) {
                buds.push(Bud {
                    face_id: face_ids[forward_idx],
                    remaining,
                });
            }
        }

        Self {
            faces,
            approach,
            buds,
            stage: BuildStage::Building,
            positions,
            pull_k_at_1m,
            face_radius,
            brick_height,
            elastic_alpha,
            elastic_omega,
            push_alpha,
            push_omega,
        }
    }

    /// Advance one frame. Call before physics dispatch.
    /// Returns true if topology changed (renderer needs update).
    pub fn tick(
        &mut self,
        physics: &mut GrowablePhysics,
        queue: &wgpu::Queue,
    ) -> bool {
        // Advance approaching intervals
        self.approach.tick(physics, queue);

        match self.stage {
            BuildStage::Building => {
                if self.approach.all_settled() {
                    if let Some(bud) = self.buds.pop() {
                        // Place next brick
                        let spin = match &bud.remaining {
                            BuildNode::Column { spin, .. } => *spin,
                            BuildNode::Open => return false,
                        };
                        let template = BrickTemplate::for_spin(spin, self.face_radius, self.brick_height);
                        let new_face_ids = placement::place_brick(
                            physics,
                            queue,
                            &mut self.approach,
                            &mut self.faces,
                            &template,
                            bud.face_id,
                            &mut self.positions,
                            self.pull_k_at_1m,
                        );

                        // Create buds for new forward faces
                        let next_remaining = match bud.remaining {
                            BuildNode::Column { count, spin } => {
                                if count > 1 {
                                    BuildNode::Column {
                                        count: count - 1,
                                        spin: spin.opposite(),
                                    }
                                } else {
                                    BuildNode::Open
                                }
                            }
                            BuildNode::Open => BuildNode::Open,
                        };

                        if !matches!(next_remaining, BuildNode::Open) {
                            for face_id in new_face_ids {
                                self.buds.push(Bud {
                                    face_id,
                                    remaining: next_remaining.clone(),
                                });
                            }
                        }

                        return true;
                    } else {
                        // All buds consumed, transition to pretensing
                        log::info!("Build complete, transitioning to pretensing");
                        self.stage = BuildStage::Pretensing;
                    }
                }
                false
            }
            BuildStage::Pretensing => {
                if self.approach.all_settled() {
                    log::info!("Pretensing complete, enabling gravity");
                    self.stage = BuildStage::Falling;
                }
                false
            }
            BuildStage::Falling => {
                // Could check for settled state and transition to Complete
                // For now, just stay in Falling
                false
            }
            BuildStage::Complete => false,
        }
    }

    /// Update cached positions from GPU readback.
    pub fn update_positions(&mut self, positions: Vec<[f32; 4]>) {
        self.positions = positions;
    }

    /// Get the current number of active joints.
    pub fn num_joints(&self) -> u32 {
        self.positions.len() as u32
    }

    pub fn is_building(&self) -> bool {
        self.stage == BuildStage::Building
    }

    pub fn stage_name(&self) -> &str {
        match self.stage {
            BuildStage::Building => "building",
            BuildStage::Pretensing => "pretensing",
            BuildStage::Falling => "falling",
            BuildStage::Complete => "complete",
        }
    }
}
