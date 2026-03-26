use glam::Vec3;

use super::approach::ApproachManager;
use super::brick::BrickTemplate;
use super::face::{FaceId, FaceRegistry};
use super::placement;
use crate::gpu::growable::GrowablePhysics;

// ── DSL types ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum SeedKind {
    SingleTwist,
    Omni,
}

#[derive(Clone, Debug)]
pub struct BuildProgram {
    pub seed: SeedKind,
    pub face_nodes: Vec<BuildNode>,
}

/// Composable build instruction tree — mirrors tensegrity-lab's DSL.
#[derive(Clone, Debug)]
pub enum BuildNode {
    /// Target a named face, then execute child node on it.
    Face { name: &'static str, node: Box<BuildNode> },
    /// Column of N bricks. scale < 1.0 = shrinking. after = post-column nodes.
    Column { count: usize, scale: f32, after: Vec<BuildNode> },
    /// Mark a face for later reference (shaping, spacing, etc.)
    Mark { name: &'static str },
    /// Add a prism ending to a face: push strut + 6 pull cables.
    Prism { outer_pct: f32 },
    /// Open face — remove radials entirely.
    Open,
}

// ── DSL builder helpers ────────────────────────────────────────────────────

pub fn face(name: &'static str, node: BuildNode) -> BuildNode {
    BuildNode::Face { name, node: Box::new(node) }
}

pub fn column(count: usize) -> ColumnBuilder {
    ColumnBuilder { count, scale: 1.0, after: vec![] }
}

pub fn open() -> BuildNode {
    BuildNode::Open
}

pub fn prism(outer_pct: f32) -> BuildNode {
    BuildNode::Prism { outer_pct }
}


/// Fluent builder for columns, matching tensegrity-lab's FaceColumnBuilder.
pub struct ColumnBuilder {
    count: usize,
    scale: f32,
    after: Vec<BuildNode>,
}

impl ColumnBuilder {
    pub fn shrink_by(mut self, pct: f32) -> Self {
        self.scale = 1.0 - pct / 100.0;
        self
    }

    pub fn mark(mut self, name: &'static str) -> Self {
        self.after.push(BuildNode::Mark { name });
        self
    }

    pub fn prism(mut self, outer_pct: f32) -> Self {
        self.after.push(BuildNode::Prism { outer_pct });
        self
    }

    pub fn then(mut self, node: BuildNode) -> Self {
        self.after.push(node);
        self
    }

    pub fn build(self) -> BuildNode {
        BuildNode::Column {
            count: self.count,
            scale: self.scale,
            after: self.after,
        }
    }
}

/// Allow implicit conversion from ColumnBuilder to BuildNode.
impl From<ColumnBuilder> for BuildNode {
    fn from(cb: ColumnBuilder) -> Self {
        cb.build()
    }
}

// ── Executor ───────────────────────────────────────────────────────────────

#[derive(Debug, PartialEq)]
pub enum BuildStage {
    Building,
    Pretensing,
    Falling,
    #[allow(dead_code)]
    Complete,
}

struct Bud {
    face_id: FaceId,
    column_remaining: usize,
    column_scale: f32,
    after: Vec<BuildNode>,
}

#[allow(dead_code)]
pub struct BuildExecutor {
    pub faces: FaceRegistry,
    pub approach: ApproachManager,
    buds: Vec<Bud>,
    pub stage: BuildStage,
    pub positions: Vec<[f32; 4]>,
    pub pull_k_at_1m: f32,
    pub marks: Vec<(&'static str, FaceId)>,
}

impl BuildExecutor {
    pub fn new(
        physics: &mut GrowablePhysics,
        queue: &wgpu::Queue,
        program: BuildProgram,
        pull_k_at_1m: f32,
    ) -> Self {
        let mut faces = FaceRegistry::new();
        let mut approach = ApproachManager::new();

        let (template, down_face) = match &program.seed {
            SeedKind::SingleTwist => (BrickTemplate::single_twist_left_baked(), None),
            SeedKind::Omni => (BrickTemplate::omni_symmetrical(), Some("OmniBot")),
        };

        let mut template = template;
        if let Some(down_name) = down_face {
            let rotation = template.down_rotation(down_name);
            template.apply_rotation(rotation);
        }

        let (face_ids, positions) = placement::place_seed_brick(
            physics, queue, &mut approach, &mut faces,
            &template, Vec3::ZERO, pull_k_at_1m,
        );

        let buds = resolve_face_nodes(&program.face_nodes, &face_ids, &template);

        log::info!("Build initialized: seed={:?}, {} joints, {} buds",
            program.seed, physics.active_joints, buds.len());

        Self {
            faces, approach, buds,
            stage: BuildStage::Building,
            positions,
            pull_k_at_1m,
            marks: Vec::new(),
        }
    }

    /// Advance one frame. Returns true if topology changed.
    pub fn tick(
        &mut self,
        physics: &mut GrowablePhysics,
        queue: &wgpu::Queue,
    ) -> bool {
        self.approach.tick(physics, queue);

        match self.stage {
            BuildStage::Building => {
                if !self.approach.all_settled() {
                    return false;
                }
                if self.buds.is_empty() {
                    log::info!("Build complete: {} joints, {} push, {} elastic",
                        physics.active_joints, physics.active_push, physics.active_elastic);
                    self.stage = BuildStage::Pretensing;
                    return false;
                }

                let current_buds: Vec<Bud> = self.buds.drain(..).collect();
                let mut next_buds = Vec::new();
                let mut any_placed = false;

                for bud in current_buds {
                    if bud.column_remaining > 0 {
                        // Place a column brick
                        let face = self.faces.get(bud.face_id)
                            .expect("bud face not found");
                        let spin = face.spin;
                        let template = BrickTemplate::for_spin_baked(spin);

                        let new_face_ids = placement::place_brick(
                            physics, queue,
                            &mut self.approach, &mut self.faces,
                            &template, bud.face_id,
                            &mut self.positions, self.pull_k_at_1m,
                        );
                        any_placed = true;

                        if let Some(&forward_id) = new_face_ids.first() {
                            next_buds.push(Bud {
                                face_id: forward_id,
                                column_remaining: bud.column_remaining - 1,
                                column_scale: bud.column_scale,
                                after: bud.after,
                            });
                        }
                    } else {
                        // Column done — execute after nodes
                        self.execute_after_nodes(
                            &bud.after, bud.face_id,
                            physics, queue, &mut next_buds, &mut any_placed,
                        );
                    }
                }

                self.buds = next_buds;
                any_placed
            }
            BuildStage::Pretensing => {
                if self.approach.all_settled() {
                    log::info!("Pretensing complete");
                    self.stage = BuildStage::Falling;
                }
                false
            }
            BuildStage::Falling | BuildStage::Complete => false,
        }
    }

    fn execute_after_nodes(
        &mut self,
        nodes: &[BuildNode],
        face_id: FaceId,
        physics: &mut GrowablePhysics,
        queue: &wgpu::Queue,
        next_buds: &mut Vec<Bud>,
        any_placed: &mut bool,
    ) {
        for node in nodes {
            match node {
                BuildNode::Column { count, scale, after } => {
                    next_buds.push(Bud {
                        face_id,
                        column_remaining: *count,
                        column_scale: *scale,
                        after: after.clone(),
                    });
                }
                BuildNode::Mark { name } => {
                    self.marks.push((name, face_id));
                    log::info!("Marked face {:?} as '{}'", face_id, name);
                }
                BuildNode::Prism { outer_pct } => {
                    placement::add_prism(
                        physics, queue,
                        &mut self.approach, &self.faces,
                        face_id, *outer_pct,
                        &mut self.positions, self.pull_k_at_1m,
                    );
                    *any_placed = true;
                    log::info!("Added prism to face {:?} (outer {}%)", face_id, outer_pct);
                }
                BuildNode::Open => {
                    // Remove face radials (set K=0)
                    if let Some(face) = self.faces.get(face_id) {
                        for &idx in &face.radial_elastic_indices {
                            physics.write_elastic_ideal_at(queue, idx, 1e6, 0.0);
                        }
                    }
                }
                BuildNode::Face { .. } => {
                    log::warn!("Face targeting in after nodes not yet supported");
                }
            }
        }
    }

    pub fn update_positions(&mut self, positions: Vec<[f32; 4]>) {
        self.positions = positions;
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

fn resolve_face_nodes(
    nodes: &[BuildNode],
    face_ids: &[FaceId],
    template: &BrickTemplate,
) -> Vec<Bud> {
    let mut buds = Vec::new();
    for node in nodes {
        match node {
            BuildNode::Face { name, node } => {
                let face_idx = template.face_index_by_name(name)
                    .or_else(|| {
                        if *name == "forward" {
                            template.faces.iter().position(|f| f.is_forward)
                        } else {
                            None
                        }
                    });
                if let Some(idx) = face_idx {
                    if idx < face_ids.len() {
                        buds.extend(resolve_single_node(node, face_ids[idx]));
                    }
                } else {
                    log::warn!("Face '{}' not found in template", name);
                }
            }
            _ => {
                log::warn!("Top-level non-Face node ignored");
            }
        }
    }
    buds
}

fn resolve_single_node(node: &BuildNode, face_id: FaceId) -> Vec<Bud> {
    match node {
        BuildNode::Column { count, scale, after } => {
            vec![Bud {
                face_id,
                column_remaining: *count,
                column_scale: *scale,
                after: after.clone(),
            }]
        }
        BuildNode::Prism { .. } | BuildNode::Mark { .. } | BuildNode::Open => {
            // These are immediate actions, wrap in a zero-column bud
            vec![Bud {
                face_id,
                column_remaining: 0,
                column_scale: 1.0,
                after: vec![node.clone()],
            }]
        }
        BuildNode::Face { .. } => {
            log::warn!("Nested Face node not yet supported");
            vec![]
        }
    }
}
