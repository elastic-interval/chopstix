use glam::Vec3;

use super::Spin;

pub struct BrickFace {
    pub corners: [usize; 3],
    pub spin: Spin,
    pub is_attach: bool,
    pub is_forward: bool,
    /// Optional name for face lookup (used by hub bricks like Omni)
    pub name: Option<&'static str>,
}

pub struct BrickTemplate {
    pub joints: Vec<Vec3>,
    /// (alpha_idx, omega_idx, strain) in template-local joint indices.
    /// `strain` is the baked equilibrium strain from tensegrity-lab:
    /// `strain = (actual - ideal) / ideal`, so `ideal = actual / (1 + strain)`.
    /// Push struts have negative strain (compressed), pulls positive (stretched).
    pub pushes: Vec<(usize, usize, f32)>,
    /// (alpha_idx, omega_idx, strain) — see `pushes` for semantics.
    pub pulls: Vec<(usize, usize, f32)>,
    pub faces: Vec<BrickFace>,
}

impl BrickTemplate {
    /// SingleTwistLeft using tensegrity-lab's baked equilibrium coordinates.
    ///
    /// Joint order: AlphaX(0), OmegaX(1), AlphaY(2), OmegaY(3), AlphaZ(4), OmegaZ(5)
    /// Bottom face (attach): [0, 2, 4] = [AlphaX, AlphaY, AlphaZ] - Left spin
    /// Top face (forward):   [5, 3, 1] = [OmegaZ, OmegaY, OmegaX] - Left spin
    ///
    /// Push struts: AlphaX-OmegaX, AlphaY-OmegaY, AlphaZ-OmegaZ
    /// Pull cables: AlphaX-OmegaY, AlphaY-OmegaZ, AlphaZ-OmegaX
    pub fn single_twist_left_baked() -> Self {
        let joints = vec![
            Vec3::new(-1.10019, -0.96247, 0.00000), // 0: AlphaX
            Vec3::new(0.95280, 0.96246, -0.55010),   // 1: OmegaX
            Vec3::new(0.55011, -0.96245, 0.95280),   // 2: AlphaY
            Vec3::new(-0.95281, 0.96245, -0.55010),   // 3: OmegaY
            Vec3::new(0.55011, -0.96246, -0.95280),   // 4: AlphaZ
            Vec3::new(-0.00001, 0.96246, 1.10020),    // 5: OmegaZ
        ];

        // Baked equilibrium strain from tensegrity-lab/src/build/dsl/brick_library/baked_bricks.rs
        // (single_twist_left_baked). Push = compression, pull = tension.
        let push_strain = -0.01509_f32;
        let pull_strain = 0.10576_f32;

        let pushes = vec![
            (0, 1, push_strain), // AlphaX → OmegaX
            (2, 3, push_strain), // AlphaY → OmegaY
            (4, 5, push_strain), // AlphaZ → OmegaZ
        ];

        let pulls = vec![
            (0, 3, pull_strain), // AlphaX → OmegaY
            (2, 5, pull_strain), // AlphaY → OmegaZ
            (4, 1, pull_strain), // AlphaZ → OmegaX
        ];

        let faces = vec![
            BrickFace {
                corners: [0, 2, 4], // AlphaX, AlphaY, AlphaZ
                spin: Spin::Left,
                is_attach: true,
                is_forward: false,
                name: None,
            },
            BrickFace {
                corners: [5, 3, 1], // OmegaZ, OmegaY, OmegaX
                spin: Spin::Left,
                is_attach: false,
                is_forward: true,
                name: None,
            },
        ];

        Self { joints, pushes, pulls, faces }
    }

    /// SingleTwistRight: mirror of left. Strain is invariant under mirroring.
    pub fn single_twist_right_baked() -> Self {
        let mut template = Self::single_twist_left_baked();

        // Mirror X coordinates
        for joint in &mut template.joints {
            joint.x = -joint.x;
        }

        // Flip spins and reverse winding
        for face in &mut template.faces {
            face.spin = face.spin.opposite();
            face.corners.swap(1, 2);
        }

        template
    }

    /// Get SingleTwist template for a given spin (baked coordinates).
    /// No pre-orientation — placement_transform handles alignment.
    pub fn for_spin_baked(spin: Spin) -> Self {
        match spin {
            Spin::Left => Self::single_twist_left_baked(),
            Spin::Right => Self::single_twist_right_baked(),
        }
    }

    /// OmniSymmetrical brick using tensegrity-lab's baked equilibrium coordinates.
    ///
    /// 12 joints, 6 push struts (2 per axis), 0 pull cables, 8 faces.
    /// This is the hub brick for the Open Claw and other branching structures.
    ///
    /// Joint order (following push definition order):
    ///   X pushes: 0=BotAlphaX, 1=BotOmegaX, 2=TopAlphaX, 3=TopOmegaX
    ///   Y pushes: 4=BotAlphaY, 5=BotOmegaY, 6=TopAlphaY, 7=TopOmegaY
    ///   Z pushes: 8=BotAlphaZ, 9=BotOmegaZ, 10=TopAlphaZ, 11=TopOmegaZ
    pub fn omni_symmetrical() -> Self {
        let joints = vec![
            Vec3::new(-1.55675,  0.00000, -0.77838), //  0: BotAlphaX
            Vec3::new( 1.55675,  0.00000, -0.77838), //  1: BotOmegaX
            Vec3::new(-1.55675,  0.00000,  0.77838), //  2: TopAlphaX
            Vec3::new( 1.55675,  0.00000,  0.77838), //  3: TopOmegaX
            Vec3::new(-0.77838, -1.55675,  0.00000), //  4: BotAlphaY
            Vec3::new(-0.77838,  1.55675,  0.00000), //  5: BotOmegaY
            Vec3::new( 0.77838, -1.55675,  0.00000), //  6: TopAlphaY
            Vec3::new( 0.77838,  1.55675,  0.00000), //  7: TopOmegaY
            Vec3::new( 0.00000, -0.77838, -1.55675), //  8: BotAlphaZ
            Vec3::new( 0.00000, -0.77838,  1.55675), //  9: BotOmegaZ
            Vec3::new( 0.00000,  0.77839, -1.55675), // 10: TopAlphaZ
            Vec3::new( 0.00000,  0.77839,  1.55675), // 11: TopOmegaZ
        ];

        // Baked strain from tensegrity-lab omni_symmetrical_baked.
        let push_strain = -0.01428_f32;
        let pushes = vec![
            (0, 1, push_strain),   // BotAlphaX → BotOmegaX
            (2, 3, push_strain),   // TopAlphaX → TopOmegaX
            (4, 5, push_strain),   // BotAlphaY → BotOmegaY
            (6, 7, push_strain),   // TopAlphaY → TopOmegaY
            (8, 9, push_strain),   // BotAlphaZ → BotOmegaZ
            (10, 11, push_strain), // TopAlphaZ → TopOmegaZ
        ];

        // Omni brick has no pull cables — all connectivity comes from face joins
        let pulls = vec![];

        // 8 faces matching tensegrity-lab's Seed(1) role aliases
        let faces = vec![
            // Face 0: OmniTop [TopOmegaX, TopOmegaY, TopOmegaZ] - Right
            BrickFace {
                corners: [3, 7, 11],
                spin: Spin::Right,
                is_attach: false,
                is_forward: false,
                name: Some("OmniTop"),
            },
            // Face 1: OmniTopX [TopOmegaX, TopAlphaY, BotOmegaZ] - Left
            BrickFace {
                corners: [3, 6, 9],
                spin: Spin::Left,
                is_attach: false,
                is_forward: false,
                name: Some("OmniTopX"),
            },
            // Face 2: OmniTopY [TopOmegaY, TopAlphaZ, BotOmegaX] - Left
            BrickFace {
                corners: [7, 10, 1],
                spin: Spin::Left,
                is_attach: false,
                is_forward: false,
                name: Some("OmniTopY"),
            },
            // Face 3: OmniTopZ [TopOmegaZ, TopAlphaX, BotOmegaY] - Left
            BrickFace {
                corners: [11, 2, 5],
                spin: Spin::Left,
                is_attach: false,
                is_forward: false,
                name: Some("OmniTopZ"),
            },
            // Face 4: OmniBotZ [BotAlphaZ, BotOmegaX, TopAlphaY] - Right
            BrickFace {
                corners: [8, 1, 6],
                spin: Spin::Right,
                is_attach: false,
                is_forward: true,
                name: Some("OmniBotZ"),
            },
            // Face 5: OmniBotY [BotAlphaY, BotOmegaZ, TopAlphaX] - Right
            BrickFace {
                corners: [4, 9, 2],
                spin: Spin::Right,
                is_attach: false,
                is_forward: true,
                name: Some("OmniBotY"),
            },
            // Face 6: OmniBotX [BotAlphaX, BotOmegaY, TopAlphaZ] - Right
            BrickFace {
                corners: [0, 5, 10],
                spin: Spin::Right,
                is_attach: false,
                is_forward: true,
                name: Some("OmniBotX"),
            },
            // Face 7: OmniBot [BotAlphaX, BotAlphaY, BotAlphaZ] - Left (open)
            BrickFace {
                corners: [0, 4, 8],
                spin: Spin::Left,
                is_attach: false,
                is_forward: false,
                name: Some("OmniBot"),
            },
        ];

        Self { joints, pushes, pulls, faces }
    }

    /// Parametric single-twist-left (original implementation).
    #[allow(dead_code)]
    pub fn single_twist_left(face_radius: f32, height: f32) -> Self {
        use std::f32::consts::PI;
        let twist = PI / 6.0;

        let j0 = Vec3::new(face_radius, 0.0, 0.0);
        let j1 = Vec3::new(
            face_radius * (2.0 * PI / 3.0).cos(),
            face_radius * (2.0 * PI / 3.0).sin(),
            0.0,
        );
        let j2 = Vec3::new(
            face_radius * (4.0 * PI / 3.0).cos(),
            face_radius * (4.0 * PI / 3.0).sin(),
            0.0,
        );
        let j3 = Vec3::new(face_radius * twist.cos(), face_radius * twist.sin(), height);
        let j4 = Vec3::new(
            face_radius * (twist + 2.0 * PI / 3.0).cos(),
            face_radius * (twist + 2.0 * PI / 3.0).sin(),
            height,
        );
        let j5 = Vec3::new(
            face_radius * (twist + 4.0 * PI / 3.0).cos(),
            face_radius * (twist + 4.0 * PI / 3.0).sin(),
            height,
        );

        let joints = vec![j0, j1, j2, j3, j4, j5];
        // Parametric fallback: no equilibrium strain (structure is defined purely by geometry).
        let pushes = vec![
            (0, 5, 0.0),
            (1, 3, 0.0),
            (2, 4, 0.0),
        ];
        let pulls = vec![
            (0, 3, 0.0),
            (1, 4, 0.0),
            (2, 5, 0.0),
        ];
        let faces = vec![
            BrickFace { corners: [0, 1, 2], spin: Spin::Left, is_attach: true, is_forward: false, name: None },
            BrickFace { corners: [3, 4, 5], spin: Spin::Right, is_attach: false, is_forward: true, name: None },
        ];
        Self { joints, pushes, pulls, faces }
    }

    #[allow(dead_code)]
    pub fn single_twist_right(face_radius: f32, height: f32) -> Self {
        let mut template = Self::single_twist_left(face_radius, height);
        for joint in &mut template.joints { joint.x = -joint.x; }
        for face in &mut template.faces {
            face.spin = face.spin.opposite();
            face.corners.swap(1, 2);
        }
        template
    }

    #[allow(dead_code)]
    pub fn for_spin(spin: Spin, face_radius: f32, height: f32) -> Self {
        match spin {
            Spin::Left => Self::single_twist_left(face_radius, height),
            Spin::Right => Self::single_twist_right(face_radius, height),
        }
    }

    /// Find a face by name.
    pub fn face_index_by_name(&self, name: &str) -> Option<usize> {
        self.faces.iter().position(|f| f.name == Some(name))
    }
}

impl BrickTemplate {
    /// Compute the face normal for a given face index, accounting for spin.
    pub fn face_normal(&self, face_idx: usize) -> Vec3 {
        let face = &self.faces[face_idx];
        let c = face.corners;
        let v1 = self.joints[c[1]] - self.joints[c[0]];
        let v2 = self.joints[c[2]] - self.joints[c[0]];
        match face.spin {
            Spin::Left => v2.cross(v1).normalize(),
            Spin::Right => v1.cross(v2).normalize(),
        }
    }

    /// Compute a rotation that orients the brick so the given "downward" face normal
    /// points in the -Y direction, matching tensegrity-lab's `down_rotation`.
    pub fn down_rotation(&self, down_face_name: &str) -> glam::Quat {
        let face_idx = self.face_index_by_name(down_face_name)
            .unwrap_or_else(|| panic!("No face named '{}'", down_face_name));
        let down_normal = self.face_normal(face_idx);
        glam::Quat::from_rotation_arc(down_normal, -Vec3::Y)
    }

    /// Apply a rotation to all joint positions. Strain is invariant under rigid rotation.
    pub fn apply_rotation(&mut self, rotation: glam::Quat) {
        for joint in &mut self.joints {
            *joint = rotation * *joint;
        }
    }
}
