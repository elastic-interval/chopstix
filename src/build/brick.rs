use glam::Vec3;
use std::f32::consts::PI;

use super::Spin;

pub struct BrickFace {
    pub corners: [usize; 3],
    pub spin: Spin,
    pub is_attach: bool,
    pub is_forward: bool,
}

pub struct BrickTemplate {
    pub joints: Vec<Vec3>,
    /// (alpha_idx, omega_idx, ideal_length) in template-local joint indices
    pub pushes: Vec<(usize, usize, f32)>,
    /// (alpha_idx, omega_idx, ideal_length) in template-local joint indices
    pub pulls: Vec<(usize, usize, f32)>,
    pub faces: Vec<BrickFace>,
}

impl BrickTemplate {
    /// Create a single-twist-left brick.
    ///
    /// 6 joints forming two triangles connected by 3 push struts (twisted left)
    /// and 3 pull cables (helical).
    ///
    /// Bottom face (attach): joints 0, 1, 2
    /// Top face (forward):   joints 3, 4, 5
    pub fn single_twist_left(face_radius: f32, height: f32) -> Self {
        let twist = PI / 6.0; // 30 degree twist

        // Bottom triangle (z = 0)
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

        // Top triangle (z = height), rotated by twist
        let j3 = Vec3::new(
            face_radius * twist.cos(),
            face_radius * twist.sin(),
            height,
        );
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

        // Push struts: left twist means bottom[i] -> top[(i+2)%3]
        let pushes = vec![
            (0, 5, (j0 - j5).length()), // j0 -> j5
            (1, 3, (j1 - j3).length()), // j1 -> j3
            (2, 4, (j2 - j4).length()), // j2 -> j4
        ];

        // Pull cables: helical, bottom[i] -> top[i]
        let pulls = vec![
            (0, 3, (j0 - j3).length()), // j0 -> j3
            (1, 4, (j1 - j4).length()), // j1 -> j4
            (2, 5, (j2 - j5).length()), // j2 -> j5
        ];

        let faces = vec![
            BrickFace {
                corners: [0, 1, 2],
                spin: Spin::Left,
                is_attach: true,
                is_forward: false,
            },
            BrickFace {
                corners: [3, 4, 5],
                spin: Spin::Right,
                is_attach: false,
                is_forward: true,
            },
        ];

        Self {
            joints,
            pushes,
            pulls,
            faces,
        }
    }

    /// Create a single-twist-right brick by mirroring X coordinates.
    pub fn single_twist_right(face_radius: f32, height: f32) -> Self {
        let mut template = Self::single_twist_left(face_radius, height);

        // Mirror X coordinates
        for joint in &mut template.joints {
            joint.x = -joint.x;
        }

        // Recompute lengths (mirroring preserves distances, but be safe)
        for push in &mut template.pushes {
            push.2 = (template.joints[push.0] - template.joints[push.1]).length();
        }
        for pull in &mut template.pulls {
            pull.2 = (template.joints[pull.0] - template.joints[pull.1]).length();
        }

        // Swap spins and attach/forward roles stay the same but spins flip
        for face in &mut template.faces {
            face.spin = face.spin.opposite();
            // Reverse winding to maintain consistent normal direction after mirror
            face.corners.swap(1, 2);
        }

        template
    }

    /// Get the appropriate template for a given spin.
    pub fn for_spin(spin: Spin, face_radius: f32, height: f32) -> Self {
        match spin {
            Spin::Left => Self::single_twist_left(face_radius, height),
            Spin::Right => Self::single_twist_right(face_radius, height),
        }
    }
}
