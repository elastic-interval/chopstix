use std::collections::HashMap;

use glam::Vec3;

use super::approach::ApproachManager;
use super::Spin;
use crate::gpu::growable::GrowablePhysics;

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct FaceId(pub u32);

#[allow(dead_code)]
pub struct Face {
    pub id: FaceId,
    pub centroid_joint: u32,
    pub corners: [u32; 3],
    pub radial_elastic_indices: [usize; 3],
    pub spin: Spin,
    pub scale: f32,
}

pub struct FaceRegistry {
    faces: HashMap<FaceId, Face>,
    next_id: u32,
}

impl FaceRegistry {
    pub fn new() -> Self {
        Self {
            faces: HashMap::new(),
            next_id: 0,
        }
    }

    /// Create a face: adds a centroid joint and 3 radial elastic intervals.
    pub fn create_face(
        &mut self,
        physics: &mut GrowablePhysics,
        queue: &wgpu::Queue,
        approach: &mut ApproachManager,
        corners: [u32; 3],
        spin: Spin,
        scale: f32,
        positions: &[[f32; 4]],
        pull_k_at_1m: f32,
    ) -> FaceId {
        let id = FaceId(self.next_id);
        self.next_id += 1;

        // Compute centroid position from corner positions
        let c0 = corner_pos(positions, corners[0]);
        let c1 = corner_pos(positions, corners[1]);
        let c2 = corner_pos(positions, corners[2]);
        let centroid = (c0 + c1 + c2) / 3.0;

        // Add centroid joint
        let centroid_joint = physics.append_joints(
            queue,
            &[[centroid.x, centroid.y, centroid.z, 0.0]],
            &[[0.0f32; 4]],
        );

        // Add 3 radial elastic intervals (centroid → each corner)
        let mut radial_indices = [0usize; 3];
        for (i, &corner) in corners.iter().enumerate() {
            let actual_length = (corner_pos(positions, corner) - centroid).length();
            let ideal = actual_length;
            let target_ideal = actual_length;
            let k = pull_k_at_1m / target_ideal;

            let elastic_idx = physics.append_elastic(
                queue,
                &[centroid_joint],
                &[corner],
                &[ideal],
                &[k],
            );
            radial_indices[i] = elastic_idx as usize;

            approach.add_elastic(elastic_idx as usize, actual_length, target_ideal, k);
        }

        self.faces.insert(
            id,
            Face {
                id,
                centroid_joint,
                corners,
                radial_elastic_indices: radial_indices,
                spin,
                scale,
            },
        );

        id
    }

    /// Join two faces: creates 6 circumference cables between their corners
    /// and "removes" both faces by zeroing radial K values.
    pub fn join_faces(
        &mut self,
        physics: &mut GrowablePhysics,
        queue: &wgpu::Queue,
        approach: &mut ApproachManager,
        face_a_id: FaceId,
        face_b_id: FaceId,
        positions: &[[f32; 4]],
        pull_k_at_1m: f32,
    ) {
        let face_a = self.faces.get(&face_a_id).expect("face_a not found");
        let face_b = self.faces.get(&face_b_id).expect("face_b not found");

        let a_corners = face_a.corners;
        let b_corners = face_b.corners;
        let a_radials = face_a.radial_elastic_indices;
        let b_radials = face_b.radial_elastic_indices;

        // Create 6 circumference cables: a[i]→b[(i+1)%3] and a[i]→b[(i+2)%3]
        for i in 0..3 {
            let j1 = (i + 1) % 3;
            let j2 = (i + 2) % 3;

            let len1 = (corner_pos(positions, a_corners[i]) - corner_pos(positions, b_corners[j1])).length();
            if len1 > 1e-6 {
                let k1 = pull_k_at_1m / len1;
                let idx1 = physics.append_elastic(
                    queue,
                    &[a_corners[i]],
                    &[b_corners[j1]],
                    &[len1 * 1.5],
                    &[k1 * 0.1],
                );
                approach.add_elastic(idx1 as usize, len1 * 1.5, len1, k1);
            }

            let len2 = (corner_pos(positions, a_corners[i]) - corner_pos(positions, b_corners[j2])).length();
            if len2 > 1e-6 {
                let k2 = pull_k_at_1m / len2;
                let idx2 = physics.append_elastic(
                    queue,
                    &[a_corners[i]],
                    &[b_corners[j2]],
                    &[len2 * 1.5],
                    &[k2 * 0.1],
                );
                approach.add_elastic(idx2 as usize, len2 * 1.5, len2, k2);
            }
        }

        // "Remove" radial intervals: set K=0 and ideal=large to avoid div-by-zero
        // in the shader's strain = (actual - ideal) / ideal computation
        for &idx in a_radials.iter().chain(b_radials.iter()) {
            physics.write_elastic_ideal_at(queue, idx, 1e6, 0.0);
        }

        self.faces.remove(&face_a_id);
        self.faces.remove(&face_b_id);
    }

    pub fn get(&self, id: FaceId) -> Option<&Face> {
        self.faces.get(&id)
    }

}


fn corner_pos(positions: &[[f32; 4]], joint: u32) -> Vec3 {
    let p = positions[joint as usize];
    Vec3::new(p[0], p[1], p[2])
}
