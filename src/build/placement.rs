use glam::{Mat4, Vec3};

use super::approach::ApproachManager;
use super::brick::BrickTemplate;
use super::face::{face_basis, FaceId, FaceRegistry};
use crate::constants::PUSH_LINEAR_DENSITY;
use crate::gpu::growable::GrowablePhysics;

/// Place a brick on an existing face.
///
/// `positions` must contain current positions for all active joints (from last readback).
/// Returns the FaceIds of any new forward-growth faces created.
pub fn place_brick(
    physics: &mut GrowablePhysics,
    queue: &wgpu::Queue,
    approach: &mut ApproachManager,
    faces: &mut FaceRegistry,
    template: &BrickTemplate,
    attach_face_id: FaceId,
    positions: &mut Vec<[f32; 4]>,
    pull_k_at_1m: f32,
) -> Vec<FaceId> {
    let face = faces.get(attach_face_id).expect("attach face not found");
    let existing_corners = [
        pos_vec3(positions, face.corners[0]),
        pos_vec3(positions, face.corners[1]),
        pos_vec3(positions, face.corners[2]),
    ];
    let existing_corner_indices = face.corners;
    let face_scale = face.scale;

    // Find the template's attach face
    let attach_face_def = template
        .faces
        .iter()
        .find(|f| f.is_attach)
        .expect("template has no attach face");
    let template_attach_corners = [
        template.joints[attach_face_def.corners[0]],
        template.joints[attach_face_def.corners[1]],
        template.joints[attach_face_def.corners[2]],
    ];

    // Compute the rigid transform: template space -> world space
    let transform = placement_transform(existing_corners, template_attach_corners);

    // Build mapping from template joint index -> global joint index
    let mut joint_map: Vec<u32> = vec![u32::MAX; template.joints.len()];

    // Map attach face corners to existing joints
    for (i, &template_idx) in attach_face_def.corners.iter().enumerate() {
        joint_map[template_idx] = existing_corner_indices[i];
    }

    // Create new joints for non-attach joints
    let mut new_positions = Vec::new();
    let mut new_velocities = Vec::new();
    let mut new_joint_template_indices = Vec::new();

    for (i, local_pos) in template.joints.iter().enumerate() {
        if joint_map[i] == u32::MAX {
            let world_pos = transform.transform_point3(*local_pos);
            new_positions.push([world_pos.x, world_pos.y, world_pos.z, 0.0]);
            new_velocities.push([0.0f32; 4]);
            new_joint_template_indices.push(i);
        }
    }

    if !new_positions.is_empty() {
        let start = physics.append_joints(queue, &new_positions, &new_velocities);
        for (offset, &template_idx) in new_joint_template_indices.iter().enumerate() {
            joint_map[template_idx] = start + offset as u32;
        }
        // Extend the positions array so face registration can use it
        positions.extend_from_slice(&new_positions);
    }

    // Create push intervals with approaching ideals
    for &(a, o, target_ideal) in &template.pushes {
        let alpha = joint_map[a];
        let omega = joint_map[o];
        let actual = (pos_vec3(positions, alpha) - pos_vec3(positions, omega)).length();
        let start_ideal = actual.max(0.1);
        let k = pull_k_at_1m / target_ideal;
        let half_mass = PUSH_LINEAR_DENSITY * target_ideal / 2.0;

        let idx = physics.append_push(
            queue,
            &[alpha],
            &[omega],
            &[start_ideal],
            &[k * 0.1],
            &[half_mass],
        );
        approach.add_push(idx as usize, start_ideal, target_ideal, k);
    }

    // Create pull intervals with approaching ideals
    for &(a, o, target_ideal) in &template.pulls {
        let alpha = joint_map[a];
        let omega = joint_map[o];
        let actual = (pos_vec3(positions, alpha) - pos_vec3(positions, omega)).length();
        let start_ideal = actual.max(0.1);
        let k = pull_k_at_1m / target_ideal;

        let idx = physics.append_elastic(
            queue,
            &[alpha],
            &[omega],
            &[start_ideal],
            &[k * 0.1],
        );
        approach.add_elastic(idx as usize, start_ideal, target_ideal, k);
    }

    physics.update_counts(queue);

    // Register new faces (non-attach faces)
    let mut new_face_ids = Vec::new();
    for face_def in &template.faces {
        if face_def.is_attach {
            continue;
        }
        let corners = [
            joint_map[face_def.corners[0]],
            joint_map[face_def.corners[1]],
            joint_map[face_def.corners[2]],
        ];
        let face_id = faces.create_face(
            physics,
            queue,
            approach,
            corners,
            face_def.spin,
            face_scale,
            positions,
            pull_k_at_1m,
        );
        // Extend positions with the new centroid joint
        let c0 = pos_vec3(positions, corners[0]);
        let c1 = pos_vec3(positions, corners[1]);
        let c2 = pos_vec3(positions, corners[2]);
        let centroid = (c0 + c1 + c2) / 3.0;
        positions.push([centroid.x, centroid.y, centroid.z, 0.0]);

        if face_def.is_forward {
            new_face_ids.push(face_id);
        }
    }

    physics.update_counts(queue);

    // Join the template's attach face with the existing face
    // Create a temporary face for the brick's attach face (same corners as existing)
    let attach_template_face_id = faces.create_face(
        physics,
        queue,
        approach,
        existing_corner_indices,
        attach_face_def.spin,
        face_scale,
        positions,
        pull_k_at_1m,
    );
    // Add centroid for this face too
    let c0 = pos_vec3(positions, existing_corner_indices[0]);
    let c1 = pos_vec3(positions, existing_corner_indices[1]);
    let c2 = pos_vec3(positions, existing_corner_indices[2]);
    let centroid = (c0 + c1 + c2) / 3.0;
    positions.push([centroid.x, centroid.y, centroid.z, 0.0]);

    physics.update_counts(queue);

    faces.join_faces(
        physics,
        queue,
        approach,
        attach_face_id,
        attach_template_face_id,
        positions,
        pull_k_at_1m,
    );

    physics.update_counts(queue);

    new_face_ids
}

/// Place the first brick (seed) at a given position.
/// Returns the FaceIds of all created faces.
pub fn place_seed_brick(
    physics: &mut GrowablePhysics,
    queue: &wgpu::Queue,
    approach: &mut ApproachManager,
    faces: &mut FaceRegistry,
    template: &BrickTemplate,
    origin: Vec3,
    pull_k_at_1m: f32,
) -> (Vec<FaceId>, Vec<[f32; 4]>) {
    let mut positions: Vec<[f32; 4]> = template
        .joints
        .iter()
        .map(|j| [j.x + origin.x, j.y + origin.y, j.z + origin.z, 0.0])
        .collect();
    let velocities = vec![[0.0f32; 4]; positions.len()];

    let start_joint = physics.append_joints(queue, &positions, &velocities);

    // Create push intervals at target (no approach for seed)
    for &(a, o, target_ideal) in &template.pushes {
        let alpha = start_joint + a as u32;
        let omega = start_joint + o as u32;
        let k = pull_k_at_1m / target_ideal;
        let half_mass = PUSH_LINEAR_DENSITY * target_ideal / 2.0;

        physics.append_push(
            queue,
            &[alpha],
            &[omega],
            &[target_ideal],
            &[k],
            &[half_mass],
        );
    }

    // Create pull intervals at target
    for &(a, o, target_ideal) in &template.pulls {
        let alpha = start_joint + a as u32;
        let omega = start_joint + o as u32;
        let k = pull_k_at_1m / target_ideal;

        physics.append_elastic(
            queue,
            &[alpha],
            &[omega],
            &[target_ideal],
            &[k],
        );
    }

    physics.update_counts(queue);

    // Register faces
    let mut face_ids = Vec::new();
    for face_def in &template.faces {
        let corners = [
            start_joint + face_def.corners[0] as u32,
            start_joint + face_def.corners[1] as u32,
            start_joint + face_def.corners[2] as u32,
        ];
        let face_id = faces.create_face(
            physics,
            queue,
            approach,
            corners,
            face_def.spin,
            1.0,
            &positions,
            pull_k_at_1m,
        );
        // Add centroid to positions
        let c0 = pos_vec3(&positions, corners[0]);
        let c1 = pos_vec3(&positions, corners[1]);
        let c2 = pos_vec3(&positions, corners[2]);
        let centroid = (c0 + c1 + c2) / 3.0;
        positions.push([centroid.x, centroid.y, centroid.z, 0.0]);

        face_ids.push(face_id);
    }

    physics.update_counts(queue);
    (face_ids, positions)
}

/// Compute the rigid transform that maps template attach corners onto existing face corners.
fn placement_transform(existing_corners: [Vec3; 3], template_corners: [Vec3; 3]) -> Mat4 {
    let existing_mat = face_basis(existing_corners);
    let template_mat = face_basis(template_corners);
    existing_mat * template_mat.inverse()
}

fn pos_vec3(positions: &[[f32; 4]], joint: u32) -> Vec3 {
    let p = positions[joint as usize];
    Vec3::new(p[0], p[1], p[2])
}
