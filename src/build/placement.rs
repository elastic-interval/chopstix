use glam::{Mat4, Vec3};

use super::approach::ApproachManager;
use super::brick::BrickTemplate;
use super::face::{FaceId, FaceRegistry};
use super::Spin;
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

    // Find the template's attach face index
    let attach_face_idx = template.faces.iter().position(|f| f.is_attach)
        .expect("template has no attach face");

    // Compute outward direction: from structure centroid toward face centroid
    let face_centroid = (existing_corners[0] + existing_corners[1] + existing_corners[2]) / 3.0;
    let struct_centroid = structure_centroid(positions);
    let outward = (face_centroid - struct_centroid).normalize_or_zero();

    // Compute the rigid transform: template space -> world space
    let transform = placement_transform(
        existing_corners, outward, template, attach_face_idx,
    );

    // Build mapping from template joint index -> global joint index
    let mut joint_map: Vec<u32> = vec![u32::MAX; template.joints.len()];

    // Map attach face corners to existing joints
    let attach_face_def = &template.faces[attach_face_idx];
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
        // Determine the correct spin for the placed face: the one whose normal
        // points outward (away from the structure centroid toward the face centroid).
        let fc = (pos_vec3(positions, corners[0])
            + pos_vec3(positions, corners[1])
            + pos_vec3(positions, corners[2])) / 3.0;
        let sc = structure_centroid(positions);
        let outward_dir = (fc - sc).normalize_or_zero();
        let n_left = face_normal_for_spin(positions, corners, Spin::Left);
        let registered_spin = if n_left.dot(outward_dir) > 0.0 { Spin::Left } else { Spin::Right };

        let face_id = faces.create_face(
            physics,
            queue,
            approach,
            corners,
            registered_spin,
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

/// Compute the rigid transform that places a brick on an existing face,
/// extending outward from the structure.
///
/// Uses empirical geometry: the brick's "growth direction" (from attach face
/// centroid toward brick body centroid) is aligned with the existing face's
/// outward normal (from structure centroid toward face centroid).
fn placement_transform(
    existing_corners: [Vec3; 3],
    outward_hint: Vec3,
    template: &BrickTemplate,
    attach_face_idx: usize,
) -> Mat4 {
    let attach_def = &template.faces[attach_face_idx];
    let template_corners = [
        template.joints[attach_def.corners[0]],
        template.joints[attach_def.corners[1]],
        template.joints[attach_def.corners[2]],
    ];

    // Existing face basis: X toward midpoint of edge(c0,c1), Y = outward, Z = X×Y
    let e_centroid = (existing_corners[0] + existing_corners[1] + existing_corners[2]) / 3.0;
    let e_y = outward_hint.normalize_or_zero();
    let e_x = (existing_corners[0] + existing_corners[1] - e_centroid * 2.0).normalize_or_zero();
    let e_z = e_x.cross(e_y).normalize_or_zero();
    let e_x = e_y.cross(e_z).normalize_or_zero(); // re-orthogonalize

    // Template attach face basis: same convention
    let t_centroid = (template_corners[0] + template_corners[1] + template_corners[2]) / 3.0;
    // Growth direction: from attach face toward brick body
    let brick_centroid: Vec3 = template.joints.iter().copied().sum::<Vec3>()
        / template.joints.len() as f32;
    let t_y = (brick_centroid - t_centroid).normalize_or_zero();
    let t_x = (template_corners[0] + template_corners[1] - t_centroid * 2.0).normalize_or_zero();
    let t_z = t_x.cross(t_y).normalize_or_zero();
    let t_x = t_y.cross(t_z).normalize_or_zero(); // re-orthogonalize

    let existing_mat = Mat4::from_cols(
        e_x.extend(0.0), e_y.extend(0.0), e_z.extend(0.0), e_centroid.extend(1.0),
    );
    let template_mat = Mat4::from_cols(
        t_x.extend(0.0), t_y.extend(0.0), t_z.extend(0.0), t_centroid.extend(1.0),
    );

    existing_mat * template_mat.inverse()
}

fn pos_vec3(positions: &[[f32; 4]], joint: u32) -> Vec3 {
    let p = positions[joint as usize];
    Vec3::new(p[0], p[1], p[2])
}

fn structure_centroid(positions: &[[f32; 4]]) -> Vec3 {
    if positions.is_empty() { return Vec3::ZERO; }
    let n = positions.len() as f32;
    let sum: Vec3 = positions.iter().map(|p| Vec3::new(p[0], p[1], p[2])).sum();
    sum / n
}

/// Add a prism to an existing face: push strut + 6 pull cables.
/// Matches tensegrity-lab's `add_face_prism`.
pub fn add_prism(
    physics: &mut GrowablePhysics,
    queue: &wgpu::Queue,
    approach: &mut ApproachManager,
    faces: &FaceRegistry,
    face_id: FaceId,
    outer_pct: f32,
    positions: &mut Vec<[f32; 4]>,
    pull_k_at_1m: f32,
) {
    let face = faces.get(face_id).expect("prism face not found");
    let corners = face.corners;
    let scale = face.scale;

    let c0 = pos_vec3(positions, corners[0]);
    let c1 = pos_vec3(positions, corners[1]);
    let c2 = pos_vec3(positions, corners[2]);
    let midpoint = (c0 + c1 + c2) / 3.0;

    // Outward normal from structure centroid
    let sc = structure_centroid(positions);
    let normal = (midpoint - sc).normalize_or_zero();

    // Radial distance (from face center to corner)
    let radial_dist = (c0 - midpoint).length();

    // Push strut length = face.scale * 1.5
    let push_length = scale.max(radial_dist) * 1.5;
    let half = push_length / 2.0;
    let inner = half;
    let outer = half * (outer_pct / 100.0);
    let total_push = inner + outer;

    // Pull cable lengths (Pythagorean: radial distance + prism depth)
    let alpha_pull = (radial_dist * radial_dist + inner * inner).sqrt();
    let omega_pull = (radial_dist * radial_dist + outer * outer).sqrt();

    // Create prism joints
    let alpha_pos = midpoint - normal * inner;
    let omega_pos = midpoint + normal * outer;

    let alpha_joint = physics.append_joints(
        queue,
        &[[alpha_pos.x, alpha_pos.y, alpha_pos.z, 0.0]],
        &[[0.0f32; 4]],
    );
    let omega_joint = physics.append_joints(
        queue,
        &[[omega_pos.x, omega_pos.y, omega_pos.z, 0.0]],
        &[[0.0f32; 4]],
    );
    positions.push([alpha_pos.x, alpha_pos.y, alpha_pos.z, 0.0]);
    positions.push([omega_pos.x, omega_pos.y, omega_pos.z, 0.0]);

    // Push strut (alpha → omega)
    let push_k = pull_k_at_1m / total_push;
    let push_half_mass = crate::constants::PUSH_LINEAR_DENSITY * total_push / 2.0;
    let push_idx = physics.append_push(
        queue,
        &[alpha_joint], &[omega_joint],
        &[total_push], &[push_k], &[push_half_mass],
    );
    approach.add_push(push_idx as usize, total_push, total_push, push_k);

    // 6 pull cables: each prism joint to each radial corner
    for &corner in &corners {
        // Alpha to corner
        let k_a = pull_k_at_1m / alpha_pull;
        let idx_a = physics.append_elastic(
            queue,
            &[alpha_joint], &[corner],
            &[alpha_pull], &[k_a],
        );
        approach.add_elastic(idx_a as usize, alpha_pull, alpha_pull, k_a);

        // Omega to corner
        let k_o = pull_k_at_1m / omega_pull;
        let idx_o = physics.append_elastic(
            queue,
            &[omega_joint], &[corner],
            &[omega_pull], &[k_o],
        );
        approach.add_elastic(idx_o as usize, omega_pull, omega_pull, k_o);
    }

    physics.update_counts(queue);
}

fn face_normal_for_spin(positions: &[[f32; 4]], corners: [u32; 3], spin: Spin) -> Vec3 {
    let c0 = pos_vec3(positions, corners[0]);
    let c1 = pos_vec3(positions, corners[1]);
    let c2 = pos_vec3(positions, corners[2]);
    let v1 = c1 - c0;
    let v2 = c2 - c0;
    match spin {
        Spin::Left => v2.cross(v1).normalize_or_zero(),
        Spin::Right => v1.cross(v2).normalize_or_zero(),
    }
}
