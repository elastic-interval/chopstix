use std::f32::consts::PI;

use crate::constants::*;
use crate::tensegrity::TensegritySphereBuffers;

/// Generate a tensegrity Möbius band.
///
/// Creates a zigzag strip that twists 180° as it goes around.
/// Joint count = 2 * segments + 1 (odd, for the Möbius twist).
/// Push intervals share joints, so this uses spring-push mode.
pub fn generate_mobius(segments: usize, pull_k_at_1m: f32) -> TensegritySphereBuffers {
    let joint_count = segments * 2 + 1;

    let band_width = 2.0;
    let radius = 5.0 + (segments as f32 * 0.1);

    // Möbius strip parametric position
    let location = |bottom: bool, angle: f32| -> [f32; 3] {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let major_x = cos_a * radius;
        let major_z = sin_a * radius;
        let out_x = cos_a;
        let out_z = sin_a;
        // Cross-section rotates by angle/2 (the Möbius twist)
        let half = angle / 2.0;
        let ray_x = out_x * half.sin();
        let ray_y = half.cos();
        let ray_z = out_z * half.sin();
        let sign = if bottom { -0.5 } else { 0.5 };
        [
            major_x + ray_x * band_width * sign,
            ray_y * band_width * sign,
            major_z + ray_z * band_width * sign,
        ]
    };

    let mut positions: Vec<[f32; 4]> = Vec::with_capacity(joint_count);
    for i in 0..joint_count {
        let angle = i as f32 / joint_count as f32 * PI * 2.0;
        let [x, y, z] = location(i % 2 == 0, angle);
        positions.push([x, y, z, 0.0]);
    }

    let mut push_alpha: Vec<u32> = Vec::new();
    let mut push_omega: Vec<u32> = Vec::new();
    let mut push_ideal: Vec<f32> = Vec::new();
    let mut push_k: Vec<f32> = Vec::new();
    let mut push_half_mass: Vec<f32> = Vec::new();
    let mut elastic_alpha: Vec<u32> = Vec::new();
    let mut elastic_omega: Vec<u32> = Vec::new();
    let mut elastic_ideal: Vec<f32> = Vec::new();
    let mut elastic_k: Vec<f32> = Vec::new();

    let push_ideal_length: f32 = 4.0;
    let pull_edge: f32 = 2.0;
    let pull_width: f32 = 2.2;

    for joint_index in 0..joint_count {
        let j = |offset: usize| ((joint_index * 2 + offset) % joint_count) as u32;

        // Pull along edge (offset 0 to 2)
        elastic_alpha.push(j(0));
        elastic_omega.push(j(2));
        elastic_ideal.push(pull_edge);
        elastic_k.push(pull_k_at_1m / pull_edge);

        // Pull across width (offset 0 to 1)
        elastic_alpha.push(j(0));
        elastic_omega.push(j(1));
        elastic_ideal.push(pull_width);
        elastic_k.push(pull_k_at_1m / pull_width);

        // Push diagonal (offset 0 to 3)
        push_alpha.push(j(0));
        push_omega.push(j(3));
        push_ideal.push(push_ideal_length);
        push_k.push(pull_k_at_1m / push_ideal_length);
        push_half_mass.push(PUSH_LINEAR_DENSITY * push_ideal_length / 2.0);
    }

    let velocities = vec![[0.0f32; 4]; joint_count];

    log::info!(
        "Generated Möbius {} segments: {} joints, {} struts (spring), {} cables",
        segments,
        joint_count,
        push_alpha.len(),
        elastic_alpha.len()
    );

    TensegritySphereBuffers {
        positions,
        velocities,
        elastic_alpha,
        elastic_omega,
        elastic_ideal,
        elastic_k,
        rigid_alpha: Vec::new(),
        rigid_omega: Vec::new(),
        rigid_length: Vec::new(),
        rigid_half_mass: Vec::new(),
        push_alpha,
        push_omega,
        push_ideal,
        push_k,
        push_half_mass,
        use_spring_push: true,
    }
}
