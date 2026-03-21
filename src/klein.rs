use rand::prelude::*;

use crate::constants::*;
use crate::tensegrity::TensegritySphereBuffers;

/// Generate a Klein bottle tensegrity structure.
///
/// Width must be even, height must be odd.
/// Default parameters: width=10, height=31, shift=0.
///
/// Push intervals are stored as spring-based push (not rigid) because
/// Klein bottles have joints shared by multiple struts.
pub fn generate_klein(width: usize, height: usize, shift: usize, pull_k_at_1m: f32) -> TensegritySphereBuffers {
    assert!(width % 2 == 0, "Klein bottle width must be even, got {}", width);
    assert!(height % 2 == 1, "Klein bottle height must be odd, got {}", height);

    let (w, h, sh) = (width as isize, height as isize, shift as isize);

    let joint = |x: isize, y: isize| -> isize {
        let flip = (y / h) % 2 == 1;
        let x_rel = if flip { sh - 1 - x } else { x };
        let x_mod = ((w * 2 + x_rel) % w + w) % w;
        let y_mod = ((y % h) + h) % h;
        (y_mod * w + x_mod) / 2
    };

    let num_joints = (w * h / 2) as usize;
    let mut rng = rand::thread_rng();

    // Random positions inside unit sphere
    let mut positions: Vec<[f32; 4]> = Vec::with_capacity(num_joints);
    for _ in 0..num_joints {
        loop {
            let x: f32 = rng.gen_range(-1.0..1.0);
            let y: f32 = rng.gen_range(-1.0..1.0);
            let z: f32 = rng.gen_range(-1.0..1.0);
            if x * x + y * y + z * z <= 1.0 {
                positions.push([x, y, z, 0.0]);
                break;
            }
        }
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

    let push_ideal_length: f32 = 8.0;
    let pull_ideal_length: f32 = 1.0;

    for y in 0..h {
        for x in 0..w {
            if (x + y) % 2 == 0 {
                let (a, b, c, d, e, f) = (
                    joint(x, y),
                    joint(x - 1, y + 1),
                    joint(x + 1, y + 1),
                    joint(x, y + 2),
                    joint(x - 1, y + 3),
                    joint(x + 1, y + 3),
                );

                // 3 pull cables
                for &(alpha, omega) in &[(a, b), (a, c), (a, d)] {
                    elastic_alpha.push(alpha as u32);
                    elastic_omega.push(omega as u32);
                    elastic_ideal.push(pull_ideal_length);
                    elastic_k.push(pull_k_at_1m / pull_ideal_length);
                }

                // 3 push struts (as springs)
                for &(alpha, omega) in &[(a, e), (a, f), (e, f)] {
                    push_alpha.push(alpha as u32);
                    push_omega.push(omega as u32);
                    push_ideal.push(push_ideal_length);
                    push_k.push(pull_k_at_1m / push_ideal_length);
                    push_half_mass.push(PUSH_LINEAR_DENSITY * push_ideal_length / 2.0);
                }
            }
        }
    }

    let velocities = vec![[0.0f32; 4]; num_joints];

    log::info!(
        "Generated Klein {}x{}: {} joints, {} struts (spring), {} cables",
        width, height,
        num_joints,
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
