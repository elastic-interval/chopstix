// Joint data (Group 0)
@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> force_x: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> force_y: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> force_z: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> masses: array<atomic<i32>>;

// Interval topology (Group 1)
@group(1) @binding(0) var<storage, read> elastic_alpha: array<u32>;
@group(1) @binding(1) var<storage, read> elastic_omega: array<u32>;
@group(1) @binding(2) var<storage, read> elastic_ideal: array<f32>;
@group(1) @binding(3) var<storage, read> elastic_k: array<f32>;
@group(1) @binding(4) var<storage, read> rigid_alpha: array<u32>;
@group(1) @binding(5) var<storage, read> rigid_omega: array<u32>;
@group(1) @binding(6) var<storage, read> rigid_length: array<f32>;
@group(1) @binding(7) var<storage, read> rigid_half_mass: array<f32>;

// Uniform params (Group 2)
struct Params {
    dt: f32,
    gravity: f32,
    drag: f32,
    viscosity: f32,
    num_joints: u32,
    num_elastic: u32,
    num_rigid: u32,
    ambient_mass: f32,
    force_scale: f32,
    ground_y: f32,
    restitution: f32,
    speed_limit: f32,
}
@group(2) @binding(0) var<uniform> params: Params;

// Check velocity against speed limit; if exceeded, zero velocity and flag the joint as frozen.
// The w component of velocity is used as a "nuked" flag (1.0 = this joint blew up).
fn enforce_speed_limit(vel: vec4<f32>) -> vec4<f32> {
    let speed_sq = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
    let limit_sq = params.speed_limit * params.speed_limit;
    if speed_sq > limit_sq {
        // Kill this joint — zero velocity, mark as nuked
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return vel;
}

const MASS_SCALE: f32 = 1e4;

// Pass 1: Half kick and drift
// v += 0.5 * (F/m) * dt; x += v * dt
@compute @workgroup_size(64)
fn half_kick_and_drift(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= params.num_joints { return; }

    let fx = f32(atomicLoad(&force_x[idx])) / params.force_scale;
    let fy = f32(atomicLoad(&force_y[idx])) / params.force_scale;
    let fz = f32(atomicLoad(&force_z[idx])) / params.force_scale;
    let m = f32(atomicLoad(&masses[idx])) / MASS_SCALE;

    var vel = velocities[idx];
    // If this joint was nuked, keep it frozen
    if vel.w > 0.5 { return; }

    let inv_m = select(0.0, 1.0 / m, m > 0.0);
    vel.x += 0.5 * fx * inv_m * params.dt;
    vel.y += 0.5 * fy * inv_m * params.dt;
    vel.z += 0.5 * fz * inv_m * params.dt;
    vel = enforce_speed_limit(vel);
    velocities[idx] = vel;

    if vel.w > 0.5 {
        // Propagate nuke flag to position so CPU can detect it
        var pos = positions[idx];
        pos.w = 1.0;
        positions[idx] = pos;
        return;
    }

    var pos = positions[idx];
    pos.x += vel.x * params.dt;
    pos.y += vel.y * params.dt;
    pos.z += vel.z * params.dt;
    positions[idx] = pos;
}

// Pass 2: SHAKE — correct positions to maintain rigid interval lengths.
// In tensegrity, no two push intervals share a joint, so all constraints
// are independent and a single pass is exact.
@compute @workgroup_size(64)
fn shake_constraints(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= params.num_rigid { return; }

    let a = rigid_alpha[idx];
    let o = rigid_omega[idx];
    let ideal = rigid_length[idx];

    var pos_a = positions[a];
    var pos_o = positions[o];

    let dx = pos_o.x - pos_a.x;
    let dy = pos_o.y - pos_a.y;
    let dz = pos_o.z - pos_a.z;
    let actual = sqrt(dx * dx + dy * dy + dz * dz);

    if actual < 0.0001 { return; }

    // Move each endpoint by half the length error along the constraint axis
    let correction = (actual - ideal) * 0.5 / actual;
    pos_a.x += dx * correction;
    pos_a.y += dy * correction;
    pos_a.z += dz * correction;
    pos_o.x -= dx * correction;
    pos_o.y -= dy * correction;
    pos_o.z -= dz * correction;

    positions[a] = pos_a;
    positions[o] = pos_o;
}

// Pass 3: Reset forces and masses
@compute @workgroup_size(64)
fn reset_forces(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= params.num_joints { return; }

    atomicStore(&force_x[idx], 0);
    atomicStore(&force_y[idx], 0);
    atomicStore(&force_z[idx], 0);
    atomicStore(&masses[idx], i32(params.ambient_mass * MASS_SCALE));
}

// Pass 4: Elastic forces (cables)
@compute @workgroup_size(64)
fn elastic_forces(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= params.num_elastic { return; }

    let a = elastic_alpha[idx];
    let o = elastic_omega[idx];
    let ideal = elastic_ideal[idx];
    let k = elastic_k[idx];

    let pos_a = positions[a];
    let pos_o = positions[o];

    let dx = pos_o.x - pos_a.x;
    let dy = pos_o.y - pos_a.y;
    let dz = pos_o.z - pos_a.z;
    let actual = sqrt(dx * dx + dy * dy + dz * dz);

    if actual < 0.0001 { return; }

    let strain = (actual - ideal) / ideal;

    // Pull cables go slack when compressed
    if strain <= 0.0 { return; }

    let force_mag = k * strain * ideal;
    let inv_actual = 1.0 / actual;
    let fx = force_mag * dx * inv_actual;
    let fy = force_mag * dy * inv_actual;
    let fz = force_mag * dz * inv_actual;

    let ifx = i32(fx * params.force_scale);
    let ify = i32(fy * params.force_scale);
    let ifz = i32(fz * params.force_scale);

    // Add force to alpha (toward omega)
    atomicAdd(&force_x[a], ifx);
    atomicAdd(&force_y[a], ify);
    atomicAdd(&force_z[a], ifz);

    // Subtract force from omega (toward alpha)
    atomicAdd(&force_x[o], -ifx);
    atomicAdd(&force_y[o], -ify);
    atomicAdd(&force_z[o], -ifz);

    // Add cable mass to endpoints
    let half_mass = i32(0.05 * actual * 0.5 * MASS_SCALE);
    atomicAdd(&masses[a], half_mass);
    atomicAdd(&masses[o], half_mass);
}

// Pass 5: Rigid mass contribution
@compute @workgroup_size(64)
fn rigid_mass(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= params.num_rigid { return; }

    let a = rigid_alpha[idx];
    let o = rigid_omega[idx];
    let hm = i32(rigid_half_mass[idx] * MASS_SCALE);

    atomicAdd(&masses[a], hm);
    atomicAdd(&masses[o], hm);
}

// Pass 6: Second half kick with gravity and damping
@compute @workgroup_size(64)
fn second_half_kick(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= params.num_joints { return; }

    let m = f32(atomicLoad(&masses[idx])) / MASS_SCALE;

    // Add gravity to force
    let gravity_force = -m * params.gravity;
    atomicAdd(&force_y[idx], i32(gravity_force * params.force_scale));

    let fx = f32(atomicLoad(&force_x[idx])) / params.force_scale;
    let fy = f32(atomicLoad(&force_y[idx])) / params.force_scale;
    let fz = f32(atomicLoad(&force_z[idx])) / params.force_scale;

    let inv_m = select(0.0, 1.0 / m, m > 0.0);

    var vel = velocities[idx];
    if vel.w > 0.5 { return; } // nuked — stay frozen

    vel.x += 0.5 * fx * inv_m * params.dt;
    vel.y += 0.5 * fy * inv_m * params.dt;
    vel.z += 0.5 * fz * inv_m * params.dt;

    // Apply drag damping: v *= (1 - drag * dt)
    let damping = 1.0 - params.drag * params.dt;
    vel.x *= damping;
    vel.y *= damping;
    vel.z *= damping;

    vel = enforce_speed_limit(vel);
    velocities[idx] = vel;
}

// Pass 7: RATTLE — project out velocity along rigid constraint axis.
// Ensures joints connected by a rigid interval have no relative velocity
// along the strut, preserving the constraint after velocity updates.
@compute @workgroup_size(64)
fn rattle_constraints(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= params.num_rigid { return; }

    let a = rigid_alpha[idx];
    let o = rigid_omega[idx];

    let pos_a = positions[a];
    let pos_o = positions[o];

    // Constraint unit vector
    let dx = pos_o.x - pos_a.x;
    let dy = pos_o.y - pos_a.y;
    let dz = pos_o.z - pos_a.z;
    let len = sqrt(dx * dx + dy * dy + dz * dz);
    if len < 0.0001 { return; }

    let ux = dx / len;
    let uy = dy / len;
    let uz = dz / len;

    var vel_a = velocities[a];
    var vel_o = velocities[o];

    // Relative velocity projected onto constraint axis
    let rel_vx = vel_o.x - vel_a.x;
    let rel_vy = vel_o.y - vel_a.y;
    let rel_vz = vel_o.z - vel_a.z;
    let vel_along = rel_vx * ux + rel_vy * uy + rel_vz * uz;

    // Remove half from each endpoint (equal mass weighting)
    let half_v = vel_along * 0.5;
    vel_a.x += ux * half_v;
    vel_a.y += uy * half_v;
    vel_a.z += uz * half_v;
    vel_o.x -= ux * half_v;
    vel_o.y -= uy * half_v;
    vel_o.z -= uz * half_v;

    velocities[a] = vel_a;
    velocities[o] = vel_o;
}

// Pass 8: Ground plane collision
@compute @workgroup_size(64)
fn ground_collision(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= params.num_joints { return; }

    var pos = positions[idx];
    if pos.y < params.ground_y {
        pos.y = params.ground_y;
        positions[idx] = pos;

        var vel = velocities[idx];
        if vel.y < 0.0 {
            vel.y = -vel.y * params.restitution;
        }
        velocities[idx] = vel;
    }
}
