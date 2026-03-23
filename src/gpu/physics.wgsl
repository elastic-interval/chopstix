// Joint data (Group 0)
@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> force_x: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> force_y: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> force_z: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> masses: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> frozen: atomic<u32>;

// Interval topology (Group 1)
@group(1) @binding(0) var<storage, read> elastic_alpha: array<u32>;
@group(1) @binding(1) var<storage, read> elastic_omega: array<u32>;
@group(1) @binding(2) var<storage, read> elastic_ideal: array<f32>;
@group(1) @binding(3) var<storage, read> elastic_k: array<f32>;
@group(1) @binding(4) var<storage, read> rigid_alpha: array<u32>;
@group(1) @binding(5) var<storage, read> rigid_omega: array<u32>;
@group(1) @binding(6) var<storage, read> rigid_length: array<f32>;
@group(1) @binding(7) var<storage, read> rigid_half_mass: array<f32>;

// Spring-based push intervals (Group 3)
@group(3) @binding(0) var<storage, read> push_alpha: array<u32>;
@group(3) @binding(1) var<storage, read> push_omega: array<u32>;
@group(3) @binding(2) var<storage, read> push_ideal: array<f32>;
@group(3) @binding(3) var<storage, read> push_k: array<f32>;
@group(3) @binding(4) var<storage, read> push_half_mass: array<f32>;

// Uniform params (Group 2)
struct Params {
    dt: f32,
    gravity: f32,
    drag: f32,
    _reserved0: f32,
    num_joints: u32,
    num_elastic: u32,
    num_rigid: u32,
    ambient_mass: f32,
    force_scale: f32,
    ground_y: f32,
    _reserved1: f32,
    speed_limit: f32,
    num_push: u32,
    surface_character: u32,
    _pad2: u32,
    _pad3: u32,
}
@group(2) @binding(0) var<uniform> params: Params;

fn check_speed_limit(vel: vec3<f32>) {
    let speed_sq = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
    if speed_sq > params.speed_limit * params.speed_limit {
        atomicStore(&frozen, 1u);
    }
}

fn is_frozen() -> bool {
    return atomicLoad(&frozen) != 0u;
}

const MASS_SCALE: f32 = 1e4;

// Half kick and drift: v += 0.5*(F/m)*dt, x += v*dt
@compute @workgroup_size(64)
fn half_kick_and_drift(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= params.num_joints || is_frozen() { return; }

    let fx = f32(atomicLoad(&force_x[idx])) / params.force_scale;
    let fy = f32(atomicLoad(&force_y[idx])) / params.force_scale;
    let fz = f32(atomicLoad(&force_z[idx])) / params.force_scale;
    let m = f32(atomicLoad(&masses[idx])) / MASS_SCALE;

    var vel = velocities[idx];
    let inv_m = select(0.0, 1.0 / m, m > 0.0);
    vel.x += 0.5 * fx * inv_m * params.dt;
    vel.y += 0.5 * fy * inv_m * params.dt;
    vel.z += 0.5 * fz * inv_m * params.dt;

    check_speed_limit(vec3<f32>(vel.x, vel.y, vel.z));
    velocities[idx] = vel;

    var pos = positions[idx];
    pos.x += vel.x * params.dt;
    pos.y += vel.y * params.dt;
    pos.z += vel.z * params.dt;
    positions[idx] = pos;
}

// SHAKE: correct positions to maintain rigid strut lengths.
// Independent constraints (no shared joints) so one pass is exact.
@compute @workgroup_size(64)
fn shake_constraints(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= params.num_rigid || is_frozen() { return; }

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

// Elastic forces: Hooke's law cables that go slack when compressed
@compute @workgroup_size(64)
fn elastic_forces(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= params.num_elastic || is_frozen() { return; }

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
    if strain <= 0.0 { return; }

    let force_mag = k * strain * ideal;
    let inv_actual = 1.0 / actual;
    let fx = force_mag * dx * inv_actual;
    let fy = force_mag * dy * inv_actual;
    let fz = force_mag * dz * inv_actual;

    let ifx = i32(fx * params.force_scale);
    let ify = i32(fy * params.force_scale);
    let ifz = i32(fz * params.force_scale);

    atomicAdd(&force_x[a], ifx);
    atomicAdd(&force_y[a], ify);
    atomicAdd(&force_z[a], ifz);
    atomicAdd(&force_x[o], -ifx);
    atomicAdd(&force_y[o], -ify);
    atomicAdd(&force_z[o], -ifz);

    let half_mass = i32(0.05 * actual * 0.5 * MASS_SCALE);
    atomicAdd(&masses[a], half_mass);
    atomicAdd(&masses[o], half_mass);
}

// Rigid mass: distribute strut mass to endpoints
@compute @workgroup_size(64)
fn rigid_mass(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= params.num_rigid || is_frozen() { return; }

    let hm = i32(rigid_half_mass[idx] * MASS_SCALE);
    atomicAdd(&masses[rigid_alpha[idx]], hm);
    atomicAdd(&masses[rigid_omega[idx]], hm);
}

// Second half kick: gravity, second velocity update, drag, force/mass reset
@compute @workgroup_size(64)
fn second_half_kick(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= params.num_joints { return; }

    let m = f32(atomicLoad(&masses[idx])) / MASS_SCALE;

    atomicAdd(&force_y[idx], i32(-m * params.gravity * params.force_scale));

    let fx = f32(atomicLoad(&force_x[idx])) / params.force_scale;
    let fy = f32(atomicLoad(&force_y[idx])) / params.force_scale;
    let fz = f32(atomicLoad(&force_z[idx])) / params.force_scale;

    atomicStore(&force_x[idx], 0);
    atomicStore(&force_y[idx], 0);
    atomicStore(&force_z[idx], 0);
    atomicStore(&masses[idx], i32(params.ambient_mass * MASS_SCALE));

    if is_frozen() { return; }

    let inv_m = select(0.0, 1.0 / m, m > 0.0);
    var vel = velocities[idx];
    vel.x += 0.5 * fx * inv_m * params.dt;
    vel.y += 0.5 * fy * inv_m * params.dt;
    vel.z += 0.5 * fz * inv_m * params.dt;

    let damping = 1.0 - params.drag * params.dt;
    vel.x *= damping;
    vel.y *= damping;
    vel.z *= damping;

    check_speed_limit(vec3<f32>(vel.x, vel.y, vel.z));
    velocities[idx] = vel;
}

// RATTLE: project out velocity along rigid strut axes
@compute @workgroup_size(64)
fn rattle_constraints(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= params.num_rigid || is_frozen() { return; }

    let a = rigid_alpha[idx];
    let o = rigid_omega[idx];

    let pos_a = positions[a];
    let pos_o = positions[o];

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

    let vel_along = (vel_o.x - vel_a.x) * ux + (vel_o.y - vel_a.y) * uy + (vel_o.z - vel_a.z) * uz;
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

// Surface character: 0=absent, 1=bouncy, 2=frozen, 3=sticky, 4=slippery
@compute @workgroup_size(64)
fn ground_collision(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= params.num_joints || is_frozen() { return; }

    var pos = positions[idx];
    if pos.y >= params.ground_y { return; }
    if params.surface_character == 0u { return; }

    var vel = velocities[idx];
    let depth = params.ground_y - pos.y;

    switch (params.surface_character) {
        case 1u: { // bouncy
            pos.y = params.ground_y;
            if vel.y < 0.0 {
                vel.y *= -0.5;
            }
            vel.x *= 0.6;
            vel.z *= 0.6;
            vel.y += params.gravity * 5.0 * params.dt;
        }
        case 2u: { // frozen
            pos.y = params.ground_y;
            vel = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        }
        case 3u: { // sticky
            let friction = select(1.0 - params.drag * params.dt, 0.8, vel.y < 0.0);
            vel.x *= friction;
            vel.z *= friction;
            vel.y += params.gravity * 50.0 * params.dt;
            if vel.y < 0.0 {
                vel.y *= 0.5;
            }
            if depth > 0.1 {
                pos.y = params.ground_y - 0.1;
                vel.y = 0.0;
            }
        }
        case 4u: { // slippery
            pos.y = params.ground_y;
            vel.y = 0.0;
            let speed_h = sqrt(vel.x * vel.x + vel.z * vel.z);
            let lin_fric = 1.0 - (50.0 + params.drag) * params.dt;
            let quad_fric = 1.0 - 2.0 * speed_h * speed_h * params.dt;
            let total = max(lin_fric * max(quad_fric, 0.0), 0.0);
            vel.x *= total;
            vel.z *= total;
        }
        default: {}
    }

    positions[idx] = pos;
    velocities[idx] = vel;
}

// Push forces: spring-based push for shared-joint topologies (no slack check)
@compute @workgroup_size(64)
fn push_forces(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= params.num_push || is_frozen() { return; }

    let a = push_alpha[idx];
    let o = push_omega[idx];
    let ideal = push_ideal[idx];
    let k = push_k[idx];
    let hm = push_half_mass[idx];

    let pos_a = positions[a];
    let pos_o = positions[o];

    let dx = pos_o.x - pos_a.x;
    let dy = pos_o.y - pos_a.y;
    let dz = pos_o.z - pos_a.z;
    let actual = sqrt(dx * dx + dy * dy + dz * dz);
    if actual < 0.0001 { return; }

    let strain = (actual - ideal) / ideal;
    let force_mag = k * strain * ideal;
    let inv_actual = 1.0 / actual;
    let fx = force_mag * dx * inv_actual;
    let fy = force_mag * dy * inv_actual;
    let fz = force_mag * dz * inv_actual;

    let ifx = i32(fx * params.force_scale);
    let ify = i32(fy * params.force_scale);
    let ifz = i32(fz * params.force_scale);

    atomicAdd(&force_x[a], ifx);
    atomicAdd(&force_y[a], ify);
    atomicAdd(&force_z[a], ifz);
    atomicAdd(&force_x[o], -ifx);
    atomicAdd(&force_y[o], -ify);
    atomicAdd(&force_z[o], -ifz);

    let ihm = i32(hm * MASS_SCALE);
    atomicAdd(&masses[a], ihm);
    atomicAdd(&masses[o], ihm);
}
