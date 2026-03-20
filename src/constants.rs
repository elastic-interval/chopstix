/// Base pull cable spring constant at 1m length (viewing preset), in N/m
/// Real Dyneema is 6.7e9; stepping toward that incrementally.
pub const PULL_K_AT_1M: f32 = 5_000_000.0;

/// Push interval (aluminum tube) linear density in kg/m
pub const PUSH_LINEAR_DENSITY: f32 = 3.0;

/// Joint ambient mass (connector hardware) in kg
pub const JOINT_AMBIENT_MASS: f32 = 2.28;

/// Velocity drag coefficient (viewing preset)
pub const DRAG: f32 = 0.5;

/// Viscosity coefficient (viewing preset)
pub const VISCOSITY: f32 = 0.0;

/// Twist angle for tensegrity strut rotation in radians
pub const TWIST_ANGLE: f32 = 0.52;

/// Gravity acceleration in m/s²
pub const GRAVITY: f32 = 9.81;

/// Default sphere radius in meters
pub const SPHERE_RADIUS: f32 = 10.0;

/// Atomic force scale factor for fixed-point accumulation
pub const FORCE_SCALE: f32 = 100.0;

/// Ground plane Y coordinate
pub const GROUND_Y: f32 = -20.0;

/// Ground collision restitution (bounciness)
pub const RESTITUTION: f32 = 0.5;

/// Only read back positions from GPU every N frames (reduces main-thread blocking)
pub const READBACK_INTERVAL: u32 = 3;

/// Fixed physics timestep
pub const ITERATION_DT: f32 = 0.25e-3;

/// Fixed iterations per frame (keep GPU dispatch count bounded)
/// 80 * 0.25ms = 20ms sim time per frame ≈ real-time at 60fps
pub const ITERATIONS_PER_FRAME: u32 = 80;

/// Joint speed limit (m/s) — if any joint exceeds this, it gets frozen (nuked).
/// Freefall from 20m gives ~20 m/s; bounces double that; allow generous headroom.
pub const SPEED_LIMIT: f32 = 100.0;

/// Iterations for the initial settling phase (high drag, no gravity)
/// Lets the pre-tensioned structure find its self-stress equilibrium
pub const SETTLE_ITERATIONS: u32 = 2000;
