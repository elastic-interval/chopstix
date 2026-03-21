# GPU Shader Physics in Chopstix

This document describes how Chopstix uses GPU compute shaders for tensegrity physics simulation, the design decisions behind the approach, and the limitations discovered through testing.

## Overview

The physics simulation runs entirely on the GPU as a sequence of compute shader dispatches within a single compute pass. Each frame, the CPU submits one compute pass containing many iterations of the full physics pipeline, then optionally reads back joint positions for rendering. The GPU executes all dispatches sequentially with implicit storage barriers between them, which means each dispatch sees the results of the previous one without any explicit synchronization.

## Data Layout

All physics state lives in GPU storage buffers, organized as struct-of-arrays for coalesced memory access:

- **Positions** and **velocities**: float arrays, one vec4 per joint. The w component of velocity serves as a "nuked" flag (explained below). The w component of position mirrors this flag for CPU-side detection without a full velocity readback.
- **Forces**: three separate arrays of atomic integers (one per axis), enabling parallel accumulation from multiple intervals writing to the same joint.
- **Masses**: an atomic integer array, also accumulated in parallel from intervals contributing mass to their endpoint joints.
- **Interval topology**: read-only arrays of endpoint indices, ideal lengths, spring constants, and strut masses. Uploaded once when the sphere is generated.

Forces and masses use **fixed-point integer atomics** because WGSL does not support atomic floating-point operations. Float forces are multiplied by a scale factor before atomic addition, then divided back after reading. This introduces quantization error proportional to 1/scale_factor and limits the maximum representable force to i32::MAX / scale_factor.

## The Integration Scheme

The simulation uses **velocity Verlet** integration with SHAKE/RATTLE geometric constraints for rigid struts. Each iteration consists of 8 dispatches executed in sequence:

### 1. Half Kick and Drift (one thread per joint)

Read the accumulated forces and mass from the previous iteration. Apply the first half of the velocity update, then advance positions:

    v += 0.5 * (F / m) * dt
    x += v * dt

If any joint exceeds the speed limit, its velocity is zeroed and it is marked as "nuked" — permanently frozen for the rest of the simulation. This prevents a single unstable joint from cascading into a full structural explosion.

### 2. SHAKE Position Correction (one thread per rigid strut)

After positions have drifted, rigid struts may have changed length. SHAKE projects each strut back to its ideal length by moving both endpoints along the strut axis:

    error = actual_length - ideal_length
    correction = error * 0.5 / actual_length
    move each endpoint by correction along the strut axis

In tensegrity, no two push struts share a joint, so all constraints are independent and a single correction pass is exact. This is a crucial structural property — general rigid body constraints would require iterative convergence.

### 3. Elastic Force Accumulation (one thread per cable)

Each cable computes its strain relative to its ideal (rest) length. Cables are one-sided: they exert force only in tension, not compression (they go slack). The force magnitude follows Hooke's law:

    strain = (actual - ideal) / ideal
    if strain <= 0: return  (cable is slack)
    force = k * strain * ideal

Force is projected along the cable direction and accumulated into both endpoint joints' force buffers using atomic integer addition. Cable mass (proportional to current length) is also distributed to endpoints.

### 4. Rigid Mass Accumulation (one thread per rigid strut)

Each strut distributes half its mass to each endpoint joint, also via atomic addition. Strut mass is pre-computed from strut length and linear density.

### 5. Second Half Kick with Force Reset (one thread per joint)

Adds gravity to the accumulated force, applies the second half of the velocity update, then applies velocity drag:

    F_y += -m * gravity
    v += 0.5 * (F / m) * dt
    v *= (1 - drag * dt)

After reading forces and masses, this dispatch **resets them to zero** (forces) and **ambient mass** (masses) in preparation for the next iteration. This merged reset eliminates what was previously a separate dispatch.

### 6. RATTLE Velocity Correction (one thread per rigid strut)

The velocity update may have introduced relative velocity along rigid strut axes. RATTLE projects this out:

    relative_velocity_along_strut = (v_omega - v_alpha) . strut_axis
    remove half from each endpoint

Like SHAKE, this is exact in one pass because tensegrity struts are independent.

### 7. Ground Collision (one thread per joint)

If a joint has penetrated below the ground plane, its y-position is clamped and its y-velocity is reflected with a restitution coefficient.

### 8. Post-Collision SHAKE (one thread per rigid strut)

Ground collision repositioned joints, potentially violating strut length constraints. A second SHAKE pass corrects this.

## Single Compute Pass Architecture

All iterations of the above 8-dispatch sequence are recorded into a **single GPU compute pass**. Bind groups (buffer references) are set once at the start; only the pipeline (which shader entry point to run) changes between dispatches.

This was a critical optimization. The original implementation used a separate compute pass (begin/end) for each dispatch — 9 passes per iteration, 720 pass transitions per frame at 80 iterations. The overhead of beginning and ending compute passes dominated at high iteration counts, causing the GPU to hang beyond roughly 80 iterations per frame.

With a single pass, the same 80 iterations require just 1 pass transition instead of 720. This enabled scaling to 333+ iterations per frame (matching the CPU reference implementation) and up to 500+ iterations per frame for high-frequency spheres.

## Settling Phase

Before the simulation begins, the structure must find its self-stress equilibrium. Cables are generated with ideal lengths 5% shorter than the initial placement distance, creating pre-tension. But the initial geometry is only approximate — the actual equilibrium shape depends on the interplay of all cable tensions and strut constraints.

The settling phase runs the same physics pipeline with high drag (100x normal), no gravity, and a distant ground plane. Over 2000 iterations, the structure converges to a stable pre-stressed shape. Settled positions are read back to the CPU and used as the initial state for the real simulation.

Settling is batched into chunks of 200 iterations per GPU submission, with a device poll between chunks, to avoid GPU timeout on long settling runs.

## Fixed-Point Force Accumulation

WGSL provides `atomicAdd` for integers but not for floats. Since multiple cables may connect to the same joint, forces must be accumulated atomically. The workaround:

    integer_force = float_force * FORCE_SCALE
    atomicAdd(force_buffer[joint], integer_force)

And on readback:

    float_force = float(atomicLoad(force_buffer[joint])) / FORCE_SCALE

The same technique is used for mass accumulation, with a separate scale factor (MASS_SCALE = 10,000).

This introduces two limitations:

- **Quantization**: forces smaller than 1/FORCE_SCALE are rounded to zero. At the current scale factor of 100, the minimum representable force is 0.01 N.
- **Overflow**: the maximum representable force is approximately 2.1 billion / FORCE_SCALE. At scale 100, this is ~21 million N per joint. With very stiff cables (K > 10^8) under high strain, multiple cables on the same joint can approach this limit.

## Runtime Configuration

Physics parameters are configurable at runtime through a `PhysicsConfig` struct rather than compile-time constants. This includes timestep, iterations per frame, cable stiffness, force scale, drag, speed limit, settle parameters, and ambient joint mass.

A `scaled_for_frequency` method adjusts parameters for higher geodesic frequencies. At higher frequencies, cables become shorter and stiffer (K is inversely proportional to ideal length), requiring a smaller timestep. The scaling reduces dt and increases iterations proportionally above a reference frequency, preserving the same simulated time per frame while maintaining numerical stability.

## Discovered Limitations

### Stability vs. Frequency

Geodesic frequency determines the mesh density. At frequency N, the sphere has 60N^2 joints, 30N^2 struts, and 90N^2 cables, with element lengths proportional to radius/N.

Cable spring constant is computed as K_at_1m / ideal_length. As frequency increases, cables shorten, and K grows inversely. The Courant-Friedrichs-Lewy (CFL) stability condition for explicit integration of a spring-mass system requires dt < 2/sqrt(K/m). Shorter, stiffer cables tighten this bound.

With the default timestep (250 microseconds) and default cable stiffness (5 million N/m), the simulation is stable through frequency 33 (65,340 joints). At frequency 34, a single joint exceeds the speed limit and is nuked. The `scaled_for_frequency` method extends stability to frequency 40+ by reducing dt above a reference frequency.

### Ambient Mass Tension

Every joint carries an "ambient mass" representing connector hardware, independent of the intervals attached to it. This creates a tension at high frequencies:

- **Too much ambient mass**: at frequency 40 with 96,000 joints, the total ambient mass dominates the structural mass. The sphere deforms significantly under gravity because the structure is carrying far more weight than its cables can support at their current stiffness.
- **Too little ambient mass**: reducing ambient mass makes joints lighter, which means the same cable forces produce larger accelerations. This tightens the CFL stability bound, causing speed limit violations at lower frequencies than with heavier joints.
- **Zero ambient mass**: without ambient mass, joints that have not yet had interval mass accumulated (at the start of each iteration, before mass accumulation dispatches run) have zero mass, causing division by zero in the velocity update.

There is no ambient mass scaling that improves both deformation and stability simultaneously without also scaling the timestep, which costs frame rate. The current approach leaves ambient mass at its default value and accepts the deformation at high frequencies as physically correct behavior (more total mass on the same-radius sphere).

### Force Scale Precision

The integer fixed-point representation limits both the minimum and maximum representable force. At higher frequencies with shorter elements, the forces per element decrease, making the quantization floor more significant. Conversely, with stiffer cables (K approaching physical Dyneema at 6.7 billion N/m), forces per element can overflow the i32 representation.

Addressing this would require either float atomics (not available in WGSL), a compare-and-swap float accumulation loop, or a fundamentally different force accumulation strategy.

### GPU Command Submission

Even with the single compute pass optimization, there is still one GPU submission per frame (or per settling chunk). The CPU must wait for the GPU to finish before reading back positions. This synchronization point means the CPU and GPU cannot overlap their work during physics frames.

Position readback is mitigated by only performing it every N frames (currently every 3 frames), allowing the GPU to run ahead on physics while the CPU renders with slightly stale positions.
