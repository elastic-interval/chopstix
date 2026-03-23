# GPU Shader Physics in Chopstix

This document describes how Chopstix uses GPU compute shaders for tensegrity physics simulation, the design decisions behind the approach, and the limitations discovered through testing.

## Overview

The physics simulation runs entirely on the GPU as a sequence of compute shader dispatches within a single compute pass. Each frame, the CPU submits one compute pass containing many iterations of the full physics pipeline, then optionally reads back joint positions for rendering. The GPU executes all dispatches sequentially with implicit storage barriers between them, which means each dispatch sees the results of the previous one without any explicit synchronization.

## Data Layout

All physics state lives in GPU storage buffers, organized as struct-of-arrays for coalesced memory access:

- **Positions** and **velocities**: float arrays, one vec4 per joint.
- **Forces**: three separate arrays of atomic integers (one per axis), enabling parallel accumulation from multiple intervals writing to the same joint.
- **Masses**: an atomic integer array, also accumulated in parallel from intervals contributing mass to their endpoint joints.
- **Frozen flag**: a single atomic u32. When any joint exceeds the speed limit, this flag is set to 1 and the entire simulation halts — all shader entry points check this flag first and early-return if set.
- **Interval topology**: read-only arrays of endpoint indices, ideal lengths, spring constants, and strut masses. Uploaded once when the shape is generated.

Forces and masses use **fixed-point integer atomics** because WGSL does not support atomic floating-point operations. Float forces are multiplied by a scale factor before atomic addition, then divided back after reading. This introduces quantization error proportional to 1/scale_factor and limits the maximum representable force to i32::MAX / scale_factor.

## Two Physics Modes

### SHAKE/RATTLE Mode (Geodesic Spheres)

Used when no two push struts share a joint. Push intervals are treated as rigid geometric constraints. 8 dispatches per iteration:

1. **half_kick_and_drift** — v += 0.5·F/m·dt, x += v·dt
2. **shake_constraints** — correct positions to maintain rigid strut lengths
3. **elastic_forces** — Hooke's law cables with slack detection
4. **rigid_mass** — distribute strut mass to endpoint joints
5. **second_half_kick** — add gravity, second half velocity update, drag, force reset
6. **rattle_constraints** — project out velocity along strut axes
7. **ground_collision** — surface interaction with four character modes
8. **shake_constraints** — fix constraint violations from ground repositioning

### Spring-Push Mode (Klein Bottles, Möbius Bands)

Used when push struts share joints (detected automatically). Push intervals are treated as stiff springs with atomic force accumulation, identical to cables but without slack detection. 5 dispatches per iteration:

1. **half_kick_and_drift**
2. **elastic_forces** (cables)
3. **push_forces** (struts as springs — no slack check)
4. **second_half_kick** (includes force reset)
5. **ground_collision**

No SHAKE/RATTLE needed — force accumulation via atomics handles shared joints naturally.

## The Integration Scheme

The simulation uses **velocity Verlet** integration.

### Half Kick and Drift (one thread per joint)

Read the accumulated forces and mass from the previous iteration. Apply the first half of the velocity update, then advance positions:

    v += 0.5 * (F / m) * dt
    x += v * dt

If any joint exceeds the speed limit, the global frozen flag is set. All subsequent dispatches (including future iterations in the same compute pass) detect this and skip their work. The CPU reads the frozen flag on the next readback and auto-pauses the simulation.

### SHAKE Position Correction (one thread per rigid strut)

After positions have drifted, rigid struts may have changed length. SHAKE projects each strut back to its ideal length by moving both endpoints along the strut axis:

    error = actual_length - ideal_length
    correction = error * 0.5 / actual_length
    move each endpoint by correction along the strut axis

In tensegrity, no two push struts share a joint, so all constraints are independent and a single correction pass is exact.

### Elastic Force Accumulation (one thread per cable)

Each cable computes its strain relative to its ideal (rest) length. Cables are one-sided: they exert force only in tension, not compression (they go slack):

    strain = (actual - ideal) / ideal
    if strain <= 0: return  (cable is slack)
    force = k * strain * ideal

Force is projected along the cable direction and accumulated into both endpoint joints' force buffers using atomic integer addition.

### Push Force Accumulation (one thread per push interval, spring-push mode only)

Identical to elastic forces but without the slack check — push intervals resist both compression and extension. Mass is also accumulated to endpoints.

### Second Half Kick with Force Reset (one thread per joint)

Adds gravity to the accumulated force, applies the second half of the velocity update, then applies velocity drag:

    F_y += -m * gravity
    v += 0.5 * (F / m) * dt
    v *= (1 - drag * dt)

After reading forces and masses, this dispatch resets them to zero (forces) and ambient mass (masses) in preparation for the next iteration. The reset always runs even if the simulation is frozen, to keep accumulators clean.

### Ground Collision with Surface Character (one thread per joint)

Five surface interaction modes, selectable at runtime:

- **Absent**: No surface interaction — joints fall through the ground plane. Used for zero-gravity shapes (Klein, Möbius).
- **Bouncy**: Elastic bounce with 50% energy loss, horizontal damping (0.6×), gentle antigravity push proportional to penetration depth.
- **Frozen**: Complete immobilization at the ground plane. Velocity zeroed, position clamped.
- **Sticky**: High friction (0.8 downward, drag-based upward), strong antigravity support (50× gravity), depth clamping at 0.1m.
- **Slippery**: Frictionless horizontal slide on the surface plane, with combined linear + quadratic speed damping.

## Shape Generators

### Geodesic Sphere (`tensegrity.rs`)

Subdivides an icosahedron to create a geodesic scaffold, then converts each edge into a push strut with twisted joint placement. Pull cables wrap circumferentially and diagonally around the struts. Pre-tension is applied by setting cable ideal lengths to 95% of initial placement distance.

### Klein Bottle (`klein.rs`)

Ported from tensegrity-lab. Creates a parametric Klein bottle surface with push struts and pull cables. Width must be even, height must be odd. Push struts share joints (3 struts meet at each joint), requiring spring-push mode. Joints start at random positions within a unit sphere and settle via approach-based ideal length interpolation.

### Möbius Band (`mobius.rs`)

Ported from tensegrity-lab. Creates a zigzag strip with 180° twist. Joint count = 2×segments+1 (odd, for the twist). Each joint connects via pull cables (edge + width) and push struts (diagonal). Uses spring-push mode.

## Settling

### Standard Settling (Spheres)

Runs physics with high drag (100×), no gravity, and a distant ground plane for 2000 iterations in chunks of 200. The pre-tensioned structure converges to its self-stress equilibrium.

### Approach-Based Settling (Klein, Möbius)

For shapes starting from random positions, ideal lengths are interpolated from actual → target over 20 steps of 2000 iterations each (40,000 total). Between steps, the CPU reads back positions, updates ideal lengths (linear interpolation based on progress), and rebuilds the PhysicsCompute with new buffers. After approach completes, 5000 additional iterations run at final target lengths.

## Single Compute Pass Architecture

All iterations of the dispatch sequence are recorded into a single GPU compute pass. Bind groups are set once; only the pipeline changes between dispatches.

This was a critical optimization. The original implementation used separate compute passes per dispatch — 720 pass transitions per frame at 80 iterations. With a single pass, the same work requires just 1 pass transition.

## Fixed-Point Force Accumulation

WGSL provides `atomicAdd` for integers but not for floats. Forces are scaled to integers before atomic addition:

    integer_force = float_force * FORCE_SCALE
    atomicAdd(force_buffer[joint], integer_force)

This introduces quantization (minimum force = 1/FORCE_SCALE) and risks i32 overflow at high stiffness. The same technique is used for mass accumulation with a separate scale factor (MASS_SCALE = 10,000).

## Runtime Configuration

Physics parameters are configurable through a `PhysicsConfig` struct: timestep, iterations per frame, cable stiffness, force scale, drag, speed limit, settle parameters, ambient joint mass, gravity, ground plane Y, and surface character.

A `scaled_for_frequency` method adjusts dt and iteration count for higher geodesic frequencies, maintaining stability as cables shorten and stiffen.

Stiffness (K) and pretension are adjustable at runtime via on-screen sliders without re-settling. Surface character can be switched at runtime. Shape selection (sphere frequency, Klein dimensions, Möbius segments) triggers a deferred regeneration and settling cycle — the old structure disappears immediately while the new one generates in the background.

The `elastic_ideal` and `elastic_k` GPU buffers are created with `COPY_DST`, allowing the CPU to update interval ideal lengths at runtime via `queue.write_buffer` without rebuilding pipelines or bind groups.

## Muscle Animation

The `Twitcher` system animates Möbius bands by modulating pull-cable ideal lengths with a traveling sine wave. Each muscle (pull-edge interval) oscillates sinusoidally with a phase offset based on its position in the sequence, creating peristaltic locomotion.

The CPU updates ideal lengths and K values before each physics dispatch. The sine wave parameters (phase speed, amplitude) are tuned for a slow, smooth contraction wave that makes the band cycle around its loop. The system is generic — any shape can provide muscle indices for different animation patterns.

## Rendering

The ground surface is rendered as a triangular lattice (three line directions at 60° angles) when a surface is active. It is hidden for zero-gravity shapes and when surface character is Absent.

Intervals are rendered as instanced cylinders. Push struts are silver/thick, pull cables are blue/thin. Radius scales inversely with shape complexity to keep larger structures from looking bulky.
