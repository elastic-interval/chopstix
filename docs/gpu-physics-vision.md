# Chopstix: GPU Tensegrity Physics — Status & Path Forward

This document captures what has been built, what we learned, and what needs to happen next to make GPU physics decisively faster than the CPU-based tensegrity-lab.

## What Exists

Chopstix is a GPU-accelerated tensegrity sphere simulator. It generates geodesic tensegrity spheres at any frequency, drops them onto a ground plane, and renders the result in real time using wgpu compute shaders.

### Architecture

**Data layout**: Struct-of-arrays for GPU-coalesced access. Joint positions/velocities are `vec4<f32>` arrays. Forces use `atomic<i32>` with a scale factor for parallel accumulation. Interval topology (endpoints, ideal lengths, spring constants) is uploaded once.

**Physics pipeline**: Velocity Verlet integration with 9 compute passes per iteration:

| Pass | Domain | Purpose |
|------|--------|---------|
| 1. half_kick_and_drift | joints | v += 0.5·F/m·dt, x += v·dt |
| 2. shake_constraints | rigid intervals | SHAKE: correct positions to maintain strut lengths |
| 3. reset_forces | joints | Zero force accumulators, reset mass to ambient |
| 4. elastic_forces | elastic intervals | Hooke's law with slack detection for cables |
| 5. rigid_mass | rigid intervals | Distribute strut mass to endpoint joints |
| 6. second_half_kick | joints | Add gravity, v += 0.5·F/m·dt, apply drag |
| 7. rattle_constraints | rigid intervals | RATTLE: project out velocity along strut axes |
| 8. ground_collision | joints | Bounce off ground plane with restitution |
| 9. post_collision_shake | rigid intervals | Fix constraint violations from ground repositioning |

**Rigid push intervals** use SHAKE/RATTLE geometric constraints instead of penalty springs. This is unconditionally stable — no spring constant, no stiffness-dependent timestep limit. In tensegrity, no two push intervals share a joint, so all constraints are independent and a single pass is exact.

**Elastic pull intervals** (cables) use Hooke's law: `F = k · strain · ideal_length`, with `k = K_AT_1M / ideal_length`. Cables go slack (zero force) when compressed. Cable mass is distributed to endpoint joints.

**Speed limit**: Joints exceeding a velocity threshold are "nuked" — velocity zeroed, position frozen, flag propagated to the position buffer's w component so the CPU can detect failure without a full readback.

**Settling phase**: Before the simulation starts, runs physics with high drag (100) and no gravity for 2000 iterations to let the pre-tensioned structure find its self-stress equilibrium. Positions are read back and used as initial state for the real simulation.

### Current Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Cable K at 1m | 5×10⁶ N/m | Real Dyneema is 6.7×10⁹ — we are 1340× softer |
| Physics dt | 250 µs | tensegrity-lab uses 50 µs |
| Iterations/frame | 80 | tensegrity-lab does ~333 |
| Sim time/frame | 20 ms | Both run ≈ real-time at 60fps |
| Drag | 0.5 | Per-step: v *= 0.99988 |
| Speed limit | 100 m/s | Catches instability before cascade |
| Pre-tension | 5% | Cable ideal = 0.95 × initial distance |
| Initial position scale | 0.95 | Places joints near cable equilibrium |

### Test Infrastructure

`tests/frequency_stress.rs` — headless GPU test that sweeps frequencies 1→20, settles each sphere, drops it, waits for ground contact + settling, and reports results. Detects explosion via nuked joints. Current results: **freq 1–14 stable** (up to 11,760 joints, 5,880 struts, 17,640 cables).

Run with: `cargo test --release --test frequency_stress --no-run` then execute the binary directly with `--nocapture` for output.

## What We Learned

### The dispatch overhead wall

This is the critical bottleneck. Each physics iteration requires 9 separate compute dispatches (one per pass). At 80 iterations/frame, that's 720 dispatches. Pushing beyond ~80 iterations causes the GPU command submission overhead to dominate — the test hangs and the app beach-balls.

This means we cannot simply reduce dt and increase iterations to support stiffer cables. At 333 iterations/frame (matching tensegrity-lab), we'd need 2,997 dispatches — roughly 4× beyond what currently works.

**This is the single biggest obstacle to outperforming tensegrity-lab.** The CPU version does 333 iterations per frame with zero dispatch overhead — it's just a tight loop. Our GPU version parallelizes each iteration across thousands of joints, but pays a heavy per-iteration tax.

### Why we're not yet faster than CPU

At the current settings, both engines cover ~20ms of sim time per frame at 60fps. The GPU parallelizes force computation across joints/intervals, but:

1. At freq ≤ 10 (~6000 joints), the per-dispatch work is small — GPU cores are underutilized
2. At freq ≥ 15 (~13,500 joints), we start hitting the elastic stability limit
3. The dispatch overhead prevents us from doing enough iterations for stiffer cables

The GPU advantage would kick in when we can do 333+ iterations/frame over 10,000+ joints. We can't get there with the current 9-dispatch-per-iteration architecture.

### Stiffness vs stability vs dispatch count

The elastic cable stability limit is `dt < 2/sqrt(k/m)`. Current numbers:

| K at 1m | ω (1m cable, 5kg) | dt_max | Required iter/frame | Dispatches/frame |
|---------|-------------------|--------|--------------------|--------------------|
| 5×10⁶ | 1,000 rad/s | 2.0 ms | 10 | 90 |
| 5×10⁷ | 3,162 | 0.63 ms | 32 | 288 |
| 5×10⁸ | 10,000 | 0.20 ms | 100 | 900 |
| 6.7×10⁹ (Dyneema) | 36,600 | 55 µs | 364 | 3,276 |

At higher frequencies, cables are shorter so k is larger (k = K/ideal), making the stability limit tighter. The table shows values for 1m cables; at freq 14 with ~0.7m cables, multiply ω by ~1.2.

### Pre-tension and settling

Tensegrity spheres need self-stress (pre-tension) to hold shape. We set cable ideal lengths to 95% of initial placement distance, then scale initial positions inward by 0.95. This places cables near their rest length and struts slightly compressed — close to equilibrium.

The settling phase (2000 iterations, drag=100, no gravity) lets the structure find its actual equilibrium. Without settling, the residual force imbalance causes visible oscillation during freefall and explosions at higher frequencies.

### Atomic force accumulation

Forces are accumulated using `atomicAdd` on `i32` with a scale factor (FORCE_SCALE = 100). Multiple cables write to the same joint. This works but introduces quantization error and risks i32 overflow at high stiffness. With K = 6.7×10⁹, max force per cable at 5% strain ≈ 3.4×10⁸ N — even FORCE_SCALE = 1 risks overflow with multiple cables per joint.

### Surface behavior

Only basic **bouncy** ground collision is implemented (reflect vy with restitution). tensegrity-lab has four modes: Frozen (stick to ground), Sticky (friction), Bouncy, Slippery (frictionless). These are straightforward to add as a per-joint GPU pass.

## Path to Outperforming CPU

### Priority 1: Reduce passes per iteration

The 9-pass architecture is the bottleneck. Key merges:

**Merge half_kick + drift + SHAKE into one pass** (joint-domain). Each joint does its own kick and drift, then each rigid interval corrects positions. Since rigid constraints are independent in tensegrity, this can be two dispatches instead of three — or even one if we give each joint knowledge of its one connected strut.

**Merge reset_forces + elastic_forces + rigid_mass**. Currently three passes. If each elastic interval atomically accumulates forces AND each rigid interval accumulates mass in a single dispatch, we save two passes. The reset can be folded into the second_half_kick (read force, then zero it).

**Target: 3–4 dispatches per iteration** instead of 9. This alone would allow ~200 iterations/frame, enough for K ≈ 5×10⁸.

### Priority 2: In-shader iteration loop

The vision doc's original suggestion: loop multiple iterations within a single dispatch, using storage buffer flushes between iterations. This requires careful ordering but could reduce dispatch count by 10-50×. The challenge is global synchronization between passes — `workgroupBarrier()` only syncs within a workgroup.

One approach: split into just 2 mega-dispatches per iteration (joint-pass and interval-pass), with the iteration loop on the CPU side but only 2 dispatches per iteration instead of 9.

### Priority 3: Float atomics for force accumulation

Replace the i32 atomic hack with proper float accumulation. Options:
- `atomicCompareExchangeWeak` loop on reinterpreted f32 (standard GPU trick)
- Separate force buffers per interval, then a reduction pass
- Use subgroup operations for partial reduction before atomic write

This eliminates the FORCE_SCALE quantization error and the i32 overflow risk, enabling arbitrarily stiff cables.

### Priority 4: Adaptive iteration count

Instead of fixed iterations/frame, compute iterations based on current maximum strain energy or velocity. Calm structures need fewer iterations; high-energy bounces need more. This lets us use stiffer cables for most of the simulation and only burn extra iterations during impact.

### Priority 5: What "remarkably better" looks like

tensegrity-lab on CPU: ~500 joints at interactive rates, single-threaded, 333 iterations/frame.

Chopstix target: **10,000+ joints at interactive rates** with stiffer cables and rigid struts. This means:
- Freq 15–20 spheres running smoothly (13,500–24,000 joints)
- Cable stiffness ≥ 10⁸ N/m (20× current, approaching physical realism)
- Real-time at 60fps

The GPU parallelism wins when per-dispatch work is large (many joints) AND dispatch count is low (merged passes). Both conditions must hold simultaneously.

## Codebase Map

| File | Purpose |
|------|---------|
| `src/main.rs` | CLI entry point, frequency argument |
| `src/app.rs` | Window, event loop, frame dispatch |
| `src/constants.rs` | All physics constants in one place |
| `src/tensegrity.rs` | Geodesic sphere → tensegrity topology generation |
| `src/sphere.rs` | Icosahedron subdivision (geodesic scaffold) |
| `src/camera.rs` | Orbit camera with mouse controls |
| `src/gpu/mod.rs` | wgpu device/surface initialization |
| `src/gpu/physics.rs` | Compute pipeline setup, dispatch, settle, readback |
| `src/gpu/physics.wgsl` | All compute shader entry points |
| `src/gpu/renderer.rs` | Cylinder instance rendering |
| `src/gpu/render.wgsl` | Vertex/fragment shaders |
| `src/lib.rs` | Library facade for test access |
| `tests/frequency_stress.rs` | Headless frequency sweep test |

## Heritage from tensegrity-lab

Concepts carried over: geodesic subdivision, tensegrity topology (push/pull intervals), pre-tension, settling, speed limit, struct-of-arrays data layout, surface interaction model.

Concepts changed: push intervals are now truly rigid (SHAKE/RATTLE) instead of stiff springs. Physics runs on GPU compute shaders instead of CPU Verlet loop. Rendering uses GPU instance drawing from the same device.

Not yet ported: Tenscript DSL, evolution framework, fabric plan executor, CSV export, multiple surface characters (Frozen/Sticky/Slippery), scaffold forces, pretensing by strut extension.
