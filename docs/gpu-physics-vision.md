# Chopstix: GPU Tensegrity Physics — Status & Path Forward

This document captures what has been built, what we learned, and what needs to happen next.

## What Exists

Chopstix is a GPU-accelerated tensegrity simulator supporting three shape types: geodesic spheres, Klein bottles, and Möbius bands. Physics runs entirely on GPU compute shaders via wgpu. All shapes can be selected at runtime through on-screen buttons.

### Architecture

**Data layout**: Struct-of-arrays for GPU-coalesced access. Joint positions/velocities are `vec4<f32>` arrays. Forces use `atomic<i32>` with a scale factor for parallel accumulation. A single `atomic<u32>` frozen flag halts the simulation globally when any joint exceeds the speed limit.

**Two physics modes**:

- **SHAKE/RATTLE** (spheres): Push intervals as rigid geometric constraints. 8 dispatches per iteration. Exploits the tensegrity property that no two struts share a joint, making all constraints independent.
- **Spring-push** (Klein, Möbius): Push intervals as stiff springs with atomic force accumulation. 5 dispatches per iteration. Handles shared-joint topologies where SHAKE would race.

**Surface characters**: Five ground interaction modes — Absent (no surface), Bouncy, Frozen, Sticky, Slippery. Switchable at runtime. The ground is rendered as a triangular lattice when active.

**Shape generators**:
- Geodesic spheres at any frequency (1–60+), with SHAKE/RATTLE rigid struts
- Klein bottles with parametric (width, height), 3 rows × 10 columns of presets varying aspect ratio at constant joint count
- Möbius bands with configurable segment count, 10 presets

### Runtime Controls

- **Shape selection**: Sphere frequency bar, Klein grid (3×10), Möbius segment bar — all always visible, click to switch
- **Stiffness slider**: Log scale K ∈ [10³, 10¹⁰] N/m
- **Pretension slider**: Cable ideal length scaling 0.5–1.0
- **Surface character**: 5 clickable buttons
- **Camera**: Mouse orbit, scroll zoom, auto-tracking centroid
- **Space**: Pause/resume (starts unpaused)

### Current Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Cable K at 1m | 5×10⁶ N/m | Real Dyneema is 6.7×10⁹ — we are 1340× softer |
| Physics dt | 250 µs | tensegrity-lab uses 50 µs |
| Iterations/frame | 80 | tensegrity-lab does ~333 |
| Sim time/frame | 20 ms | Both run ≈ real-time at 60fps |
| Drag | 0.5 | Per-step: v *= 0.99988 |
| Speed limit | 100 m/s | Triggers global freeze |
| Pre-tension | 5% | Cable ideal = 0.95 × initial distance |

### Test Infrastructure

- `tests/frequency_stress.rs` — headless GPU test sweeping sphere frequencies with fixed and scaled configs. Detects explosion via frozen flag. Current max stable: freq 32 (fixed dt), freq 40+ (scaled).
- `tests/klein_stress.rs` — headless Klein bottle generation, approach-based settling, and physics stability over 300 frames.

## What We Learned

### The dispatch overhead wall

Each physics iteration requires multiple compute dispatches. At 80 iterations/frame with 8 dispatches each, that's 640 dispatches. The single compute pass optimization (all dispatches in one pass, one submission) was critical — the original per-pass architecture hung beyond ~80 iterations.

Spring-push mode helps here: 5 dispatches instead of 8, enabling more iterations at the same overhead budget.

### Approach-based settling for random-start topologies

Klein bottles and Möbius bands start from random joint positions. Direct settling with target ideal lengths causes violent forces. The approach system interpolates ideal lengths from actual → target over 20 steps of 2000 iterations each. This settled reliably in ~1-2 seconds wall time.

### Surface character as a shader concern

The four surface characters from tensegrity-lab (plus Absent) mapped directly to a switch statement in the ground_collision shader. No CPU-side abstraction needed — just a u32 in the params uniform.

### Speed limit as global freeze

The original per-joint "nuked" flag was complex (w-component flags, recovery logic, per-joint frozen state). Replacing it with a single atomic frozen flag simplified the shader significantly and better matches the physics reality: if any joint exceeds the speed limit, the simulation state is compromised and should halt entirely.

## Path Forward

### Reduce passes per iteration

The 8-pass SHAKE/RATTLE architecture could be reduced to 3-4 by merging joint-domain passes. The force reset is already merged into second_half_kick.

### Higher stiffness

The CFL stability limit for explicit integration is dt < 2/sqrt(k/m). Supporting physical Dyneema (6.7×10⁹ N/m) requires dt ≈ 55µs and ~364 iterations/frame. This needs fewer dispatches per iteration or in-shader iteration loops.

### Float atomics

Replace i32 atomic force accumulation with compare-and-swap float accumulation or subgroup reductions. Eliminates quantization error and overflow risk.

### More shapes

tensegrity-lab has a full DSL (Tenscript) for constructing arbitrary tensegrity fabrics. Porting the algorithmic shapes (sphere, Klein, Möbius) was straightforward; a GPU-compatible fabric builder could enable custom constructions.

## Codebase Map

| File | Purpose |
|------|---------|
| `src/main.rs` | Entry point, ShapeConfig enum |
| `src/app.rs` | Window, event loop, shape/surface/slider handling |
| `src/constants.rs` | All physics constants |
| `src/tensegrity.rs` | Geodesic sphere topology generation, TensegritySphereBuffers |
| `src/sphere.rs` | Icosahedron subdivision (geodesic scaffold) |
| `src/klein.rs` | Klein bottle topology generator |
| `src/mobius.rs` | Möbius band topology generator |
| `src/twitcher.rs` | Muscle animation — traveling sine wave contraction |
| `src/camera.rs` | Orbit camera with mouse controls |
| `src/gpu/mod.rs` | wgpu device/surface initialization |
| `src/gpu/physics.rs` | Compute pipeline setup, dispatch modes, settle, readback |
| `src/gpu/physics.wgsl` | All compute shader entry points |
| `src/gpu/renderer.rs` | Cylinder instance rendering, triangular ground grid |
| `src/gpu/render.wgsl` | Vertex/fragment shaders |
| `src/gpu/hud.rs` | Text-based HUD with shape bars, sliders, surface buttons |
| `src/lib.rs` | Library facade for test access |
| `tests/frequency_stress.rs` | Headless sphere frequency sweep |
| `tests/klein_stress.rs` | Headless Klein bottle stress test |

## Heritage from tensegrity-lab

Ported: geodesic sphere generation, Klein bottle generator, Möbius band generator with muscle twitching animation, surface character physics (Absent/Frozen/Sticky/Bouncy/Slippery), pre-tension model, settling, speed limit concept.

Changed: push intervals use SHAKE/RATTLE (rigid) for spheres, spring-push for shared-joint topologies. Physics runs on GPU compute shaders. Speed limit triggers global freeze. Elastic ideal lengths updatable at runtime via `queue.write_buffer` for muscle animation without pipeline rebuild.

Not yet ported: Tenscript DSL, evolution framework, fabric plan executor, CSV export, faces/radials, scaffold forces, pretensing by strut extension, dual chirality Möbius bands.
