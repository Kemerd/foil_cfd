# Integration notes — render module (viz.* / particles.* / assets/shaders)

Status: render module is COMPLETE (no stubs remain in src/render). This file
records the interface decisions other agents bind against.

## Facts (already true in the tree, not proposals)

### VizSettings grew fields the UI agent should expose (src/render/viz.h)
- `float particleSpeedColorScale = 0.15f` / `float particleVorticityColorScale
  = 0.05f` — value mapped to the palette end for the particle color key
  (lattice units). Optional sliders; defaults are sensible for u_lat ~0.08.
- `SliceConfig::cell` default changed to **-1 = "domain center"**; the
  renderer clamps any explicit value into range. `SliceConfig::rangeScale`
  (default 1.0) is a contrast multiplier on the field's default
  normalization (|u| 0.15, vorticity-z +/-0.04, pressure +/-0.01 lattice).
- **One slice texture per axis**: at most one ENABLED slice per axis draws;
  if two enabled slots share an axis the lower-indexed slot wins (documented
  on SliceConfig). Default slices[] pre-arms X/Y/Z one each, mid-span Z
  vorticity enabled, so toggling `showSlices` shows content immediately.
- Q-criterion raycast (mode 3 / hotkey 3) toggles:
  `bool showQRaycast` (default false), `float qThreshold` (0.15, on the
  NORMALIZED Q in [0,1]), `float qScale` (5e-5, lattice Q mapped to 1.0),
  `int qUpdateEveryNFrames` (4). The 3D volume texture (~1 B/cell) is created
  lazily on first enable; if the q shaders fail to compile the mode degrades
  to unavailable without affecting modes 1/2 (stderr notes why).

### ParticleAdvectParams semantics (src/render/particles.cuh)
- `inletSpawnFraction` (default now 0.3) = probability that a LIFETIME/SOLID
  death respawns at the inlet rather than re-seeding the fluid volume.
  Outflow exits ALWAYS respawn at the inlet (that is the rate balance).
- `colorScale` added: kernel writes colorKeys already normalized to [0,1];
  the GLSL side is a pure palette lookup.
- New kernel APIs in the same header (implemented in particles.cu):
  `launchSliceFill` (RGBA8 surface, texel->cell mapping documented in the
  header) and `launchQCriterionVolume` (R8 3D surface).
- `launchAdvectParticles` (sim/lbm_core.cuh forwarder) is implemented here,
  unchanged signature.

### Renderer behavior contracts
- Every CUDA<->GL registration happens exactly once per resource lifetime:
  particle pos/key VBOs at init (re-registered ONLY inside
  `resizeParticlePool`, where the GL buffers themselves are recreated; old
  pool survives a failed resize), 3 slice textures at init, Q volume at lazy
  creation. Verified by code path: no register call is reachable per-frame.
- `updateFields` brackets ALL GL-touching kernels in ONE
  map/unmap per frame on the app's single stream (plan 9.3). dtSteps is
  capped at 8 lattice steps per frame for visual continuity (selftest passes
  200). Null `vel.u` (solver not ready) is a clean no-op.
- `uploadGeometry` mirrors the voxelizer transform CONTRACT from
  geom/voxelizer.h doc comments: rotate chord-unit section about (0.25, 0)
  by **-aoa_deg**, scale by `layout.chordCells`, anchor quarter-chord at
  (anchorX, anchorY). VG vanes use `surfaceFrameAt(airfoil, x_c, upper=true)`
  + the same rotation; vane geometry mirrors vg.h's doc (plates h tall along
  the local normal, l = length_h*h, yaw +/-beta about the normal, pairs
  gap_h*h apart with commonFlowDown sign flip, arrays pitch_c apart centered
  mid-span, Ramp = right-triangular wedge h wide). **Voxelizer/vg agents: if
  your implementation deviates from your own header docs, the rendered mesh
  will visibly detach from the voxel solid — flag it here.**
- Shaders load from `assets/shaders/` found by walking up from the exe dir
  (handles build/Release) and the CWD. A distribution layout should ship
  `assets/` next to the exe.
- `screenshotPNG` reads the BACK buffer; call before swap (selftest does).

### GL level
Everything fits the bundled glad 3.3-core loader (no >3.3 entry points used;
shaders are `#version 330 core`). No loader extension needed.

## Requests to other agents
- **UI agent**: bind the new VizSettings fields above; hotkey 3 ->
  `showQRaycast`. Slice UI should treat axis as fixed-per-slot (X/Y/Z) or
  respect the one-per-axis rule.
- **Solver agent**: `velocityField()`/`deviceRho()`/`deviceFlags()` are
  consumed exactly as declared in lbm_solver.h — keep macroscopic arrays
  refreshed when a render mode is active (StepParams::writeMacro).
