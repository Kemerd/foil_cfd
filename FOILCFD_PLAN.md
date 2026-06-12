# FoilCFD — Implementation Plan

> **Hand this file to Claude Code as the project specification.**
> Target: a real-time GPU wind tunnel. CUDA lattice-Boltzmann solver + interactive 3D
> visualizer. Load any NACA/UIUC airfoil (or a custom STL), bolt on parametric vortex
> generators, and watch the flow live.

---

## 1. Project identity

- **Name:** FoilCFD
- **License:** PolyForm Noncommercial 1.0.0 (source-available, free for any noncommercial use).
  Put the full license text in `LICENSE` and a one-line notice in every source header.
- **Platform:** Windows-first (MSVC + CUDA). Keep code Linux-compatible (no Win32 calls outside
  one platform shim), but do not spend effort testing Linux.
- **GPU:** NVIDIA only, CUDA Toolkit **>= 12.8** (Blackwell/sm_120 support). Primary dev card is an
  RTX 5090 (32 GB). Compile for `-arch=native` by default with explicit fallback list
  `86;89;120` in CMake for release builds.
- **Language:** C++20 host code, CUDA C++ kernels. No Python anywhere in the runtime.

### Mission statement (OVERRIDES any "qualitative-only" framing below)
FoilCFD exists so **EAA/EAB homebuilders can run CFD on *their* airfoil and get a defensible
STARTING POINT for VG installation** — chordwise station, height, spacing, incidence — not a
certification number, but trends and deltas solid enough to act on and then verify with
tuft/cotton-tape testing. It is *also* a gorgeous real-time visualizer. Engineers may use it
in the future. Accuracy-relevant consequences:
- **Trust deltas, not absolutes:** VG-on vs VG-off on the identical grid/settings is the
  product. The UI and README frame results this way.
- **High Fidelity mode (toggle):** Fine/Ultra chord-resolution presets, `u_lat` lowered to
  0.05, force EMA over many flow-throughs, and the dual-resolution Richardson trend readout
  (§12.3, promoted from stretch to v1.x priority) with an error bar on Cl/Cd deltas.
- **VG placement guidance overlays** from published correlations (Lin 2002: place VGs 5–10 h
  upstream of separation onset, h = 0.1–1.0 δ99; Strausak flight-proven recipe: x/c ≈ 0.07,
  ±15° counter-rotating pairs, l = 3h). Compute δ99 along the suction surface from the sim
  and display the recommended h band at the selected station.
- **Validation anchors** (see §15): a small set of digitized NASA lift curves + XFOIL polars
  (copied from `L:\Dev\glasair_vg_sim\validation`) serve as **developer-side solver QA only**
  — they anchor the solver against one well-documented airfoil family; they are NOT a
  user-facing feature and we make no claim of covering all airfoils. The scalable per-airfoil
  reference path is **optional XFOIL integration (v1.x)**: shell out to xfoil.exe to generate
  a polar for whatever foil the user loaded and overlay it as a reference curve.
- Honesty stays: display effective vs target Re, gate force readouts until converged, and be
  explicit in the README that a user should tuft-test before drilling holes in their wing.

### Goals (in priority order)
1. Real-time-feeling 3D flow around an airfoil section, rendered in-app (no file export needed).
2. Airfoil sources: built-in NACA 4-digit generator + UIUC Selig `.dat` loader (drop files in
   a folder) + a fetch script for the full UIUC database (§15).
3. Parametric vortex generator (VG) placement on the suction surface; instantly see the change.
4. Warm-start cache so adding/editing VGs does NOT restart the flow from scratch.
5. Engineering-useful accuracy path: High Fidelity mode, δ99/VG-guidance overlay, validation
   anchors (see Mission statement).
6. Secondary: custom STL import (watertight solids), with reduced feature support (see §7.4).

### Non-goals / honesty (write these in the README too)
- v1 runs at a reduced effective Reynolds number (~1e4–1e5) with a Smagorinsky subgrid model
  and stair-step walls. Vortex topology, placement trends, separation behavior, and
  configuration *deltas* are meaningful and actionable; absolute Cl/Cd at flight Reynolds
  numbers carry uncertainty — verify installs with tuft testing. Wall functions (§12.4) are
  the path to higher effective Re in v1.x.
- No adaptive mesh refinement, no multi-GPU, no compressible flow in v1.

---

## 2. Tech stack

| Concern | Choice | Notes |
|---|---|---|
| Build | CMake >= 3.24 | `enable_language(CUDA)`, separable compilation OFF (single TU for kernels is fine) |
| Window/input | GLFW 3.4 | vendored via FetchContent |
| GL loader | glad (GL 4.6 core) | generated header committed to repo |
| UI | Dear ImGui (docking branch) | vendored via FetchContent |
| CUDA<->GL | `cudaGraphicsGLRegisterBuffer` / `RegisterImage` | register ONCE at init, map/unmap per frame |
| Math | tiny header (float3 helpers) | no GLM dependency needed, but GLM is acceptable |
| STL parse | hand-rolled (binary + ASCII) | trivial format, no library |

No other dependencies. Everything must build with `cmake -B build && cmake --build build --config Release`.

---

## 3. Repository layout

```
FoilCFD/
  CMakeLists.txt
  LICENSE                      # PolyForm Noncommercial 1.0.0
  README.md                    # screenshots, quickstart, honesty section on fidelity
  airfoils/                    # user drops UIUC .dat files here; ship 5-6 examples
  assets/shaders/              # GLSL for particles, slices, raycast
  src/
    main.cpp                   # owns the loop: input -> sim steps -> render
    app/ui.cpp/.h              # ImGui panels
    app/camera.cpp/.h          # orbit camera
    sim/lbm_core.cu/.cuh       # kernels: init, stream-collide, forces, watchdog, particle advect
    sim/lbm_solver.cpp/.h      # host orchestration, memory, snapshots, lattice scaling
    sim/units.h                # physical <-> lattice unit conversion in one place
    geom/airfoil.cpp/.h        # NACA generator + .dat loader (Selig & Lednicer)
    geom/stl.cpp/.h            # STL load (binary+ASCII), normalize, repair-check
    geom/voxelizer.cpp/.h      # polygon-extrusion path + generic STL ray-parity path
    geom/vg.cpp/.h             # VG parametric types + placement + voxelization
    render/viz.cpp/.h          # GL resources, interop, particle system, slice planes
    platform/                  # the only place OS-specific code may live
  tests/                       # small CTest set: dat parsing, voxelizer parity, units math
```

---

## 4. Solver core (sim/)

### 4.1 Method
- **Lattice:** D3Q19, FP32 throughout. Weights/velocities as standard (Krüger et al. ordering;
  document the ordering in `lbm_core.cuh` and never change it — snapshots depend on it).
- **Collision:** TRT (two-relaxation-time). Magic parameter `Lambda = 3/16` (best wall accuracy
  with bounce-back). BGK is acceptable for the very first bring-up milestone only.
- **Turbulence:** Smagorinsky LES, `Cs = 0.12`. Compute local strain rate from the
  non-equilibrium populations (standard trick — no finite differences needed), add eddy
  viscosity to the relaxation rate per cell.
- **Storage:** SoA — `f[q][cell]` as 19 contiguous arrays. Two full buffers (ping-pong),
  swap pointers per step. (AA-pattern in-place is a stretch goal, §12.)
- **Kernel structure:** ONE fused kernel per step, pull scheme:
  read neighbors' post-collision values -> compute rho,u -> TRT collide -> write to local cell
  of the destination buffer. One thread per cell, 1D grid over `nx*ny*nz`, block 128 or 256.
- **Cell flags:** `uint8 flags[cell]`: FLUID, SOLID, INLET, OUTLET, SLIP_TOP, SLIP_BOTTOM.
  No divergent per-q branching in the hot path: handle boundaries by flag lookup on the
  *neighbor* cell during the pull (bounce-back = read own opposite direction).

### 4.2 Boundary conditions
- **Inlet (x=0):** equilibrium Dirichlet at `u_in` (uniform; AoA is applied to the *geometry*,
  not the inflow — simpler voxel cache keying).
- **Outlet (x=nx-1):** zero-gradient copy of populations from the neighbor column.
- **Spanwise (z):** periodic.
- **Top/bottom (y):** free-slip (specular reflection).
- **Solids (foil + VGs):** half-way bounce-back against stair-step voxels.

### 4.3 Units and scaling (`units.h` — single source of truth)
- User specifies: chord `c_phys` [m], airspeed `U_phys` [m/s], air `nu_phys = 1.48e-5`.
- Lattice: chord resolution `N_c` cells (default 256), `u_lat = 0.08` (hard cap 0.12).
- `dx = c_phys / N_c`, `dt = u_lat * dx / U_phys`.
- Target `Re_phys = U*c/nu` is typically ~2.5e6 — **unreachable**. Compute the *effective*
  achievable Re from the minimum stable viscosity and DISPLAY BOTH numbers in the UI
  ("simulating at Re 8.4e4 (target 2.6e6)"). Clamp `tau >= 0.5005`; the Smagorinsky term
  provides additional stabilization.
- **Startup ramp:** begin each fresh run at 4x the target viscosity, ramp linearly to target
  over the first `2 * nx` steps (kills the initialization shock).

### 4.4 Forces (Cl/Cd readout)
- Momentum-exchange method summed over solid-adjacent links, reduced on GPU
  (`cub`-style block reduction or atomics into a small buffer; either fine at this scale).
- Display Cl, Cd, and L/D as exponential moving averages (window ~ one flow-through).
  Label the readout "qualitative" in the UI tooltip.

### 4.5 Robustness
- **NaN watchdog:** every 200 steps, kernel checks a strided sample of cells; on NaN ->
  pause sim, surface an ImGui error with the likely cause (too high u_lat / tau clamp hit).
- **Windows TDR:** never launch unbounded work; cap sim work per present to ~10 ms worth of
  steps. README must mention the TdrDelay registry key for users who push huge grids.

### 4.6 Default domain & memory budget (sanity table — verify at runtime, warn if >80% VRAM)
- Default grid: **768 x 320 x 96** (~23.6 M cells), chord = 256 cells, foil centered at
  x = 0.3*nx. Spanwise depth 96 cells ~ 0.375c — enough for several VG pairs.
- Memory: 19 floats x 2 buffers = 152 B/cell -> 3.6 GB. + flags + macroscopic (rho,u,v,w for
  rendering, 16 B) + particle buffers. Total < 5 GB. Leave headroom: snapshots (§8) and the
  user's browser tabs exist.
- Expected throughput on a 5090: 5–9 GLUPS -> 200–380 steps/s at default grid. Run N sim
  steps per rendered frame (adaptive N targeting ~60 fps presentation).

---

## 5. Geometry: airfoils (geom/airfoil.*)

### 5.1 NACA 4-digit generator
Standard equations; cosine point spacing (~200 points per surface); closed trailing edge
variant of the thickness polynomial (`-0.1036` last coefficient). Input is the 4-digit string
("2412", "0012"). 5-digit support is a stretch goal — do not block on it.

### 5.2 UIUC `.dat` loader
- **Selig format** (most common): name line, then x,y pairs running TE -> upper -> LE -> lower -> TE.
- **Lednicer format**: name, point-count line ("NU. NL."), upper surface LE->TE, blank, lower LE->TE.
  Detect by checking whether line 2 parses as two values > 1.
- Normalize: chord to [0,1], LE at origin. Reject files with < 20 points. Handle CRLF, stray
  blank lines, and duplicate LE points (the database is messy — be tolerant).
- App scans `airfoils/` at startup and on a "refresh" button; dropdown lists filename + the
  name line.

### 5.3 Airfoil voxelization (fast path)
- Apply AoA by rotating the 2D polygon about quarter-chord.
- For each (x,y) lattice column: point-in-polygon parity test on cell centers -> SOLID flag;
  extrude identical mask across all z. O(nx*ny) tests, instant. Re-voxelization happens on
  every AoA/airfoil change and on every VG edit (VG voxels are OR'd in, §6).

---

## 6. Vortex generators (geom/vg.*)

### 6.1 Parametric model
```cpp
enum class VGType { SingleVane, CounterRotatingPair, CoRotatingArray, Ramp };
struct VGParams {
  VGType type;
  float x_c;        // chordwise station, 0..1 (typical 0.05–0.30)
  float height_c;   // device height / chord (typical 0.005–0.02)
  float length_h;   // vane length in heights (default 3.0)
  float beta_deg;   // incidence to local flow (default 16, range ±30)
  float pitch_c;    // spanwise spacing between units (arrays/pairs)
  float gap_h;      // intra-pair gap in heights (pairs; default 2.5)
  int   count;      // number of units across the span
  bool  commonFlowDown; // pair orientation
};
```
- Vanes are thin boxes (1–2 cells thick) rotated by ±beta about the local surface normal,
  sitting on the **upper** surface at station `x_c`. Compute the local surface point and
  tangent/normal frame from the airfoil polygon; the vane root follows the surface.
- **Resolution guard:** if `height_c * N_c < 8` cells, show a UI warning ("VG under-resolved —
  increase chord resolution or VG size") rather than silently rendering noise.
- Ramp type = right-triangular wedge prism; same placement frame.

### 6.2 Editing flow
VG add/edit/delete -> rebuild flags (airfoil mask OR VG voxels) -> **warm restart** (§8).
Never cold-restart on a VG edit.

---

## 7. Custom STL import (geom/stl.*) — secondary feature

### 7.1 Loader
Binary and ASCII STL. Drag-and-drop onto the window (GLFW drop callback) or file dialog.
Show triangle count; refuse > 2M triangles in v1.

### 7.2 Normalization UI
STLs arrive in random units/orientation. On import show a small modal: bounding box, axis
remap presets (Y-up/Z-up), uniform scale so the longest x-extent maps to a user-chosen chord
in cells, and translation to center the object at the standard foil position.

### 7.3 Voxelization (generic path)
Ray-parity test: for each (y,z) row cast a +x ray, collect triangle crossings, sort, fill
between odd/even crossings. Do it on the GPU (one thread per row, triangles in a flat buffer;
a uniform-grid triangle binning pass is a stretch optimization — brute force is acceptable
up to ~200k triangles). **Non-watertight meshes will produce garbage; detect obviously bad
parity (odd crossing counts) and warn the user instead of crashing.**

### 7.4 Graceful degradation (document in README)
For STL geometry: VG surface-frame auto-placement is unavailable (no parametric surface) —
instead offer free 3D placement gizmo for vanes; warm-cache keys on the STL file hash; Cl/Cd
uses the same momentum exchange (fine); spanwise-periodic BC may be physically inappropriate
for a full 3D object — expose a toggle to switch z to free-slip walls in STL mode.

---

## 8. Warm-start cache (the "don't recompute the wing" feature)

**Purpose:** the clean-foil converged flow is the expensive part. Cache it; VG edits restart
from it and only the suction-side region must re-adjust (a few flow-throughs, seconds).

- **Snapshot contents (exact restart):** full `f` (19 floats/cell) + flags hash + scaling
  params. Size at default grid: 19 x 4 B x 23.6M = **1.8 GB** -> keep **one** VRAM-resident
  clean-foil snapshot (device-to-device copy, instant restore).
- **Compact variant (disk + extra slots):** macroscopic-only (rho, u, v, w = 16 B/cell,
  ~380 MB; ~150 MB zstd-compressed). Restore = set f to equilibrium(rho,u) + Smagorinsky
  re-init; costs a short transient (~1000 steps), still vastly better than cold start.
  Note for the README: a snapshot is ONE state, hundreds of MB — the multi-TB figures people
  associate with CFD data are for *every-timestep transient exports*, which FoilCFD never does.
- **Keying:** (airfoil id or STL hash, AoA bucket of 0.5°, u_lat, grid dims, solver version).
- **Flow on VG edit:** restore clean snapshot -> apply new flags -> cells newly SOLID get
  zeroed, cells newly FLUID get equilibrium from neighbors -> run. The restored field is
  wrong only near the new vanes, exactly where the solver fixes it first.
- **Disk cache:** `cache/` next to the exe, LRU capped at a user-set GB budget (default 10 GB),
  saved on a worker thread. Load on app start if key matches.

---

## 9. Rendering & UI (render/, app/)

### 9.1 Modes (hotkeys 1/2/3, all can overlay)
1. **Particles (the hero mode, build first):** 1,000,000 particles in a GL buffer registered
   with CUDA. CUDA kernel advects with RK2 through trilinearly-sampled velocity; respawn at
   inlet (rate-balanced) and on entering solids; age-fade. Color by local vorticity magnitude
   or speed (two palettes). Render as `GL_POINTS`, additive blending, depth-tested against
   the foil mesh. This single mode makes vortices, separation, and VG streaks vividly visible.
2. **Slice planes:** axis-aligned movable slices textured with |u|, vorticity-z, or pressure;
   CUDA writes into a registered GL texture (surface object). Diverging colormap for
   vorticity, sequential for speed. NO rainbow/jet as default; use viridis-like + coolwarm.
3. **Q-criterion isosurface look (stretch):** screen-space volume raycast in a fragment
   shader sampling a 3D texture of Q (computed by a CUDA kernel every N frames). Threshold
   slider. This is the "screenshot mode"; do it last.
- Foil + VGs rendered as extracted triangle mesh (marching the 2D polygon outline x extrusion
  is sufficient; do not marching-cubes the voxels).

### 9.2 ImGui panel
- Airfoil: dropdown (scanned `.dat` files) + NACA text box + "Load STL…" button; AoA slider
  (-5°…20°); airspeed; chord resolution preset (Fast 192 / Default 256 / Fine 320 with grid
  dims scaling accordingly).
- VG editor: list of VG entries with the §6.1 params, add/duplicate/delete; live re-voxelize
  on slider release (not per-tick).
- Sim: run/pause, single-step, reset (cold), "Save clean state" / auto-cache toggle;
  steps-per-frame indicator; MLUPS; effective-vs-target Re display; tau and u_lat readouts.
- Readouts: Cl, Cd, L/D (EMA), with the qualitative-disclaimer tooltip.
- View: mode toggles, particle count, colormap, slice positions, screenshot button (PNG).

### 9.3 Frame loop
`while: poll input -> (if running) N sim steps -> update particle kernel -> map interop ->
draw -> ImGui -> present`. Adaptive N: increase until frame time ~16 ms, decrease on misses.
Single CUDA stream; interop map/unmap brackets only the GL-touching kernels.

---

## 10. Validation milestones (build in this order; each is a CTest or a manual checklist item)

- **M0** Headless solver, 64³ Taylor–Green-like decay: kinetic energy decays monotonically,
  no NaN for 50k steps. (Automated test.)
- **M1** Lid-driven cavity 128², Re 1000 (run a 2D-thin 128x128x4 grid): centerline velocity
  profiles within ~5% of Ghia et al. reference points hard-coded in the test.
- **M2** Cylinder (voxelized, D=40 cells) at Re 150: vortex street visible; Strouhal from lift
  FFT in 0.17–0.21. (Automated-ish: log lift, assert peak frequency range.)
- **M3** NACA 0012 at AoA 0/8/16: attached flow, then TE separation, then full stall —
  visually sane in particle mode; Cl trend monotone 0->8 and drops by 16+.
- **M4** Add a counter-rotating VG pair at x/c 0.10 on a stalled-ish 2412 at high AoA:
  streamwise vortex streaks visible in particles; separation visibly delayed downstream of
  the array vs. clean. (Screenshot pair for the README.)
- **M5** Warm restart: VG edit converges (force EMA settles) in < 15% of cold-start steps.

---

## 11. Performance notes for the kernel author

- The solver is memory-bandwidth bound. Lattice update reads 19 + writes 19 floats: at
  152 B/cell and ~1.5 TB/s effective, the ceiling is ~10 GLUPS; 5–7 is a good real number.
  If you measure < 3, the streaming reads are uncoalesced — check the SoA indexing
  (`q*ncells + cell`, cell fastest) and the pull-neighbor index math.
- Precompute the 19 neighbor offsets as compile-time constants; periodic-z via index wrap,
  not modulo in the hot loop (use `if` on the boundary tile or padded ghost layer — padded
  ghosts in z are simplest).
- Branchless boundaries: the pull reads `f_opp` from self for SOLID neighbors via a select,
  not an `if/else` ladder over flags.
- Keep macroscopic (rho, u) written every step only if a render mode needs it that frame;
  otherwise skip the store (flag from host).
- FP16 *storage* with FP32 compute (Lehmann-style) is the single biggest stretch win
  (~2x): design `load_f()/store_f()` accessors now so the swap is one typedef later.

## 12. Stretch goals (post-v1, in rough order of value)
1. FP16 population storage (+~2x speed, halves memory).
2. AA-pattern streaming (halves memory again; enables Fine presets on smaller cards).
3. Dual-resolution Richardson readout: run Fast grid alongside, extrapolate Cl/Cd trend, show error bar.
4. Wall-function boundary for higher effective Re.
5. Body-force (jBAY) VG mode for sub-resolution vanes.
6. NACA 5-digit; airfoil morphing slider between two foils (pure fun).
7. Flow-field A/B diff view: clean vs VG'd snapshot, rendered as difference colors.

## 13. Known pitfalls (read before coding)
- **.dat zoo:** UIUC files have inconsistent headers, point counts, and direction; the loader
  must be defensive and report *why* a file was rejected.
- **Interop:** register GL buffers once; re-registering per frame silently tanks performance.
- **TE thinness:** voxelized trailing edges thinner than 1 cell create leaky single-cell gaps;
  after voxelization, run a one-pass closure: any FLUID cell with SOLID on both ±y neighbors
  becomes SOLID.
- **AoA changes invalidate the warm cache** (geometry rotated) — that's why the cache is
  keyed per AoA bucket. Changing airspeed does NOT invalidate (rescale via units, restart ramp).
- **Don't trust first 2 flow-throughs** of any cold start for forces; gate the Cl/Cd display.
- **ImGui sliders:** re-voxelize on release/`IsItemDeactivatedAfterEdit`, not every frame.

## 14.5 Build environment facts (this machine — bake into BUILDING.md)
- GPU: RTX 5090 (32 GB), driver 596.36. CUDA Toolkits installed: 10.1, 11.8, 12.3, 12.6, **12.9**.
- **The `nvcc` on PATH is CUDA 10.1 — do NOT use it.** Target CUDA 12.9 at
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9`. With the Visual Studio generator,
  select it via toolset: `cmake -B build -G "Visual Studio 17 2022" -A x64 -T cuda=12.9`.
  If MSVC 17.14 is rejected by nvcc, add `--allow-unsupported-compiler` to CUDA flags.
- Visual Studio 2022 Enterprise 17.14 (MSBuild + MSVC v143).
- Document the exact working configure/build commands in `BUILDING.md` once verified.

## 15. Reference data, citations, and the UIUC database (new)
> **Scope note:** the copied NASA/XFOIL data below is *developer validation* for the solver
> (milestone QA on the LS(1)-0413 family) — not a user-facing database, and intentionally not
> "all NASA lift curves" (no such digitized dataset exists; each curve is hand-digitized from
> PDF figures). Per-airfoil user-facing reference polars come later via optional XFOIL
> integration (v1.x): generate a polar on demand for the loaded foil and overlay it.
- **Copy from `L:\Dev\glasair_vg_sim`:**
  - `docs\CITATIONS.md` -> `docs/CITATIONS.md` (keep provenance, extend — see below).
  - `validation\nasa\digitized\*.csv` -> `validation/nasa/digitized/` (NASA TM X-72843 LS(1)-0413
    lift curves at Re 2.2/4.3/6.4e6; Wentz CR-145139 GA(W)-2 curve).
  - `validation\nasa\SOURCES.md` -> `validation/nasa/SOURCES.md`.
  - `validation\xfoil\polar_*.csv` -> `validation/xfoil/` (LS(1)-0413 XFOIL polars, Re 1.5/3/6e6,
    free + tripped; CSV contract `alpha,cl,cd,cdp,cm,xtr_top,xtr_bot`).
  - `geometry\ls413.dat` -> `airfoils/ls413.dat` (LS(1)-0413, Lednicer format — good loader test).
- **Extend CITATIONS.md with the Lehmann LBM references** (these inform §11 FP16 accessors and
  a future in-place streaming upgrade):
  - Lehmann, M.: *Esoteric Pull and Esoteric Push: Two Simple In-Place Streaming Schemes for the
    Lattice Boltzmann Method on GPUs.* Computation 10, 92 (2022).
  - Lehmann, M., Krause, M., Amati, G., Sega, M., Harting, J., Gekle, S.: *Accuracy and performance
    of the lattice Boltzmann method with 64-bit, 32-bit, and customized 16-bit number formats.*
    Phys. Rev. E 106, 015308 (2022).
  - Lehmann, M.: *Computational study of microplastic transport at the water-air interface with a
    memory-optimized lattice Boltzmann method.* PhD thesis (2023).
  - Lehmann, M.: *Combined scientific CFD simulation and interactive raytracing with OpenCL.*
    IWOCL'22 (2022).
  - Lehmann, M.: *High Performance Free Surface LBM on GPUs.* Master's thesis (2019).
  - Plus the VG-placement anchors already cited there (Lin 2002; Strausak 2021; Wentz & Seetharam;
    McGhee & Beasley TM X-72843).
- **UIUC database fetch:** `scripts/fetch_uiuc_airfoils.py` (helper script, NOT runtime; Windows
  console / UTF-8 safe). Scrapes `https://m-selig.ae.illinois.edu/ads/coord_database.html`,
  downloads every `.dat` into `airfoils/uiuc/` (~1600 files, ~15 MB) with polite rate limiting,
  idempotent/resumable (skip existing). The database is small enough to commit to the repo once
  fetched. App scans `airfoils/` recursively.

## 16. README requirements
Screenshots (M4 pair), 60-second quickstart, the fidelity-honesty section (§1 non-goals),
UIUC database link + note that users drop `.dat` files into `airfoils/`, STL caveats (§7.4),
license badge (PolyForm NC), and a short "how it works" with a one-paragraph LBM explainer.
