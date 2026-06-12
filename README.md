# FoilCFD

**A real-time GPU wind tunnel for people who build airplanes — or are just fascinated by what makes them fly.**

[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20NC%201.0.0-blue.svg)](LICENSE)
[![Platform: Windows](https://img.shields.io/badge/platform-Windows%20%7C%20NVIDIA-green.svg)](#requirements)
[![CUDA 12.8+](https://img.shields.io/badge/CUDA-12.8%2B-76b900.svg)](#requirements)

![Preview of software](preview.jpg)

Load your airfoil. Set your angle of attack. Watch a million particles stream over the wing
in real time. Bolt on vortex generators, drag a slider, and see the separation bubble react
*right now* — not after an overnight mesh-and-solve marathon.

FoilCFD is a CUDA lattice-Boltzmann solver (D3Q19, TRT, Smagorinsky LES) strapped to an
interactive 3D visualizer. On an RTX 5090 it pushes **~8 billion lattice updates per second**
on a 23.6M-cell grid — about 340 full solver steps every second, while rendering. That's not
"fast for CFD." That's a wind tunnel with a framerate.

### Sim time vs. wall time

One lattice step advances physical time by `dt = u_lat × dx / airspeed` (derived in
`src/sim/units.h`, the single unit-conversion authority). At the default setup — 1.2 m
chord, 30 m/s airspeed, 256 cells/chord, lattice speed 0.08 — that's dt ≈ 12.5 µs, so
**~80,000 solver steps equal one real-world second of flow**. The general formula:

```
steps per real second = airspeed [m/s] × cells_per_chord / (u_lat × chord [m])
```

At ~340 steps/s on an RTX 5090, the tunnel runs at roughly **1/235 of real time** — one
second of physical flow takes about four minutes of wall clock at the default grid. The
app itself doesn't slow down to match: per-frame solver work is capped at 10 ms (TDR
safety), so the visualizer stays at interactive framerates (typically **60+ FPS**,
render-bound), advancing ~5–6 solver steps per rendered frame. Faster airspeeds and
finer grids both shrink dt, so the steps-per-real-second count grows linearly with each.

*(Screenshots: clean vs. VG'd wing comparison pair coming with milestone M4 —
`screenshots/m4_clean.png` / `screenshots/m4_vg.png`)*

---

## Who this is for

You're building a Glasair, a Lancair, an RV — something with your name on the data plate —
and you're staring at a bag of vortex generators wondering *where exactly do these go?*

FoilCFD gives you a defensible **starting point**: chordwise station, height, spacing, and
incidence, grounded in your actual airfoil at your actual angle of attack, guided by
published correlations (Lin 2002, flight-proven recipes — see [docs/CITATIONS.md](docs/CITATIONS.md)).
Run the clean wing, save the converged state, drop a VG array, and compare. The delta is
the product.

Then go tuft-test it. Cotton tape is cheap. Trust, but verify — this tool exists to make
your first guess a *good* guess, not to replace the roll of tape.

Not drilling holes in anything yet? It's still mostly about airplanes — but the tunnel
doesn't check your pilot's license. Any watertight STL flies here (see
[Custom STL import](#custom-stl-import)): drones, fairings, whatever you're curious about.

## The honesty section

Read this before you drill holes in your wing:

- **Trust deltas, not absolutes.** VG-on vs. VG-off on the identical grid is meaningful.
  An absolute Cl at flight Reynolds number is not a certification value.
- **Effective Reynolds number is displayed, always.** Your wing flies at Re ~2-6 million;
  the solver runs at the highest stable effective Re it can (typically 1e4–1e5 at default
  resolution) and tells you both numbers. Separation trends, vortex topology, and VG
  placement behavior carry over; boundary-layer-resolved drag counts do not.
- **High Fidelity mode** trades interactivity for accuracy: finer grids (Fine/Ultra
  presets), lower lattice Mach, force averaging over many flow-throughs. Dual-resolution
  Richardson trend extrapolation with an error bar on deltas is on the v1.x roadmap.
- **Walls are stair-stepped voxels** with half-way bounce-back. Good enough for trends;
  not a panel-method polish.
- Forces are gated until the flow has actually developed (two flow-throughs minimum) —
  the readout refuses to lie to you while the tunnel is still spinning up.

## 60-second quickstart

```powershell
git clone <this-repo> foil_cfd
cd foil_cfd
cmake -B build -G "Visual Studio 17 2022" -A x64 -T cuda=12.9
cmake --build build --config Release
.\build\Release\FoilCFD.exe
```

Exact build details and troubleshooting: [BUILDING.md](BUILDING.md).

1. Pick an airfoil — NACA 4-digit text box, or any of the **1,587 UIUC database foils**
   shipped in `airfoils/uiuc/`, or search your *aircraft* by name in the Aircraft menu
   (Glasair III → LS(1)-0413, like magic).
2. Set AoA and airspeed. The flow starts immediately.
3. Press `1` for the particle view. Find where the flow separates.
4. Save the clean state (one click — it's cached in VRAM and on disk).
5. Add VGs in the VG editor. The sim warm-restarts from the clean state in seconds —
   it never recomputes the whole wing from scratch.
6. Compare Cl/Cd/L-over-D deltas and the separation line. Iterate.

## Airfoils in, three ways

- **NACA generator** — type "2412", get a wing section.
- **UIUC database** — the full coordinate database is committed in `airfoils/uiuc/`
  (refresh it anytime: `python scripts/fetch_uiuc_airfoils.py`). Drop your own `.dat`
  files (Selig or Lednicer format) into `airfoils/` — the loader is defensive about the
  database's, let's say, *artisanal* formatting, and tells you exactly why a file was
  rejected if it can't be parsed.
- **Aircraft manifest** — `airfoils/aircraft_manifest.csv` maps popular aircraft
  (homebuilts first) to their sections, sourced from Lednicer's *Incomplete Guide to
  Airfoil Usage*. Searchable in-app. User-editable CSV — add your own bird.

## Vortex generators

Parametric VG types: single vane, counter-rotating pairs, co-rotating arrays, ramps.
Place by chordwise station; height, length, incidence, pitch, and gap are all live
parameters. The vane root follows the actual airfoil surface.

The **guidance panel** shows the simulated boundary-layer thickness (δ99) at your selected
station and the published recommended ranges: VG height 0.1–1.0 δ99 (low-profile sweet spot
0.2–0.5), placement 5–10 vane-heights upstream of separation onset (Lin 2002), with a
one-click flight-proven preset (x/c ≈ 0.07, ±15° counter-rotating, l = 3h — Strausak 2021).
If your VG is under-resolved on the current grid, the UI says so instead of rendering noise.

## Custom STL import

Drag a watertight STL onto the window. You get axis remap, scaling to a chosen chord, GPU
ray-parity voxelization, and the same force readouts. Caveats (by design, v1):

- Non-watertight meshes produce garbage; FoilCFD detects bad parity and warns instead of
  pretending. Fix your mesh.
- No parametric surface means no automatic VG surface placement — you get a free 3D
  placement gizmo instead.
- Spanwise-periodic boundaries may be wrong for a full 3D body; there's a toggle to switch
  the z-boundaries to free-slip walls in STL mode.
- Triangle cap: 2M.

## Mesh refinement

FoilCFD's solver operates on a uniform logical grid — every cell has the same LBM collision
math, same memory layout, same FLOPS cost. Mesh refinement doesn't change that. Instead it
bends the **physical coordinate map**: a tanh-based stretching function that packs more cells
per metre into regions where resolution matters most, without adding a single cell or
consuming any extra VRAM.

The **Mesh** panel (tabbed with Sim/View, right column) exposes four presets:

| Preset | What it does |
|---|---|
| **Off** | Perfectly uniform — today's default, zero overhead |
| **Balanced** | 2× denser at the leading edge and suction surface BL; 4× in the VG zone; 0.5× in the far field |
| **Aggressive** | 4× LE, 8× VG zone, 0.3× far field — maximum safe contrast |
| **Custom** | All seven zone-weight sliders unlocked |

**Seven refinement zones**, each with an independent density weight:

1. **Far upstream** — inlet far-field; coarser saves cells for where they count
2. **Near upstream** — ~0.8c ahead of leading edge; transition ramp
3. **Leading edge** — suction peak, stagnation point, LE curvature; high accuracy here matters for Cl
4. **Top surface BL** — wall-normal direction on the suction side; resolves the boundary layer that the VG guidance panel reads
5. **VG zone** — ultra-fine band centered on the VG station (auto-tracks the VG editor); this is where the vortex generation mechanism lives
6. **Near wake** — 0–1.5c behind the trailing edge; the vortex shedding and VG vortex rollup
7. **Far wake** — downstream freestream; coarser

**How it works:** the stretching is computed once at startup as a piecewise tanh density
function integrated into a coordinate CDF. This gives a lookup table `physX[i]` mapping each
logical cell index to its physical x position. The voxelizer uses these tables when stamping
the airfoil and VGs, so geometry lands at the right physical location regardless of the
stretching profile. The LBM hot path (collision + streaming kernels) is completely untouched.

**Accuracy note:** LBM's collision operator assumes `dx = dy = dz`. Stretching introduces a
metric correction of order `(stretch_ratio - 1) × Ma²` — at the default lattice Mach 0.08
and ratios up to 8:1 this is around 0.5% of the existing compressibility error, well within
acceptable tolerance for VG placement decisions. The zone-weight bounds in the UI are capped
at 10:1 to stay firmly in that regime. Don't use Aggressive mode for high-AoA stall studies;
do use it for VG height/station sensitivity sweeps.

The VG zone centre auto-tracks the first VG station in the editor (configurable in the Mesh
panel). A live domain diagram in the panel shows relative cell density across the domain so
you can see exactly what you're buying before committing.

## How it works (one paragraph)

FoilCFD solves the lattice-Boltzmann equation on a D3Q19 lattice: every cell holds 19
particle-population values that stream to neighbors and collide each step, and the
macroscopic flow (density, velocity, forces) emerges from their moments. Collisions use a
two-relaxation-time (TRT) operator for clean wall behavior, plus a Smagorinsky subgrid model
computed locally from the non-equilibrium populations so the unresolvable turbulent scales
dissipate physically instead of blowing up. The whole update is one fused CUDA kernel — LBM
is embarrassingly parallel and memory-bound, which is why a consumer GPU does in
milliseconds what a CPU RANS solve does in hours. The memory-layout and precision tricks
follow Lehmann's work on GPU-LBM (Esoteric Pull, mixed-precision storage — see
[docs/CITATIONS.md](docs/CITATIONS.md)).

## Requirements

- Windows 10/11, NVIDIA GPU (Ampere or newer recommended; ~5 GB VRAM at default grid)
- CUDA Toolkit 12.8+ (12.9 tested), Visual Studio 2022, CMake 3.24+
- **Big grids + slow GPUs:** Windows will kill any kernel that hogs the display GPU for ~2 s
  (TDR). FoilCFD caps per-frame solver work to stay under it, but if you push Fine/Ultra
  presets on a smaller card, raise `TdrDelay` in
  `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers` (REG_DWORD, seconds, reboot).

## Validation

The solver isn't taken on faith: the CTest suite runs Taylor–Green decay (M0), lid-driven
cavity vs. Ghia et al. reference profiles (M1), and cylinder vortex shedding vs. the
canonical Strouhal band (M2) on the GPU, plus parser/units/voxelizer unit tests. Measured
results live in [notes/VALIDATION.md](notes/VALIDATION.md). Solver-QA anchors (digitized
NASA lift curves for the LS(1)-0413 family, XFOIL polars) live in
[validation/](validation/README.md).

## License

[PolyForm Noncommercial 1.0.0](LICENSE) — source-available, free for any noncommercial use.
Building a plane? Free. Selling CFD seats? Talk to us.
