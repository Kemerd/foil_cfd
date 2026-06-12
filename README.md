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

## Mesh refinement (two-level grid)

The LBM lattice is intrinsically uniform — you can't stretch it without changing the
physics it solves. So FoilCFD refines the honest way: a **2×-finer nested lattice patch**
covering the leading edge, suction-surface boundary layer, VG zone, and near wake, two-way
coupled to the base grid every step (the multi-level scheme of the Filippova–Hänel /
Dupuis–Chopard family).

**How it works:** the fine level runs two half-`dt` steps per coarse step at half the cell
size, with the same physical viscosity (`tau_f = 2·(tau_c − ½) + ½`). At the patch boundary
a two-cell interface shell receives trilinearly-interpolated populations from the coarse
grid (time-blended between coarse steps), with the non-equilibrium part rescaled to the
fine level so the strain rate — and therefore the stress — carries across the interface
correctly. After its two sub-steps the fine solution is restricted (averaged, inverse-
rescaled) back onto the coarse grid. The fused collision kernel itself is untouched: it
runs identically on both levels.

**What you get:**
- **2× wall resolution** on the foil, the VG vanes, and the boundary layer the VG guidance
  reads — a vane that voxelized at 8 cells now gets 16.
- **Forces measured on the fine grid** whenever the patch covers every solid cell (the Mesh
  panel shows which grid the momentum exchange ran on).
- Honesty note: the effective Reynolds number is **shared across levels** — the patch buys
  resolution at the same Re, not a higher Re. Wall-function boundary conditions (a separate
  technique that models, rather than resolves, the inner boundary layer) are possible future
  work.

The **Mesh** panel (tabbed with Sim/View) has the enable toggle, four margin sliders
(upstream/wake/above/below, in chords around the foil+VG bounding box), a live diagram of
the patch box, and the VRAM bill. At the Default preset the patch adds roughly 6 GB; if the
allocation fails the sim simply continues on the base grid and tells you.

## Faster startups (mesh sequencing)

Every cold start (geometry, AoA, airspeed, VG edits) first converges a **4×-coarser
companion sim** — 1/64th the cells, a second or two of wall time — then trilinearly
upsamples that developed flow onto the full grid and continues. The wake and circulation
are already in place instead of convecting in from an impulsive start, so the Cl/Cd gate
opens far sooner. The readouts stay gated through a settle window while the full-resolution
grid re-adjusts the upsampled field; the usual trust-deltas rule applies unchanged. Toggle
it in the Mesh panel (on by default; sections only — STL imports start cold).

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
