# Project citations & reference library

Everything the Glasair III VG study leans on, with where each source is used.
Local copies (where downloaded) live in `validation/nasa/` and `ref/`.

## Vortex generator theory & design guidance

| Citation | Use in project | Link / location |
|---|---|---|
| Lin, J.C., *Review of Research on Low-Profile Vortex Generators to Control Boundary-Layer Separation*, Progress in Aerospace Sciences 38 (2002) 389-420; NASA Langley. | THE design-guidance anchor (cited in the project spec): place VGs 5-10 vane-heights upstream of a relatively fixed separation onset; low-profile h = 0.1-0.5 delta vs conventional h = delta; drag trade. | [NTRS 20030000983](https://ntrs.nasa.gov/citations/20030000983) — not yet downloaded |
| Wentz, W.H. & Seetharam, H.C. (Wichita State, NASA grant), VG tests on the GA(W)-1 airfoil. | Closest experimental VG-on/off anchor for this airfoil family (GA(W)-1 = 17% sibling of our GA(W)-2): VGs suppressed separation, raised lift-curve slope and Clmax, but LOWERED the stall angle — the stall is reshaped, not just postponed. | TO ACQUIRE — find the specific WSU/NASA CR and add to validation/nasa/ |
| *Experimental Study on the Effect of Vortex Generator on the Aerodynamic Characteristics of NASA LS-0417 Airfoil*, Applied Mechanics and Materials 758 (2015). | Modern VG-on-GA(W)-1 experiment; secondary corroboration. | [scientific.net AMM.758.63](https://www.scientific.net/AMM.758.63) |
| Strausak, V., *Vortex Generators in Practical Testing*, IE Impulse #74 (Jan 2021), pp. 7-9 (Igo Etrich Club Austria; machine-translated). | Flight-proven recipe on a Lancair Legacy (same problem class): wing VGs at 7% chord, 50 mm pitch outboard / 90 mm inboard, 15 deg vanes, counter-rotating; elevator VGs 100 mm ahead of hinge, 30 mm pitch. Stall -22%, full aileron authority to the stall. Centers of all our sweep ranges. | `ref/Impulse74_Translated.pdf` (+ .txt extraction) |

## Airfoil & aircraft data

| Citation | Use in project | Link / location |
|---|---|---|
| McGhee, R.J. & Beasley, W.D., *Effects of Thickness on the Aerodynamic Characteristics of an Initial Low-Speed Family of Airfoils for General Aviation Applications*, NASA TM X-72843. | Primary clean-section experimental anchor: LS(1)-0413 lift curves at Re 2.2/4.3/6.4e6 (fixed transition). Digitized into `validation/nasa/digitized/`. | `validation/nasa/NASA_TMX72843_thickness_effects_GAW_family.pdf` ([NTRS 19790004829](https://ntrs.nasa.gov/citations/19790004829)) |
| Wentz, W.H., *Wind Tunnel Tests of the GA(W)-2 Airfoil with 20% Aileron, 25% Slotted Flap, 30% Fowler Flap and 10% Slot-Lip Spoiler*, NASA CR-145139 / WSU AR 76-2 (1977). | Deflected-aileron experimental anchor — the Glasair aileron measures 19.87% chord on this exact section. Basic lift curve digitized; aileron series digitization deferred. | `validation/nasa/NASA_GAW2_aileron_flap_spoiler.pdf` ([NTRS 19790001850](https://ntrs.nasa.gov/citations/19790001850)) |
| McGhee, R.J. & Beasley, W.D., *Low-Speed Wind Tunnel Results for a Modified 13-Percent-Thick Airfoil*, NASA. | Backup if the as-built wing deviates from the published section (VT slides cite t/c = 12.5% vs published 13%). | `validation/nasa/NASA_modified_13pct_airfoil.pdf` ([NTRS 19790016789](https://ntrs.nasa.gov/citations/19790016789)) |
| UIUC Airfoil Coordinates Database, LS(1)-0413 (GA(W)-2). | Committed section coordinates (`geometry/ls413.dat`). | [ls413.dat](https://m-selig.ae.illinois.edu/ads/coord/ls413.dat) · [reference index](https://m-selig.ae.illinois.edu/ads/ref/ls0413_ref.html) |
| Carobine, A., Fitzwater, R., Jackson, D., *Glasair III* analysis slides, Virginia Tech (2005-03-30). | Aircraft-level data: weights, areas, performance, stability; several values superseded by DXF measurement. | `ref/virginia_tech_GlassairIII.pdf` (+ .txt) |
| Stoddard-Hamilton Glasair III factory 3-view, drawing rev G (1990-04-26); owner-converted DXFs. | Measured planform truth: chords, taper, aileron/elevator/rudder geometry, hinge lines (see `geometry/dxf/measured.yaml`, extracted by `scripts/measure_dxf.py`). | `ref/Glasair-III-3-View-Drawing.{pdf,dwg,ai}`, `geometry/dxf/glasair_{top,side,front}view.dxf` |

## Tools & solvers

| Citation | Use in project | Link |
|---|---|---|
| Lehmann, M., *FluidX3D* (LBM, OpenCL), v3.7. | Active GPU solver arm (visual/qualitative; FP32/FP16S). Custom Glasair setup in `gpu/fluidx3d/`. Non-commercial license. | [github.com/ProjectPhysX/FluidX3D](https://github.com/ProjectPhysX/FluidX3D) |
| OpenFOAM v2506 (ESI/OpenCFD). | Parked wall-resolved RANS pipeline (kOmegaSSTLM transition); pinned in aircraft.yaml. | [openfoam.com](https://www.openfoam.com/news/main-news/openfoam-v2506) |
| Drela, M., *XFOIL* 6.99 (MIT). | 2D panel/e^N baseline polars (`validation/xfoil/`); binary provenance in `tools/xfoil/SOURCE.md`. | [web.mit.edu/drela/Public/web/xfoil](https://web.mit.edu/drela/Public/web/xfoil/) |
| Jirasek, A., jBAY vane source-term model (AIAA J. Aircraft 2005). | Planned VG model for the RANS pipeline (spec milestone M2, not yet built). | TO ACQUIRE at M2 |

## GPU-CFD landscape (evaluated 2026-06-11/12, decisions in docs/BACKLOG.md)

| Source | Verdict |
|---|---|
| [RapidCFD (SimFlowCFD)](https://github.com/SimFlowCFD/RapidCFD-dev) | Rejected for this study: no transition model (verified in source), 12-year-old base. Parked: cloned in WSL, sm_120-patched, CUDA 12.9 installed. |
| NVIDIA/ESI AmgX + PETSc4FOAM + FOAM2CSR (8th OpenFOAM Conference; Martineau, Posey, Spiga 2020) | Candidate pressure-solve offload at Phase-3 mesh sizes (~4-8x on the linear solve, ~2-3x wallclock, case dependent). |
| gpuFOAM / stdpar porting (Hartree Centre et al.) | Watch item: mainline OpenFOAM GPU offload targeting ~v2606. |
| Dickenmann, N., *Fifteen Years of FP64 Segmentation...* (blog, 2026-02-18) | Context for consumer-GPU FP64 limits and Ozaki-scheme emulation (GEMM-only — not applicable to sparse CFD). |

## FoilCFD additions

References adopted for the FoilCFD CUDA lattice-Boltzmann solver. The Esoteric
Pull / Esoteric Push and FP16 number-format papers below directly inform the
FoilCFD plan section-11 `load_f()/store_f()` accessor design: population access
is funneled through one accessor pair from day one so that FP16 storage with
FP32 compute, and a later in-place (AA/Esoteric) streaming upgrade, are each a
typedef/index-function swap rather than a kernel rewrite.

| Citation | Use in FoilCFD |
|---|---|
| Lehmann, M.: *Esoteric Pull and Esoteric Push: Two Simple In-Place Streaming Schemes for the Lattice Boltzmann Method on GPUs.* Computation 10, 92 (2022). | In-place streaming scheme (halves population memory vs ping-pong); target of the stretch upgrade behind the section-11 accessors. |
| Lehmann, M., Krause, M., Amati, G., Sega, M., Harting, J., Gekle, S.: *Accuracy and performance of the lattice Boltzmann method with 64-bit, 32-bit, and customized 16-bit number formats.* Phys. Rev. E 106, 015308 (2022). | Justifies FP16 population *storage* with FP32 *compute* (~2x bandwidth win at acceptable accuracy); the reason the accessors exist. |
| Lehmann, M.: *Computational study of microplastic transport at the water-air interface with a memory-optimized lattice Boltzmann method.* PhD thesis (2023). | Consolidated reference for the memory-optimized GPU LBM techniques (formats + in-place streaming) used above. |
| Lehmann, M.: *Combined scientific CFD simulation and interactive raytracing with OpenCL.* IWOCL'22 (2022). | Pattern for fusing a live renderer with a running GPU LBM solver (FoilCFD's CUDA-GL interop visualizer). |
| Lehmann, M.: *High Performance Free Surface LBM on GPUs.* Master's thesis (2019). | Background on GPU LBM kernel structure and performance characteristics. |

VG-placement anchors (Lin 2002; Strausak 2021; Wentz & Seetharam; McGhee &
Beasley TM X-72843) carry over from the tables above and drive the FoilCFD
VG-guidance overlay (Lin: VGs 5-10 h upstream of separation onset, h = 0.1-1.0
delta99; Strausak: x/c ~ 0.07, +/-15 deg counter-rotating pairs, l = 3h).

## House findings worth citing internally

- Quasi-2D span-periodic LBM domains never reach a stall break (lift keeps
  rising past 20 deg via coherent spanwise rollers); a single mid-span bump
  ("speck") breaks the artificial coherence — see suite reports under
  `gpu/fluidx3d/results/suite/`.
- FluidX3D's voxelizer parity-counts along the minimum-cross-section axis;
  thin plates along that axis can vanish wholesale. Solids smaller than one
  lattice cell in any dimension must be stamped analytically (see the
  stamper in `gpu/fluidx3d/setup_glasair.cpp`) with a one-cell floor.
