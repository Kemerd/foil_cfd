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
| Wendt, B.J.: *Initial Circulation and Peak Vorticity Behavior of Vortices Shed from Airfoil Vortex Generators.* NASA/CR-2001-211144 (2001). | Source of the shed-circulation correlation behind the vortex-strength audit (`geom/vg_audit.h`): Gamma = k1 alpha V c tanh(k3 h/delta) / (1 + k2/AR), AR = 8h/(pi c), k1 = 1.61, k2 = 0.48, k3 = 1.41. The audit compares the LBM's measured crossflow circulation against this prediction as the resolved-vane honesty meter. |
| Dudek, J.C.: *An Empirical Model for Vane-Type Vortex Generators in a Navier-Stokes Code.* AIAA-2005-1003 / AIAA Journal 44(8) (2006). | Implementation-grade statement of the Wendt correlation (Eq. 3) and its validity ranges (0.13 < h/c < 2.62, 0.12 < h/delta < 2.60); the constants in `vg_audit.cpp` follow this paper. |
| Asmuth, H., Olivares-Espinosa, H., Nilsson, K., Ivanell, S.: *Wall-modeled lattice Boltzmann large-eddy simulation of neutral atmospheric boundary layers.* Phys. Fluids 33, 105111 (2021). | The inverse momentum exchange method (iMEM): enforcing a wall-function shear stress by prescribing a slip velocity on bounce-back links, without eddy-viscosity reconstruction — the design FoilCFD's wall model (`sim/lbm_wallmodel.cuh`, `sim/wall_model.h`) follows. |
| Ponsin, J., Lozano, C.: *Wall boundary conditions for Lattice Boltzmann simulations of turbulent flows with wall functions.* (INTA, 2025; DOI 10.1063/5.0283930). | Systematic comparison showing slip-velocity bounce-back wall functions beat regularized reconstructions on accuracy and grid-coarsening robustness — the evidence basis for choosing the bounce-back family. |
| Reichardt, H.: *Vollständige Darstellung der turbulenten Geschwindigkeitsverteilung in glatten Leitungen.* ZAMM 31 (1951). | The all-y+ law of the wall the wall model's Newton solve inverts (kappa = 0.41, C = 7.8 — smooth from the viscous sublayer through the log layer, needed because first-cell y+ spans ~1 to ~300 along the chord). |
| Flamm, M.H., Kalal, Z.: *Near-wall resolution and wall modeling for lattice Boltzmann simulations of mechanically agitated vessels.* Chem. Eng. Sci. 264, 118172 (2022). | Evidence that high-Re LBM WITHOUT a wall model mispredicts wall-driven flow wholesale (premature jet detachment) unless resolution is extreme — the motivating failure mode for adding the wall function. |
| Filippova, O., Hänel, D.: *Grid Refinement for Lattice-BGK Models.* J. Comput. Phys. 147(1), 219-228 (1998); DOI 10.1006/jcph.1998.6089. | The original equilibrium/non-equilibrium split at a refinement interface: feq transfers unchanged (continuity of rho,u), fneq is rescaled by the tau-ratio to keep the viscous stress continuous — the family the FoilCFD coupling (`sim/lbm_refine.cu`) belongs to. |
| Dupuis, A., Chopard, B.: *Theory and applications of an alternative lattice Boltzmann grid refinement algorithm.* Phys. Rev. E 67, 066707 (2003); DOI 10.1103/PhysRevE.67.066707. | The fneq rescale the fill/restrict kernels use, `neqScale = tau_f/(m*tau_c)` and its inverse, valid for all tau (rescale-before-collision) — needed because the Smagorinsky tau varies cell-to-cell. |
| Latt, J., Chopard, B.: *Lattice Boltzmann method with regularized pre-collision distribution functions.* Math. Comput. Simul. 72(2-4), 165-168 (2006); DOI 10.1016/j.matcom.2006.05.017. | The interface fneq REGULARIZATION (`regularizeFneq()` in `sim/lbm_refine.cu`): project fneq onto the second-order Hermite/stress space, fneq_q = (w_q/2cs^4) Q_q:Pi_neq (their Eq. 10), discarding the non-hydrodynamic ghost moments that become the interface speckle. Conserves mass and momentum exactly. |
| Lagrava, D., Malaspinas, O., Latt, J., Chopard, B.: *Advances in multi-domain lattice Boltzmann grid refinement.* J. Comput. Phys. 231(14), 4808-4822 (2012); DOI 10.1016/j.jcp.2012.03.015. | The 1/19 box filter on fneq during fine->coarse restriction (their Eq. 33, fneq only): averaging fneq over the node + 18 lattice neighbours stops fine-grid structures polluting the coarse grid — the change that makes the nested (high-ratio) VG patch stable instead of divergent. |
| Astoul, T., Wissocq, G., Boussuge, J.-F., Sengissen, A., Sagaut, P.: *Analysis and reduction of spurious noise generated at grid refinement interfaces with the lattice Boltzmann method.* J. Comput. Phys. 418, 109645 (2020); DOI 10.1016/j.jcp.2020.109645. | Diagnosis behind the two fixes above: the interface vortex/speckle signature is aliasing energy dumped into non-hydrodynamic ghost modes at the resolution jump — a collision/reconstruction problem, not a geometry one. Justifies regularizing + filtering fneq rather than chasing a sharper interpolation. |

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
