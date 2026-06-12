# Integration notes — tests module (tests/*)

No signature changes are proposed: every test codes strictly against the
scaffold headers as published. The entries below are BEHAVIORAL contracts the
tests depend on. All six tests now PASS against the landed sim + geom
implementations (full suite green, 2026-06-11), so most entries have been
verified in-tree; they remain recorded so later refactors do not silently
break the assumptions.

## src/sim/lbm_core.cuh : CellFlag::Inlet semantics (behavioral)
Contract:
    (no signature change) The fused step kernel treats ANY cell flagged
    CellFlag::Inlet as an equilibrium Dirichlet cell at u = (StepParams::uInlet,
    0, 0), regardless of where in the grid it sits — not only on the x = 0 plane.
Reason: m1_cavity realizes the moving lid of the Re-1000 lid-driven cavity by
flagging the y = ny-1 row as Inlet (a lid sliding in +x IS an equilibrium
Dirichlet at (u_lat, 0, 0)). The landed lbm_core.cu implements exactly this
(position-independent flag dispatch) and m1 passes within 1.5% of Ghia.
Status: VERIFIED IN TREE — keep the semantics position-independent.

## src/sim/lbm_solver.h : custom flag fields + snapshot-injection path (behavioral)
Contract:
    (no signature change) LBMSolver accepts arbitrary flag fields (cavity:
    Solid on three walls, Inlet lid, NO Outlet cells; or a plain open 64^3
    box) without assuming the wind-tunnel layout, and the documented snapshot
    path — cudaMemcpy into activeDeviceF() followed by
    notifySnapshotRestored(true) — is a supported way to inject an arbitrary
    externally-built f state. init(..., stream = nullptr) (default stream)
    is supported.
Reason: all three GPU milestone tests inject host-built equilibrium initial
conditions (Taylor-Green field, quiescent cavity, seeded cylinder inflow)
through exactly this path.
Status: VERIFIED IN TREE.

## src/sim/lbm_core.cuh : launchForceReduction mid-run usage (behavioral)
Contract:
    (no signature change) launchForceReduction is callable by external host
    code between stepN() calls, on a DeviceLatticeView assembled from
    LBMSolver::activeDeviceF()/deviceFlags()/dims(), with a caller-owned
    3-float device accumulator; the wrapper zeroes the accumulator before
    accumulating (as its Doxygen states).
Reason: m2_cylinder samples the raw (un-EMA'd) lift every 16 steps to build
the shedding spectrum — the solver's forces() readout is EMA-smoothed over
~a flow-through and would destroy the oscillation. Measured St = 0.202 at
Re ~150 (10% blockage), inside the 0.17-0.21 acceptance band.
Status: VERIFIED IN TREE.

## src/sim/lbm_solver.h : activeDeviceF()/fBufferBytes() layout (informational)
The landed solver stores f with two z ghost planes per direction
(per-direction stride nx*ny*(nz+2), real cell 0 one plane in) and
fBufferBytes() returns the PADDED size — the lbm_solver.h Doxygen still says
"kQ * cellCount floats" and should be updated to match. The tests do not
hard-code either convention: tests/lbm_test_util.h fLayout() derives the
layout from fBufferBytes() at runtime and handles both (uploads fill ghost
planes by periodic z-wrap). If a third layout ever appears (e.g. FP16
population storage changing sizeof per entry), fLayout() must learn it.
Status: INFORMATIONAL — doc fix in lbm_solver.h belongs to the sim agent.

## src/geom/airfoil.h : loadAirfoilDat skips '#' comment lines (behavioral)
Contract:
    (no signature change) The .dat loader ignores lines whose first
    non-whitespace character is '#'.
Reason: the in-tree reference file airfoils/ls413.dat (plan §15 calls it a
"good loader test"; test_airfoil parses it) ships with an 8-line '#'-prefixed
provenance header before the Lednicer name line. The landed loader handles it.
Status: VERIFIED IN TREE.

## src/geom/voxelizer.h : conventions test_voxelizer pins (informational)
- The diamond-parity test classifies cells only when BOTH center conventions
  (p = (i,j) and p = (i+0.5,j+0.5)) agree with ~0.9 cells of clearance from
  the polygon edge, so either cell-center convention passes. Cells near the
  outline are deliberately unasserted.
- closeTrailingEdgeGaps: the test asserts the EXACT return count (a single
  +-y pass over the deliberate gap pattern closes exactly 2 cells on an
  8x8x2 grid) and that single-sided solids do not convert.
- buildBoundaryFlags corner priority (inlet/outlet columns win over slip
  rows) is asserted as the scaffold established it.
Status: INFORMATIONAL.

## Stale comments in files the tests agent does not own (cleanup request)
- Root CMakeLists.txt (tests section header) and BUILDING.md still describe
  the six tests as intentionally-failing placeholders. The placeholders are
  gone: tests/not_implemented.cpp is deleted, all six tests are real, and the
  full suite passes. cuda_smoke is unchanged but now carries the "gpu" CTest
  label alongside m0/m1/m2 (it needs a physical card) — `ctest -LE gpu`
  skips every device-touching test on CI-less machines; per-test TIMEOUTs
  are set (M0 180 s, M1/M2 300 s; observed: 1.7 s / 2.2 s / 11.7 s on the
  dev 5090).
Status: PROPOSED (doc-only edits for the integration agent).
