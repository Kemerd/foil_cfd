# Integration notes — sim module (lbm_core, lbm_solver, units, snapshot)

Status: IMPLEMENTED (not proposals — these are live in the headers I own).
Listed here so the integration agent and downstream modules can reconcile.

## Public additions to headers I own

### src/sim/lbm_solver.h
- `bool LBMSolver::restoreFromMacroscopic(const float* rho, const float* u,
  const float* v, const float* w, std::string* error)` — compact-snapshot
  restore entry point: uploads host macroscopic arrays (cellCount() floats
  each, unpadded x + nx*(y + ny*z) layout), sets both f buffers to
  equilibrium(rho,u), and performs the notifySnapshotRestored(false)
  bookkeeping internally. Called by `CompactSnapshot::restore`.
- `const std::vector<std::uint8_t>& LBMSolver::hostFlags() const` — host copy
  of the current (unpadded) flag field; snapshot capture hashes it.
- `void LBMSolver::setForceEmaWindow(float flowThroughs)` — force EMA window
  in flow-through times. Default 1.0 (StandardPreset). **UI agent:** call
  with `HighFidelityPreset::forceEmaFlowThroughs` (8.0) when the user toggles
  High Fidelity mode, and back to 1.0 when toggling off.
  [WIRED by the integration pass 2026-06-11: main.cpp sets it right after
  every solver init (initSimulation + the airspeed shutdown/init workaround),
  selecting between HighFidelityPreset and StandardPreset values — a fresh
  solver resets the window, so the call lives next to init(), not the toggle.]

### src/sim/snapshot.h
- `void DiskSnapshotCache::storeAsync(std::shared_ptr<const CompactSnapshot>)`
  — enqueues a write on the cache's internal worker thread (plan section 8
  "saved on a worker thread"); failures log to stderr. Synchronous `store()`
  remains and is internally serialized.

### src/sim/lbm_core.cuh
- `kMirY[19]` — y-mirrored direction table for the free-slip boundary.
- `using FPop = float;` + `load_f()/store_f()` device accessors (FP16 storage
  later = typedef swap + accessor edits only; plan section 11, Lehmann
  PRE 106 015308). `DeviceLatticeView::f` is now `FPop*` (today identical to
  `float*`).
- `GridDims::paddedCellCount()` and new launch wrappers:
  `launchRefreshGhostZ`, `launchRefreshGhostZFlags`,
  `launchInitFromMacroscopic`, `launchApplyFlagEdits`.

## Memory-layout contract (IMPORTANT for anyone touching raw solver buffers)

The f and flag device buffers carry one ghost nx*ny plane at EACH z end
(periodic span via padded ghosts, plan section 11 — no modulo in the hot
loop). Padded index(x,y,z) = x + nx*(y + ny*(z+1)).

What this means for OTHER modules — almost nothing, by design:
- `LBMSolver::deviceFlags()` returns a pointer already offset past the low
  ghost plane, so the documented unpadded indexing x + nx*(y + ny*z) works
  unchanged for particle/render kernels (z in [0, nz)).
- `DeviceVelocityField` / `deviceRho()` arrays are genuinely unpadded
  (cellCount() floats) — unchanged contract.
- `activeDeviceF()` / `fBufferBytes()` refer to the PADDED buffer
  (kQ * paddedCellCount() * sizeof(FPop)). Only snapshot code (also mine)
  consumes these; treat the blob as opaque bytes.

## Behavioral notes
- Cold start injects a deterministic spanwise "speck" perturbation
  (amplitude 1e-3 * u_lat, fixed seed) — required to break quasi-2D spanwise
  coherence so stall can develop 3D structure. Identical reruns reproduce
  identical fields.
- Force readout gates: cold start 2.0 flow-throughs; VG warm edit +0.5;
  compact restore +1000 steps + 0.5 flow-through; VRAM restore +0.25.
- `adaptiveStepsForBudget()` returns 0 while the NaN watchdog is latched;
  `nanDiagnosis()` has the user-facing explanation string.
- `extractSuctionDelta99()` / `separationOnsetXc()` sample the mid-span plane
  (one ~2 MB D2H + stream sync per sim step, cached) — call at UI rate
  (a few Hz), not per frame, per the header docs.
- Compact snapshot disk format: magic "FCS1", header (key, flags hash,
  scaling), raw FP32 payload; a compression tag field is reserved (currently
  0 = raw) so a compressed writer can be added without breaking readers.
