// Host solver orchestration: device buffer ownership (padded ghost-z layout),
// stepping with the startup viscosity ramp, TDR-safe adaptive pacing, force
// EMA gating, NaN watchdog plumbing, warm-restart flag editing, and the
// suction-surface delta99 / separation-onset extraction for the VG overlay.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include "lbm_solver.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace foilcfd {

namespace {

/// Watchdog cadence in steps (plan 4.5: "every 200 steps").
constexpr int kWatchdogPeriodSteps = 200;

/// Warm restarts (VG edits on a restored field) reopen the force gate after
/// this many additional flow-throughs — the field is already developed, only
/// the vane neighborhood must re-adjust (plan section 8 / milestone M5).
constexpr float kWarmGateFlowThroughs = 0.5f;

/// Settling transient after a compact (equilibrium re-init) restore, in
/// steps (plan section 8: "costs a short transient (~1000 steps)").
constexpr long long kCompactSettleSteps = 1000;

/// Upper bound on steps per stepN batch, independent of the time budget —
/// a hard backstop so a bogus timing sample can never queue minutes of work.
constexpr int kMaxStepsPerBatch = 4096;

/// Smoothing factor for the per-step wall-time estimate (EMA over batches).
constexpr double kTimingEmaAlpha = 0.25;

/// Spanwise speck perturbation amplitude relative to u_lat (house finding:
/// quasi-2D LBM never reaches stall break without breaking the artificial
/// spanwise coherence of a uniform init).
constexpr float kSpeckAmplitudeFrac = 1e-3f;

/// Fixed seed for the speck perturbation — reruns must be reproducible.
constexpr unsigned int kSpeckSeed = 0x5eedu;

} // namespace

// ===========================================================================
// Impl: every device handle, all async-readback bookkeeping, and the host
// caches for the delta99/separation extraction.
// ===========================================================================

struct LBMSolver::Impl {
    GridDims       dims{};
    LatticeScaling scaling{};
    cudaStream_t   stream = nullptr;

    // Derived sizes, computed once at init (q*ncellsPad overflows 32-bit at
    // the default grid — everything stays 64-bit).
    long long ncells    = 0; ///< Real cells (nx*ny*nz).
    long long nxny      = 0; ///< One z-plane.
    long long ncellsPad = 0; ///< Real cells + 2 ghost z-planes.

    // ---- device memory ----------------------------------------------------
    FPop*         f[2]   = {nullptr, nullptr}; ///< Ping-pong padded f buffers.
    std::uint8_t* flags  = nullptr;            ///< Padded flag field.
    float*        rho    = nullptr;            ///< Unpadded macroscopic fields.
    float*        u      = nullptr;
    float*        v      = nullptr;
    float*        w      = nullptr;
    float*        force  = nullptr;            ///< 3-float momentum-exchange sum.
    int*          nanDev = nullptr;            ///< Watchdog trip flag.
    std::uint8_t* editMask = nullptr;          ///< Unpadded VG-edit mask (reused).

    // ---- pinned host readback + events (all readbacks are poll-don't-stall:
    // the frame loop must never block on a D2H copy) ------------------------
    float* hForce = nullptr; ///< Pinned 3-float force sample.
    int*   hNan   = nullptr; ///< Pinned watchdog flag.
    cudaEvent_t evT0 = nullptr, evT1 = nullptr; ///< Batch timing pair.
    cudaEvent_t evForce = nullptr;              ///< Force-readback fence.
    cudaEvent_t evNan   = nullptr;              ///< Watchdog-readback fence.

    // ---- sim state ---------------------------------------------------------
    int       src   = 0;     ///< Which f buffer holds the current state.
    long long steps = 0;     ///< Steps since last cold start.
    bool      initialized = false;

    // Startup ramp (plan 4.3): active only after cold starts. Snapshot
    // restores and warm flag edits disable it — their fields are developed.
    bool      rampActive    = true;
    long long rampStartStep = 0;

    // Force EMA + convergence gate (plan 4.4 / 13).
    long long gateOpenAtSteps     = 0;     ///< forces().valid once steps pass this.
    float     emaWindowFlowThroughs = 1.0f;///< StandardPreset default.
    float     emaFx = 0.0f, emaFy = 0.0f;  ///< Smoothed lattice forces.
    bool      emaSeeded = false;
    bool      forcePending = false;        ///< A reduction readback is in flight.
    long long stepsAtForceLaunch   = 0;
    long long stepsAtLastForceFold = 0;

    // Adaptive pacing (plan 4.5/9.3).
    bool   timingPending = false;
    int    timingBatchN  = 0;
    double emaStepMs     = 0.0;
    double lastStepMs    = 0.0;
    int    lastChosenN   = 0;

    // NaN watchdog.
    long long stepsSinceWatchdog = 0;
    bool      nanPending = false;
    bool      nanLatched = false;
    float     tauAtTrip  = 0.0f;

    // Host flag copy (snapshot hashing, surface extraction).
    std::vector<std::uint8_t> hostFlags;

    // Clean-foil (VG-free) reference flags for the suction-surface extraction.
    // The live hostFlags include VG voxels, and a vane crossing the mid-span
    // plane would raise topSolidY to the vane crest — delta99 would then be
    // measured from the vane and the vane's own recirculation would read as a
    // false separation onset, corrupting the Lin-2002 guidance exactly in
    // VG-on configurations. When empty, hostFlags is used (no VGs possible).
    std::vector<std::uint8_t> surfaceRefFlags;

    // ---- extraction caches (refreshed lazily from const readout methods,
    // hence mutable) ---------------------------------------------------------
    mutable std::vector<float> midU, midV; ///< Mid-span planes of u, v.
    mutable long long midStamp = -1;       ///< steps value the planes match.
    std::vector<int> topSolidY;            ///< Per-x top solid row at mid-span.
    int  xLE = 0, xTE = 0;                 ///< Solid extent at mid-span.
    bool surfValid = false;

    // STL slip-z mode (plan 7.4): when the z faces carry SlipFront/SlipBack
    // walls, those two planes are marker cells — the body and all force-
    // generating links live in only nz-2 planes, so the force-coefficient
    // reference area must use the FLUID span, not the raw nz (a raw-nz
    // normalization reads ~2/nz low on absolute Cl/Cd in that mode).
    bool zSlipWalls = false;

    // ------------------------------------------------------------------

    /// @brief View over one f buffer (padded pointers, real dims).
    DeviceLatticeView view(int which) const {
        return DeviceLatticeView{f[which], flags, dims};
    }

    /// @brief Steps in one flow-through at the current scaling.
    float flowThroughSteps() const { return scaling.flowThroughSteps(dims.nx); }

    /// @brief tau for the step about to run, honoring the ramp state.
    float effectiveTau() const {
        return rampActive ? rampedTau(scaling, steps - rampStartStep, dims.nx)
                          : scaling.tau;
    }

    /// @brief Poll an event, absorbing ONLY the expected cudaErrorNotReady
    /// from the per-thread last-error slot (it is flow control, not a
    /// failure). Genuine errors are deliberately left in place so they keep
    /// surfacing through stepN's return path — an unconditional
    /// cudaGetLastError() here used to wipe real launch failures every frame.
    static bool eventDone(cudaEvent_t ev) {
        const cudaError_t st = cudaEventQuery(ev);
        if (st == cudaErrorNotReady) (void)cudaGetLastError();
        return st == cudaSuccess;
    }

    /// @brief Drain any completed async readbacks (timing, force, watchdog)
    /// WITHOUT stalling: events that are still in flight are left alone.
    void pollAsync() {
        // Batch timing -> per-step estimate for the pacer + MLUPS readout.
        if (timingPending && eventDone(evT1)) {
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, evT0, evT1) == cudaSuccess
                && timingBatchN > 0) {
                lastStepMs = static_cast<double>(ms) / timingBatchN;
                emaStepMs = (emaStepMs > 0.0)
                    ? emaStepMs + kTimingEmaAlpha * (lastStepMs - emaStepMs)
                    : lastStepMs;
            }
            timingPending = false;
        }
        // Momentum-exchange sample -> EMA fold. The EMA weight covers the
        // steps elapsed since the previous folded sample so the effective
        // window is the configured flow-through count regardless of how many
        // steps each frame ran.
        if (forcePending && eventDone(evForce)) {
            const float windowSteps =
                std::max(1.0f, emaWindowFlowThroughs * flowThroughSteps());
            const float dSteps =
                static_cast<float>(stepsAtForceLaunch - stepsAtLastForceFold);
            const float alpha = 1.0f - std::exp(-dSteps / windowSteps);
            if (!emaSeeded) {
                emaFx = hForce[0]; emaFy = hForce[1];
                emaSeeded = true;
            } else {
                emaFx += alpha * (hForce[0] - emaFx);
                emaFy += alpha * (hForce[1] - emaFy);
            }
            stepsAtLastForceFold = stepsAtForceLaunch;
            forcePending = false;
        }
        // Watchdog verdict.
        if (nanPending && eventDone(evNan)) {
            if (*hNan != 0) {
                nanLatched = true;
                tauAtTrip = effectiveTau();
            }
            nanPending = false;
        }
    }

    /// @brief Upload an unpadded host flag field into the padded device
    /// buffer and refresh the ghost planes; refresh the host copy + the
    /// surface-extraction cache as well.
    cudaError_t uploadFlags(const std::vector<std::uint8_t>& newFlags) {
        // Interior region of the padded buffer starts one ghost plane in.
        if (auto err = cudaMemcpyAsync(flags + nxny, newFlags.data(),
                                       static_cast<std::size_t>(ncells),
                                       cudaMemcpyHostToDevice, stream);
            err != cudaSuccess)
            return err;
        if (auto err = launchRefreshGhostZFlags(flags, dims, stream);
            err != cudaSuccess)
            return err;
        hostFlags = newFlags;
        // Detect STL slip-z mode from the z=0 plane (the import flow stamps
        // every fluid cell of that plane SlipFront): determines the fluid
        // span used by the force-coefficient normalization.
        const auto slipF = static_cast<std::uint8_t>(CellFlag::SlipFront);
        zSlipWalls = false;
        for (long long i = 0; i < nxny; ++i) {
            if (hostFlags[static_cast<std::size_t>(i)] == slipF) {
                zSlipWalls = true;
                break;
            }
        }
        rebuildSurfaceCache();
        midStamp = -1; // velocity planes are stale relative to new geometry
        return cudaSuccess;
    }

    /// @brief Recompute the mid-span suction-surface description: per-column
    /// topmost solid row plus the solid x extent. Reads the CLEAN-FOIL
    /// reference flags when the app has provided them (so VG voxels in the
    /// live flags can never contaminate the delta99/separation readouts);
    /// falls back to the live flags otherwise. O(nx*ny) on one plane —
    /// instant, runs only on flag changes.
    void rebuildSurfaceCache() {
        topSolidY.assign(static_cast<std::size_t>(dims.nx), -1);
        xLE = dims.nx; xTE = -1;
        const int zmid = dims.nz / 2;
        const long long planeBase = nxny * zmid;
        const auto solid = static_cast<std::uint8_t>(CellFlag::Solid);
        // The reference is only trusted when it matches the live grid size —
        // a stale reference from a previous resolution must never index OOB.
        const std::vector<std::uint8_t>& src =
            (surfaceRefFlags.size() == hostFlags.size()) ? surfaceRefFlags
                                                         : hostFlags;
        for (int y = 0; y < dims.ny; ++y) {
            const long long rowBase = planeBase + static_cast<long long>(dims.nx) * y;
            for (int x = 0; x < dims.nx; ++x) {
                if (src[static_cast<std::size_t>(rowBase + x)] == solid) {
                    topSolidY[static_cast<std::size_t>(x)] = y; // y ascends: last write wins = topmost
                    xLE = std::min(xLE, x);
                    xTE = std::max(xTE, x);
                }
            }
        }
        surfValid = (xTE > xLE);
    }

    /// @brief Ensure midU/midV hold the mid-span planes of the CURRENT
    /// macroscopic field. One 2 * nx * ny float D2H copy + a stream sync —
    /// a couple of MB, intended for UI-rate callers (a few Hz).
    bool ensureMidplane() const {
        if (!initialized) return false;
        if (midStamp == steps && !midU.empty()) return true;
        midU.resize(static_cast<std::size_t>(nxny));
        midV.resize(static_cast<std::size_t>(nxny));
        const long long off = nxny * (dims.nz / 2);
        if (cudaMemcpyAsync(midU.data(), u + off, nxny * sizeof(float),
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess)
            return false;
        if (cudaMemcpyAsync(midV.data(), v + off, nxny * sizeof(float),
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess)
            return false;
        if (cudaStreamSynchronize(stream) != cudaSuccess) return false;
        midStamp = steps;
        return true;
    }

    /// @brief Release everything; safe to call repeatedly / when empty.
    void freeAll() {
        for (FPop*& p : f) { cudaFree(p); p = nullptr; }
        cudaFree(flags); flags = nullptr;
        cudaFree(rho); rho = nullptr;
        cudaFree(u); u = nullptr;
        cudaFree(v); v = nullptr;
        cudaFree(w); w = nullptr;
        cudaFree(force); force = nullptr;
        cudaFree(nanDev); nanDev = nullptr;
        cudaFree(editMask); editMask = nullptr;
        cudaFreeHost(hForce); hForce = nullptr;
        cudaFreeHost(hNan); hNan = nullptr;
        for (cudaEvent_t* ev : {&evT0, &evT1, &evForce, &evNan}) {
            if (*ev) { cudaEventDestroy(*ev); *ev = nullptr; }
        }
        hostFlags.clear();
        surfaceRefFlags.clear();
        midU.clear(); midV.clear(); midStamp = -1;
        topSolidY.clear(); surfValid = false;
        initialized = false;
    }
};

// ===========================================================================
// Lifecycle
// ===========================================================================

LBMSolver::LBMSolver() : impl_(std::make_unique<Impl>()) {}
LBMSolver::~LBMSolver() { shutdown(); }

bool LBMSolver::init(const GridDims& dims, const LatticeScaling& scaling,
                     const std::vector<std::uint8_t>& flags, cudaStream_t stream,
                     std::string* error) {
    shutdown();
    Impl& s = *impl_;
    s.dims = dims;
    s.scaling = scaling;
    s.stream = stream;
    s.ncells = dims.cellCount();
    s.nxny = static_cast<long long>(dims.nx) * dims.ny;
    s.ncellsPad = dims.paddedCellCount();

    if (s.ncells <= 0 || flags.size() != static_cast<std::size_t>(s.ncells)) {
        if (error) *error = "flag field size does not match grid dimensions";
        return false;
    }

    auto fail = [&](const char* what, cudaError_t err) {
        if (error) *error = std::string(what) + ": " + cudaGetErrorString(err);
        impl_->freeAll();
        return false;
    };

    // Allocation order: largest first so OOM fails fast before fragmentation.
    // Per-buffer size includes the two spanwise ghost planes (plan 11).
    const std::size_t fBytes =
        static_cast<std::size_t>(kQ) * s.ncellsPad * sizeof(FPop);
    for (FPop*& p : s.f) {
        if (auto err = cudaMalloc(&p, fBytes); err != cudaSuccess)
            return fail("f buffer allocation failed", err);
    }
    if (auto err = cudaMalloc(&s.flags, static_cast<std::size_t>(s.ncellsPad));
        err != cudaSuccess)
        return fail("flag allocation failed", err);
    for (float** p : {&s.rho, &s.u, &s.v, &s.w}) {
        if (auto err = cudaMalloc(p, s.ncells * sizeof(float)); err != cudaSuccess)
            return fail("macroscopic allocation failed", err);
    }
    if (auto err = cudaMalloc(&s.editMask, static_cast<std::size_t>(s.ncells));
        err != cudaSuccess)
        return fail("edit mask allocation failed", err);
    if (auto err = cudaMalloc(&s.force, 3 * sizeof(float)); err != cudaSuccess)
        return fail("force accumulator allocation failed", err);
    if (auto err = cudaMalloc(&s.nanDev, sizeof(int)); err != cudaSuccess)
        return fail("watchdog flag allocation failed", err);

    // Pinned host readback slots — async D2H copies into pageable memory
    // silently serialize, which would defeat the poll-don't-stall design.
    if (auto err = cudaMallocHost(&s.hForce, 3 * sizeof(float)); err != cudaSuccess)
        return fail("pinned force readback allocation failed", err);
    if (auto err = cudaMallocHost(&s.hNan, sizeof(int)); err != cudaSuccess)
        return fail("pinned watchdog readback allocation failed", err);

    // Timing events keep timestamps; the readback fences don't need them.
    for (auto [ev, flags_] : {std::pair{&s.evT0, 0u}, {&s.evT1, 0u},
                              {&s.evForce, (unsigned)cudaEventDisableTiming},
                              {&s.evNan, (unsigned)cudaEventDisableTiming}}) {
        if (auto err = cudaEventCreateWithFlags(ev, flags_); err != cudaSuccess)
            return fail("event creation failed", err);
    }

    // Macroscopic fields start at quiescent defaults so render/extraction
    // reads are defined before the first stepped frame writes real values.
    // Density primes to the LBM rest value 1.0, NOT 0: the pressure slice
    // shows cs^2*(rho-1) (so rho=0 would render a saturated -1/3 field), and
    // a compact snapshot captured before the first step must restore to a
    // valid quiescent state — equilibrium(rho=0) is all-zero populations and
    // the next collide would divide by zero, NaN-flooding the whole field.
    launchFillFloat(s.rho, s.ncells, 1.0f, stream);
    cudaMemsetAsync(s.u, 0, s.ncells * sizeof(float), stream);
    cudaMemsetAsync(s.v, 0, s.ncells * sizeof(float), stream);
    cudaMemsetAsync(s.w, 0, s.ncells * sizeof(float), stream);

    if (auto err = s.uploadFlags(flags); err != cudaSuccess)
        return fail("flag upload failed", err);

    s.initialized = true;
    reset();

    // Surface any launch failure from the reset sequence now, while the
    // caller still has its error channel open.
    if (auto err = cudaGetLastError(); err != cudaSuccess)
        return fail("initial field setup failed", err);
    return true;
}

void LBMSolver::shutdown() {
    if (impl_) impl_->freeAll();
}

void LBMSolver::reset() {
    Impl& s = *impl_;
    if (!s.initialized) return;

    // Cold-start bookkeeping: step counter, ramp, force gate, watchdog.
    s.steps = 0;
    s.src = 0;
    s.rampActive = true;
    s.rampStartStep = 0;
    s.gateOpenAtSteps =
        static_cast<long long>(kForceGateFlowThroughs * s.flowThroughSteps());
    s.emaFx = s.emaFy = 0.0f;
    s.emaSeeded = false;
    s.forcePending = false;
    s.stepsAtForceLaunch = s.stepsAtLastForceFold = 0;
    s.nanLatched = false;
    s.nanPending = false;
    s.stepsSinceWatchdog = 0;
    s.midStamp = -1;
    cudaMemsetAsync(s.nanDev, 0, sizeof(int), s.stream);

    // Both ping-pong buffers get the equilibrium inflow + the spanwise speck
    // perturbation (fresh starts must break quasi-2D coherence). The init
    // kernels run over the padded domain with z-periodic formulas, so the
    // ghost planes are consistent by construction — no refresh needed here.
    for (int i = 0; i < 2; ++i) {
        launchInitEquilibrium(s.view(i), s.scaling.u_lat, s.stream);
        launchSpanwisePerturbation(s.view(i),
                                   kSpeckAmplitudeFrac * s.scaling.u_lat,
                                   kSpeckSeed, s.stream);
    }
}

void LBMSolver::setFlags(const std::vector<std::uint8_t>& flags) {
    Impl& s = *impl_;
    if (!s.initialized) return;
    if (flags.size() != static_cast<std::size_t>(s.ncells)) return;
    if (s.uploadFlags(flags) != cudaSuccess) return;
    reset(); // geometry replaced wholesale -> cold restart (plan 9.2)
}

void LBMSolver::setSurfaceReference(const std::vector<std::uint8_t>& cleanFlags) {
    Impl& s = *impl_;
    if (!s.initialized) return;
    if (cleanFlags.size() != static_cast<std::size_t>(s.ncells)) return;
    s.surfaceRefFlags = cleanFlags;
    // The live flags may already include VG voxels (init/setFlags receive the
    // merged field) — re-derive the surface description from the clean foil.
    s.rebuildSurfaceCache();
}

void LBMSolver::applyEditedFlags(const std::vector<std::uint8_t>& flags) {
    Impl& s = *impl_;
    if (!s.initialized) return;
    if (flags.size() != static_cast<std::size_t>(s.ncells)) return;

    // Diff old vs new flags into the edit mask BEFORE the host copy is
    // replaced: 1 = newly solid (zero out), 2 = newly fluid (equilibrium
    // fill from neighbors) — plan section 8 VG-edit flow.
    const auto solid = static_cast<std::uint8_t>(CellFlag::Solid);
    std::vector<std::uint8_t> mask(static_cast<std::size_t>(s.ncells), 0);
    long long edited = 0;
    for (long long i = 0; i < s.ncells; ++i) {
        const bool wasSolid = s.hostFlags[static_cast<std::size_t>(i)] == solid;
        const bool isSolid = flags[static_cast<std::size_t>(i)] == solid;
        if (wasSolid == isSolid) continue;
        mask[static_cast<std::size_t>(i)] = isSolid ? 1 : 2;
        ++edited;
    }

    if (s.uploadFlags(flags) != cudaSuccess) return;

    if (edited > 0) {
        cudaMemcpyAsync(s.editMask, mask.data(),
                        static_cast<std::size_t>(s.ncells),
                        cudaMemcpyHostToDevice, s.stream);
        launchApplyFlagEdits(s.view(s.src), s.f[1 - s.src], s.editMask, s.stream);
        // Edits near the span boundary touch cells whose ghost images are
        // now stale — refresh both buffers once (cheap, cold path).
        launchRefreshGhostZ(s.f[0], s.dims, s.stream);
        launchRefreshGhostZ(s.f[1], s.dims, s.stream);
    }

    // Warm-restart bookkeeping: the field is developed, so no viscosity ramp;
    // the force EMA restarts and the gate reopens after a short re-settle.
    s.rampActive = false;
    s.emaSeeded = false;
    s.forcePending = false;
    s.stepsAtLastForceFold = s.steps;
    s.gateOpenAtSteps =
        s.steps + static_cast<long long>(kWarmGateFlowThroughs * s.flowThroughSteps());
}

// ===========================================================================
// Stepping + pacing
// ===========================================================================

cudaError_t LBMSolver::stepN(int n) {
    Impl& s = *impl_;
    if (!s.initialized) return cudaErrorNotReady;
    if (n <= 0) return cudaSuccess;
    s.pollAsync();
    if (s.nanLatched) return cudaSuccess; // paused; UI shows the diagnosis

    // Bracket the batch with timing events when the previous measurement has
    // been consumed (re-recording a still-pending event would corrupt it).
    const bool timeThisBatch = !s.timingPending;
    if (timeThisBatch) cudaEventRecord(s.evT0, s.stream);

    for (int i = 0; i < n; ++i) {
        StepParams params;
        params.tau = s.effectiveTau();
        params.magicLambda = kTRTMagicLambda;
        params.smagorinskyCs = kSmagorinskyCs;
        params.uInlet = s.scaling.u_lat;
        // Macroscopic stores feed rendering once per frame: only the batch's
        // final step pays the extra 16 B/cell of write traffic (plan 11).
        params.writeMacro = (i == n - 1);

        const cudaError_t err = launchStreamCollide(
            s.view(s.src), s.view(1 - s.src), params,
            s.rho, s.u, s.v, s.w, s.stream);
        if (err != cudaSuccess) return err;

        // The freshly written buffer needs its spanwise ghost planes synced
        // before it is pulled from next step (periodic z, plan 4.2/11).
        if (auto err2 = launchRefreshGhostZ(s.f[1 - s.src], s.dims, s.stream);
            err2 != cudaSuccess)
            return err2;

        s.src = 1 - s.src; // ping-pong swap
        ++s.steps;

        // NaN watchdog every 200 steps (plan 4.5): launch the strided check
        // plus an async readback; verdicts are folded in by pollAsync().
        if (++s.stepsSinceWatchdog >= kWatchdogPeriodSteps && !s.nanPending) {
            s.stepsSinceWatchdog = 0;
            cudaMemsetAsync(s.nanDev, 0, sizeof(int), s.stream);
            // The step counter rotates the sampled residue class (977 is
            // coprime to the watchdog's prime stride), so repeated checks
            // sweep different cells instead of re-probing one fixed set.
            launchNaNWatchdog(s.view(s.src), s.nanDev, s.steps * 977, s.stream);
            cudaMemcpyAsync(s.hNan, s.nanDev, sizeof(int),
                            cudaMemcpyDeviceToHost, s.stream);
            cudaEventRecord(s.evNan, s.stream);
            s.nanPending = true;
        }
    }

    if (timeThisBatch) {
        cudaEventRecord(s.evT1, s.stream);
        s.timingPending = true;
        s.timingBatchN = n;
    }

    // One momentum-exchange sample per batch (per frame), folded into the
    // EMA asynchronously. The reduction reads the post-collision buffer the
    // batch just produced.
    if (!s.forcePending) {
        DeviceForceAccumulator acc{s.force};
        if (launchForceReduction(s.view(s.src), acc, s.stream) == cudaSuccess) {
            cudaMemcpyAsync(s.hForce, s.force, 3 * sizeof(float),
                            cudaMemcpyDeviceToHost, s.stream);
            cudaEventRecord(s.evForce, s.stream);
            s.forcePending = true;
            s.stepsAtForceLaunch = s.steps;
        }
    }
    return cudaSuccess;
}

int LBMSolver::adaptiveStepsForBudget(double budgetMs) {
    Impl& s = *impl_;
    if (!s.initialized) return 0;
    s.pollAsync();
    if (s.nanLatched) return 0; // paused by the watchdog

    int n = 1; // first frames probe conservatively (header contract)
    if (s.emaStepMs > 0.0) {
        n = static_cast<int>(budgetMs / s.emaStepMs);
        // Grow at most 2x per frame so a single optimistic timing sample
        // can't jump straight to a TDR-risky batch; shrinking is immediate.
        if (s.lastChosenN > 0) n = std::min(n, 2 * s.lastChosenN);
        n = std::clamp(n, 1, kMaxStepsPerBatch);
    }
    s.lastChosenN = n;
    return n;
}

// ===========================================================================
// Readouts
// ===========================================================================

ForceReadout LBMSolver::forces() const {
    const Impl& s = *impl_;
    ForceReadout r;
    if (!s.initialized) return r;
    r.flowThroughs = flowThroughsCompleted();
    // Coefficient normalization lives in units.h (single conversion source):
    // drag is the streamwise (x) force, lift the vertical (y) force — AoA is
    // baked into the geometry, so the inflow stays axis-aligned (plan 4.2).
    // Reference span: with periodic z the geometry truly spans all nz cells;
    // with STL slip-z walls the z=0/nz-1 planes are markers and the body
    // occupies only nz-2 fluid planes (plan 7.4).
    const int spanCells = s.zSlipWalls ? std::max(1, s.dims.nz - 2) : s.dims.nz;
    r.cd = s.scaling.forceToCoefficient(s.emaFx, spanCells);
    r.cl = s.scaling.forceToCoefficient(s.emaFy, spanCells);
    r.liftToDrag = (std::fabs(r.cd) > 1e-6f) ? r.cl / r.cd : 0.0f;
    r.valid = s.emaSeeded && (s.steps >= s.gateOpenAtSteps);
    return r;
}

std::vector<Delta99Sample> LBMSolver::extractSuctionDelta99(
    const std::vector<float>& stations_xc) const {
    const Impl& s = *impl_;
    std::vector<Delta99Sample> out(stations_xc.size());
    for (std::size_t i = 0; i < stations_xc.size(); ++i)
        out[i].x_c = stations_xc[i];
    if (!s.initialized || !s.surfValid || !s.ensureMidplane()) return out;

    const int nx = s.dims.nx, ny = s.dims.ny;
    const float chordSpan = static_cast<float>(s.xTE - s.xLE);

    for (auto& sample : out) {
        // Map x/c onto the lattice using the PROJECTED solid extent — this
        // tracks the AoA-rotated foil without needing the layout struct.
        const int ix = std::clamp(
            static_cast<int>(std::lround(s.xLE + sample.x_c * chordSpan)),
            0, nx - 1);
        const int ys = s.topSolidY[static_cast<std::size_t>(ix)];
        if (ys < 0 || ys >= ny - 3) continue; // no surface under this station

        // Probe window: from the first fluid cell up to half a chord above
        // the surface — generously past any boundary layer we can resolve.
        const int yTop = std::min(ny - 2, ys + s.scaling.chordCells / 2);

        // Pass 1: edge velocity = peak in-plane speed along the wall-normal
        // ray. Over a suction surface the profile rises monotonically through
        // the BL to the (locally accelerated) edge value, so the max is a
        // robust ue estimate even with the curved external flow above it.
        float ue = 0.0f;
        for (int y = ys + 1; y <= yTop; ++y) {
            const std::size_t idx = static_cast<std::size_t>(ix)
                                  + static_cast<std::size_t>(nx) * y;
            const float spd = std::hypot(s.midU[idx], s.midV[idx]);
            ue = std::max(ue, spd);
        }
        sample.ueEdge = ue;
        if (ue < 0.05f * s.scaling.u_lat) continue; // dead air: separated/stalled

        // Pass 2: first height reaching 99% of ue. The wall sits half a cell
        // below the first fluid center (half-way bounce-back convention).
        float d99Cells = -1.0f;
        for (int y = ys + 1; y <= yTop; ++y) {
            const std::size_t idx = static_cast<std::size_t>(ix)
                                  + static_cast<std::size_t>(nx) * y;
            if (std::hypot(s.midU[idx], s.midV[idx]) >= 0.99f * ue) {
                d99Cells = static_cast<float>(y - ys) - 0.5f;
                break;
            }
        }
        if (d99Cells <= 0.0f) continue;

        // Vertical distance -> wall-normal distance: scale by the cosine of
        // the local surface slope (central difference over +/-2 columns of
        // the stair-step crest). Matters near the leading edge.
        float cosTheta = 1.0f;
        if (ix >= 2 && ix + 2 < nx) {
            const int yl = s.topSolidY[static_cast<std::size_t>(ix - 2)];
            const int yr = s.topSolidY[static_cast<std::size_t>(ix + 2)];
            if (yl >= 0 && yr >= 0) {
                const float slope = static_cast<float>(yr - yl) / 4.0f;
                cosTheta = 1.0f / std::sqrt(1.0f + slope * slope);
            }
        }
        sample.delta99_c =
            d99Cells * cosTheta / static_cast<float>(s.scaling.chordCells);

        // Separated stations report invalid (the Lin guidance needs the
        // ATTACHED boundary layer upstream of separation): flag reverse
        // near-wall streamwise flow.
        const std::size_t nearWall = static_cast<std::size_t>(ix)
                                   + static_cast<std::size_t>(nx) * (ys + 1);
        sample.valid = (s.midU[nearWall] > -0.02f * s.scaling.u_lat);
    }
    return out;
}

float LBMSolver::separationOnsetXc() const {
    const Impl& s = *impl_;
    if (!s.initialized || !s.surfValid || !s.ensureMidplane()) return -1.0f;

    const int nx = s.dims.nx;
    const float chordSpan = static_cast<float>(s.xTE - s.xLE);
    // Reverse-flow threshold: a couple percent of u_lat rejects numerical
    // jitter; demand three consecutive reversed columns so a single noisy
    // column can't fake a separation point.
    const float uRev = -0.02f * s.scaling.u_lat;
    int consecutive = 0;

    // Skip the first few percent of chord: the stagnation region under the
    // (rotated) leading edge legitimately carries tiny negative u.
    const int ixStart = s.xLE + std::max(2, static_cast<int>(0.03f * chordSpan));
    for (int ix = ixStart; ix <= s.xTE && ix < nx; ++ix) {
        const int ys = s.topSolidY[static_cast<std::size_t>(ix)];
        if (ys < 0 || ys + 2 >= s.dims.ny) { consecutive = 0; continue; }
        const std::size_t idx = static_cast<std::size_t>(ix)
                              + static_cast<std::size_t>(nx) * (ys + 2);
        if (s.midU[idx] < uRev) {
            if (++consecutive >= 3) {
                const float ixOnset = static_cast<float>(ix - consecutive + 1);
                return (ixOnset - static_cast<float>(s.xLE)) / chordSpan;
            }
        } else {
            consecutive = 0;
        }
    }
    return -1.0f; // attached all the way to the trailing edge
}

bool LBMSolver::nanDetected() const { return impl_->nanLatched; }

std::string LBMSolver::nanDiagnosis() const {
    const Impl& s = *impl_;
    if (!s.nanLatched) return {};
    char buf[512];
    // The two usual suspects (plan 4.5): lattice Mach too high, or the run
    // is pinned at the tau stability clamp (target Re unreachable).
    std::snprintf(buf, sizeof buf,
        "NaN detected at step %lld. u_lat = %.4f (cap %.2f)%s; tau at trip = "
        "%.5f (clamp %.4f)%s. Likely cause: %s. Try a lower airspeed, a finer "
        "chord resolution, or the Fast preset.",
        s.steps, s.scaling.u_lat, kMaxULat,
        (s.scaling.u_lat > 0.1f ? " — HIGH" : ""),
        s.tauAtTrip, kMinTau,
        (s.scaling.tauClamped ? " — CLAMPED" : ""),
        s.scaling.u_lat > 0.1f
            ? "lattice velocity too close to the stability cap"
            : (s.scaling.tauClamped
                   ? "viscosity pinned at the stability clamp"
                   : "a transient too sharp for the current resolution"));
    return buf;
}

SolverPerfStats LBMSolver::perfStats() const {
    const Impl& s = *impl_;
    SolverPerfStats p;
    p.lastStepsPerFrame = s.lastChosenN;
    p.lastStepMs = s.lastStepMs;
    if (s.lastStepMs > 0.0)
        p.mlups = static_cast<double>(s.ncells) / (s.lastStepMs * 1000.0);
    return p;
}

// ===========================================================================
// State access for rendering and snapshots
// ===========================================================================

DeviceVelocityField LBMSolver::velocityField() const {
    return DeviceVelocityField{impl_->u, impl_->v, impl_->w, impl_->dims};
}

const float* LBMSolver::deviceRho() const { return impl_->rho; }

const std::uint8_t* LBMSolver::deviceFlags() const {
    // External consumers (particle kernels) index flags with the UNPADDED
    // convention x + nx*(y + ny*z). The real-domain region of the padded
    // buffer is contiguous and starts one ghost plane in, so offsetting the
    // base pointer makes their indexing exact — no second flag copy.
    return impl_->flags ? impl_->flags + impl_->nxny : nullptr;
}

DeviceLatticeView LBMSolver::latticeView() const {
    // Padded bases, real dims — exactly what the launch wrappers index with.
    return impl_->view(impl_->src);
}

float* LBMSolver::activeDeviceF() { return impl_->f[impl_->src]; }

std::size_t LBMSolver::fBufferBytes() const {
    // Padded size: ghost planes ARE part of the snapshot payload (they are
    // consistent copies of the edge planes; restoring them keeps the very
    // first post-restore pull valid without a refresh pass).
    return static_cast<std::size_t>(kQ) * impl_->ncellsPad * sizeof(FPop);
}

void LBMSolver::notifySnapshotRestored(bool fullState) {
    Impl& s = *impl_;
    if (!s.initialized) return;
    // A restored field is developed: never re-run the startup ramp on it.
    s.rampActive = false;
    s.midStamp = -1;
    s.emaSeeded = false;
    s.forcePending = false;
    s.stepsAtLastForceFold = s.steps;
    s.nanLatched = false;
    s.nanPending = false;
    if (fullState) {
        // Exact restore: only a short gate, mostly to let the EMA refill.
        s.gateOpenAtSteps = s.steps
            + static_cast<long long>(0.25f * s.flowThroughSteps());
    } else {
        // Equilibrium re-init: schedule the documented settling transient
        // (plan section 8) plus a half flow-through for the EMA.
        s.gateOpenAtSteps = s.steps + kCompactSettleSteps
            + static_cast<long long>(kWarmGateFlowThroughs * s.flowThroughSteps());
    }
}

bool LBMSolver::restoreFromMacroscopic(const float* rho, const float* u,
                                       const float* v, const float* w,
                                       std::string* error) {
    Impl& s = *impl_;
    if (!s.initialized) {
        if (error) *error = "solver not initialized";
        return false;
    }
    auto fail = [&](const char* what, cudaError_t err) {
        if (error) *error = std::string(what) + ": " + cudaGetErrorString(err);
        return false;
    };
    // Upload into the solver's own macroscopic arrays — rendering then shows
    // the restored field immediately, before the first new step.
    const std::size_t bytes = static_cast<std::size_t>(s.ncells) * sizeof(float);
    struct { const float* src; float* dst; } uploads[] = {
        {rho, s.rho}, {u, s.u}, {v, s.v}, {w, s.w}};
    for (const auto& up : uploads) {
        if (auto err = cudaMemcpyAsync(up.dst, up.src, bytes,
                                       cudaMemcpyHostToDevice, s.stream);
            err != cudaSuccess)
            return fail("macroscopic upload failed", err);
    }
    // Equilibrium re-init of BOTH ping-pong buffers (the kernel covers the
    // ghost planes itself via z-wrapped sampling).
    for (int i = 0; i < 2; ++i) {
        if (auto err = launchInitFromMacroscopic(s.view(i), s.rho, s.u, s.v,
                                                 s.w, s.stream);
            err != cudaSuccess)
            return fail("equilibrium re-init failed", err);
    }
    notifySnapshotRestored(false);
    return true;
}

const std::vector<std::uint8_t>& LBMSolver::hostFlags() const {
    return impl_->hostFlags;
}

void LBMSolver::setForceEmaWindow(float flowThroughs) {
    impl_->emaWindowFlowThroughs = std::max(0.05f, flowThroughs);
}

// ===========================================================================
// Misc accessors
// ===========================================================================

GridDims LBMSolver::dims() const { return impl_->dims; }

const LatticeScaling& LBMSolver::scaling() const { return impl_->scaling; }

long long LBMSolver::stepCount() const { return impl_->steps; }

float LBMSolver::flowThroughsCompleted() const {
    const Impl& s = *impl_;
    if (!s.initialized || s.dims.nx <= 0) return 0.0f;
    return static_cast<float>(s.steps) / s.flowThroughSteps();
}

float LBMSolver::currentTau() const {
    const Impl& s = *impl_;
    if (!s.initialized) return 0.0f;
    return s.effectiveTau();
}

} // namespace foilcfd
