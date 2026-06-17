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
#include <cstring>

#include "lbm_refine.cuh"
#include "lbm_wallmodel.cuh"
#include "wall_model.h"

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
    float*        mass   = nullptr;            ///< 2-float mass diagnostic {sum rho, count}.
    int*          nanDev = nullptr;            ///< Watchdog trip flag.
    std::uint8_t* editMask = nullptr;          ///< Unpadded VG-edit mask (reused).

    // ---- pinned host readback + events (all readbacks are poll-don't-stall:
    // the frame loop must never block on a D2H copy) ------------------------
    float* hForce = nullptr; ///< Pinned 3-float force sample.
    float* hMass  = nullptr; ///< Pinned 2-float mass sample {sum rho, count}.
    int*   hNan   = nullptr; ///< Pinned watchdog flag.
    cudaEvent_t evT0 = nullptr, evT1 = nullptr; ///< Batch timing pair.
    cudaEvent_t evForce = nullptr;              ///< Force-readback fence.
    cudaEvent_t evMass  = nullptr;              ///< Mass-readback fence.
    cudaEvent_t evNan   = nullptr;              ///< Watchdog-readback fence.

    // ---- sim state ---------------------------------------------------------
    int       src   = 0;     ///< Which f buffer holds the current state.
    long long steps = 0;     ///< Steps since last cold start.
    bool      initialized = false;

    // Startup ramp (plan 4.3): active only after cold starts. Snapshot
    // restores and warm flag edits disable it — their fields are developed.
    bool      rampActive    = true;
    long long rampStartStep = 0;
    // Master switch for the startup ramps + rest-init (setStartupRampEnabled).
    // Off restores the legacy instant start for steady-state tests.
    bool      rampEnabled   = true;

    // Force EMA + convergence gate (plan 4.4 / 13).
    long long gateOpenAtSteps     = 0;     ///< forces().valid once steps pass this.
    float     emaWindowFlowThroughs = 1.0f;///< StandardPreset default.
    float     emaFx = 0.0f, emaFy = 0.0f;  ///< Smoothed lattice forces.
    bool      emaSeeded = false;
    bool      forcePending = false;        ///< A reduction readback is in flight.
    long long stepsAtForceLaunch   = 0;
    long long stepsAtLastForceFold = 0;

    // Mass-drift diagnostic (graded-refinement health monitor, 2026-06-16).
    // A poll-don't-stall reduction sums rho over fluid cells once per batch.
    // meanMassBaseline is latched on the first post-ramp sample (the field is
    // developed by then); meanMassNow tracks the latest sample, and the host
    // exposes the relative drift (now/baseline - 1) in the readout. DIAGNOSTIC
    // ONLY — nothing here rescales the field (a global rescale would smear a
    // structured near-wall leak into the force/pressure integration).
    bool      massPending      = false;    ///< A mass readback is in flight.
    bool      massBaselineSet  = false;    ///< Baseline latched after the ramp.
    float     meanMassBaseline = 1.0f;     ///< Mean fluid rho at baseline.
    float     meanMassNow      = 1.0f;     ///< Mean fluid rho, latest sample.

    /// @brief One sample stored in the rolling average ring buffer.
    struct ForceSample {
        long long step = 0;   ///< Solver step at which this sample was folded.
        float     fx   = 0.0f; ///< Raw lattice Fx at this sample.
        float     fy   = 0.0f; ///< Raw lattice Fy at this sample.
    };

    /// Ring buffer holding the most recent force samples for the 1-flow-through
    /// trailing average (Cl(a)/Cd(a)).  256 slots is far more than one flow-
    /// through ever produces (force readbacks fire at most once per frame).
    static constexpr int kAvgRingSize = 256;
    ForceSample avgRing[kAvgRingSize]{};
    int  avgRingHead = 0;   ///< Next write index (mod kAvgRingSize).
    int  avgRingCount = 0;  ///< Valid entries (capped at kAvgRingSize).

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
    int       nanTripKind = 0;   ///< 1 = NaN/Inf, 2 = velocity runaway.
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

    // ---- two-level refinement patch (plan M-refine) ------------------------
    // The fine level owns its own ping-pong f pair and (padded) flag field;
    // it has NO macroscopic arrays — the restriction writes the fine-derived
    // moments straight into the coarse macro arrays, so every macro consumer
    // (renderer, particles, delta99 extraction) is fine-aware for free.
    struct FineLevel {
        bool           active = false;
        int            factor = kRefineFactor; ///< m: sub-steps + cell ratio.
        PatchBox       box;
        GridDims       dims{};
        LatticeScaling scaling{};            ///< refinedScaling(coarse, m).
        FPop*          f[2] = {nullptr, nullptr};
        std::uint8_t*  flags = nullptr;      ///< Padded ghost-z layout.
        int            src = 0;
        long long      ncells = 0, nxny = 0, ncellsPad = 0;
        bool           forcesFromFine = false;
    } fine;

    /// @brief View over one FINE f buffer (padded pointers, fine dims).
    DeviceLatticeView fineView(int which) const {
        return DeviceLatticeView{fine.f[which], fine.flags, fine.dims};
    }

    // ---- N-level cascade (graded refinement, 2026-06-16) -------------------
    // The historical design has exactly two refinement levels: `fine` (rung 0,
    // 2x the coarse grid, hugging the foil) and `finer` (rung 1, nested 2x the
    // fine grid around the VGs). The cascade generalizes this to an arbitrary
    // staircase of integer-2x rungs WITHOUT touching the level-agnostic coupling
    // kernels: each rung is an exact 2x of its parent, tau telescopes
    // (fineTauFor composes), and the coarse-to-fine fill / fine-to-coarse
    // restrict pair couples any (parent, child) pair identically.
    //
    // To keep the ~170 existing references to `fine`/`finer` (and the proven
    // 2-level tests) working verbatim, the first two rungs REMAIN the named
    // `fine`/`finer` members; rungs at depth >= 2 live in `extraLevels`. The
    // `chain` vector is the unifying iteration order, rebuilt by syncChain()
    // whenever a level is (de)allocated: chain[0] = &fine, chain[1] = &finer
    // (only while active), chain[2..] = &extraLevels[i]. Parallel vectors hold
    // each deep rung's wall-model + q-LIBB mirror and host flag copies, indexed
    // the same way (chain depth d >= 2 -> extra* [d-2]).
    std::vector<FineLevel> extraLevels;        ///< Rungs at cascade depth >= 2.
    std::vector<FineLevel*> chain;             ///< Active rungs, parent->child.

    // ---- ISLBM stretched-mesh mode (continuous-gradient refinement) -------
    // Mutually exclusive with the cascade: when this is active no refinement
    // levels are allocated and stepN routes the single coarse grid through the
    // interpolated-gather kernel. Rebuilt from the wall-distance field on every
    // geometry edit; the host copy is kept so setFlags can rebuild it.
    StretchMesh stretch;                       ///< Device mesh + diagnostics.
    std::vector<float> stretchWallDist;        ///< Host wall-distance field copy.

    /// @brief View over one buffer of cascade rung @p depth (0 = fine).
    DeviceLatticeView chainView(int depth, int which) const {
        const FineLevel* L = chain[depth];
        return DeviceLatticeView{L->f[which], L->flags, L->dims};
    }

    /// @brief Rebuild `chain` from the live level storage. chain[0]=&fine when
    /// fine is active, then &finer if active, then each active extra rung. The
    /// strict nesting invariant (a rung is active only if its parent is) means
    /// the active prefix is contiguous, so iteration stops at the first gap.
    void syncChain() {
        chain.clear();
        if (!fine.active) return;
        chain.push_back(&fine);
        if (!finer.active) return;
        chain.push_back(&finer);
        for (FineLevel& e : extraLevels) {
            if (!e.active) break;
            chain.push_back(&e);
        }
    }

    /// @brief This rung's wall-model slip view (named members at depth 0/1, the
    /// wmExtra vector beyond — the same indexing syncChain uses).
    WallSlipView chainSlip(int depth) const {
        return (depth == 0) ? wmFine.slipView()
             : (depth == 1) ? wmFiner.slipView()
                            : wmExtra[depth - 2].slipView();
    }
    /// @brief This rung's q-LIBB link view (same depth indexing).
    QLinkView chainQLink(int depth) const {
        return (depth == 0) ? qFine.view()
             : (depth == 1) ? qFiner.view()
                            : qExtra[depth - 2].view();
    }

    /// @brief Recursively advance cascade rung @p depth by its factor sub-steps
    /// against its parent, then restrict it back into the parent buffers. The
    /// depth-general form of the historical fine/finer block; couples any
    /// (parent, child) pair with the level-agnostic fill/restrict kernels
    /// (tau telescopes via fineTauFor). Recursion-before-swap is preserved so
    /// both of a rung's time-level buffers stay valid for its child's fill.
    /// @param depth        Index into `chain` (0 = fine, coupled to coarse).
    /// @param parentParams The parent rung's StepParams (its tau is the base
    ///                     for this rung's fineTauFor).
    /// @param parentT0     Parent buffer view at time t (pre-swap source).
    /// @param parentT1Buf  Parent f buffer at time t+1 (post-collision).
    /// @param parentDstBuf Parent f buffer the restriction overwrites.
    /// @param macroRho/U/V/W Coarse macro arrays to write on a render step
    ///                     (depth 0 only; null deeper and on non-render steps).
    cudaError_t advanceCascadeRung(int depth, const StepParams& parentParams,
                                   DeviceLatticeView parentT0, FPop* parentT1Buf,
                                   FPop* parentDstBuf, float* macroRho,
                                   float* macroU, float* macroV, float* macroW) {
        FineLevel& L = *chain[depth];
        const int   m    = L.factor;
        const float tauP = parentParams.tau;
        const float tauL = fineTauFor(tauP, m);
        StepParams lp = parentParams;
        lp.tau        = tauL;
        lp.writeMacro = false; // refinement levels never own macro arrays

        const WallSlipView slip  = chainSlip(depth);
        const QLinkView    qlink = chainQLink(depth);
        const bool hasChild = (depth + 1) < static_cast<int>(chain.size());

        for (int k = 0; k < m; ++k) {
            // Fill: sub-step k sees the parent field time-interpolated at
            // t + k/m (the two parent buffers bracket it).
            if (auto e = launchCoarseToFineFill(
                    parentT0, parentT1Buf, chainView(depth, L.src), L.box, m,
                    static_cast<float>(k) / static_cast<float>(m), tauP, tauL,
                    /*fullVolume=*/false, stream);
                e != cudaSuccess)
                return e;
            if (auto e = launchRefreshGhostZ(L.f[L.src], L.dims, stream);
                e != cudaSuccess)
                return e;
            if (auto e = launchStreamCollide(
                    chainView(depth, L.src), chainView(depth, 1 - L.src), lp,
                    nullptr, nullptr, nullptr, nullptr, stream, slip, qlink);
                e != cudaSuccess)
                return e;
            if (auto e = launchRefreshGhostZ(L.f[1 - L.src], L.dims, stream);
                e != cudaSuccess)
                return e;

            // Recurse into the child BEFORE swapping this rung: both of this
            // rung's buffers (t = L.f[L.src], t+1 = L.f[1-L.src]) are valid and
            // ghost-refreshed, the contract the child's fill needs. The child
            // restricts into this rung's t+1 buffer, carried up by the restrict
            // below.
            if (hasChild) {
                if (auto e = advanceCascadeRung(
                        depth + 1, lp, chainView(depth, L.src),
                        L.f[1 - L.src], L.f[1 - L.src],
                        nullptr, nullptr, nullptr, nullptr);
                    e != cudaSuccess)
                    return e;
            }

            L.src = 1 - L.src; // swap only after the child subtree finished
        }

        // Restrict this rung -> the parent's post-collision buffer (+ the
        // coarse macro arrays at depth 0 on a render step; deeper rungs carry
        // their moments up through the chain of restrictions).
        if (auto e = launchFineToCoarseRestrict(
                chainView(depth, L.src),
                DeviceLatticeView{parentDstBuf, parentT0.flags, parentT0.dims},
                L.box, m, tauP, tauL, macroRho, macroU, macroV, macroW, stream);
            e != cudaSuccess)
            return e;
        if (auto e = launchRefreshGhostZ(parentDstBuf, parentT0.dims, stream);
            e != cudaSuccess)
            return e;
        return cudaSuccess;
    }

    // ---- nested VG patch (third level) -------------------------------------
    // The finer level is a FineLevel reused one rung down: its `box` is in FINE
    // cells, its parent is `fine` (not the coarse grid), and it couples to the
    // fine grid with the SAME level-agnostic kernels. It can only be active
    // while `fine` is — freeFine() cascades into freeFiner().
    FineLevel finer;

    /// @brief View over one FINER f buffer (padded pointers, finer dims).
    DeviceLatticeView finerView(int which) const {
        return DeviceLatticeView{finer.f[which], finer.flags, finer.dims};
    }

    // ---- iMEM wall model (lbm_wallmodel.cuh + wall_model.h) ----------------
    // One device mirror of the host-built wall-cell list per level, plus the
    // half-precision slip field the hot kernels read. Everything here exists
    // only while the wall model is enabled; the slip view degrades to the
    // inactive (plain bounce-back) state whenever a level has no list.
    struct WMLevel {
        bool           active = false;
        int            count  = 0;       ///< Listed wall cells.
        long long*     dCellIdx    = nullptr;
        long long*     dSampleIdx  = nullptr;
        float*         dNormalX    = nullptr;
        float*         dNormalY    = nullptr;
        float*         dNormalZ    = nullptr;
        float*         dSampleDist = nullptr;
        std::uint32_t* dLinkMask   = nullptr;
        float*         dUTau       = nullptr; ///< Persistent EMA state.
        std::uint16_t* dUwx = nullptr;        ///< Slip field, half bits,
        std::uint16_t* dUwy = nullptr;        ///< level-cellCount entries
        std::uint16_t* dUwz = nullptr;        ///< each (zero = plain BB).
        WallModelDeviceStats* dStats = nullptr;
        WallListStats  buildStats{};          ///< Host-side build diagnostics.

        /// @brief Slip view for the hot kernels (inactive when unbuilt).
        WallSlipView slipView() const {
            return active ? WallSlipView{dUwx, dUwy, dUwz} : WallSlipView{};
        }

        /// @brief Device list view for the update kernel.
        WallCellListView listView() const {
            return WallCellListView{dCellIdx, dSampleIdx, dNormalX, dNormalY,
                                    dNormalZ, dSampleDist, dLinkMask, dUTau,
                                    count};
        }

        /// @brief Release every device allocation of this level's mirror.
        void free() {
            cudaFree(dCellIdx);    cudaFree(dSampleIdx);
            cudaFree(dNormalX);    cudaFree(dNormalY);
            cudaFree(dNormalZ);    cudaFree(dSampleDist);
            cudaFree(dLinkMask);   cudaFree(dUTau);
            cudaFree(dUwx);        cudaFree(dUwy);
            cudaFree(dUwz);        cudaFree(dStats);
            *this = WMLevel{};
        }
    };
    bool    wmEnabled = false; ///< User/app switch (survives re-init).
    WMLevel wmCoarse;
    WMLevel wmFine;
    WMLevel wmFiner; ///< Nested VG patch wall list (VGs live here at the
                     ///< finest resolution — this is the level whose wall
                     ///< stress the readout reports when active).
    /// Wall-model mirrors for cascade rungs at depth >= 2 (parallel to
    /// extraLevels; wmExtra[i] belongs to extraLevels[i]).
    std::vector<WMLevel> wmExtra;

    // ---- interpolated bounce-back (q-LIBB) per level -----------------------
    // The dense per-cell-per-direction cut-fraction field the hot kernel reads
    // (QLinkView). Only allocated on levels that carry RESOLVED vanes — in
    // practice the fine and nested (finer) patches; the coarse grid's vanes are
    // too under-resolved for q-LIBB to mean anything, so it stays unallocated.
    struct QLevel {
        bool           active = false;
        std::uint8_t*  dQFrac  = nullptr; ///< [ncells*kQ] cut fraction bytes.
        std::uint32_t* dFFMask = nullptr; ///< [ncells] bit q: x_ff fluid.
        int            links   = 0;       ///< Resolved q-links (UI readout).
        int            fallback = 0;      ///< Thin-vane links left at half-way.

        QLinkView view() const {
            return active ? QLinkView{dQFrac, dFFMask} : QLinkView{};
        }
        void free() {
            cudaFree(dQFrac);  cudaFree(dFFMask);
            *this = QLevel{};
        }
    };
    bool   qlibbEnabled = false; ///< User/app switch (survives re-init).
    QLevel qCoarse;              ///< (Normally inactive — vanes unresolved here.)
    QLevel qFine;
    QLevel qFiner;
    /// q-LIBB mirrors for cascade rungs at depth >= 2 (parallel to extraLevels).
    std::vector<QLevel> qExtra;
    // NOTE: the analytic vane slabs + q-link construction live in the GEOM layer
    // (vg.h buildVaneQLinks/densifyQLinks); the app builds the dense fields and
    // hands them here via setFineQLinks/setFinerQLinks, so the solver stays
    // geom-free and just owns the device mirror.

    // Host copies the FINE wall-list rebuild needs (the coarse analogs are
    // hostFlags / surfaceRefFlags above): the live fine flags as uploaded by
    // initRefinement/setRefinedFlags, and the clean (VG-free) fine reference.
    std::vector<std::uint8_t> fineHostFlags;
    std::vector<std::uint8_t> fineSurfaceRefFlags;

    // Finer-level analogs (the live finer flags + the clean VG-free reference).
    std::vector<std::uint8_t> finerHostFlags;
    std::vector<std::uint8_t> finerSurfaceRefFlags;

    // Host flag copies for cascade rungs at depth >= 2 (live + clean reference),
    // parallel to extraLevels — the wall-list rebuild needs them per rung.
    std::vector<std::vector<std::uint8_t>> extraHostFlags;
    std::vector<std::vector<std::uint8_t>> extraSurfaceRefFlags;

    /// @brief Build one level's wall-cell list on the host and mirror it to
    /// the device. Frees the previous mirror first; leaves the level inactive
    /// when the list is empty or any allocation fails (the sim then simply
    /// runs plain bounce-back there — never fatal).
    cudaError_t buildWMLevel(WMLevel& wm, const GridDims& d,
                             const std::vector<std::uint8_t>& active,
                             const std::vector<std::uint8_t>& clean,
                             int sampleCells, long long levelNcells) {
        wm.free();
        const WallCellList list =
            buildWallCellList(d, active, clean, sampleCells, &wm.buildStats);
        if (list.size() == 0) return cudaSuccess;
        const auto n = static_cast<long long>(list.size());

        // Allocations: list arrays sized by the surface, slip field by the
        // level's full cell count (the hot kernels index it per cell).
        struct Alloc { void** p; std::size_t bytes; };
        const Alloc allocs[] = {
            {reinterpret_cast<void**>(&wm.dCellIdx),    n * sizeof(long long)},
            {reinterpret_cast<void**>(&wm.dSampleIdx),  n * sizeof(long long)},
            {reinterpret_cast<void**>(&wm.dNormalX),    n * sizeof(float)},
            {reinterpret_cast<void**>(&wm.dNormalY),    n * sizeof(float)},
            {reinterpret_cast<void**>(&wm.dNormalZ),    n * sizeof(float)},
            {reinterpret_cast<void**>(&wm.dSampleDist), n * sizeof(float)},
            {reinterpret_cast<void**>(&wm.dLinkMask),   n * sizeof(std::uint32_t)},
            {reinterpret_cast<void**>(&wm.dUTau),       n * sizeof(float)},
            {reinterpret_cast<void**>(&wm.dUwx), static_cast<std::size_t>(levelNcells) * sizeof(std::uint16_t)},
            {reinterpret_cast<void**>(&wm.dUwy), static_cast<std::size_t>(levelNcells) * sizeof(std::uint16_t)},
            {reinterpret_cast<void**>(&wm.dUwz), static_cast<std::size_t>(levelNcells) * sizeof(std::uint16_t)},
            {reinterpret_cast<void**>(&wm.dStats), sizeof(WallModelDeviceStats)},
        };
        for (const Alloc& a : allocs) {
            if (auto err = cudaMalloc(a.p, a.bytes); err != cudaSuccess) {
                wm.free();
                return err;
            }
        }

        // Upload the list; prime u_tau and the slip field to zero (the model
        // ramps in over the first updates through the EMA — the very first
        // batch after a build runs plain bounce-back values everywhere).
        struct Up { void* dst; const void* src; std::size_t bytes; };
        const Up ups[] = {
            {wm.dCellIdx,    list.cellIdx.data(),    n * sizeof(long long)},
            {wm.dSampleIdx,  list.sampleIdx.data(),  n * sizeof(long long)},
            {wm.dNormalX,    list.normalX.data(),    n * sizeof(float)},
            {wm.dNormalY,    list.normalY.data(),    n * sizeof(float)},
            {wm.dNormalZ,    list.normalZ.data(),    n * sizeof(float)},
            {wm.dSampleDist, list.sampleDist.data(), n * sizeof(float)},
            {wm.dLinkMask,   list.linkMask.data(),   n * sizeof(std::uint32_t)},
        };
        for (const Up& u_ : ups) {
            if (auto err = cudaMemcpyAsync(u_.dst, u_.src, u_.bytes,
                                           cudaMemcpyHostToDevice, stream);
                err != cudaSuccess) {
                // The host vectors live until the function returns; sync so
                // the failed-state teardown cannot race in-flight copies.
                cudaStreamSynchronize(stream);
                wm.free();
                return err;
            }
        }
        cudaMemsetAsync(wm.dUTau, 0, n * sizeof(float), stream);
        cudaMemsetAsync(wm.dUwx, 0, levelNcells * sizeof(std::uint16_t), stream);
        cudaMemsetAsync(wm.dUwy, 0, levelNcells * sizeof(std::uint16_t), stream);
        cudaMemsetAsync(wm.dUwz, 0, levelNcells * sizeof(std::uint16_t), stream);
        cudaMemsetAsync(wm.dStats, 0, sizeof(WallModelDeviceStats), stream);
        // The async uploads read the list's host vectors, which die when this
        // function returns — fence them before the buffers go away.
        if (auto err = cudaStreamSynchronize(stream); err != cudaSuccess) {
            wm.free();
            return err;
        }
        wm.count = static_cast<int>(n);
        wm.active = true;
        return cudaSuccess;
    }

    /// @brief Upload a densified q-LIBB field into a level's device mirror.
    /// Frees the previous mirror first; leaves the level inactive (plain
    /// bounce-back) when the field is empty/disabled or any allocation fails.
    /// @param ql        The level's q mirror to (re)build.
    /// @param qFrac     Dense per-cell-per-dir cut bytes (ncells*kQ), or empty.
    /// @param ffMask    Dense per-cell two-node-usable mask (ncells).
    /// @param levelNcells Cell count of the level.
    /// @param links     Resolved-link count (UI readout).
    /// @param fallback  Thin-vane links left at half-way (UI readout).
    cudaError_t uploadQLevel(QLevel& ql,
                             const std::vector<std::uint8_t>& qFrac,
                             const std::vector<std::uint32_t>& ffMask,
                             long long levelNcells, int links, int fallback) {
        ql.free();
        if (!qlibbEnabled || links == 0
            || qFrac.size() != static_cast<std::size_t>(levelNcells) * kQ
            || ffMask.size() != static_cast<std::size_t>(levelNcells))
            return cudaSuccess;

        const std::size_t qBytes =
            static_cast<std::size_t>(levelNcells) * kQ * sizeof(std::uint8_t);
        const std::size_t mBytes =
            static_cast<std::size_t>(levelNcells) * sizeof(std::uint32_t);
        if (auto err = cudaMalloc(reinterpret_cast<void**>(&ql.dQFrac), qBytes);
            err != cudaSuccess) { ql.free(); return err; }
        if (auto err = cudaMalloc(reinterpret_cast<void**>(&ql.dFFMask), mBytes);
            err != cudaSuccess) { ql.free(); return err; }
        cudaMemcpyAsync(ql.dQFrac, qFrac.data(), qBytes,
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(ql.dFFMask, ffMask.data(), mBytes,
                        cudaMemcpyHostToDevice, stream);
        if (auto err = cudaStreamSynchronize(stream); err != cudaSuccess) {
            ql.free();
            return err;
        }
        ql.active   = true;
        ql.links    = links;
        ql.fallback = fallback;
        return cudaSuccess;
    }

    /// @brief (Re)build the coarse wall-cell list from the current host
    /// flags. No-op (free only) when the wall model is off.
    void rebuildWallModelCoarse() {
        wmCoarse.free();
        if (!wmEnabled || hostFlags.size() != static_cast<std::size_t>(ncells))
            return;
        const std::vector<std::uint8_t>& clean =
            (surfaceRefFlags.size() == hostFlags.size()) ? surfaceRefFlags
                                                         : hostFlags;
        buildWMLevel(wmCoarse, dims, hostFlags, clean,
                     kWallSampleCellsCoarse, ncells);
    }

    /// @brief (Re)build the fine wall-cell list (needs the fine host flag
    /// copy kept by initRefinement/setRefinedFlags).
    void rebuildWallModelFine() {
        wmFine.free();
        if (!wmEnabled || !fine.active
            || fineHostFlags.size() != static_cast<std::size_t>(fine.ncells))
            return;
        const std::vector<std::uint8_t>& clean =
            (fineSurfaceRefFlags.size() == fineHostFlags.size())
                ? fineSurfaceRefFlags
                : fineHostFlags;
        buildWMLevel(wmFine, fine.dims, fineHostFlags, clean,
                     kWallSampleCellsFine, fine.ncells);
    }

    /// @brief (Re)build the FINER (nested VG patch) wall-cell list. The VGs
    /// live on this level, so this is the list that carries the vane wall
    /// stress; it keys normals + VG exclusion off the clean finer reference.
    void rebuildWallModelFiner() {
        wmFiner.free();
        if (!wmEnabled || !finer.active
            || finerHostFlags.size() != static_cast<std::size_t>(finer.ncells))
            return;
        const std::vector<std::uint8_t>& clean =
            (finerSurfaceRefFlags.size() == finerHostFlags.size())
                ? finerSurfaceRefFlags
                : finerHostFlags;
        buildWMLevel(wmFiner, finer.dims, finerHostFlags, clean,
                     kWallSampleCellsFine, finer.ncells);
    }

    /// @brief Zero the persistent wall-model state (u_tau EMA + slip field)
    /// of both levels — cold starts must not inherit the previous run's wall
    /// stress.
    void resetWallModelState() {
        // (WMLevel*, cell-count) for the coarse grid + every active cascade
        // rung. The deep rungs (depth >= 2) pull their count from extraLevels.
        auto zero = [this](WMLevel& wm, long long n) {
            if (!wm.active) return;
            cudaMemsetAsync(wm.dUTau, 0, wm.count * sizeof(float), stream);
            cudaMemsetAsync(wm.dUwx, 0, n * sizeof(std::uint16_t), stream);
            cudaMemsetAsync(wm.dUwy, 0, n * sizeof(std::uint16_t), stream);
            cudaMemsetAsync(wm.dUwz, 0, n * sizeof(std::uint16_t), stream);
        };
        zero(wmCoarse, ncells);
        zero(wmFine, fine.ncells);
        zero(wmFiner, finer.ncells);
        for (std::size_t i = 0; i < wmExtra.size(); ++i)
            zero(wmExtra[i], extraLevels[i].ncells);
    }

    /// @brief Release one deep cascade rung's device memory (helper for the
    /// extraLevels storage; the named fine/finer rungs use freeFine/freeFiner).
    void freeExtraLevel(int i) {
        FineLevel& e = extraLevels[i];
        for (FPop*& p : e.f) { cudaFree(p); p = nullptr; }
        cudaFree(e.flags); e.flags = nullptr;
        e = FineLevel{};
        wmExtra[i].free();
        qExtra[i].free();
        extraHostFlags[i].clear();
        extraSurfaceRefFlags[i].clear();
    }

    /// @brief Release every deep cascade rung (depth >= 2), deepest first so the
    /// nesting invariant (child freed before parent) holds. Leaves fine/finer.
    void freeExtraLevels() {
        for (int i = static_cast<int>(extraLevels.size()) - 1; i >= 0; --i)
            freeExtraLevel(i);
        extraLevels.clear();
        wmExtra.clear();
        qExtra.clear();
        extraHostFlags.clear();
        extraSurfaceRefFlags.clear();
    }

    /// @brief Release only the FINER level's device memory. The fine + coarse
    /// sim is untouched (the overlap simply stops receiving finer restrictions).
    /// Cascade rungs deeper than `finer` nest inside it, so they go first.
    void freeFiner() {
        freeExtraLevels();
        for (FPop*& p : finer.f) { cudaFree(p); p = nullptr; }
        cudaFree(finer.flags); finer.flags = nullptr;
        finer = FineLevel{};
        wmFiner.free();
        qFiner.free();
        finerHostFlags.clear();
        finerSurfaceRefFlags.clear();
        syncChain();
    }

    /// @brief Release only the fine level's device memory.
    void freeFine() {
        // Lifecycle invariant: the finer level nests inside the fine level and
        // indexes fine-derived data — it can never outlive its parent.
        freeFiner();
        for (FPop*& p : fine.f) { cudaFree(p); p = nullptr; }
        cudaFree(fine.flags); fine.flags = nullptr;
        fine = FineLevel{};
        // The fine wall-model + q-LIBB mirrors index fine-level cells; they die
        // with the level (a later initRefinement rebuilds them).
        wmFine.free();
        qFine.free();
        fineHostFlags.clear();
        fineSurfaceRefFlags.clear();
        syncChain();
    }

    /// @brief Seed the fine level from the CURRENT coarse field: full-volume
    /// coarse-to-fine fill of both fine ping-pong buffers + ghost refresh.
    /// Valid at any point (cold start, restore, flag edit) — the fill derives
    /// everything from the coarse populations.
    cudaError_t seedFineFromCoarse() {
        if (!fine.active) return cudaSuccess;
        const float tauC = effectiveTau();
        const float tauF = fineTauFor(tauC, fine.factor);
        for (int i = 0; i < 2; ++i) {
            // Both time levels point at the same buffer (weight 0): seeding
            // is a snapshot, not a time interpolation.
            if (auto err = launchCoarseToFineFill(
                    view(src), f[src], fineView(i), fine.box, fine.factor,
                    /*timeWeight=*/0.0f, tauC, tauF,
                    /*fullVolume=*/true, stream);
                err != cudaSuccess)
                return err;
            if (auto err = launchRefreshGhostZ(fine.f[i], fine.dims, stream);
                err != cudaSuccess)
                return err;
        }
        return cudaSuccess;
    }

    /// @brief Seed the FINER level from the CURRENT fine field: full-volume
    /// fine-to-finer fill of both finer ping-pong buffers + ghost refresh.
    /// Mirrors seedFineFromCoarse one rung down — the parent is `fine`, and the
    /// tau pair is the composed (tauFine, tauFiner). The neq rescale composes
    /// exactly (telescopes to tauFiner/(m*m2*tauCoarse)), so passing the
    /// fine/finer pair here is the same shape as the coarse/fine seeding above.
    cudaError_t seedFinerFromFine() {
        if (!finer.active) return cudaSuccess;
        const float tauC     = effectiveTau();
        const float tauF     = fineTauFor(tauC, fine.factor);
        const float tauFiner = fineTauFor(tauF, finer.factor);
        for (int i = 0; i < 2; ++i) {
            // Both fine time levels point at the same fine buffer (weight 0):
            // seeding is a snapshot, not a time interpolation.
            if (auto err = launchCoarseToFineFill(
                    fineView(fine.src), fine.f[fine.src], finerView(i),
                    finer.box, finer.factor,
                    /*timeWeight=*/0.0f, tauF, tauFiner,
                    /*fullVolume=*/true, stream);
                err != cudaSuccess)
                return err;
            if (auto err = launchRefreshGhostZ(finer.f[i], finer.dims, stream);
                err != cudaSuccess)
                return err;
        }
        return cudaSuccess;
    }

    /// @brief Composed base tau at cascade depth @p depth: the coarse tau
    /// folded through every 2x hop down to that rung (fineTauFor telescopes, so
    /// folding factor-by-factor is exact). depth -1 returns the coarse tau.
    float tauAtDepth(int depth) const {
        float tau = effectiveTau();
        for (int d = 0; d <= depth && d < static_cast<int>(chain.size()); ++d)
            tau = fineTauFor(tau, chain[d]->factor);
        return tau;
    }

    /// @brief Seed a deep cascade rung (depth >= 2, the i-th extra level) from
    /// its CURRENT parent field — the depth-general form of seedFinerFromFine.
    /// The parent is chain[depth-1]; both parent time levels point at the same
    /// buffer (weight 0: a snapshot, not a time interpolation).
    cudaError_t seedExtraFromParent(int i) {
        const int depth = i + 2;            // extra[i] lives at cascade depth i+2
        if (depth >= static_cast<int>(chain.size())) return cudaSuccess;
        FineLevel& L = extraLevels[i];
        const float tauParent = tauAtDepth(depth - 1);
        const float tauL      = fineTauFor(tauParent, L.factor);
        const DeviceLatticeView parent = chainView(depth - 1, chain[depth - 1]->src);
        FPop* parentBuf = chain[depth - 1]->f[chain[depth - 1]->src];
        for (int b = 0; b < 2; ++b) {
            if (auto err = launchCoarseToFineFill(
                    parent, parentBuf, chainView(depth, b), L.box, L.factor,
                    /*timeWeight=*/0.0f, tauParent, tauL,
                    /*fullVolume=*/true, stream);
                err != cudaSuccess)
                return err;
            if (auto err = launchRefreshGhostZ(L.f[b], L.dims, stream);
                err != cudaSuccess)
                return err;
        }
        return cudaSuccess;
    }

    /// @brief (Re)build a deep cascade rung's wall-cell list (depth >= 2). Same
    /// shape as rebuildWallModelFiner, keyed off the per-rung host flag copies.
    void rebuildWallModelExtra(int i) {
        wmExtra[i].free();
        FineLevel& L = extraLevels[i];
        if (!wmEnabled || !L.active
            || extraHostFlags[i].size() != static_cast<std::size_t>(L.ncells))
            return;
        const std::vector<std::uint8_t>& clean =
            (extraSurfaceRefFlags[i].size() == extraHostFlags[i].size())
                ? extraSurfaceRefFlags[i]
                : extraHostFlags[i];
        buildWMLevel(wmExtra[i], L.dims, extraHostFlags[i], clean,
                     kWallSampleCellsFine, L.ncells);
    }

    /// @brief Recompute whether the patch covers every coarse Solid cell
    /// (excluding the restriction band): when true, the momentum-exchange
    /// force reduction runs on the fine grid for 2x wall resolution.
    void updateForcesFromFine() {
        if (!fine.active) { fine.forcesFromFine = false; return; }
        const auto solid = static_cast<std::uint8_t>(CellFlag::Solid);
        const int x0 = fine.box.x0 + kRestrictBandCoarse;
        const int x1 = fine.box.x1 - kRestrictBandCoarse;
        const int y0 = fine.box.y0 + kRestrictBandCoarse;
        const int y1 = fine.box.y1 - kRestrictBandCoarse;
        for (int z = 0; z < dims.nz; ++z) {
            const long long zBase = nxny * z;
            for (int y = 0; y < dims.ny; ++y) {
                const long long rowBase = zBase + static_cast<long long>(dims.nx) * y;
                for (int x = 0; x < dims.nx; ++x) {
                    if (hostFlags[static_cast<std::size_t>(rowBase + x)] != solid)
                        continue;
                    if (x < x0 || x >= x1 || y < y0 || y >= y1) {
                        fine.forcesFromFine = false;
                        return; // a solid pokes out of the patch: coarse forces
                    }
                }
            }
        }
        fine.forcesFromFine = true;
    }

    // ------------------------------------------------------------------

    /// @brief View over one f buffer (padded pointers, real dims).
    DeviceLatticeView view(int which) const {
        return DeviceLatticeView{f[which], flags, dims};
    }

    /// @brief Steps in one flow-through at the current scaling.
    float flowThroughSteps() const { return scaling.flowThroughSteps(dims.nx); }

    /// @brief tau for the step about to run, honoring the ramp state.
    float effectiveTau() const {
        return (rampActive && rampEnabled)
                   ? rampedTau(scaling, steps - rampStartStep, dims.nx)
                   : scaling.tau;
    }

    /// @brief Inlet/freestream velocity for the step about to run, honoring the
    /// startup velocity ramp (eases from rest so the impulsive-start pressure
    /// shock never forms). Outside the ramp (or when disabled) it is u_lat.
    float effectiveU() const {
        return (rampActive && rampEnabled)
                   ? rampedU(scaling, steps - rampStartStep, dims.nx)
                   : scaling.u_lat;
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

            // Push this raw sample into the trailing-average ring buffer.
            avgRing[avgRingHead] = {stepsAtForceLaunch, hForce[0], hForce[1]};
            avgRingHead  = (avgRingHead + 1) % kAvgRingSize;
            if (avgRingCount < kAvgRingSize) ++avgRingCount;

            stepsAtLastForceFold = stepsAtForceLaunch;
            forcePending = false;
        }
        // Mass-drift sample -> mean fluid density. hMass = {sum rho, count};
        // the mean is the conserved quantity to watch (exact streaming holds it
        // to round-off). The baseline is latched once the startup ramp is over
        // (rampActive false) so it reflects a developed field, not the rest
        // init; thereafter meanMassNow / meanMassBaseline - 1 is the drift the
        // readout reports. No rescale — diagnostic only.
        if (massPending && eventDone(evMass)) {
            if (hMass[1] > 0.0f) {
                meanMassNow = hMass[0] / hMass[1];
                if (!massBaselineSet && !rampActive) {
                    meanMassBaseline = meanMassNow;
                    massBaselineSet  = true;
                }
            }
            massPending = false;
        }
        // Watchdog verdict. Flag 1 = non-finite populations, flag 2 = a cell
        // pinned at the collision limiter's velocity cap (diverged-but-finite
        // runaway) — both latch the pause; the diagnosis text distinguishes.
        if (nanPending && eventDone(evNan)) {
            if (*hNan != 0) {
                nanLatched = true;
                nanTripKind = *hNan;
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
        // Geometry changed: the wall-cell list is keyed to the flags.
        rebuildWallModelCoarse();
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
        freeFine();
        stretch.free();  // ISLBM device mesh (no-op when uniform/cascade).
        stretchWallDist.clear();
        wmCoarse.free(); // wmFine already died with freeFine(); the wmEnabled
                         // SETTING survives so re-init reapplies the policy.
        qCoarse.free();  // qFine/qFiner died with freeFine(); the setting survives.
        for (FPop*& p : f) { cudaFree(p); p = nullptr; }
        cudaFree(flags); flags = nullptr;
        cudaFree(rho); rho = nullptr;
        cudaFree(u); u = nullptr;
        cudaFree(v); v = nullptr;
        cudaFree(w); w = nullptr;
        cudaFree(force); force = nullptr;
        cudaFree(mass); mass = nullptr;
        cudaFree(nanDev); nanDev = nullptr;
        cudaFree(editMask); editMask = nullptr;
        cudaFreeHost(hForce); hForce = nullptr;
        cudaFreeHost(hMass); hMass = nullptr;
        cudaFreeHost(hNan); hNan = nullptr;
        for (cudaEvent_t* ev : {&evT0, &evT1, &evForce, &evMass, &evNan}) {
            if (*ev) { cudaEventDestroy(*ev); *ev = nullptr; }
        }
        // In-flight async readbacks died with their events/buffers above, so
        // the pending latches must drop with them. A stale 'pending' that
        // survives into the next init() polls FRESH, never-recorded events:
        // cudaEventQuery treats those as complete, and cudaEventElapsedTime
        // then returns cudaErrorInvalidResourceHandle — worse, it leaves that
        // error in the sticky per-thread slot, so the next launch wrapper's
        // cudaGetLastError() reports a phantom "sim step" failure (the
        // resolution/HiFi/airspeed re-init crash).
        timingPending = false;
        timingBatchN  = 0;
        forcePending  = false;
        massPending     = false;
        massBaselineSet = false;
        nanPending    = false;
        // Pacing estimates are per-grid: a per-step time measured on a smaller
        // grid would let the first post-reinit batch overshoot the TDR budget
        // (adaptiveStepsForBudget grows from lastChosenN). Re-probe from 1.
        emaStepMs   = 0.0;
        lastStepMs  = 0.0;
        lastChosenN = 0;
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
    if (auto err = cudaMalloc(&s.mass, 2 * sizeof(float)); err != cudaSuccess)
        return fail("mass accumulator allocation failed", err);
    if (auto err = cudaMalloc(&s.nanDev, sizeof(int)); err != cudaSuccess)
        return fail("watchdog flag allocation failed", err);

    // Pinned host readback slots — async D2H copies into pageable memory
    // silently serialize, which would defeat the poll-don't-stall design.
    if (auto err = cudaMallocHost(&s.hForce, 3 * sizeof(float)); err != cudaSuccess)
        return fail("pinned force readback allocation failed", err);
    if (auto err = cudaMallocHost(&s.hMass, 2 * sizeof(float)); err != cudaSuccess)
        return fail("pinned mass readback allocation failed", err);
    if (auto err = cudaMallocHost(&s.hNan, sizeof(int)); err != cudaSuccess)
        return fail("pinned watchdog readback allocation failed", err);

    // Timing events keep timestamps; the readback fences don't need them.
    for (auto [ev, flags_] : {std::pair{&s.evT0, 0u}, {&s.evT1, 0u},
                              {&s.evForce, (unsigned)cudaEventDisableTiming},
                              {&s.evMass, (unsigned)cudaEventDisableTiming},
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
    // Clear the trailing-average ring on every cold start so Cl(a)/Cd(a)
    // don't bleed samples from the previous run into the new one.
    s.avgRingHead  = 0;
    s.avgRingCount = 0;
    s.nanLatched = false;
    s.nanPending = false;
    s.nanTripKind = 0;
    s.stepsSinceWatchdog = 0;
    s.midStamp = -1;
    cudaMemsetAsync(s.nanDev, 0, sizeof(int), s.stream);

    // Both ping-pong buffers initialize AT REST (u = 0), not at freestream:
    // the inlet ramp (effectiveU) then accelerates the flow from zero so the
    // pressure field develops smoothly around the body instead of ringing an
    // impulsive shock off the edges. The spanwise speck still seeds the 3D
    // coherence-breaking ripple (scaled by the target u_lat so its relative
    // strength is unchanged once the flow is up to speed). The init kernels run
    // over the padded domain with z-periodic formulas, so the ghost planes are
    // consistent by construction — no refresh needed here.
    // Rest-init only when the ramp is enabled; otherwise legacy instant start
    // at full freestream (the inlet drives u_lat from step 1 too, via the
    // effectiveU gate below).
    const float initU = s.rampEnabled ? 0.0f : s.scaling.u_lat;
    for (int i = 0; i < 2; ++i) {
        launchInitEquilibrium(s.view(i), initU, s.stream);
        launchSpanwisePerturbation(s.view(i),
                                   kSpeckAmplitudeFrac * s.scaling.u_lat,
                                   kSpeckSeed, s.stream);
    }
    // Fine level restarts from the freshly initialized coarse field, then the
    // finer level restarts from the freshly seeded fine field (ordering: a
    // child always seeds AFTER its parent so it pulls developed parent data).
    s.seedFineFromCoarse();
    s.seedFinerFromFine();
    // Wall-model EMA state and slip field restart with the flow.
    s.resetWallModelState();
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
    // The wall list keys normals and VG exclusion off the clean reference.
    s.rebuildWallModelCoarse();
}

void LBMSolver::setWallModelEnabled(bool enabled) {
    Impl& s = *impl_;
    if (s.wmEnabled == enabled) return;
    s.wmEnabled = enabled;
    if (!s.initialized) return; // setting persists; init's flag upload builds
    // Build or tear down both levels; the slip view goes (in)active with
    // them, which is the entire switch the hot kernels see.
    s.rebuildWallModelCoarse();
    s.rebuildWallModelFine();
    s.rebuildWallModelFiner();
}

bool LBMSolver::wallModelEnabled() const { return impl_->wmEnabled; }

void LBMSolver::setRefinedSurfaceReference(
    const std::vector<std::uint8_t>& fineCleanFlags) {
    Impl& s = *impl_;
    if (!s.initialized || !s.fine.active) return;
    if (fineCleanFlags.size() != static_cast<std::size_t>(s.fine.ncells)) return;
    s.fineSurfaceRefFlags = fineCleanFlags;
    s.rebuildWallModelFine();
}

WallModelReadout LBMSolver::wallModelReadout() const {
    const Impl& s = *impl_;
    WallModelReadout r;
    r.enabled = s.wmEnabled;
    if (!s.initialized || !s.wmEnabled) return r;
    // Report the level that carries the finest wall physics: the nested VG
    // patch when it is wall-modeling (the VGs live there), else the fine patch,
    // else the coarse level. fromFine flags "a refinement patch level".
    const Impl::WMLevel& wm = s.wmFiner.active ? s.wmFiner
                            : s.wmFine.active  ? s.wmFine
                                               : s.wmCoarse;
    r.fromFine   = s.wmFiner.active || s.wmFine.active;
    r.excludedVG = wm.buildStats.excludedVG;
    r.degenerate = wm.buildStats.degenerate;
    if (!wm.active) return r;

    // Small synchronous read; the readout is a UI-rate call (header note).
    WallModelDeviceStats st{};
    if (cudaMemcpy(&st, wm.dStats, sizeof st, cudaMemcpyDeviceToHost)
        != cudaSuccess)
        return r;
    r.cells = st.activeCells;
    if (st.activeCells > 0) {
        r.meanYplus = st.sumYplus / static_cast<float>(st.activeCells);
        // maxYplusBits carries a positive float's bit pattern (atomicMax
        // over int is monotone there) — undo the punning.
        float maxY = 0.0f;
        static_assert(sizeof maxY == sizeof st.maxYplusBits);
        std::memcpy(&maxY, &st.maxYplusBits, sizeof maxY);
        r.maxYplus = maxY;
        r.clampedFrac = static_cast<float>(st.clampedCells)
                      / static_cast<float>(st.activeCells);
    }
    return r;
}

// ------ interpolated bounce-back (q-LIBB) ------

void LBMSolver::setQLIBBEnabled(bool enabled) {
    Impl& s = *impl_;
    if (s.qlibbEnabled == enabled) return;
    s.qlibbEnabled = enabled;
    if (enabled) return; // uploads (re)build the fields; nothing to do here.
    // Turning off: free every level's q field immediately so the next step
    // runs the plain bounce-back kernel instantiation.
    s.qCoarse.free();
    s.qFine.free();
    s.qFiner.free();
}

bool LBMSolver::qlibbEnabled() const { return impl_->qlibbEnabled; }

void LBMSolver::setCoarseQLinks(const std::vector<std::uint8_t>& qFrac,
                                const std::vector<std::uint32_t>& ffMask,
                                int links, int fallback) {
    Impl& s = *impl_;
    if (!s.initialized) return;
    s.uploadQLevel(s.qCoarse, qFrac, ffMask, s.ncells, links, fallback);
}

void LBMSolver::setFineQLinks(const std::vector<std::uint8_t>& qFrac,
                              const std::vector<std::uint32_t>& ffMask,
                              int links, int fallback) {
    Impl& s = *impl_;
    if (!s.initialized || !s.fine.active) return;
    s.uploadQLevel(s.qFine, qFrac, ffMask, s.fine.ncells, links, fallback);
}

void LBMSolver::setFinerQLinks(const std::vector<std::uint8_t>& qFrac,
                               const std::vector<std::uint32_t>& ffMask,
                               int links, int fallback) {
    Impl& s = *impl_;
    if (!s.initialized || !s.finer.active) return;
    s.uploadQLevel(s.qFiner, qFrac, ffMask, s.finer.ncells, links, fallback);
}

long long LBMSolver::fineCellCount() const {
    return impl_->fine.active ? impl_->fine.ncells : 0;
}

long long LBMSolver::finerCellCount() const {
    return impl_->finer.active ? impl_->finer.ncells : 0;
}

QLIBBReadout LBMSolver::qlibbReadout() const {
    const Impl& s = *impl_;
    QLIBBReadout r;
    r.enabled = s.qlibbEnabled;
    if (!s.qlibbEnabled) return r;
    for (const Impl::QLevel* q : {&s.qCoarse, &s.qFine, &s.qFiner}) {
        if (!q->active) continue;
        r.links    += q->links;
        r.fallback += q->fallback;
    }
    return r;
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
// N-level cascade advance (graded refinement). Recursively advances cascade
// rung `depth` (chain[depth]) by its factor sub-steps against its parent, then
// restricts it back into the parent's post-collision buffer. This is the
// depth-general form of the historical two-level fine/finer block: depth 0 is
// the fine patch coupled to the coarse grid, depth 1 the nested VG patch, and
// any deeper rung couples to its immediate parent with the SAME level-agnostic
// fill/restrict kernels (tau telescopes via fineTauFor). The recursion-before-
// swap ordering is preserved exactly: a rung does not swap its ping-pong until
// its entire child subtree has advanced and restricted, so both of the rung's
// time-level buffers stay valid for the child's interface fill.
// ===========================================================================

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
        // Ramped inlet: eases from rest over the startup window so the inlet
        // never over-drives the still-developing field (no impulsive shock).
        params.uInlet = s.effectiveU();
        // Macroscopic stores feed rendering once per frame: only the batch's
        // final step pays the extra 16 B/cell of write traffic (plan 11).
        params.writeMacro = (i == n - 1);

        // ISLBM: the per-cell tau lives in the stretch field, so the startup
        // viscosity ramp can't ride the scalar `params.tau` — it rides
        // tauRampMul, the ratio of the ramped to the target viscosity. The
        // scalar tau is still set (the collar cells, all at the finest spacing,
        // use it and it equals the wall field value by construction).
        if (s.stretch.active) {
            const float nuTarget = s.scaling.tau - 0.5f;
            const float nuNow    = s.effectiveTau() - 0.5f;
            params.tauRampMul = (nuTarget > 1e-9f) ? (nuNow / nuTarget) : 1.0f;
            params.tau        = s.scaling.tau; // wall reference for the collar
        }

        const cudaError_t err = launchStreamCollide(
            s.view(s.src), s.view(1 - s.src), params,
            s.rho, s.u, s.v, s.w, s.stream, s.wmCoarse.slipView(),
            s.qCoarse.view(), s.stretch.view());
        if (err != cudaSuccess) return err;

        // The freshly written buffer needs its spanwise ghost planes synced
        // before it is pulled from next step (periodic z, plan 4.2/11).
        if (auto err2 = launchRefreshGhostZ(s.f[1 - s.src], s.dims, s.stream);
            err2 != cudaSuccess)
            return err2;

        // ---- N-level cascade coupling (graded refinement): advance the whole
        // refinement staircase recursively. chain[0] (the fine patch) couples
        // to the coarse grid here; advanceCascadeRung recurses into every
        // deeper rung before each swap and restricts each rung back into its
        // parent. The coarse ping-pong pair provides both interface-fill time
        // levels (f[src] holds t, f[1-src] just received t+1). The depth-0
        // restriction writes the coarse macro arrays on render steps, so every
        // macro consumer still sees fine-derived data for free. ------------
        if (!s.chain.empty()) {
            if (auto e = s.advanceCascadeRung(
                    /*depth=*/0, params, s.view(s.src), s.f[1 - s.src],
                    s.f[1 - s.src],
                    params.writeMacro ? s.rho : nullptr,
                    params.writeMacro ? s.u : nullptr,
                    params.writeMacro ? s.v : nullptr,
                    params.writeMacro ? s.w : nullptr);
                e != cudaSuccess)
                return e;
        }

        s.src = 1 - s.src; // ping-pong swap
        ++s.steps;

        // NaN watchdog every 200 steps (plan 4.5): launch the strided check
        // plus an async readback; verdicts are folded in by pollAsync().
        // With the patch active the fine buffer is sampled too — a fine-level
        // divergence would otherwise hide until it bled into the overlap.
        if (++s.stepsSinceWatchdog >= kWatchdogPeriodSteps && !s.nanPending) {
            s.stepsSinceWatchdog = 0;
            cudaMemsetAsync(s.nanDev, 0, sizeof(int), s.stream);
            // The step counter rotates the sampled residue class (977 is
            // coprime to the watchdog's prime stride), so repeated checks
            // sweep different cells instead of re-probing one fixed set.
            launchNaNWatchdog(s.view(s.src), s.nanDev, s.steps * 977, s.stream);
            // Sample every cascade rung too — a rung-level divergence would
            // otherwise hide until it bled into the overlap. flagBase = 2*(d+1)
            // so a verdict identifies the grid: coarse 1/2, fine 3/4, nested
            // 5/6, deeper rungs 7/8, ... (the diagnosis text reads base+1/+2).
            for (int d = 0; d < static_cast<int>(s.chain.size()); ++d) {
                launchNaNWatchdog(s.chainView(d, s.chain[d]->src), s.nanDev,
                                  s.steps * 977, s.stream,
                                  /*flagBase=*/2 * (d + 1));
            }
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
    // batch just produced — on the FINE grid when the patch covers every
    // solid cell (2x wall resolution is the patch's whole purpose; the
    // coefficient normalization switches to the fine scaling in forces()).
    if (!s.forcePending) {
        DeviceForceAccumulator acc{s.force};
        const bool fineForces = s.fine.active && s.fine.forcesFromFine;
        const DeviceLatticeView forceView =
            fineForces ? s.fineView(s.fine.src) : s.view(s.src);
        // The reduction must see the same slip field the stream-collide of
        // its grid used, or the modeled wall stress drops out of Cd.
        const WallSlipView forceSlip =
            fineForces ? s.wmFine.slipView() : s.wmCoarse.slipView();
        if (launchForceReduction(forceView, acc, s.stream, forceSlip)
            == cudaSuccess) {
            cudaMemcpyAsync(s.hForce, s.force, 3 * sizeof(float),
                            cudaMemcpyDeviceToHost, s.stream);
            cudaEventRecord(s.evForce, s.stream);
            s.forcePending = true;
            s.stepsAtForceLaunch = s.steps;
        }
    }

    // One mass-drift diagnostic sample per batch, same poll-don't-stall shape
    // as the force reduction. Reads the COARSE post-collision buffer: total
    // fluid mass on the base grid is the global conserved quantity (the patch
    // restriction writes fine-derived moments back into it on render steps, so
    // a fine-level leak surfaces here too). Diagnostic only — see pollAsync.
    if (!s.massPending) {
        if (launchMassReduction(s.view(s.src), s.mass, s.stream)
            == cudaSuccess) {
            cudaMemcpyAsync(s.hMass, s.mass, 2 * sizeof(float),
                            cudaMemcpyDeviceToHost, s.stream);
            cudaEventRecord(s.evMass, s.stream);
            s.massPending = true;
        }
    }

    // One wall-model update per batch (u_tau is a slow quantity; per-step
    // updates would only feed the u_tau <-> u_w loop). Reads the buffer the
    // batch just produced; the refreshed slip field takes effect next batch.
    if (s.wmCoarse.active) {
        WallModelParams wmp;
        const float tauNow = s.effectiveTau();
        wmp.nuLat    = (tauNow - 0.5f) / 3.0f;
        wmp.utCutoff = 1e-3f * s.scaling.u_lat;
        launchWallModelUpdate(s.wmCoarse.listView(), s.view(s.src),
                              s.wmCoarse.dUwx, s.wmCoarse.dUwy, s.wmCoarse.dUwz,
                              wmp, s.wmCoarse.dStats, s.stream);
    }
    // Every cascade rung's wall model, in depth order. The rung's viscosity is
    // the coarse tau composed through each 2x hop down to it (fineTauFor
    // telescopes, so folding factor-by-factor down the chain is exact). The
    // mirror lives in the named members for depth 0/1 and wmExtra beyond.
    {
        float tau = s.effectiveTau();
        for (int d = 0; d < static_cast<int>(s.chain.size()); ++d) {
            tau = fineTauFor(tau, s.chain[d]->factor); // compose down this hop
            Impl::WMLevel& wm = (d == 0) ? s.wmFine
                              : (d == 1) ? s.wmFiner
                                         : s.wmExtra[d - 2];
            if (!wm.active) continue;
            WallModelParams wmp;
            wmp.nuLat    = (tau - 0.5f) / 3.0f;
            wmp.utCutoff = 1e-3f * s.scaling.u_lat;
            launchWallModelUpdate(wm.listView(), s.chainView(d, s.chain[d]->src),
                                  wm.dUwx, wm.dUwy, wm.dUwz, wmp, wm.dStats,
                                  s.stream);
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
    // When the momentum exchange runs on the FINE grid (plan M-refine), the
    // raw lattice force is in fine units: normalize with the fine scaling and
    // the fine span (coefficients are dimensionless per level, so this is the
    // entire conversion).
    const bool fineForces = s.fine.active && s.fine.forcesFromFine;
    const LatticeScaling& fscale = fineForces ? s.fine.scaling : s.scaling;
    const int spanCells = fineForces
        ? s.fine.dims.nz
        : (s.zSlipWalls ? std::max(1, s.dims.nz - 2) : s.dims.nz);
    r.cd = fscale.forceToCoefficient(s.emaFx, spanCells);
    r.cl = fscale.forceToCoefficient(s.emaFy, spanCells);
    r.liftToDrag = (std::fabs(r.cd) > 1e-6f) ? r.cl / r.cd : 0.0f;
    r.valid = s.emaSeeded && (s.steps >= s.gateOpenAtSteps);

    // Overall average Cl/Cd/L/D: collect every ring sample after the first
    // 1.0 flow-through (the startup transient is discarded), then compute
    // mean, min, max, and median over the surviving set.  The ring is cleared
    // on every cold start, so all entries belong to the current run.
    if (r.valid && s.avgRingCount > 0) {
        // Samples at step < 1 FT are pure startup transient — skip them.
        const long long oneFTSteps =
            static_cast<long long>(s.flowThroughSteps());

        // Collect converted coefficients into temporary arrays for statistics.
        std::vector<float> clVec, cdVec, ldVec;
        clVec.reserve(static_cast<std::size_t>(s.avgRingCount));
        cdVec.reserve(static_cast<std::size_t>(s.avgRingCount));
        ldVec.reserve(static_cast<std::size_t>(s.avgRingCount));

        // Walk the entire valid portion of the ring (newest to oldest) so we
        // capture ALL post-1FT samples, not just the most recent window.
        for (int i = 0; i < s.avgRingCount; ++i) {
            const int idx = (s.avgRingHead - 1 - i + Impl::kAvgRingSize)
                            % Impl::kAvgRingSize;
            const Impl::ForceSample& smp = s.avgRing[idx];

            // Discard samples from the first flow-through (startup transient).
            if (smp.step < oneFTSteps) continue;

            const float clSmp = fscale.forceToCoefficient(smp.fy, spanCells);
            const float cdSmp = fscale.forceToCoefficient(smp.fx, spanCells);
            const float ldSmp = (std::fabs(cdSmp) > 1e-6f)
                                    ? clSmp / cdSmp : 0.0f;
            clVec.push_back(clSmp);
            cdVec.push_back(cdSmp);
            ldVec.push_back(ldSmp);
        }

        if (!clVec.empty()) {
            // --- Mean ---
            float sumCl = 0.0f, sumCd = 0.0f, sumLd = 0.0f;
            for (std::size_t k = 0; k < clVec.size(); ++k) {
                sumCl += clVec[k];
                sumCd += cdVec[k];
                sumLd += ldVec[k];
            }
            const float n  = static_cast<float>(clVec.size());
            r.clAvg = sumCl / n;
            r.cdAvg = sumCd / n;
            r.ldAvg = sumLd / n;

            // --- Min / Max ---
            r.clMin = *std::min_element(clVec.begin(), clVec.end());
            r.clMax = *std::max_element(clVec.begin(), clVec.end());
            r.cdMin = *std::min_element(cdVec.begin(), cdVec.end());
            r.cdMax = *std::max_element(cdVec.begin(), cdVec.end());
            r.ldMin = *std::min_element(ldVec.begin(), ldVec.end());
            r.ldMax = *std::max_element(ldVec.begin(), ldVec.end());

            // --- Median (sort copies in-place; data is small, ~tens of items) ---
            auto medianOf = [](std::vector<float> v) -> float {
                std::sort(v.begin(), v.end());
                const std::size_t m = v.size() / 2;
                return (v.size() % 2 == 0)
                           ? 0.5f * (v[m - 1] + v[m])
                           : v[m];
            };
            r.clMedian = medianOf(clVec);
            r.cdMedian = medianOf(cdVec);
            r.ldMedian = medianOf(ldVec);
        }
    }

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

bool LBMSolver::downloadCrossflowPlane(int ix, std::vector<float>& v,
                                       std::vector<float>& w) const {
    const Impl& s = *impl_;
    if (!s.initialized || ix < 0 || ix >= s.dims.nx) return false;
    const std::size_t count = static_cast<std::size_t>(s.dims.ny)
                            * static_cast<std::size_t>(s.dims.nz);
    v.resize(count);
    w.resize(count);
    // One strided D2H copy per component: the plane at fixed x is a column
    // of ny*nz elements spaced nx floats apart in the unpadded macro arrays.
    const std::size_t spitch = static_cast<std::size_t>(s.dims.nx)
                             * sizeof(float);
    struct { const float* src; float* dst; } planes[] = {
        {s.v + ix, v.data()}, {s.w + ix, w.data()}};
    for (const auto& p : planes) {
        if (cudaMemcpy2DAsync(p.dst, sizeof(float), p.src, spitch,
                              sizeof(float), count, cudaMemcpyDeviceToHost,
                              s.stream) != cudaSuccess)
            return false;
    }
    return cudaStreamSynchronize(s.stream) == cudaSuccess;
}

int LBMSolver::suctionSurfaceY(int ix) const {
    const Impl& s = *impl_;
    if (!s.initialized || !s.surfValid || ix < 0 || ix >= s.dims.nx) return -1;
    return s.topSolidY[static_cast<std::size_t>(ix)];
}

int LBMSolver::latticeXForChordStation(float xc) const {
    const Impl& s = *impl_;
    if (!s.initialized || !s.surfValid) return -1;
    const float span = static_cast<float>(s.xTE - s.xLE);
    return std::clamp(
        static_cast<int>(std::lround(static_cast<float>(s.xLE) + xc * span)),
        0, s.dims.nx - 1);
}

bool LBMSolver::nanDetected() const { return impl_->nanLatched; }

std::string LBMSolver::nanDiagnosis() const {
    const Impl& s = *impl_;
    if (!s.nanLatched) return {};
    char buf[512];
    // Verdict decode: 1/2 = coarse grid, 3/4 = the 2x refinement patch,
    // 5/6 = the nested VG patch (watchdog flagBase contract); odd = NaN/Inf,
    // even = velocity runaway.
    const bool onFiner  = s.nanTripKind >= 5;
    const bool onFine   = s.nanTripKind >= 3 && !onFiner;
    const bool runaway  = (s.nanTripKind % 2) == 0;
    const char* level   = onFiner ? " on the nested VG refinement patch"
                        : onFine  ? " on the 2x refinement patch"
                                  : "";
    const bool onPatch  = onFine || onFiner;
    // The two usual suspects (plan 4.5): lattice Mach too high, or the run
    // is pinned at the tau stability clamp (target Re unreachable).
    std::snprintf(buf, sizeof buf,
        "%s%s at step %lld. u_lat = %.4f (cap %.2f)%s; tau at trip = "
        "%.5f (clamp %.4f)%s. Likely cause: %s. Try a lower airspeed, a finer "
        "chord resolution, or the Fast preset%s.",
        runaway ? "Velocity runaway (field saturated at the stability limiter)"
                : "NaN detected",
        level,
        s.steps, s.scaling.u_lat, kMaxULat,
        (s.scaling.u_lat > 0.1f ? " — HIGH" : ""),
        s.tauAtTrip, kMinTau,
        (s.scaling.tauClamped ? " — CLAMPED" : ""),
        s.scaling.u_lat > 0.1f
            ? "lattice velocity too close to the stability cap"
            : (s.scaling.tauClamped
                   ? "viscosity pinned at the stability clamp"
                   : "a transient too sharp for the current resolution"),
        onPatch ? "; disabling the refinement patch also isolates the issue"
                : "");
    return buf;
}

SolverPerfStats LBMSolver::perfStats() const {
    const Impl& s = *impl_;
    SolverPerfStats p;
    p.lastStepsPerFrame = s.lastChosenN;
    p.lastStepMs = s.lastStepMs;
    if (s.lastStepMs > 0.0) {
        // Cell updates per coarse step: every coarse cell once, every fine
        // cell m times (m sub-steps of dt/m), and every finer cell m*m2 times
        // (m2 finer sub-steps inside each of the m fine sub-steps).
        double updates = static_cast<double>(s.ncells);
        if (s.fine.active)
            updates += static_cast<double>(s.fine.factor)
                     * static_cast<double>(s.fine.ncells);
        if (s.finer.active)
            updates += static_cast<double>(s.fine.factor)
                     * static_cast<double>(s.finer.factor)
                     * static_cast<double>(s.finer.ncells);
        p.mlups = updates / (s.lastStepMs * 1000.0);
    }
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

// ===========================================================================
// Two-level refinement patch (plan M-refine).
// ===========================================================================

bool LBMSolver::initRefinement(const PatchBox& box, int factor,
                               const std::vector<std::uint8_t>& fineFlags,
                               std::string* error) {
    Impl& s = *impl_;
    if (!s.initialized) {
        if (error) *error = "solver not initialized";
        return false;
    }
    // Refinement levels and ISLBM are mutually exclusive: the stretched mesh IS
    // the (continuous) refinement, so a discrete patch on top is meaningless.
    if (s.stretch.active) {
        if (error) *error = "ISLBM stretched mode active (no discrete patches)";
        return false;
    }
    shutdownRefinement(); // replace any previous fine level

    if (factor < 2 || factor > kMaxRefineFactor) {
        if (error) *error = "refinement factor out of range (2..4)";
        return false;
    }
    if (!box.valid() || box.x0 < 1 || box.y0 < 1 || box.x1 > s.dims.nx - 1
        || box.y1 > s.dims.ny - 1) {
        if (error) *error = "refinement patch box out of range";
        return false;
    }

    Impl::FineLevel& fl = s.fine;
    fl.factor = factor;
    fl.box  = box;
    fl.dims = fineDimsFor(box, s.dims, factor);
    fl.ncells    = fl.dims.cellCount();
    fl.nxny      = static_cast<long long>(fl.dims.nx) * fl.dims.ny;
    fl.ncellsPad = fl.dims.paddedCellCount();
    fl.scaling   = refinedScaling(s.scaling, factor);
    fl.src = 0;

    if (fineFlags.size() != static_cast<std::size_t>(fl.ncells)) {
        if (error) *error = "fine flag field size does not match the patch";
        fl = Impl::FineLevel{};
        return false;
    }

    auto fail = [&](const char* what, cudaError_t err) {
        if (error) *error = std::string(what) + ": " + cudaGetErrorString(err);
        s.freeFine();
        return false;
    };

    // Fine f pair + padded flags. OOM here leaves the coarse sim untouched —
    // the caller reports it and the run continues unrefined.
    const std::size_t fBytes =
        static_cast<std::size_t>(kQ) * fl.ncellsPad * sizeof(FPop);
    for (FPop*& p : fl.f) {
        if (auto err = cudaMalloc(&p, fBytes); err != cudaSuccess)
            return fail("fine f buffer allocation failed", err);
    }
    if (auto err = cudaMalloc(&fl.flags, static_cast<std::size_t>(fl.ncellsPad));
        err != cudaSuccess)
        return fail("fine flag allocation failed", err);

    // Upload flags one ghost plane in, then sync the flag ghosts (periodic z).
    if (auto err = cudaMemcpyAsync(fl.flags + fl.nxny, fineFlags.data(),
                                   static_cast<std::size_t>(fl.ncells),
                                   cudaMemcpyHostToDevice, s.stream);
        err != cudaSuccess)
        return fail("fine flag upload failed", err);
    if (auto err = launchRefreshGhostZFlags(fl.flags, fl.dims, s.stream);
        err != cudaSuccess)
        return fail("fine flag ghost refresh failed", err);

    fl.active = true;
    s.syncChain(); // fine is now chain[0]
    s.updateForcesFromFine();

    // Keep the host copy the fine wall-cell list rebuild scans, then build
    // that list (no-op while the wall model is off).
    s.fineHostFlags = fineFlags;
    s.rebuildWallModelFine();

    // Seed the fine state from the current coarse field (valid mid-run).
    if (auto err = s.seedFineFromCoarse(); err != cudaSuccess)
        return fail("fine level seeding failed", err);
    return true;
}

void LBMSolver::shutdownRefinement() {
    if (impl_) impl_->freeFine();
}

void LBMSolver::setRefinedFlags(const std::vector<std::uint8_t>& fineFlags) {
    Impl& s = *impl_;
    if (!s.initialized || !s.fine.active) return;
    if (fineFlags.size() != static_cast<std::size_t>(s.fine.ncells)) return;
    if (cudaMemcpyAsync(s.fine.flags + s.fine.nxny, fineFlags.data(),
                        static_cast<std::size_t>(s.fine.ncells),
                        cudaMemcpyHostToDevice, s.stream) != cudaSuccess)
        return;
    launchRefreshGhostZFlags(s.fine.flags, s.fine.dims, s.stream);
    s.updateForcesFromFine();
    // Fine geometry changed: refresh the wall list's host source and rebuild.
    s.fineHostFlags = fineFlags;
    s.rebuildWallModelFine();
    // Geometry changed: re-derive the whole fine state from the (freshly
    // cold-restarted) coarse field so no stale vane flow survives the edit.
    s.seedFineFromCoarse();
}

bool LBMSolver::initFinerRefinement(const PatchBox& finerBox, int m2,
                                    const std::vector<std::uint8_t>& finerFlags,
                                    std::string* error) {
    Impl& s = *impl_;
    if (!s.initialized) {
        if (error) *error = "solver not initialized";
        return false;
    }
    // The finer level couples to the fine grid: it cannot exist without one.
    if (!s.fine.active) {
        if (error) *error = "fine level not active (finer level nests in it)";
        return false;
    }
    shutdownFinerRefinement(); // replace any previous finer level

    if (m2 < 2 || m2 > kMaxRefineFactor) {
        if (error) *error = "finer factor out of range (2..4)";
        return false;
    }
    // The box is in FINE cells; clamp against the FINE dims, leaving the same
    // 1-cell guard initRefinement uses against the parent faces.
    if (!finerBox.valid() || finerBox.x0 < 1 || finerBox.y0 < 1
        || finerBox.x1 > s.fine.dims.nx - 1
        || finerBox.y1 > s.fine.dims.ny - 1) {
        if (error) *error = "finer patch box out of range";
        return false;
    }

    Impl::FineLevel& fr = s.finer;
    fr.factor    = m2;
    fr.box       = finerBox;
    fr.dims      = fineDimsFor(finerBox, s.fine.dims, m2); // full-z: m2*fine.nz
    fr.ncells    = fr.dims.cellCount();
    fr.nxny      = static_cast<long long>(fr.dims.nx) * fr.dims.ny;
    fr.ncellsPad = fr.dims.paddedCellCount();
    fr.scaling   = refinedScaling(s.fine.scaling, m2);
    fr.src = 0;
    // Forces stay on the fine grid (the finer box only covers the VG zone and
    // cannot produce total foil Cl/Cd) — never integrate forces here.
    fr.forcesFromFine = false;

    if (finerFlags.size() != static_cast<std::size_t>(fr.ncells)) {
        if (error) *error = "finer flag field size does not match the patch";
        fr = Impl::FineLevel{};
        return false;
    }

    auto fail = [&](const char* what, cudaError_t err) {
        if (error) *error = std::string(what) + ": " + cudaGetErrorString(err);
        s.freeFiner();
        return false;
    };

    // Finer f pair + padded flags. OOM here leaves the two-level sim untouched.
    const std::size_t fBytes =
        static_cast<std::size_t>(kQ) * fr.ncellsPad * sizeof(FPop);
    for (FPop*& p : fr.f) {
        if (auto err = cudaMalloc(&p, fBytes); err != cudaSuccess)
            return fail("finer f buffer allocation failed", err);
    }
    if (auto err = cudaMalloc(&fr.flags, static_cast<std::size_t>(fr.ncellsPad));
        err != cudaSuccess)
        return fail("finer flag allocation failed", err);

    if (auto err = cudaMemcpyAsync(fr.flags + fr.nxny, finerFlags.data(),
                                   static_cast<std::size_t>(fr.ncells),
                                   cudaMemcpyHostToDevice, s.stream);
        err != cudaSuccess)
        return fail("finer flag upload failed", err);
    if (auto err = launchRefreshGhostZFlags(fr.flags, fr.dims, s.stream);
        err != cudaSuccess)
        return fail("finer flag ghost refresh failed", err);

    fr.active = true;
    s.syncChain(); // finer is now chain[1]

    // Keep the host copy the finer wall-cell list rebuild scans, then build it.
    s.finerHostFlags = finerFlags;
    s.rebuildWallModelFiner();

    // Seed the finer state from the current fine field (valid mid-run).
    if (auto err = s.seedFinerFromFine(); err != cudaSuccess)
        return fail("finer level seeding failed", err);
    return true;
}

void LBMSolver::shutdownFinerRefinement() {
    if (impl_) impl_->freeFiner();
}

void LBMSolver::setRefinedFinerFlags(
    const std::vector<std::uint8_t>& finerFlags) {
    Impl& s = *impl_;
    if (!s.initialized || !s.finer.active) return;
    if (finerFlags.size() != static_cast<std::size_t>(s.finer.ncells)) return;
    if (cudaMemcpyAsync(s.finer.flags + s.finer.nxny, finerFlags.data(),
                        static_cast<std::size_t>(s.finer.ncells),
                        cudaMemcpyHostToDevice, s.stream) != cudaSuccess)
        return;
    launchRefreshGhostZFlags(s.finer.flags, s.finer.dims, s.stream);
    s.finerHostFlags = finerFlags;
    s.rebuildWallModelFiner();
    // Re-derive the finer state from the fine field (which the caller just
    // cold-restarted + re-seeded via setRefinedFlags).
    s.seedFinerFromFine();
}

void LBMSolver::setRefinedFinerSurfaceReference(
    const std::vector<std::uint8_t>& finerCleanFlags) {
    Impl& s = *impl_;
    if (!s.initialized || !s.finer.active) return;
    if (finerCleanFlags.size() != static_cast<std::size_t>(s.finer.ncells))
        return;
    s.finerSurfaceRefFlags = finerCleanFlags;
    s.rebuildWallModelFiner();
}

// ---------------------------------------------------------------------------
// N-level cascade: rungs at depth >= 2.
// ---------------------------------------------------------------------------

bool LBMSolver::appendCascadeLevel(const PatchBox& box, int m,
                                   const std::vector<std::uint8_t>& flags,
                                   const std::vector<std::uint8_t>& cleanFlags,
                                   std::string* error) {
    Impl& s = *impl_;
    if (!s.initialized) {
        if (error) *error = "solver not initialized";
        return false;
    }
    // A deep rung needs a parent: finer must be active (so the cascade already
    // has depth 2's parent). chain.back() is the deepest active rung = parent.
    if (!s.finer.active || s.chain.empty()) {
        if (error) *error = "finer level not active (deep rung has no parent)";
        return false;
    }
    if (m < 2 || m > kMaxRefineFactor) {
        if (error) *error = "cascade factor out of range (2..4)";
        return false;
    }
    Impl::FineLevel& parent = *s.chain.back();
    if (!box.valid() || box.x0 < 1 || box.y0 < 1
        || box.x1 > parent.dims.nx - 1 || box.y1 > parent.dims.ny - 1) {
        if (error) *error = "cascade patch box out of range for its parent";
        return false;
    }

    // Reserve the parallel slots for this new rung up front so the indices line
    // up (extra[i] <-> wmExtra[i] <-> qExtra[i] <-> extraHostFlags[i]).
    const int i = static_cast<int>(s.extraLevels.size());
    s.extraLevels.emplace_back();
    s.wmExtra.emplace_back();
    s.qExtra.emplace_back();
    s.extraHostFlags.emplace_back();
    s.extraSurfaceRefFlags.emplace_back();

    Impl::FineLevel& L = s.extraLevels[i];
    L.factor    = m;
    L.box       = box;
    L.dims      = fineDimsFor(box, parent.dims, m);
    L.ncells    = L.dims.cellCount();
    L.nxny      = static_cast<long long>(L.dims.nx) * L.dims.ny;
    L.ncellsPad = L.dims.paddedCellCount();
    L.scaling   = refinedScaling(parent.scaling, m);
    L.src = 0;
    L.forcesFromFine = false; // deep rungs cover only a sub-region, never forces

    auto fail = [&](const char* what, cudaError_t err) {
        if (error) *error = std::string(what) + ": " + cudaGetErrorString(err);
        s.freeExtraLevel(i);
        // Drop the just-added (now-empty) slots so the vectors stay in lockstep.
        s.extraLevels.pop_back(); s.wmExtra.pop_back(); s.qExtra.pop_back();
        s.extraHostFlags.pop_back(); s.extraSurfaceRefFlags.pop_back();
        s.syncChain();
        return false;
    };

    if (flags.size() != static_cast<std::size_t>(L.ncells)) {
        if (error) *error = "cascade flag field size does not match the patch";
        return fail("size mismatch", cudaErrorInvalidValue);
    }

    const std::size_t fBytes =
        static_cast<std::size_t>(kQ) * L.ncellsPad * sizeof(FPop);
    for (FPop*& p : L.f) {
        if (auto err = cudaMalloc(&p, fBytes); err != cudaSuccess)
            return fail("cascade f buffer allocation failed", err);
    }
    if (auto err = cudaMalloc(&L.flags, static_cast<std::size_t>(L.ncellsPad));
        err != cudaSuccess)
        return fail("cascade flag allocation failed", err);
    if (auto err = cudaMemcpyAsync(L.flags + L.nxny, flags.data(),
                                   static_cast<std::size_t>(L.ncells),
                                   cudaMemcpyHostToDevice, s.stream);
        err != cudaSuccess)
        return fail("cascade flag upload failed", err);
    if (auto err = launchRefreshGhostZFlags(L.flags, L.dims, s.stream);
        err != cudaSuccess)
        return fail("cascade flag ghost refresh failed", err);

    L.active = true;
    s.syncChain(); // the new rung is now chain.back()

    // Host flag copies + wall list, then seed from the parent's current field.
    s.extraHostFlags[i]       = flags;
    if (cleanFlags.size() == static_cast<std::size_t>(L.ncells))
        s.extraSurfaceRefFlags[i] = cleanFlags;
    s.rebuildWallModelExtra(i);
    if (auto err = s.seedExtraFromParent(i); err != cudaSuccess)
        return fail("cascade level seeding failed", err);
    return true;
}

void LBMSolver::shutdownDeepLevels() {
    if (!impl_) return;
    impl_->freeExtraLevels();
    impl_->syncChain();
}

int LBMSolver::cascadeDepth() const {
    return impl_ ? static_cast<int>(impl_->chain.size()) : 0;
}

void LBMSolver::setCascadeQLinks(int depth,
                                 const std::vector<std::uint8_t>& qFrac,
                                 const std::vector<std::uint32_t>& ffMask,
                                 int links, int fallback) {
    Impl& s = *impl_;
    if (!s.initialized || depth < 2) return;
    const int i = depth - 2;
    if (i < 0 || i >= static_cast<int>(s.extraLevels.size())) return;
    if (!s.extraLevels[i].active) return;
    s.uploadQLevel(s.qExtra[i], qFrac, ffMask, s.extraLevels[i].ncells, links,
                   fallback);
}

// ---------------------------------------------------------------------------
// ISLBM stretched-mesh mode.
// ---------------------------------------------------------------------------

bool LBMSolver::initStretchMode(const std::vector<float>& wallDist,
                                std::string* error) {
    Impl& s = *impl_;
    if (!s.initialized) {
        if (error) *error = "solver not initialized";
        return false;
    }
    // Mutually exclusive with the cascade: tear down every refinement level
    // first (the single stretched grid replaces them).
    s.freeFine();
    // Build + upload the stretched mesh from the wall-distance field. Keep the
    // host copy so setFlags/geometry edits can rebuild without re-deriving it.
    if (!buildStretchMesh(s.stretch, s.dims, s.scaling, wallDist, s.stream,
                          error)) {
        s.stretchWallDist.clear();
        return false; // graceful: mode stays uniform, coarse sim runs on
    }
    s.stretchWallDist = wallDist;
    return true;
}

void LBMSolver::shutdownStretchMode() {
    if (!impl_) return;
    impl_->stretch.free();
    impl_->stretchWallDist.clear();
}

bool LBMSolver::stretchActive() const {
    return impl_ && impl_->stretch.active;
}

StretchInfo LBMSolver::stretchInfo() const {
    StretchInfo info;
    const Impl& s = *impl_;
    if (!s.stretch.active) return info;
    info.active          = true;
    info.dxMin           = s.stretch.dxMin;
    info.dxMax           = s.stretch.dxMax;
    info.growthX         = s.stretch.growthX;
    info.growthY         = s.stretch.growthY;
    info.tauWall         = s.stretch.tauWall;
    info.tauFar          = s.stretch.tauFar;
    info.tauFloorClamped = s.stretch.tauFloorClamped;
    info.fluidCellSaving = s.stretch.fluidCellSaving;
    // Foot LUTs (2*nx + 2*ny entries of float+int8) + the per-cell tau field.
    info.vramBytes =
        static_cast<double>(2 * s.dims.nx + 2 * s.dims.ny) * (sizeof(float) + 1)
        + static_cast<double>(s.ncells) * sizeof(float);
    return info;
}

RefinementInfo LBMSolver::refinementInfo() const {
    const Impl& s = *impl_;
    RefinementInfo info;
    if (!s.fine.active) return info;
    info.active         = true;
    info.factor         = s.fine.factor;
    info.box            = s.fine.box;
    info.fineDims       = s.fine.dims;
    info.fineScaling    = s.fine.scaling;
    info.forcesFromFine = s.fine.forcesFromFine;
    info.vramBytes =
        2.0 * static_cast<double>(kQ) * static_cast<double>(s.fine.ncellsPad)
            * sizeof(FPop)
        + static_cast<double>(s.fine.ncellsPad);

    // Nested VG patch (third level), when present.
    if (s.finer.active) {
        info.finerActive  = true;
        info.finerFactor  = s.finer.factor;
        info.finerBox     = s.finer.box;
        info.finerDims    = s.finer.dims;
        info.finerScaling = s.finer.scaling;
        info.finerVramBytes =
            2.0 * static_cast<double>(kQ)
                * static_cast<double>(s.finer.ncellsPad) * sizeof(FPop)
            + static_cast<double>(s.finer.ncellsPad);
    }

    // Cascade rungs at depth >= 2 (graded refinement). Walk the chain so the
    // effective (cumulative-vs-coarse) factor accumulates down the staircase.
    auto rungVram = [](long long ncellsPad) {
        return 2.0 * static_cast<double>(kQ) * static_cast<double>(ncellsPad)
                   * sizeof(FPop)
             + static_cast<double>(ncellsPad);
    };
    info.cascadeDepth   = static_cast<int>(s.chain.size());
    info.totalVramBytes = info.vramBytes + info.finerVramBytes;
    int effective = s.fine.active ? s.fine.factor : 1;
    if (s.finer.active) effective *= s.finer.factor;
    for (std::size_t i = 0; i < s.extraLevels.size(); ++i) {
        const Impl::FineLevel& L = s.extraLevels[i];
        if (!L.active) break; // active prefix is contiguous (nesting invariant)
        effective *= L.factor;
        RefinementInfo::LevelInfo li;
        li.factor          = L.factor;
        li.effectiveFactor = effective;
        li.box             = L.box;
        li.dims            = L.dims;
        li.vramBytes       = rungVram(L.ncellsPad);
        info.totalVramBytes += li.vramBytes;
        info.deepLevels.push_back(li);
    }
    return info;
}

float LBMSolver::massDrift() const {
    const Impl& s = *impl_;
    // Undefined until the baseline is latched (first post-ramp sample) and
    // until a positive baseline exists — report a healthy 0 in the interim.
    if (!s.massBaselineSet || s.meanMassBaseline <= 0.0f) return 0.0f;
    return s.meanMassNow / s.meanMassBaseline - 1.0f;
}

bool LBMSolver::seedFromCoarse(const LBMSolver& presolver, std::string* error) {
    Impl& s = *impl_;
    const Impl& p = *presolver.impl_;
    if (!s.initialized || !p.initialized) {
        if (error) *error = "both solvers must be initialized";
        return false;
    }
    auto fail = [&](const char* what, cudaError_t err) {
        if (error) *error = std::string(what) + ": " + cudaGetErrorString(err);
        return false;
    };
    // Trilinear upsample of the presolver's macroscopic field straight into
    // this solver's device arrays (same context — no host round trip), then
    // the standard compact-restore path: equilibrium re-init + bookkeeping.
    if (auto err = launchUpsampleMacro(p.rho, p.u, p.v, p.w, p.dims,
                                       s.rho, s.u, s.v, s.w, s.dims, s.stream);
        err != cudaSuccess)
        return fail("macroscopic upsample failed", err);
    for (int i = 0; i < 2; ++i) {
        if (auto err = launchInitFromMacroscopic(s.view(i), s.rho, s.u, s.v,
                                                 s.w, s.stream);
            err != cudaSuccess)
            return fail("equilibrium re-init failed", err);
    }
    notifySnapshotRestored(false);
    // CRITICAL difference from a same-grid snapshot restore: an upsampled
    // 4x-coarser field carries interpolation roughness a tau-clamped grid
    // cannot absorb cold (observed: velocity runaway at ~0.36 flow-throughs
    // in deep stall). Re-arm the startup viscosity ramp over the seeded
    // field — 2*nx steps of elevated viscosity is cheap insurance, and the
    // fine level follows automatically through fineTauFor(effectiveTau()).
    s.rampActive    = true;
    s.rampStartStep = s.steps;
    // The fine level re-derives from the freshly seeded coarse field, then the
    // finer level re-derives from the fine field (child after parent).
    if (auto err = s.seedFineFromCoarse(); err != cudaSuccess)
        return fail("fine level re-seed failed", err);
    if (auto err = s.seedFinerFromFine(); err != cudaSuccess)
        return fail("finer level re-seed failed", err);
    return true;
}

void LBMSolver::setForceEmaWindow(float flowThroughs) {
    impl_->emaWindowFlowThroughs = std::max(0.05f, flowThroughs);
}

void LBMSolver::setStartupRampEnabled(bool enabled) {
    impl_->rampEnabled = enabled;
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
