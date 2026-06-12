// Host-side solver orchestrator: owns the ping-pong device buffers, runs the
// startup viscosity ramp, paces sim work against the TDR-safe frame budget,
// gates the force EMA until the flow is trustworthy, and extracts the
// suction-surface delta99 profile for the VG-guidance overlay (plan section 4).
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "lbm_core.cuh"
#include "lbm_refine.cuh"
#include "units.h"

namespace foilcfd {

/// Per-present sim budget in milliseconds (plan 4.5): never launch more step
/// work than this between presents, or Windows TDR may kill the context.
inline constexpr double kDefaultStepBudgetMs = 10.0;

/// Cold starts are untrustworthy for forces until the wake has flushed through
/// the domain twice (plan section 13: "don't trust first 2 flow-throughs").
inline constexpr float kForceGateFlowThroughs = 2.0f;

/// @brief Aerodynamic force readout (EMA-smoothed momentum-exchange result).
struct ForceReadout {
    float cl = 0.0f;          ///< Lift coefficient (EMA).
    float cd = 0.0f;          ///< Drag coefficient (EMA).
    float liftToDrag = 0.0f;  ///< L/D convenience ratio (0 when cd ~ 0).
    bool  valid = false;      ///< False until kForceGateFlowThroughs completed
                              ///< since the last cold start (UI greys readout).
    float flowThroughs = 0.0f;///< Flow-throughs completed since last cold start.

    /// Average Cl/Cd over the most recent 1.0 flow-through window.
    /// Only valid when valid == true; zero otherwise.
    float clAvg = 0.0f;
    float cdAvg = 0.0f;
};

/// @brief One station of the suction-surface boundary-layer profile used by
/// the Lin-2002 VG guidance overlay (Mission statement). delta99 is measured
/// wall-normal from the stair-step surface to where |u| reaches 99% of the
/// local edge velocity.
struct Delta99Sample {
    float x_c       = 0.0f;  ///< Chordwise station this sample was taken at.
    float delta99_c = 0.0f;  ///< Boundary-layer thickness in chord units.
    float ueEdge    = 0.0f;  ///< Local edge velocity (lattice units) — lets the
                             ///< guidance code spot separated stations (ue ~ 0).
    bool  valid     = false; ///< False if the wall-normal probe left the domain
                             ///< or the station sits inside separated reverse flow.
};

/// @brief Live performance counters for the UI sim panel (plan 9.2).
struct SolverPerfStats {
    int    lastStepsPerFrame = 0;   ///< N chosen by the adaptive pacer last frame.
    double lastStepMs        = 0.0; ///< Measured wall time per single step [ms].
    double mlups             = 0.0; ///< Million lattice updates per second
                                    ///< (counts coarse + 2x fine when the
                                    ///< refinement patch is active).
};

/// @brief Status of the two-level refinement patch (plan M-refine), for the
/// UI Mesh panel. All fields are derived at initRefinement() time.
struct RefinementInfo {
    bool           active = false;    ///< Fine level allocated and stepping.
    int            factor = 0;        ///< Refinement factor m (2..4); 0 = off.
    PatchBox       box;               ///< Patch in coarse cells.
    GridDims       fineDims;          ///< Fine grid dimensions.
    LatticeScaling fineScaling;       ///< Fine-level scaling (refinedScaling).
    double         vramBytes = 0.0;   ///< Fine f-pair + flag allocation.
    bool           forcesFromFine = false; ///< Momentum exchange runs on the
                                      ///< fine grid (true when the patch
                                      ///< covers every coarse solid cell).
};

/// @brief Host orchestrator for the D3Q19 TRT-Smagorinsky solver.
///
/// Lifecycle: construct -> init() -> [reset()] -> stepN() in the frame loop.
/// All geometry edits (airfoil, AoA, VG add/remove/move) flow through
/// setFlags() which always does a full cold restart.
/// All GPU work runs on the single stream passed to init (plan 9.3).
class LBMSolver {
public:
    LBMSolver();
    ~LBMSolver();

    // Owns ~4 GB of device memory at default grid — never copy.
    LBMSolver(const LBMSolver&) = delete;
    LBMSolver& operator=(const LBMSolver&) = delete;

    /// @brief Allocate device buffers and prime the field (cold start).
    /// Allocates: 2x full f (ping-pong), flags, macroscopic rho/u/v/w, force
    /// accumulator, watchdog flag. Fails gracefully (no throw) on OOM —
    /// the UI suggests a smaller preset.
    /// @param dims    Grid dimensions.
    /// @param scaling Unit scaling for this run (from computeScaling()).
    /// @param flags   Host flag field, dims.cellCount() bytes (CellFlag values).
    /// @param stream  The app's single CUDA stream.
    /// @param error   On failure, receives a human-readable reason.
    /// @return True on success.
    bool init(const GridDims& dims, const LatticeScaling& scaling,
              const std::vector<std::uint8_t>& flags, cudaStream_t stream,
              std::string* error);

    /// @brief Release all device memory (also runs from the destructor).
    void shutdown();

    /// @brief Cold restart: re-initialize to equilibrium inflow, re-apply the
    /// spanwise speck perturbation (fresh starts must break quasi-2D coherence
    /// — see launchSpanwisePerturbation), restart the viscosity ramp, zero the
    /// step counter, and invalidate the force EMA gate.
    void reset();

    /// @brief Replace the flag field and cold-restart.
    /// Used for all geometry changes: airfoil, AoA, and VG edits.
    /// @param flags New host flag field (must match init dims).
    void setFlags(const std::vector<std::uint8_t>& flags);

    /// @brief Internal warm-restart flag swap (kept for potential snapshot
    /// restore flows). External callers should use setFlags() for a clean cold
    /// restart; this path skips the viscosity ramp and may produce transients.
    /// @param flags New host flag field (clean foil mask OR'd with VG voxels).
    void applyEditedFlags(const std::vector<std::uint8_t>& flags);

    // ------ two-level refinement patch (plan M-refine) ------

    /// @brief Allocate and seed the fine level over the given patch box at
    /// the chosen refinement factor (2..kMaxRefineFactor; the fine grid runs
    /// `factor` sub-steps per coarse step at 1/factor the cell size).
    /// Call after init()/setFlags() with the matching fine flag field (from
    /// buildFinePatchFlags + VG stamping). The fine state is seeded from the
    /// current coarse field via the full-volume coarse-to-fine fill, so this
    /// is valid at any point of a run. Replaces any previous fine level.
    /// Fails gracefully on OOM — the coarse-only sim keeps running.
    /// @param box       Patch in coarse cells (derivePatchBox output).
    /// @param factor    Refinement factor m (2..kMaxRefineFactor).
    /// @param fineFlags Fine flag field, fineDimsFor(box, dims(), factor)
    ///                  .cellCount() bytes, with the Interface shell stamped.
    /// @param error     On failure, receives a human-readable reason.
    /// @return True on success.
    bool initRefinement(const PatchBox& box, int factor,
                        const std::vector<std::uint8_t>& fineFlags,
                        std::string* error);

    /// @brief Release the fine level (no-op when inactive). The coarse sim
    /// continues unaffected — the overlap region simply stops receiving
    /// fine-grid restrictions.
    void shutdownRefinement();

    /// @brief Replace the fine-level flag field (geometry/VG edits at fixed
    /// patch box) and re-seed the fine state from the coarse field. Callers
    /// run setFlags() (coarse, cold restart) first, then this.
    /// @param fineFlags New fine flag field (must match the active fine dims).
    void setRefinedFlags(const std::vector<std::uint8_t>& fineFlags);

    /// @brief Refinement status for the UI (zeroed RefinementInfo when off).
    RefinementInfo refinementInfo() const;

    /// @brief Mesh-sequencing seed (plan M-refine part 2): trilinearly
    /// upsample the @p presolver's macroscopic field onto this solver's grid,
    /// equilibrium re-init both ping-pong buffers from it, and run the
    /// compact-restore bookkeeping (no viscosity ramp, settle-transient force
    /// gate). When the refinement patch is active its state is re-seeded from
    /// the freshly restored coarse field. Both solvers must share the CUDA
    /// context/stream (they do — the app runs a single stream).
    /// @param presolver Converged coarse companion sim (any smaller grid).
    /// @param error     On failure, receives a human-readable reason.
    /// @return True on success.
    bool seedFromCoarse(const LBMSolver& presolver, std::string* error);

    /// @brief Provide the CLEAN-FOIL (VG-free) flag field the suction-surface
    /// extraction (extractSuctionDelta99 / separationOnsetXc) measures from.
    /// The live flags may carry VG voxels, and a vane crossing the mid-span
    /// plane would otherwise become the "surface": delta99 would be measured
    /// from the vane crest and the vane's own recirculation would register as
    /// a false separation onset — corrupting the Lin-2002 guidance precisely
    /// in VG-on configurations. Call after init()/setFlags() whenever the
    /// clean geometry changes; VG-only edits do not need a new reference.
    /// @param cleanFlags Clean-foil host flag field (must match init dims).
    void setSurfaceReference(const std::vector<std::uint8_t>& cleanFlags);

    /// @brief Run @p n fused stream-collide steps, swapping the ping-pong
    /// buffers each step. Applies the ramped tau (units.h rampedTau) while the
    /// startup ramp is active. Polls the NaN watchdog every 200 steps; on
    /// trip, stops early and latches nanDetected().
    /// @param n Step count (the adaptive pacer chooses this; see below).
    /// @return First CUDA error encountered, or cudaSuccess.
    cudaError_t stepN(int n);

    /// @brief Adaptive steps-per-frame pacer (plan 4.6/9.3): from the measured
    /// per-step time, pick N so total launch work stays under @p budgetMs
    /// (TDR safety) while filling as much of the frame as possible. Starts
    /// conservative (N=1) and converges over a few frames.
    /// @param budgetMs Per-present wall budget; default kDefaultStepBudgetMs.
    /// @return Recommended N for the next stepN() call (>= 1, or 0 if paused
    ///         by the NaN watchdog).
    int adaptiveStepsForBudget(double budgetMs = kDefaultStepBudgetMs);

    // ------ readouts ------

    /// @brief EMA-smoothed Cl/Cd/L-over-D plus 1-flow-through trailing averages
    /// (clAvg/cdAvg). `valid` stays false until kForceGateFlowThroughs
    /// flow-throughs have completed since the last cold start.
    /// EMA window comes from the active preset's forceEmaFlowThroughs (units.h).
    ForceReadout forces() const;

    /// @brief Extract the suction-surface boundary-layer thickness profile
    /// for the VG-guidance overlay. For each requested chordwise station this
    /// walks wall-normal from the upper stair-step surface through the
    /// macroscopic velocity field (downloaded once per call — call at UI rate,
    /// a few Hz, not per frame).
    /// @param stations_xc Chordwise stations (x/c in [0,1]) to sample at.
    /// @return One Delta99Sample per requested station, same order.
    std::vector<Delta99Sample> extractSuctionDelta99(
        const std::vector<float>& stations_xc) const;

    /// @brief Chordwise station of separation onset on the suction surface
    /// (first station where near-wall flow reverses), or < 0 when attached.
    /// Feeds vg.h recommendedStationBand (Lin 2002: VGs 5-10 h upstream).
    float separationOnsetXc() const;

    /// @brief True once the NaN watchdog has tripped; sim is paused.
    bool nanDetected() const;

    /// @brief Human-readable likely cause for the watchdog trip (plan 4.5):
    /// reports whether u_lat is over the cap and whether the tau clamp is
    /// active, since those are the usual suspects.
    std::string nanDiagnosis() const;

    /// @brief Live performance counters for the UI.
    SolverPerfStats perfStats() const;

    // ------ state access for rendering and snapshots ------

    /// @brief Velocity-field view over the macroscopic device arrays (valid
    /// after init; contents refresh on steps with StepParams::writeMacro).
    DeviceVelocityField velocityField() const;

    /// @brief Device macroscopic density array (cellCount floats), for slice
    /// rendering (pressure ~ cs^2 * (rho - 1)) and compact snapshots.
    const float* deviceRho() const;

    /// @brief Device flag field (cellCount bytes) — particle kernels need it
    /// for solid-entry respawn. NOTE: this pointer is ghost-OFFSET for the
    /// UNPADDED x + nx*(y + ny*z) convention; it is NOT the padded base that
    /// DeviceLatticeView::flags requires. External code building a lattice
    /// view (e.g. to call launchForceReduction) must use latticeView() — a
    /// view assembled from this pointer reads every flag one z-plane high.
    const std::uint8_t* deviceFlags() const;

    /// @brief Lattice view over the CURRENT source f buffer with the PADDED
    /// base pointers the lbm_core.cuh launch wrappers require (kernels index
    /// flags as flags[cell + nx*ny]). This is the supported way for external
    /// callers (tests, future tools) to run launchForceReduction or other
    /// core kernels against the live state.
    DeviceLatticeView latticeView() const;

    /// @brief Device pointer to the CURRENT source f buffer. The buffer is
    /// z-ghost PADDED: kQ * GridDims::paddedCellCount() FPop entries (frozen
    /// Krueger q ordering, per-q stride nx*ny*(nz+2), real cell (x,y,z) one
    /// nx*ny plane in — see the layout contract in lbm_core.cuh). Snapshot
    /// capture reads from here; snapshot restore writes here (ghosts included,
    /// so restores need no refresh). Treat the blob as opaque bytes unless you
    /// replicate the padded indexing exactly. After a restore, call
    /// notifySnapshotRestored() so internal gating resets.
    float* activeDeviceF();

    /// @brief Byte size of one full PADDED f buffer (for snapshot D2D copies):
    /// kQ * paddedCellCount() * sizeof(FPop).
    std::size_t fBufferBytes() const;

    /// @brief Inform the solver its f field was externally replaced.
    /// @param fullState True for an exact full-f restore (no transient);
    ///                  false for a compact equilibrium re-init (the solver
    ///                  schedules the short ~1000-step settling transient and
    ///                  keeps the force gate closed until it passes).
    void notifySnapshotRestored(bool fullState);

    /// @brief Compact-snapshot restore entry point (plan section 8): upload
    /// host macroscopic fields into the solver's device arrays, set both f
    /// buffers to equilibrium(rho, u), and run the notifySnapshotRestored
    /// (false) bookkeeping internally. Arrays are cellCount() floats each,
    /// unpadded layout x + nx*(y + ny*z).
    /// @param rho Host density array.
    /// @param u   Host x-velocity (lattice units).
    /// @param v   Host y-velocity.
    /// @param w   Host z-velocity.
    /// @param error On failure, receives a human-readable reason.
    /// @return True on success.
    bool restoreFromMacroscopic(const float* rho, const float* u,
                                const float* v, const float* w,
                                std::string* error);

    /// @brief Host-side copy of the current flag field (unpadded, cellCount()
    /// bytes). Snapshot capture hashes this; the UI may inspect it. Valid
    /// after init(); updated by setFlags().
    const std::vector<std::uint8_t>& hostFlags() const;

    /// @brief Set the force EMA window in flow-through times (plan 4.4 /
    /// HighFidelityPreset::forceEmaFlowThroughs). Default 1.0 (the standard
    /// preset); High Fidelity mode passes 8.0.
    void setForceEmaWindow(float flowThroughs);

    // ------ misc accessors ------

    /// @brief Grid dimensions this solver was initialized with.
    GridDims dims() const;

    /// @brief Unit scaling this solver was initialized with.
    const LatticeScaling& scaling() const;

    /// @brief Total steps since the last cold start.
    long long stepCount() const;

    /// @brief Flow-throughs completed since the last cold start.
    float flowThroughsCompleted() const;

    /// @brief tau in effect right now (shows the ramp progressing in the UI).
    float currentTau() const;

private:
    // Pimpl keeps cuda_runtime types and the 4 GB of buffer handles out of
    // every TU that includes the solver (most of the app does).
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace foilcfd
