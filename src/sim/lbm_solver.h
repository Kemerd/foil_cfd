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

    /// Overall mean Cl/Cd/L/D accumulated after the first 1.0 flow-through,
    /// with the noisy startup transient discarded.  Only valid when valid==true.
    float clAvg    = 0.0f;
    float clMin    = 0.0f;
    float clMax    = 0.0f;
    float clMedian = 0.0f;

    float cdAvg    = 0.0f;
    float cdMin    = 0.0f;
    float cdMax    = 0.0f;
    float cdMedian = 0.0f;

    float ldAvg    = 0.0f;   ///< L/D derived from the averaged samples.
    float ldMin    = 0.0f;
    float ldMax    = 0.0f;
    float ldMedian = 0.0f;
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

/// @brief Live status of the iMEM slip-velocity wall function for the UI
/// sim panel: how many surface cells are wall-modeled, where the sampled y+
/// sits (the honesty number — y+ far above ~300 means even the log layer is
/// marginal), and how often the slip safety clamp engaged (a healthy run
/// clamps essentially never; a large fraction means the model is fighting
/// the resolved flow and its output should not be trusted).
struct WallModelReadout {
    bool  enabled   = false; ///< Wall model switched on by the app.
    bool  fromFine  = false; ///< Stats below describe the fine level.
    int   cells     = 0;     ///< Wall cells carrying a live u_tau last update.
    float meanYplus = 0.0f;  ///< Mean sampled y+ across those cells.
    float maxYplus  = 0.0f;  ///< Max sampled y+.
    float clampedFrac = 0.0f;///< Fraction of cells whose slip hit the clamp.
    int   excludedVG  = 0;   ///< Build: cells dropped for touching VG vanes.
    int   degenerate  = 0;   ///< Build: cells dropped for unresolvable normals.
};

/// @brief Live status of the interpolated bounce-back (q-LIBB) sub-cell vane
/// surfaces, for the UI. Reports how many vane links resolve to a true
/// sub-cell cut versus how many fell back to plain half-way bounce-back (thin
/// vanes / out-of-range cuts), summed across the active levels.
struct QLIBBReadout {
    bool enabled  = false; ///< q-LIBB switched on by the app.
    int  links    = 0;     ///< Vane links carrying an analytic cut fraction.
    int  fallback = 0;     ///< Vane links left at half-way (thin/clamped).
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

    // ---- nested VG patch (third level): tiny box at 2x the fine factor,
    // hugging the VG vanes. Active only when VGs exist and the fine patch is on.
    bool           finerActive = false;  ///< Finer level allocated and stepping.
    int            finerFactor = 0;      ///< Factor m2 relative to the FINE
                                         ///< grid (effective vs coarse = m*m2).
    PatchBox       finerBox;             ///< Nested box, in FINE cells.
    GridDims       finerDims;            ///< Finer grid dimensions.
    LatticeScaling finerScaling;         ///< refinedScaling(fineScaling, m2).
    double         finerVramBytes = 0.0; ///< Finer f-pair + flag allocation.

    // ---- N-level cascade: rungs at depth >= 2 (graded refinement). The fine
    // (depth 0) and finer (depth 1) fields above stay populated as before; this
    // vector carries any deeper rungs of the staircase for the UI readout.
    struct LevelInfo {
        int            factor          = 0;   ///< Factor vs the parent rung.
        int            effectiveFactor = 0;   ///< Cumulative factor vs coarse.
        PatchBox       box;                   ///< Box in the parent rung's cells.
        GridDims       dims;                  ///< This rung's grid dimensions.
        double         vramBytes        = 0.0;///< f-pair + flag allocation.
    };
    std::vector<LevelInfo> deepLevels;        ///< Rungs at cascade depth >= 2.
    int    cascadeDepth   = 0;                ///< Total active rungs (1=fine,...).
    double totalVramBytes = 0.0;              ///< Sum over all refinement rungs.
};

/// @brief ISLBM stretched-mesh status for the UI Mesh panel (zeroed when off).
struct StretchInfo {
    bool   active   = false;  ///< Stretched mesh allocated and stepping.
    float  dxMin    = 0.0f;   ///< Finest spacing (wall) [m].
    float  dxMax    = 0.0f;   ///< Coarsest spacing (far field) [m].
    float  growthX  = 1.0f;   ///< Achieved per-cell growth ratio, X.
    float  growthY  = 1.0f;   ///< Achieved per-cell growth ratio, Y.
    float  tauWall  = 0.0f;   ///< tau at the finest cell.
    float  tauFar   = 0.0f;   ///< tau at the coarsest cell (>= kMinTau).
    bool   tauFloorClamped = false; ///< dxMax reduced to keep tauFar valid.
    double fluidCellSaving = 0.0;   ///< Fraction of cells coarser than the wall.
    double vramBytes = 0.0;   ///< Foot LUTs + per-cell tau field.
    float  nearWallFactor = 1.0f;   ///< Sub-base refinement k: the finest cell
                                    ///< is k-times finer than the base grid
                                    ///< (dxMin = base/k). 1 = legacy base-wall.
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

    // ------ nested VG patch (third level) ------

    /// @brief Allocate and seed the FINER level over the given box (in FINE
    /// cells) at factor @p m2 relative to the fine grid — a tiny patch hugging
    /// the VG vanes, where the finest wall treatment and vortex resolution
    /// live. Requires an active fine level (the finer level couples to it the
    /// same way the fine level couples to the coarse grid). Seeded from the
    /// current fine field, so this is valid at any point of a run. Replaces any
    /// previous finer level and fails gracefully on OOM — the two-level sim
    /// keeps running.
    /// @param finerBox   Nested box in FINE cells (deriveVGPatchBoxFine output).
    /// @param m2         Factor relative to the fine grid (typically 2).
    /// @param finerFlags Finer flag field, fineDimsFor(finerBox, fineDims, m2)
    ///                   .cellCount() bytes, with the Interface shell stamped.
    /// @param error      On failure, receives a human-readable reason.
    /// @return True on success.
    bool initFinerRefinement(const PatchBox& finerBox, int m2,
                             const std::vector<std::uint8_t>& finerFlags,
                             std::string* error);

    /// @brief Release the finer level (no-op when inactive). The fine and
    /// coarse sim continue unaffected.
    void shutdownFinerRefinement();

    /// @brief Replace the finer-level flag field (VG edit at fixed box) and
    /// re-seed the finer state from the fine field. Callers run setFlags() +
    /// setRefinedFlags() first.
    /// @param finerFlags New finer flag field (must match the active finer dims).
    void setRefinedFinerFlags(const std::vector<std::uint8_t>& finerFlags);

    /// @brief Finer-level analog of setRefinedSurfaceReference: the VG-free
    /// finer flag field the finer wall list derives foil normals and VG
    /// exclusion from. Call after initFinerRefinement()/setRefinedFinerFlags()
    /// whenever the clean finer geometry changes.
    /// @param finerCleanFlags Clean finer flag field (must match finer dims).
    void setRefinedFinerSurfaceReference(
        const std::vector<std::uint8_t>& finerCleanFlags);

    // ------ N-level cascade: rungs at depth >= 2 (graded refinement) ------
    // The fine (depth 0) and finer (depth 1) levels keep their dedicated API
    // above. Deeper rungs of the staircase are appended through these calls:
    // each is an integer-2x (or up to kMaxRefineFactor) refinement of the rung
    // one level shallower, coupled by the same level-agnostic kernels. A deep
    // rung requires every shallower rung to already be active (strict nesting).

    /// @brief Append a cascade rung at the next depth (>= 2): a patch over the
    /// given box, expressed in the PARENT rung's cells, at factor @p m relative
    /// to the parent. Requires the parent rung (depth-1) to be active. Seeded
    /// from the current parent field, so valid mid-run. Fails gracefully on OOM
    /// (the shallower cascade keeps running). The first call appends depth 2,
    /// the next depth 3, and so on; shutdownDeepLevels() clears them all.
    /// @param box       Patch box in the PARENT rung's cells.
    /// @param m         Factor relative to the parent (2..kMaxRefineFactor).
    /// @param flags     Rung flag field, fineDimsFor(box, parentDims, m)
    ///                  .cellCount() bytes, with the Interface shell stamped.
    /// @param cleanFlags VG-free reference flags (wall-model normals); same dims.
    /// @param error     On failure, receives a human-readable reason.
    /// @return True on success.
    bool appendCascadeLevel(const PatchBox& box, int m,
                            const std::vector<std::uint8_t>& flags,
                            const std::vector<std::uint8_t>& cleanFlags,
                            std::string* error);

    /// @brief Release every cascade rung at depth >= 2 (the fine/finer levels
    /// are untouched). Deepest first, preserving the nesting invariant.
    void shutdownDeepLevels();

    /// @brief Number of active refinement rungs (0 = coarse only, 1 = fine,
    /// 2 = fine+finer, 3+ = deeper cascade). The cascade depth for the UI.
    int cascadeDepth() const;

    /// @brief Upload the dense q-LIBB cut-fraction field for cascade rung
    /// @p depth (>= 2; depth 0/1 use setFineQLinks/setFinerQLinks). No-op when
    /// the rung is inactive or q-LIBB is disabled.
    void setCascadeQLinks(int depth, const std::vector<std::uint8_t>& qFrac,
                          const std::vector<std::uint32_t>& ffMask, int links,
                          int fallback);

    // ------ ISLBM stretched-mesh mode (continuous-gradient refinement) ------
    // A single smoothly-stretched grid replaces the discrete refinement levels:
    // the bulk runs the interpolated-gather kernel, a thin collar around walls
    // keeps the exact pull. Mutually exclusive with the cascade — enabling it
    // tears down every refinement level; initRefinement refuses while it is on.

    /// @brief Enable ISLBM mode: build the stretched mesh from @p wallDist (a
    /// buildWallDistanceField result over THIS grid) anchored on the base
    /// scaling, upload it, and route stepN through the gather kernel. Tears down
    /// any cascade first. Rebuild on every geometry/AoA/STL edit (the mesh
    /// tracks the wall distance). Fails gracefully on OOM (reverts to Uniform).
    /// @param wallDist Per-cell wall distance, dims().cellCount() floats.
    /// @param nearWallFactor Sub-base refinement k (>= 1; 1 = base-wall, the
    ///                 legacy behavior). k > 1 makes the finest near-wall cells
    ///                 k-times finer than the base grid (true local refinement).
    /// @param error    On failure, receives a human-readable reason.
    /// @return True on success (mode now ISLBM).
    bool initStretchMode(const std::vector<float>& wallDist,
                         float nearWallFactor, std::string* error);

    /// @brief Disable ISLBM mode and free the stretched mesh (reverts to a
    /// uniform grid). No-op when ISLBM is off.
    void shutdownStretchMode();

    /// @brief True when ISLBM stretched-mesh mode is active.
    bool stretchActive() const;

    /// @brief Toggle the ISLBM gather quality live (no mesh rebuild): true =
    /// FAST raw-population gather (near-uniform speed, qualitative forces, the
    /// interactive default); false = ACCURATE per-tap feq+rescaled-fneq gather
    /// (correct Cl/Cd, ~4-5x the gather cost). No-op when ISLBM is off.
    void setStretchFastGather(bool fast);

    // ------ virtual transition strip (boundary-layer trip) ------

    /// @brief Arm the virtual transition strip: upload a per-cell trip mask
    /// (nonzero = trip-band cell) and set the fluctuation intensity. Every step
    /// thereafter, launchTransitionTrip injects localized turbulent fluctuations
    /// into the flagged cells to force laminar->turbulent transition the wall
    /// model cannot sustain alone. The mask is typically a thin near-leading-
    /// edge band over the suction surface (built host-side from the geometry).
    /// Pass an empty mask or intensity <= 0 to disable. Survives stepping;
    /// rebuilt on geometry edits by the caller.
    /// @param tripMask  Per-cell mask, dims().cellCount() bytes (nonzero = trip).
    /// @param intensity Fluctuation amplitude in lattice units (~0.05*u_lat).
    /// @param error     On failure, receives a human-readable reason.
    /// @return True on success (or clean disable).
    bool setTransitionTrip(const std::vector<std::uint8_t>& tripMask,
                           float intensity, std::string* error = nullptr);

    /// @brief True when a transition strip is armed (mask uploaded, intensity>0).
    bool transitionTripActive() const;

    /// @brief Stretched-mesh status for the UI (zeroed when ISLBM is off):
    /// dx range, achieved growth, wall/far tau, and the fluid-cell saving.
    StretchInfo stretchInfo() const;

    /// @brief Device inputs the slice renderer needs to colormap the LOCAL CELL
    /// SIZE (grid-resolution view). The per-cell tau encodes dx via
    /// tau = 0.5 + 3*nuWall*(dxMin/dxEff)^2, so the renderer inverts it to
    /// dxEff/dxMin and maps it over [1, dxRatioMax]. tauField is null (and the
    /// field renders flat) when ISLBM is inactive.
    struct GridResolutionField {
        const float* tauField = nullptr; ///< Per-cell tau (device, ncells).
        float nuWall          = 0.0f;    ///< Wall (finest-cell) lattice viscosity.
        float dxRatioMax      = 1.0f;    ///< dxMax/dxMin (palette coarse end).
    };
    GridResolutionField gridResolutionField() const;

    /// @brief Refinement status for the UI (zeroed RefinementInfo when off).
    RefinementInfo refinementInfo() const;

    /// @brief Relative total-mass drift since the post-ramp baseline, a
    /// graded-refinement health diagnostic (2026-06-16). Returns
    /// meanFluidRho_now / meanFluidRho_baseline - 1: ~0 for a healthy run
    /// (exact streaming conserves mass to round-off), growing in magnitude as
    /// an interface/coupling instability leaks mass — it trips here before the
    /// strided NaN watchdog catches the blow-up. Diagnostic ONLY: the solver
    /// never rescales the field from this number. Returns 0 until the baseline
    /// is latched (first post-ramp sample).
    float massDrift() const;

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

    /// @brief Enable/disable the startup ramps (viscosity + inlet velocity) and
    /// the rest-initialization that goes with them. ON (default) is the app
    /// behavior: a fresh field starts at REST and the inlet eases up to u_lat,
    /// so no impulsive pressure shock rings off the body. OFF restores the
    /// legacy instant start (field initialized to full freestream, inlet at
    /// u_lat from step 1) — used by the steady-state interface/physics tests,
    /// which validate the converged coupling, not the startup transient, and
    /// would otherwise have to march many extra flow-throughs for the ramp to
    /// flush. Takes effect on the next reset()/setFlags().
    void setStartupRampEnabled(bool enabled);

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

    // ------ iMEM slip-velocity wall model (lbm_wallmodel.cuh) ------

    /// @brief Switch the wall-function boundary treatment on or off. ON
    /// builds the wall-cell lists for the coarse level (and the fine level
    /// when the patch is active) and allocates the slip fields; OFF frees
    /// everything and the solver runs the exact plain bounce-back kernels.
    /// Lists rebuild automatically on every flag change while enabled.
    /// The setting survives re-init (the app applies its Auto policy).
    void setWallModelEnabled(bool enabled);

    /// @brief Current wall-model switch state.
    bool wallModelEnabled() const;

    /// @brief Fine-level analog of setSurfaceReference: the VG-free fine
    /// flag field (buildFinePatchFlags WITHOUT VG stamping) the fine wall
    /// list derives foil normals and VG exclusion from. Call after
    /// initRefinement()/setRefinedFlags() whenever the clean fine geometry
    /// changes; without it the live fine flags serve as the reference and
    /// VG exclusion at the fine level degrades to "no exclusion".
    /// @param fineCleanFlags Clean fine flag field (must match fine dims).
    void setRefinedSurfaceReference(const std::vector<std::uint8_t>& fineCleanFlags);

    /// @brief Live wall-model diagnostics (zeroed struct when disabled).
    /// Performs a small synchronous device read — call at UI rate, not per
    /// frame.
    WallModelReadout wallModelReadout() const;

    // ------ interpolated bounce-back (q-LIBB) sub-cell vane surfaces ------

    /// @brief Switch sub-cell vane surfaces (interpolated bounce-back) on/off.
    /// ON makes the next setQLinks*() uploads take effect; OFF frees the q
    /// fields and every link reverts to plain half-way bounce-back. The setting
    /// survives re-init (the app re-applies it after a geometry rebuild).
    void setQLIBBEnabled(bool enabled);

    /// @brief Current q-LIBB switch state.
    bool qlibbEnabled() const;

    /// @brief Upload the densified q-LIBB cut-fraction field for a refinement
    /// level (the app builds it via geom: buildVaneQLinks + densifyQLinks). The
    /// solver owns the device mirror and frees it with the level. Coarse-level
    /// vanes are too under-resolved for q-LIBB, so in practice only the fine and
    /// nested levels are uploaded — but the API is uniform.
    /// @param qFrac    Dense per-cell-per-direction cut bytes (ncells*kQ).
    /// @param ffMask   Dense per-cell two-node-usable mask (ncells).
    /// @param links    Resolved link count (UI readout).
    /// @param fallback Thin-vane/clamped links left at half-way (UI readout).
    void setCoarseQLinks(const std::vector<std::uint8_t>& qFrac,
                         const std::vector<std::uint32_t>& ffMask,
                         int links, int fallback);
    void setFineQLinks(const std::vector<std::uint8_t>& qFrac,
                       const std::vector<std::uint32_t>& ffMask,
                       int links, int fallback);
    void setFinerQLinks(const std::vector<std::uint8_t>& qFrac,
                        const std::vector<std::uint32_t>& ffMask,
                        int links, int fallback);

    /// @brief Cell count of a level (for the app to size the dense q arrays):
    /// coarse = dims().cellCount(); fine/finer from refinementInfo(). Convenience
    /// so the app never duplicates the padded/unpadded bookkeeping.
    long long fineCellCount() const;
    long long finerCellCount() const;

    /// @brief Live q-LIBB diagnostics (zeroed when disabled).
    QLIBBReadout qlibbReadout() const;

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

    // ------ crossflow access for the VG vortex audit (geom/vg_audit.h) ------

    /// @brief Download the macroscopic (v, w) crossflow plane at lattice
    /// column ix into ny*nz host arrays (index y + ny*z). One small D2H
    /// strided copy + stream sync — UI-rate calls only, like the delta99
    /// extraction. In the refinement-patch overlap the coarse macro arrays
    /// already carry fine-derived moments (restriction contract), so the
    /// plane is patch-accurate for free.
    /// @return False before init or on a copy failure.
    bool downloadCrossflowPlane(int ix, std::vector<float>& v,
                                std::vector<float>& w) const;

    /// @brief Topmost solid row of the CLEAN suction surface at column ix
    /// (from the setSurfaceReference field, so vanes don't raise it), or -1
    /// when no surface exists there.
    int suctionSurfaceY(int ix) const;

    /// @brief Lattice x column for a chordwise station x/c, mapped through
    /// the projected solid extent (tracks the AoA-rotated foil exactly like
    /// extractSuctionDelta99 does), or -1 when no geometry is loaded.
    int latticeXForChordStation(float xc) const;

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

    /// @brief Pre-steps warmup progress for the status bar: the current
    /// pre-step index in [0, total) while the zero-wind super-viscous phase is
    /// running, with @p total set to the phase length. Returns -1 (and leaves
    /// @p total untouched) once the wind has been released / outside a ramp.
    long long preStepProgress(long long& total) const;

private:
    // Pimpl keeps cuda_runtime types and the 4 GB of buffer handles out of
    // every TU that includes the solver (most of the app does).
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace foilcfd
