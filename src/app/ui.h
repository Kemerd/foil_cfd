// ImGui panel layer (plan section 9.2). The UI never mutates the sim/render
// modules directly: it reads a UIContext of live readouts, edits the shared
// parameter structs, and raises UIEvents that main.cpp's frame loop applies
// in a defined order (voxelize -> snapshot -> solver). That keeps re-entrancy
// and edit ordering (slider release, warm vs cold restart) in ONE place.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <string>
#include <vector>

#include "../geom/airfoil.h"
#include "../geom/stl.h"
#include "../geom/vg.h"
#include "../geom/vg_audit.h"
#include "../render/viz.h"
#include "../sim/lbm_solver.h"
#include "../sim/units.h"
#include "aircraft_manifest.h"

struct GLFWwindow; // forward-declared; only init/shutdown touch the window.

namespace foilcfd {

/// @brief Which airfoil source the user has selected.
enum class AirfoilSource { NacaDigits, DatFile, StlImport };

/// @brief Display unit for the airspeed field. The solver always stores
/// airspeed in m/s (UIParams::airspeedMs); this only changes how the number
/// is shown and edited. Mph is the default — it matches the EAB POH/ASI
/// convention most homebuilders think in (knots stays one click away).
enum class SpeedUnit { Knots, Mph, Ms };

/// @brief Per-second conversion factor: multiply m/s by this to get the unit.
inline float speedUnitPerMs(SpeedUnit u) {
    switch (u) {
        case SpeedUnit::Knots: return 1.943844f; // 1 m/s = 1.943844 kn
        case SpeedUnit::Mph:   return 2.236936f; // 1 m/s = 2.236936 mph
        case SpeedUnit::Ms:    return 1.0f;
    }
    return 1.0f;
}

/// @brief Short label for the airspeed unit (slider suffix + button text).
inline const char* speedUnitLabel(SpeedUnit u) {
    switch (u) {
        case SpeedUnit::Knots: return "kn";
        case SpeedUnit::Mph:   return "mph";
        case SpeedUnit::Ms:    return "m/s";
    }
    return "m/s";
}

/// @brief State of the STL import modal (plan 7.2). main.cpp owns the heavy
/// StlMesh itself (triangle soups can be ~100 MB — they never pass through
/// the UI structs); the modal only displays metadata and edits the import
/// choices the user makes before confirming.
struct StlImportUI {
    bool open = false;            ///< Modal visible this frame.
    // -- read-only metadata main.cpp fills after loadStl() succeeds --
    std::string fileName;         ///< Filename shown in the modal title line.
    std::string solidName;        ///< StlMesh::name.
    std::uint32_t triangleCount = 0;
    Aabb bounds;                  ///< File-coordinate bounding box.
    bool wasBinary = false;       ///< Detected format (informational).
    // -- user choices the modal edits --
    StlAxisPreset axisPreset = StlAxisPreset::XYZ; ///< Axis remap preset.
    int  chordCells = 256;        ///< Longest x-extent maps to this many cells.
    bool zFreeSlipWalls = false;  ///< Plan 7.4 z-boundary mode: false =
                                  ///< spanwise-periodic (airfoil-like), true =
                                  ///< free-slip walls (full 3D object).
};

/// @brief Everything the panels EDIT. main.cpp owns one instance; the sim is
/// rebuilt from it when events ask for it. Defaults match plan section 4.6.
struct UIParams {
    // -- airfoil panel --
    AirfoilSource source = AirfoilSource::NacaDigits;
    std::string nacaDigits = "2412";       ///< NACA text box contents.
    int   selectedDatIndex = -1;           ///< Index into the scanned catalog.
    std::string airfoilFilter;             ///< Filter box over the ~1600-file catalog.
    std::string aircraftFilter;            ///< Aircraft search box (manufacturer+model).
    int   selectedAircraftIndex = -1;      ///< Manifest row highlighted in the
                                           ///< Aircraft section (anchors the
                                           ///< root/tip selector); reset by
                                           ///< main.cpp on manifest reload.
    float aoaDeg = 4.0f;                   ///< AoA slider, -5..25 deg (plan 9.2;
                                           ///< extended past 20 for post-stall
                                           ///< VG testing — see blockage note).
    float airspeedMs = 35.7632f;           ///< Physical airspeed [m/s] (canonical);
                                           ///< default = 80 mph, a typical EAB
                                           ///< approach/climb regime.
    SpeedUnit speedUnit = SpeedUnit::Mph;  ///< Display unit for the airspeed
                                           ///< field; m/s stays the stored value.
    float chordM = 1.2f;                   ///< Physical chord [m].
    ResolutionPreset resolution = ResolutionPreset::Fast; ///< Interactive 192
                                           ///< cells/chord out of the box; the
                                           ///< finer grids are one click away.

    // -- High Fidelity mode (Mission statement) --
    HighFidelityPreset highFidelity;       ///< .enabled is the UI toggle; the
                                           ///< panel wires .resolution to Ultra.

    // -- VG editor --
    std::vector<VGParams> vgs;             ///< Add/duplicate/delete list.
    int selectedVG = -1;                   ///< Entry the guidance panel anchors to.

    /// Allowed x/c range for VG placement (clamped when sliders move).
    /// Lets the user restrict VG stations to a sensible chord region, e.g.
    /// 5–40 % by default. Exposed as percentage inputs in the VG editor header.
    float vgXcMin = 0.01f;   ///< Lower bound, 0..1 (default 1 %).
    float vgXcMax = 0.70f;   ///< Upper bound, 0..1 (default 70 %).

    /// Warm-restart policy for VG edits. When true, add/edit/delete keeps the
    /// developed flow field and only patches the changed cells in place
    /// (LBMSolver::applyEditedFlags) — the force gate reopens after a short
    /// re-settle instead of a full from-scratch convergence. Off by default:
    /// the cold restart keeps force histories clean, which is the trustworthy
    /// baseline for VG-on/VG-off delta comparisons (the mission's readout).
    bool vgWarmRestart = false;

    // -- STL import (plan 7.2/7.4) --
    StlImportUI stlImport;                 ///< Modal state; mesh lives in main.cpp.

    // -- mesh panel: refinement patch + startup pre-convergence --
    /// Two-level refinement patch settings (plan M-refine). Margins are in
    /// chord fractions around the solid bbox; the patch derivation clamps
    /// them against the domain faces. Default 2x — the VRAM readout in the
    /// panel (and the graceful init-failure path) protects smaller cards.
    struct RefinementUIParams {
        int   factor    = 2;     ///< Patch resolution: 1 = off (uniform grid),
                                 ///< 2..4 = fine-level factor (cells shrink by
                                 ///< 1/m, cost grows ~m^4 — see the panel's
                                 ///< VRAM readout before going past 2x).
        bool  showPatchBox = true; ///< Draw the patch bounding box in the 3D
                                 ///< scene (projected line overlay).
        float upstreamC = 0.20f; ///< Margin ahead of the solid bbox [chords].
        float wakeC     = 0.50f; ///< Margin behind (near-wake coverage).
        float aboveC    = 0.20f; ///< Margin above (suction-surface BL + VGs).
        float belowC    = 0.10f; ///< Margin below (pressure side).
        bool  autoVGFactor = false; ///< Raise the WHOLE fine patch to whatever
                                 ///< recommendedRefineFactorForVGs() asks
                                 ///< (vane >= 8 cells tall). OFF by default now
                                 ///< the nested VG patch (finerVGPatch) handles
                                 ///< vane resolution locally — leaving this on
                                 ///< would stack on top of the nested 2x and
                                 ///< over-refine (e.g. fine auto-raised to 4x,
                                 ///< then nested 2x -> an 8x VG box).
        bool  finerVGPatch = true; ///< Build the nested 4x box hugging the VGs
                                 ///< (2x the fine factor) when VGs are on. Only
                                 ///< effective with VGs + an active fine patch;
                                 ///< degrades gracefully (OOM / tiny box) to the
                                 ///< 2x-only behavior.
        bool  qlibbVanes = true; ///< Interpolated bounce-back (q-LIBB): place
                                 ///< the vane wall at its TRUE sub-cell position
                                 ///< instead of stair-stepping to the voxel grid
                                 ///< (Bouzidi 2001). Effective only on the fine /
                                 ///< nested levels where vanes are resolved; thin
                                 ///< vanes fall back to plain bounce-back per link.

        /// Patch active at all (factor 1 = pure uniform grid).
        bool enabled() const { return factor >= 2; }
    };
    RefinementUIParams refine;

    /// Mesh-sequencing startup (plan M-refine part 2): cold starts first
    /// converge a 4x-coarser companion sim (~seconds), then upsample its
    /// macroscopic field onto the full grid. DISABLED by default and removed
    /// from the UI: the rest-init + inlet-velocity ramp starts runs cleanly
    /// without the seed (whose upsampled field could carry a wall kink that
    /// rings on the foil). The code path stays behind this flag, just dormant.
    bool preconvergeCoarse = false;

    // -- sim panel --
    bool running = true;                   ///< Run/pause.
    int  wallModel = 0;                    ///< iMEM wall-function policy:
                                           ///< 0 = Auto (on when the first
                                           ///< cell cannot resolve the viscous
                                           ///< sublayer, i.e. estimated
                                           ///< y+ > 2 on the finest level),
                                           ///< 1 = forced On, 2 = forced Off.

    // -- view panel --
    VizSettings viz;
    bool voxelView = false;                ///< Replace the smooth foil/VG mesh
                                           ///< with the SOLVER'S voxelization:
                                           ///< coarse stair-step cubes, plus
                                           ///< half-size cubes inside the
                                           ///< refinement patch — exactly the
                                           ///< walls the lattice bounces off.
    bool showVGGuidanceOverlay = true;     ///< Lin-2002 band overlay toggle.
    bool showVelocityLegend = true;        ///< Speed colorbar overlay at the
                                           ///< right edge of the render (on by
                                           ///< default; collapses to a small
                                           ///< re-show button when hidden).
};

/// @brief Read-only live numbers the panels DISPLAY each frame.
struct UIReadouts {
    // -- honesty numbers (plan 4.3: display BOTH) --
    LatticeScaling scaling;                ///< Source of reEffective/reTarget/tau/u_lat.
    GridDims dims;                         ///< Active grid (slice slider ranges).
    std::string loadedAirfoilName;         ///< Display name of the active geometry.

    // -- sim health/perf --
    SolverPerfStats perf;                  ///< steps/frame, MLUPS (plan 9.2).
    long long stepCount = 0;
    float flowThroughs = 0.0f;
    float currentTau = 0.0f;
    int    gpuUtilPercent = -1;            ///< Whole-GPU load [0..100]; -1 =
                                           ///< driver library unavailable.
    double simElapsedMs = 0.0;             ///< PHYSICAL time simulated since
                                           ///< the last cold start [ms]
                                           ///< (= steps * dt, not wall time).
    double etaSeconds = -1.0;              ///< Estimated wall seconds until
                                           ///< the force gate opens (readouts
                                           ///< converged); 0 = open, <0 = no
                                           ///< estimate yet (paused/cold).
    bool  nanTripped = false;
    std::string nanDiagnosis;              ///< Shown in the watchdog error popup.
    bool  cudaErrorTripped = false;        ///< Frame loop latched a CUDA failure
                                           ///< (TDR/launch error) — sim stopped.
    std::string cudaErrorDiagnosis;        ///< cudaGetErrorString + failing stage.
    float vramUsedFraction = 0.0f;         ///< Warn above 0.8 (plan 4.6).

    // -- forces (EMA, gated) --
    ForceReadout forces;                   ///< .valid=false renders greyed out
                                           ///< with "converging..." text.

    // -- VG guidance overlay inputs (Mission statement) --
    std::vector<Delta99Sample> delta99Profile; ///< Suction-surface BL profile.
    float separationXc = -1.0f;            ///< <0 = attached.
    GuidanceBand heightBand;               ///< Lin h-band at the selected station.
    GuidanceBand stationBand;              ///< Lin station band for current h.
    VGAuditReadout vgAudit;                ///< Vortex-strength honesty meter
                                           ///< (measured vs Wendt-predicted
                                           ///< circulation for the selected
                                           ///< VG; .valid false when off).

    // -- iMEM wall-function status (sim panel telemetry) --
    WallModelReadout wallModel;

    // -- refinement patch status (plan M-refine) --
    struct RefinementReadout {
        bool     active = false;       ///< Fine level allocated and stepping.
        int      factor = 0;           ///< Live refinement factor (0 = off).
        GridDims fineDims;             ///< Fine grid dimensions.
        double   fineVramGB = 0.0;     ///< Fine-level f-pair + flags estimate.
        float    fineReEffective = 0.0f; ///< Effective Re at the fine level.
        bool     forcesFromFine = false; ///< Momentum exchange runs on fine grid.
        int      patchX0 = 0, patchY0 = 0, patchX1 = 0, patchY1 = 0; ///< Coarse cells (diagram).

        // -- nested VG patch (third level) --
        bool     finerActive = false;  ///< Nested VG level allocated + stepping.
        int      finerFactor = 0;      ///< Factor vs the fine grid (m2).
        int      finerEffectiveFactor = 0; ///< Factor vs coarse (m * m2).
        GridDims finerDims;            ///< Finer grid dimensions.
        double   finerVramGB = 0.0;    ///< Finer-level f-pair + flags estimate.
        // Nested box corners in COARSE lattice cells (for the 3D overlay):
        // patch origin + finerBox(fine cells) / factor.
        float    finerCX0 = 0, finerCY0 = 0, finerCX1 = 0, finerCY1 = 0;

        // -- q-LIBB sub-cell vane surfaces --
        bool     qlibbActive   = false; ///< Interpolated bounce-back live.
        int      qlibbLinks    = 0;     ///< Vane links with a sub-cell cut.
        int      qlibbFallback = 0;     ///< Thin-vane links left at half-way.
    };
    RefinementReadout refine;

    // -- pre-convergence status (plan M-refine part 2) --
    float preconvergeProgress = -1.0f; ///< 0..1 while running; < 0 = idle.
};

/// @brief One-frame edge-triggered commands the panels raise; main.cpp
/// consumes (and clears) them after ImGui rendering. Bools, not callbacks,
/// so the apply order is explicit in the frame loop.
struct UIEvents {
    bool reloadAirfoil   = false; ///< Airfoil/NACA selection changed -> rebuild + cold start.
    bool aoaChanged      = false; ///< Slider RELEASED (plan 13: not per-tick) -> revoxel + cold start.
    bool airspeedChanged = false; ///< Airspeed OR chord edited: pure units
                                  ///< rescale (grid/u_lat/flags untouched),
                                  ///< restart ramp; cache stays valid (plan 13).
    bool resolutionChanged = false; ///< Preset/chord switch -> full re-init.
    bool vgEdited        = false; ///< VG list/params changed on release -> WARM restart (plan 6.2).
    bool resetCold       = false; ///< Explicit reset button.
    bool singleStep      = false; ///< Single-step button (pauses + steps once).
    bool refreshAirfoils = false; ///< Re-scan airfoils/ directory.
    bool loadStlRequested = false;///< "Load STL..." button -> open file dialog.
    bool stlImportConfirmed = false; ///< Modal "Import": voxelize the pending STL.
    bool stlImportCancelled = false; ///< Modal "Cancel": drop the pending STL.
    bool screenshot      = false; ///< Screenshot button -> Visualizer::screenshotPNG.
    bool voxelViewToggled = false; ///< Voxel-view checkbox -> rebuild + upload
                                   ///< the render mesh (voxel soup or smooth).
    bool highFidelityToggled     = false; ///< HiFi switch flipped -> re-init with preset.
    bool meshRefinementChanged   = false; ///< Zone weights / preset changed -> rebuild stretch + cold restart.
    bool particleCountChanged = false; ///< Pool slider released -> resizeParticlePool.
    bool frameFoilView   = false; ///< "Focus foil" -> camera.frameRegion on the foil.
    bool wallModelChanged = false; ///< Wall-model combo changed -> re-apply policy.

    /// @brief Clear all events (main.cpp calls after applying).
    void reset() { *this = UIEvents{}; }
};

/// @brief Bundle passed into drawUI every frame.
struct UIContext {
    UIParams*   params   = nullptr; ///< Edited in place by the panels.
    UIReadouts* readouts = nullptr; ///< Read-only display values.
    UIEvents*   events   = nullptr; ///< Raised by the panels this frame.
    const std::vector<AirfoilCatalogEntry>* airfoilCatalog = nullptr; ///< Dropdown contents.
    const std::vector<AircraftEntry>* aircraftManifest = nullptr; ///< Plan 15.5
                                    ///< Aircraft section rows (resolved +
                                    ///< catalog-linked); null/empty hides it.
    std::string statusMessage;      ///< Transient status line (load errors etc.).
    const Mat4f* viewProj = nullptr; ///< Camera view-projection for world-space
                                    ///< line overlays (the patch bounding box);
                                    ///< null hides them. main.cpp refreshes it
                                    ///< per frame from the orbit camera.
};

/// @brief Create the ImGui context (docking branch), install the GLFW +
/// OpenGL3 backends, and apply the FoilCFD style. Call once after the GL
/// context is current.
/// @return False if backend init failed.
bool uiInit(GLFWwindow* window);

/// @brief Tear down ImGui and its backends.
void uiShutdown();

/// @brief Begin a new ImGui frame (call before drawUI, after glfwPollEvents).
void uiBeginFrame();

/// @brief Draw every panel (plan 9.2): Airfoil (filtered catalog list + NACA
/// box + the plan-15.5 searchable Aircraft section that loads a plane's
/// airfoil through the catalog path + Load STL + AoA/airspeed/resolution +
/// High Fidelity), VG editor (list
/// with section 6.1 params, add/duplicate/delete, under-resolution warnings),
/// VG Guidance (sim-derived delta99 at the selected station, the Lin-2002
/// recommended h band, Strausak flight-proven preset button — Mission
/// statement), Sim (run/pause/single-step/reset,
/// steps-per-frame, MLUPS, EFFECTIVE-VS-TARGET Re line "simulating at
/// Re %.1e (target %.1e)", tau/u_lat, NaN-watchdog error surface), Readouts
/// (Cl/Cd/L-over-D EMA with the trust-deltas tooltip: trust VG-on/VG-off
/// DELTAS on identical settings, verify with tuft testing before drilling —
/// gated until 2 flow-throughs), View (mode toggles with hotkeys 1/2/3,
/// particle count/colormap, slice positions, screenshot), and the STL import
/// modal (plan 7.2 incl. the 7.4 z-boundary mode toggle).
void drawUI(UIContext& ctx);

/// @brief Render the accumulated ImGui draw data (call after Visualizer::
/// drawFrame, before swap).
void uiEndFrame();

/// @brief True when ImGui wants the mouse/keyboard (frame loop suppresses
/// camera input while interacting with panels).
bool uiWantsInput();

} // namespace foilcfd
