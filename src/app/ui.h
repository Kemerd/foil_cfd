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
#include "../render/viz.h"
#include "../sim/lbm_solver.h"
#include "../sim/units.h"

struct GLFWwindow; // forward-declared; only init/shutdown touch the window.

namespace foilcfd {

/// @brief Which airfoil source the user has selected.
enum class AirfoilSource { NacaDigits, DatFile, StlImport };

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
    float aoaDeg = 4.0f;                   ///< AoA slider, -5..20 deg (plan 9.2).
    float airspeedMs = 30.0f;              ///< Physical airspeed [m/s].
    float chordM = 1.2f;                   ///< Physical chord [m].
    ResolutionPreset resolution = ResolutionPreset::Default;

    // -- High Fidelity mode (Mission statement) --
    HighFidelityPreset highFidelity;       ///< .enabled is the UI toggle; the
                                           ///< panel wires .resolution to Ultra.

    // -- VG editor --
    std::vector<VGParams> vgs;             ///< Add/duplicate/delete list.
    int selectedVG = -1;                   ///< Entry the guidance panel anchors to.

    // -- STL import (plan 7.2/7.4) --
    StlImportUI stlImport;                 ///< Modal state; mesh lives in main.cpp.

    // -- sim panel --
    bool running = true;                   ///< Run/pause.
    bool autoCacheClean = true;            ///< Auto-capture clean snapshot toggle.

    // -- view panel --
    VizSettings viz;
    bool showVGGuidanceOverlay = true;     ///< Lin-2002 band overlay toggle.
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
    bool saveCleanState  = false; ///< "Save clean state" button -> capture snapshots.
    bool refreshAirfoils = false; ///< Re-scan airfoils/ directory.
    bool loadStlRequested = false;///< "Load STL..." button -> open file dialog.
    bool stlImportConfirmed = false; ///< Modal "Import": voxelize the pending STL.
    bool stlImportCancelled = false; ///< Modal "Cancel": drop the pending STL.
    bool screenshot      = false; ///< Screenshot button -> Visualizer::screenshotPNG.
    bool highFidelityToggled = false; ///< HiFi switch flipped -> re-init with preset.
    bool particleCountChanged = false; ///< Pool slider released -> resizeParticlePool.
    bool frameFoilView   = false; ///< "Focus foil" -> camera.frameRegion on the foil.

    /// @brief Clear all events (main.cpp calls after applying).
    void reset() { *this = UIEvents{}; }
};

/// @brief Bundle passed into drawUI every frame.
struct UIContext {
    UIParams*   params   = nullptr; ///< Edited in place by the panels.
    UIReadouts* readouts = nullptr; ///< Read-only display values.
    UIEvents*   events   = nullptr; ///< Raised by the panels this frame.
    const std::vector<AirfoilCatalogEntry>* airfoilCatalog = nullptr; ///< Dropdown contents.
    std::string statusMessage;      ///< Transient status line (load errors etc.).
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
/// box + Load STL + AoA/airspeed/resolution + High Fidelity), VG editor (list
/// with section 6.1 params, add/duplicate/delete, under-resolution warnings),
/// VG Guidance (sim-derived delta99 at the selected station, the Lin-2002
/// recommended h band, Strausak flight-proven preset button — Mission
/// statement), Sim (run/pause/single-step/reset, save-clean-state, auto-
/// cache, steps-per-frame, MLUPS, EFFECTIVE-VS-TARGET Re line "simulating at
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
