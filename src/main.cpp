// Application entry point: owns the plan 9.3 frame loop
// (poll input -> N sim steps -> particle update -> map interop -> draw ->
// ImGui -> present), the UI-event application in its defined order
// (voxelize -> solver), the session log (logs/foilcfd.log), the GLFW drop
// routing (.dat -> airfoil loader, .stl -> import modal), and the --selftest
// mode that proves the GL + CUDA + sim plumbing end-to-end.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include <glad/gl.h>
// GLFW must come after the loader so it does not inject its own GL header.
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "app/aircraft_manifest.h"
#include "app/camera.h"
#include "app/sweep.h"
#include "app/ui.h"
#include "geom/airfoil.h"
#include "geom/stl.h"
#include "geom/vg.h"
#include "geom/vg_audit.h"
#include "geom/voxelizer.h"
#include "geom/voxelizer_stl.cuh"
#include "platform/platform.h"
#include "render/viz.h"
#include "sim/lbm_solver.h"
#include "sim/units.h"

namespace {

using namespace foilcfd;

/// Default window size; the renderer is fully resize-aware.
constexpr int kWindowW = 1600;
constexpr int kWindowH = 900;

/// How long a transient status message stays in the overlay [s].
constexpr double kStatusLifetime = 8.0;

/// Guidance-overlay refresh period [s]: extractSuctionDelta99 downloads the
/// velocity field, so it runs at UI rate (a few Hz), never per frame.
constexpr double kGuidancePeriod = 0.5;

/// Chordwise stations sampled for the delta99 profile / guidance overlay.
constexpr int kGuidanceStations = 32;

/// Default ISLBM (Stretch) near-wall refinement factor when nothing else drives
/// it: the leading edge / wall is always refined at least this much finer than
/// the base grid, so Stretch mode genuinely concentrates resolution at the
/// action (and the resolution view shows a real gradient) rather than sitting at
/// a flat base-dx grid. VGs raise it further via the resolution target.
constexpr int kIslbmDefaultLEFactor = 2;

// z-face free-slip flags for STL mode (plan 7.4): CellFlag::SlipFront/SlipBack
// landed in lbm_core.cuh and the fused kernel mirrors them via kMirZ, so the
// stamped planes behave as proper specular walls.
constexpr std::uint8_t kCellFlagSlipFront =
    static_cast<std::uint8_t>(CellFlag::SlipFront); ///< z = 0 plane.
constexpr std::uint8_t kCellFlagSlipBack =
    static_cast<std::uint8_t>(CellFlag::SlipBack);  ///< z = nz-1 plane.

/// Build the default domain layout for a chord resolution and a set of domain
/// margins. The in-plane box is sized DIRECTLY from the margins so it hugs the
/// foil instead of using the old fixed 3x-chord proportions — fewer empty
/// cells, less wasted compute (cost scales with the cell count). The spanwise
/// (z) extent stays proportional to the chord: z is periodic and exists to
/// resolve 3D vortex structure, not far-field, so it does not follow the
/// in-plane margins. The quarter-chord anchor is set ABSOLUTELY from the
/// upstream margin (LE sits `upstreamC` chords from the inlet; quarter-chord
/// is 0.25 chord aft of the LE), and vertically centered between the side
/// margins — this is what keeps the foil pinned correctly when the box shrinks
/// rather than drifting with a fixed nx-fraction.
/// @param chordCells Chord length N_c in cells.
/// @param m          Per-side fluid margins in chords (see UIParams::domain).
/// @brief Spanwise cell count that fits a WHOLE number of VG periods, so the
/// periodic z-tiling of a VG array is seamless (no partial unit clipped at the
/// z=0/nz seam). For a periodic slice one period = one UNIT = pitch_c*chord
/// cells (a counter-rotating pair's gap is INTRA-unit, so a pair is still one
/// period). We snap to count * round(pitch_cells): integer cells-per-period
/// first, then multiply, so voxelizeVG's centered array tiles cleanly. Returns 0
/// (caller keeps the default proportion) when no enabled VG defines a period or
/// the result would be too thin for 3D vortex resolution.
int spanCellsForVGs(const std::vector<VGParams>& vgs, int chordCells,
                    int defaultNz) {
    // Use the first ENABLED VG's array as the period source (the common case is
    // a single array; multiple incommensurate arrays can't all tile one nz).
    const VGParams* arr = nullptr;
    for (const VGParams& vg : vgs) if (vg.enabled) { arr = &vg; break; }
    if (!arr) return 0;

    const int count = std::max(1, arr->count);
    const int cellsPerPeriod =
        std::max(1, static_cast<int>(std::lround(
                        arr->pitch_c * static_cast<float>(chordCells))));
    const int nz = count * cellsPerPeriod;
    // Guard: never shrink below the default proportion's worth of span — a too-
    // thin domain starves the 3D vortex and the refinement patch's clearance.
    // Round UP to whole periods if the snapped span is below the floor.
    if (nz < defaultNz) {
        const int periodsNeeded =
            (defaultNz + cellsPerPeriod - 1) / cellsPerPeriod;
        return periodsNeeded * cellsPerPeriod;
    }
    return nz;
}

DomainLayout defaultLayout(int chordCells, const UIParams::DomainMargins& m,
                           const std::vector<VGParams>* vgs = nullptr) {
    DomainLayout layout;
    layout.chordCells = chordCells;

    const float nc = static_cast<float>(chordCells);

    // In-plane dimensions come straight from the margin spans (rounded to whole
    // cells, +1 so the boundary flag planes never clip the fluid region).
    layout.dims.nx = static_cast<int>(std::lround(m.spanXc() * nc)) + 1;
    layout.dims.ny = static_cast<int>(std::lround(m.spanYc() * nc)) + 1;
    // Spanwise extent: the original 0.375*N_c proportion, UNLESS VGs are present
    // — then snap it to a whole number of VG periods so the periodic z-tiling of
    // the array is seamless (no clipped partial unit at the seam).
    const int defaultNz = (3 * chordCells) / 8;
    layout.dims.nz = defaultNz;
    if (vgs) {
        const int vgNz = spanCellsForVGs(*vgs, chordCells, defaultNz);
        if (vgNz > 0) layout.dims.nz = vgNz;
    }

    // Pin the foil with an ABSOLUTE anchor instead of an nx-fraction so the
    // upstream gap stays exactly `upstreamC` chords regardless of box size.
    // Quarter-chord = LE + 0.25c; LE = upstream margin from the inlet.
    layout.anchorAbsolute = true;
    layout.anchorXCells = (m.upstreamC + 0.25f) * nc;
    // Vertically center the section in the available height (below margin from
    // the bottom slip wall, + half the thickness allowance).
    layout.anchorYCells = (m.belowC + 0.06f) * nc;
    return layout;
}

/// Everything the frame loop owns, grouped so selftest and interactive mode
/// share one initialization path.
struct App {
    GLFWwindow*  window = nullptr;
    cudaStream_t stream = nullptr;
    LBMSolver    solver;
    Visualizer   viz;
    OrbitCamera  camera;
    UIParams     params;
    UIReadouts   readouts;
    UIEvents     events;
    AirfoilGeometry airfoil;
    DomainLayout layout;
    std::vector<AirfoilCatalogEntry> catalog;
    AircraftManifest aircraft;           ///< Plan 15.5 aircraft -> airfoil rows.
    std::filesystem::path airfoilsDir;   ///< Scanned recursively (incl. uiuc/).

    // -- geometry/flag state --
    std::string geometryId;              ///< "naca:..", "dat:..", "stl:<hash>".
    std::vector<std::uint8_t> cleanFlags;  ///< Clean foil (no VGs) at current AoA.
    std::vector<std::uint8_t> activeFlags; ///< cleanFlags OR VG voxels = live flags.

    // -- refinement patch state (plan M-refine) --
    PatchBox patchBox;                   ///< Active patch in coarse cells.
    std::vector<std::uint8_t> fineFlags; ///< Fine-level flag field (2x, shell stamped).

    // -- nested VG patch state (third level) --
    PatchBox finerBox;                   ///< Active nested box, in FINE cells.
    std::vector<std::uint8_t> finerFlags;///< Finer-level flag field (shell stamped).

    // NOTE: the plan-8 snapshot warm-start cache (VRAM clean slot + disk LRU)
    // was removed: its capture path could freeze the session right as the
    // force gate opened ("clean flow cached" — then nothing). Every geometry/
    // AoA/airspeed edit now simply restarts the sim, which is predictable
    // and plenty fast on this hardware. VG edits restart cold too by default,
    // with an opt-in warm continuation (UIParams::vgWarmRestart) that patches
    // the flags in place — no capture/cache involved, so none of the snapshot
    // path's failure modes apply.

    bool nanLogged = false;              ///< One log line per watchdog trip.
    double lastHeartbeatT = 0.0;         ///< Telemetry heartbeat pacing.

    // -- STL import state (plan 7) --
    StlMesh stlMeshRaw;                  ///< As loaded, pre-normalization (kept
                                         ///< so resolution changes re-voxelize).
    bool stlActive = false;

    // -- custom-VG-STL state (2026-06-19) --
    // Unit-normalized VG meshes (normalizeVgMeshUnit), indexed by
    // VGParams::stlMeshId. These ride a parametric foil — independent of
    // stlActive (which replaces the whole foil). Heavy triangle soup lives here,
    // NOT in UIParams, so VGParams stays a cheap-to-copy POD.
    std::vector<StlMesh> vgMeshes;       ///< index == VGParams::stlMeshId.
    std::vector<std::string> vgMeshNames;///< Parallel display names for the combo.
    bool vgStlPending = false;           ///< Drop/dialog stashed a mesh in
                                         ///< stlMeshRaw for VG import (the modal
                                         ///< is showing with StlImportUI::forVg).
    bool vgMeshMissingLogged = false;    ///< One log line when a CustomStl VG
                                         ///< references an unresolved mesh id.

    // -- testing-suite run state (2026-06-19) --
    // The in-frame sweep state machine. It drives the SAME solver one case at a
    // time (set params -> applyGeometryCold -> step to convergence -> record ->
    // advance), ticked once per frame so the UI stays live and the user can
    // watch each case run. See tickSweep().
    enum class SweepPhase { Idle, Converging, Recording, Done, Cancelled };
    struct SweepRun {
        SweepPhase phase = SweepPhase::Idle;
        SweepParams params;                  ///< Snapshot taken at start.
        std::vector<SweepCase> cases;
        std::vector<SweepResult> results;
        int       current     = -1;          ///< Index of the case in flight.
        long long caseStartStep = 0;         ///< solver.stepCount() at setup.
        double    caseStartWall = 0.0;        ///< wall seconds at setup.
        bool      paused      = false;
        int       scrubLoadedCase = -1;      ///< Result loaded for live preview.
    };
    SweepRun sweep;

    // -- frame-loop bookkeeping --
    std::vector<std::string> droppedFiles; ///< Queued by the GLFW drop callback.
    double lastFrameT = 0.0;
    double lastGuidanceT = 0.0;
    double statusT = 0.0;                ///< When the status message was set.
    double lastCursorX = 0.0, lastCursorY = 0.0;
    float  scrollPending = 0.0f;         ///< Accumulated by the scroll callback.
    int    vramPollCounter = 0;
    bool   uiActive = false;             ///< ImGui context exists (interactive).
    std::string status;

    // -- latched CUDA failure (plan 4.5 TDR robustness) --
    // After a TDR/device-removed/sticky context error every subsequent launch
    // fails too; retrying each frame would silently spin on a frozen field.
    // The first failure stops the sim for the session and surfaces its
    // diagnosis through the Sim panel (mirrors the NaN-watchdog pattern).
    bool        cudaFailure = false;
    std::string cudaFailureMsg;
};

/// Set the transient status line (load errors, cache notes) with its decay
// ===========================================================================
// Session log: every status message, error, and lifecycle event is appended
// to logs/foilcfd.log next to the exe, timestamped — so "it froze and the
// status bar already faded" is never an unreproducible mystery again. The
// file is truncated per launch (one session per file keeps it shareable).
// ===========================================================================

std::FILE* gLogFile = nullptr;

/// Open (truncate) the session log. Failure is non-fatal: logging falls back
/// to stderr only.
void logOpen() {
    const auto dir = platform::executableDirectory() / "logs";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const auto path = dir / "foilcfd.log";
#ifdef _WIN32
    _wfopen_s(&gLogFile, path.wstring().c_str(), L"wb");
#else
    gLogFile = std::fopen(path.string().c_str(), "wb");
#endif
}

/// Timestamped write to the log file only (no console mirror) — used by the
/// periodic telemetry heartbeat so the console stays readable. Flushes
/// immediately: the log's whole purpose is surviving a hang or crash.
void logLineToFileOnly(const std::string& msg) {
    if (!gLogFile) return;
    const std::time_t t = std::time(nullptr);
    std::tm tmv{};
#ifdef _WIN32
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif
    char stamp[16];
    std::snprintf(stamp, sizeof stamp, "%02d:%02d:%02d", tmv.tm_hour,
                  tmv.tm_min, tmv.tm_sec);
    std::fprintf(gLogFile, "[%s] %s\n", stamp, msg.c_str());
    std::fflush(gLogFile);
}

/// Append one timestamped line to the session log and mirror it to stderr
/// (keeps the console-watching workflow intact).
void logLine(const std::string& msg) {
    std::fprintf(stderr, "[foilcfd] %s\n", msg.c_str());
    logLineToFileOnly(msg);
}

void logClose() {
    if (gLogFile) {
        std::fclose(gLogFile);
        gLogFile = nullptr;
    }
}

/// timestamp so the overlay clears itself.
void setStatus(App& app, std::string msg) {
    logLine(msg); // every status line is also a log line
    app.status  = std::move(msg);
    app.statusT = platform::timerSeconds();
}

/// Latch a fatal CUDA failure: stop stepping for the session and surface the
/// first error's diagnosis in the Sim panel. Without this, the cudaError_t
/// chain (kernel wrapper -> stepN/updateFields -> here) would terminate in a
/// silently discarded return value at the only call sites that matter.
void latchCudaFailure(App& app, const char* stage, cudaError_t err) {
    if (app.cudaFailure) return; // keep the FIRST failure's diagnosis
    app.cudaFailure = true;
    app.cudaFailureMsg = std::string(stage) + " failed: "
                       + cudaGetErrorString(err)
                       + " (" + cudaGetErrorName(err) + ")";
    app.params.running = false;
    setStatus(app, "GPU error — " + app.cudaFailureMsg);
}

// ===========================================================================
// Derived-parameter helpers: chord resolution, lattice speed, and the exact
// snapshot key (plan 8) all flow from UIParams through these so every call
// site agrees.
// ===========================================================================

/// One-line configuration summary for the session log: everything needed to
/// reproduce a reported incident (geometry, AoA, airspeed, grid, scaling)
/// without asking the user follow-up questions.
std::string configSummary(const App& app) {
    char buf[384];
    std::snprintf(buf, sizeof buf,
                  "config: %s | AoA %.1f deg | airspeed %.2f m/s | chord %.2f m"
                  " | grid %dx%dx%d (%d cells/chord) | u_lat %.3f | tau %.5f%s"
                  " | Re_eff %.2e (target %.2e) | VGs %d | HiFi %s",
                  app.readouts.loadedAirfoilName.empty()
                      ? app.geometryId.c_str()
                      : app.readouts.loadedAirfoilName.c_str(),
                  app.params.aoaDeg, app.params.airspeedMs, app.params.chordM,
                  app.layout.dims.nx, app.layout.dims.ny, app.layout.dims.nz,
                  app.layout.chordCells, app.readouts.scaling.u_lat,
                  app.readouts.scaling.tau,
                  app.readouts.scaling.tauClamped ? " (clamped)" : "",
                  app.readouts.scaling.reEffective, app.readouts.scaling.reTarget,
                  static_cast<int>(app.params.vgs.size()),
                  app.params.highFidelity.enabled ? "on" : "off");
    std::string line(buf);

    // Mesh mode + refinement depth belong in every incident report: a steep
    // high-factor VG box is the prime suspect for a from-rest seam blow-up, so
    // the dump spells out the exact cascade depth / stretch state / VG target.
    const auto& rp = app.params.refine;
    const char* modeStr = rp.meshMode == UIParams::MeshMode::Stretch ? "stretch"
                        : rp.meshMode == UIParams::MeshMode::Uniform ? "uniform"
                                                                     : "cascade";
    char mbuf[160];
    std::snprintf(mbuf, sizeof mbuf,
                  " | mesh %s (factor %d, finerVG %s, vgTargetAuto %s, target %d"
                  " cells)",
                  modeStr, rp.factor, rp.finerVGPatch ? "on" : "off",
                  rp.vgTargetAuto ? "on" : "off", rp.vgTargetCells);
    line += mbuf;

    // Live cascade levels (what ACTUALLY allocated, factors and dims) — the
    // effective VG-box factor is the headline number for a seam divergence.
    if (app.readouts.refine.active) {
        char rbuf[128];
        std::snprintf(rbuf, sizeof rbuf, " | fine %dx %dx%dx%d (forces %s)",
                      app.readouts.refine.factor,
                      app.readouts.refine.fineDims.nx,
                      app.readouts.refine.fineDims.ny,
                      app.readouts.refine.fineDims.nz,
                      app.readouts.refine.forcesFromFine ? "fine" : "coarse");
        line += rbuf;
    }
    if (app.readouts.refine.finerActive) {
        char fbuf[96];
        std::snprintf(fbuf, sizeof fbuf, " | nestedVG %dx (coarse) %dx%dx%d",
                      app.readouts.refine.finerEffectiveFactor,
                      app.readouts.refine.finerDims.nx,
                      app.readouts.refine.finerDims.ny,
                      app.readouts.refine.finerDims.nz);
        line += fbuf;
    }
    if (app.readouts.stretch.active) {
        char sbuf[128];
        std::snprintf(sbuf, sizeof sbuf,
                      " | stretch dx %.3g->%.3g mm tau %.3f->%.3f%s",
                      app.readouts.stretch.dxMin * 1e3f,
                      app.readouts.stretch.dxMax * 1e3f,
                      app.readouts.stretch.tauWall, app.readouts.stretch.tauFar,
                      app.readouts.stretch.tauFloorClamped ? " [clamped]" : "");
        line += sbuf;
    }

    // Transition trip: it injects fluctuations near the LE/VG, so its state is
    // load-bearing for a "pulse from the VG" report.
    if (app.readouts.tripActive) {
        char tbuf[64];
        std::snprintf(tbuf, sizeof tbuf, " | trip on (%d cells)",
                      app.readouts.tripCells);
        line += tbuf;
    }

    // Per-VG one-liners: arrangement, profile, station, height — so a divergence
    // tied to a specific VG (e.g. the Strausak preset) is fully reproducible.
    for (std::size_t i = 0; i < app.params.vgs.size(); ++i) {
        const VGParams& vg = app.params.vgs[i];
        static const char* kArr[] = {"single", "pair", "corot"};
        static const char* kProf[] = {"rect", "delta", "trap", "parab",
                                      "airfoil", "wedge", "stl"};
        const int ai = std::clamp(static_cast<int>(effectiveArrangement(vg)), 0, 2);
        const int pi = std::clamp(static_cast<int>(effectiveProfile(vg)), 0, 6);
        char vbuf[128];
        std::snprintf(vbuf, sizeof vbuf,
                      " | VG%zu %s/%s x/c %.3f h/c %.4f beta %.0f n %d%s",
                      i, kArr[ai], kProf[pi], vg.x_c, vg.height_c, vg.beta_deg,
                      vg.count, vg.enabled ? "" : " [off]");
        line += vbuf;
    }
    return line;
}

/// Append a divergence incident to logs/divergence.log: a timestamped block
/// with the NaN diagnosis and the full settings line, so the user can paste a
/// self-contained "here's exactly what blew up" report without digging through
/// the heartbeat-laden session log. Best-effort — a write failure is silent.
void writeDivergenceReport(const App& app, const std::string& summary) {
    const auto dir = platform::executableDirectory() / "logs";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const auto path = dir / "divergence.log";

    // Timestamp the block (local time, like the session log).
    const std::time_t t = std::time(nullptr);
    std::tm tmv{};
#ifdef _WIN32
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif
    char stamp[32];
    std::snprintf(stamp, sizeof stamp, "%04d-%02d-%02d %02d:%02d:%02d",
                  tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday, tmv.tm_hour,
                  tmv.tm_min, tmv.tm_sec);

    std::string block = "==== DIVERGENCE ";
    block += stamp;
    block += " ====\n";
    block += "step: " + std::to_string(app.readouts.stepCount) + "\n";
    block += "diagnosis: " + app.readouts.nanDiagnosis + "\n";
    block += summary + "\n\n";

    // Append (don't truncate): keep the history of every blow-up this session.
    std::FILE* f = nullptr;
#ifdef _WIN32
    _wfopen_s(&f, path.wstring().c_str(), L"ab");
#else
    f = std::fopen(path.string().c_str(), "ab");
#endif
    if (!f) return;
    std::fwrite(block.data(), 1, block.size(), f);
    std::fclose(f);
}

/// Active chord resolution: the HiFi bundle overrides the standard preset.
int currentChordCells(const UIParams& p) {
    return p.highFidelity.enabled ? chordCellsFor(p.highFidelity.resolution)
                                  : chordCellsFor(p.resolution);
}

/// Active lattice inflow speed: HiFi lowers u_lat to 0.05 (units.h).
float currentULat(const UIParams& p) {
    return p.highFidelity.enabled ? p.highFidelity.u_lat : kDefaultULat;
}

/// Lattice scaling for the current UI parameters (the only constructor path).
LatticeScaling currentScaling(const UIParams& p) {
    PhysicalParams phys;
    phys.chord_m     = p.chordM;
    phys.airspeed_ms = p.airspeedMs;
    return computeScaling(phys, currentChordCells(p), currentULat(p));
}

// ===========================================================================
// GLFW plumbing.
// ===========================================================================

/// GLFW error callback: log to stderr — there is no UI yet when most of
/// these can fire.
void glfwErrorCallback(int code, const char* desc) {
    std::fprintf(stderr, "[glfw] error %d: %s\n", code, desc ? desc : "?");
}

/// Drop callback: queue paths for the frame loop (plan 7.1 drag-and-drop).
/// Routing (.dat vs .stl) happens in processDroppedFiles, on the frame
/// boundary, so geometry rebuilds never run mid-poll.
void dropCallback(GLFWwindow* window, int count, const char** paths) {
    App* app = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (!app) return;
    for (int i = 0; i < count; ++i) {
        if (paths[i]) app->droppedFiles.emplace_back(paths[i]);
    }
}

/// Scroll callback: accumulate for the camera. Installed BEFORE uiInit so the
/// ImGui GLFW backend chains to it; the accumulated value is only consumed
/// when ImGui does not want the mouse.
void scrollCallback(GLFWwindow* window, double /*dx*/, double dy) {
    App* app = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (app) app->scrollPending += static_cast<float>(dy);
}

/// Create window + GL 4.6 core context + load GL via the GLFW-bundled glad2.
/// @param visible False for --selftest (no flashing window on CI-style runs).
bool initWindowAndGL(App& app, bool visible, std::string* error) {
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit()) {
        *error = "glfwInit failed";
        return false;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, visible ? GLFW_TRUE : GLFW_FALSE);
    app.window = glfwCreateWindow(kWindowW, kWindowH, "FoilCFD", nullptr, nullptr);
    if (!app.window) {
        *error = "glfwCreateWindow failed (GL 4.6 core context unavailable?)";
        return false;
    }
    glfwMakeContextCurrent(app.window);
    glfwSwapInterval(1); // vsync; the adaptive pacer targets the frame budget
    if (gladLoadGL(glfwGetProcAddress) == 0) {
        *error = "gladLoadGL failed";
        return false;
    }
    // Input callbacks go in BEFORE uiInit: ImGui's GLFW backend chains any
    // previously-installed callbacks, so both consumers stay wired.
    glfwSetWindowUserPointer(app.window, &app);
    glfwSetDropCallback(app.window, dropCallback);
    glfwSetScrollCallback(app.window, scrollCallback);
    return true;
}

/// Initialize the CUDA runtime up front so the first frame doesn't absorb
/// the (slow) lazy context creation, and verify a device actually exists.
bool initCuda(App& app, std::string* error) {
    int deviceCount = 0;
    if (auto err = cudaGetDeviceCount(&deviceCount);
        err != cudaSuccess || deviceCount == 0) {
        *error = std::string("no CUDA device: ")
               + (err != cudaSuccess ? cudaGetErrorString(err) : "count is 0");
        return false;
    }
    if (auto err = cudaSetDevice(0); err != cudaSuccess) {
        *error = std::string("cudaSetDevice failed: ") + cudaGetErrorString(err);
        return false;
    }
    // Single CUDA stream for the whole app (plan 9.3).
    if (auto err = cudaStreamCreate(&app.stream); err != cudaSuccess) {
        *error = std::string("cudaStreamCreate failed: ") + cudaGetErrorString(err);
        return false;
    }
    return true;
}

// ===========================================================================
// Geometry resolution + flag building.
// ===========================================================================

/// Locate the airfoils/ data directory: dev runs launch from the repo root,
/// installed runs sit next to the exe, and the build tree is two levels deep
/// (build/Release/FoilCFD.exe) — probe all three.
std::filesystem::path findAirfoilsDirectory() {
    namespace fs = std::filesystem;
    const fs::path exeDir = platform::executableDirectory();
    const fs::path candidates[] = {
        fs::current_path() / "airfoils",
        exeDir / "airfoils",
        exeDir / ".." / ".." / "airfoils",
    };
    for (const fs::path& c : candidates) {
        std::error_code ec;
        if (fs::is_directory(c, ec)) return fs::weakly_canonical(c, ec);
    }
    return exeDir / "airfoils"; // scan will just return empty
}

/// (Re)load the plan-15.5 aircraft manifest and link its resolved dat files
/// into the current catalog. Called at startup and from the airfoil Rescan
/// button (the CSV is user-editable, so refresh re-reads it) — always AFTER
/// scanAirfoilDirectory, since the linkage maps onto catalog indices.
void reloadAircraftManifest(App& app) {
    app.aircraft = loadAircraftManifest(app.airfoilsDir);
    linkManifestToCatalog(app.aircraft.entries, app.catalog);
    // Manifest indices may have shifted across the reload; a stale selection
    // would anchor the Aircraft section's root/tip selector to the wrong row.
    app.params.selectedAircraftIndex = -1;
}

/// Geometry/cache id for a .dat file: "dat:" + the canonicalized FULL path
/// as UTF-8. Keying on the filename alone would collide for identically
/// named files in different directories (airfoils/e423.dat vs
/// airfoils/uiuc/e423.dat) — the flagsHash check keeps such collisions
/// CORRECT, but they silently evict each other's warm-cache entries. UTF-8
/// via the platform helper (never path::string(), which converts through
/// the ANSI code page on MSVC: lossy at best, throwing at worst).
std::string datGeometryId(const std::filesystem::path& path) {
    std::error_code ec;
    const std::filesystem::path canon =
        std::filesystem::weakly_canonical(path, ec);
    return "dat:" + platform::pathToUtf8(ec ? path : canon);
}

/// Resolve the airfoil section from the UI source selection (NACA digits or
/// catalog .dat entry). Fills airfoil, geometryId, and the display name; on
/// failure the previous geometry stays active and the reason is reported.
bool resolveAirfoil(App& app, std::string* error) {
    AirfoilLoadResult res;
    std::string id;
    switch (app.params.source) {
        case AirfoilSource::NacaDigits:
            res = generateNACA4(app.params.nacaDigits);
            id  = "naca:" + app.params.nacaDigits;
            break;
        case AirfoilSource::DatFile: {
            const int idx = app.params.selectedDatIndex;
            if (idx < 0 || idx >= static_cast<int>(app.catalog.size())) {
                *error = "no .dat file selected";
                return false;
            }
            const auto& path = app.catalog[static_cast<size_t>(idx)].path;
            res = loadAirfoilDat(path);
            id  = datGeometryId(path);
            break;
        }
        case AirfoilSource::StlImport:
            // STL geometry never flows through here — applyStlImport owns it.
            *error = "internal: resolveAirfoil called in STL mode";
            return false;
    }
    if (!res.ok) {
        *error = res.rejectionReason;
        return false;
    }
    app.airfoil    = res.airfoil;
    app.geometryId = id;
    app.stlActive  = false;
    app.readouts.loadedAirfoilName = res.airfoil.name;
    return true;
}

/// Log once (per session) when any enabled CustomStl VG points at a mesh id we
/// can't resolve — the vane voxelizes to nothing, so the user needs to know it
/// silently dropped rather than thinking a small VG just isn't visible. Resets
/// the latch when everything resolves so a later bad edit warns again.
void warnIfVgMeshMissing(App& app) {
    bool missing = false;
    for (const VGParams& vg : app.params.vgs) {
        if (vg.enabled && effectiveProfile(vg) == VGProfile::CustomStl
            && (vg.stlMeshId < 0
                || vg.stlMeshId >= static_cast<int>(app.vgMeshes.size())
                || app.vgMeshes[vg.stlMeshId].triangles.empty())) {
            missing = true;
            break;
        }
    }
    if (missing && !app.vgMeshMissingLogged) {
        logLine("warning: a custom-STL VG has no mesh loaded — it voxelizes to "
                "nothing. Load a VG mesh in the VG editor.");
        app.vgMeshMissingLogged = true;
    } else if (!missing) {
        app.vgMeshMissingLogged = false; // re-arm for the next bad edit
    }
}

/// Rebuild clean + active flag fields for the current airfoil/AoA/VG state.
/// The clean field is kept separately so VG slider ticks never repeat the
/// O(nx*ny) airfoil parity tests (plan 6.2).
void rebuildFlagFields(App& app) {
    app.cleanFlags = buildCleanFoilFlags(app.airfoil, app.params.aoaDeg,
                                         app.layout);
    app.activeFlags = app.params.vgs.empty()
        ? app.cleanFlags
        : buildFlagsWithVGs(app.params.vgs, app.airfoil, app.params.aoaDeg,
                            app.layout, app.cleanFlags, /*slabsOut=*/nullptr,
                            &app.vgMeshes);
    warnIfVgMeshMissing(app);
}

/// Apply the wall-model policy (UIParams::wallModel) to the solver: forced
/// On/Off pass straight through; Auto enables the wall function whenever the
/// FINEST level's first cell cannot resolve the viscous sublayer. The y+
/// estimate uses the flat-plate correlation Cf = 0.0576 Re^(-1/5) at the
/// EFFECTIVE Re (the flow the lattice actually runs), u_tau = u sqrt(Cf/2),
/// y+ = 0.5 u_tau / nu in the level's lattice units. Below y+ ~ 2 the
/// sublayer is genuinely resolved and plain bounce-back is the better wall.
void applyWallModelPolicy(App& app) {
    bool enable = false;
    if (app.params.wallModel == 1) {
        enable = true;
    } else if (app.params.wallModel == 0) {
        const RefinementInfo ri = app.solver.refinementInfo();
        // Use the FINEST active level's scaling: the nested VG patch when it is
        // present, else the fine patch, else the coarse grid. The finest cells
        // give the smallest y+, which is what decides if the sublayer resolves.
        const LatticeScaling& sc =
            ri.finerActive ? ri.finerScaling
          : ri.active      ? ri.fineScaling
                           : app.solver.scaling();
        if (sc.u_lat > 0.0f && sc.nu_lat > 0.0f) {
            const float re = std::max(sc.reEffective, 1.0e3f);
            const float cf = 0.0576f * std::pow(re, -0.2f);
            const float uTau = sc.u_lat * std::sqrt(0.5f * cf);
            const float yPlusFirst = 0.5f * uTau / sc.nu_lat;
            enable = yPlusFirst > 2.0f;
        }
    }
    app.solver.setWallModelEnabled(enable);
}

/// Build + arm (or disable) the virtual transition strip: a thin near-leading-
/// edge band on the SUCTION surface where the solver injects turbulent
/// fluctuations every step to trip the boundary layer (the wall model can't
/// sustain turbulence on a smooth surface alone — the Reddit pain point). The
/// band is sized in chord fractions from the UI; for each column in the
/// streamwise window we mark the few fluid cells just above the topmost foil
/// solid (the suction-side boundary layer), across the full span.
void applyTransitionTrip(App& app) {
    const UIParams::TransitionTrip& t = app.params.trip;
    if (!t.enabled || app.cleanFlags.empty()) {
        app.solver.setTransitionTrip({}, 0.0f); // disable / free
        app.readouts.tripActive = false;
        app.readouts.tripCells = 0;
        return;
    }
    const GridDims& d = app.layout.dims;
    const int nc = app.layout.chordCells;
    // Leading-edge x in cells: the quarter-chord anchor minus a quarter chord.
    const float xLE = app.layout.anchorX() - 0.25f * static_cast<float>(nc);
    const int x0 = static_cast<int>(std::lround(
        xLE + (t.xc - 0.5f * t.widthC) * static_cast<float>(nc)));
    const int x1 = static_cast<int>(std::lround(
        xLE + (t.xc + 0.5f * t.widthC) * static_cast<float>(nc)));
    // A few cells of boundary-layer band above the surface (~2% chord, >=3).
    const int bandCells = std::max(3, static_cast<int>(std::lround(
        0.02f * static_cast<float>(nc))));

    std::vector<std::uint8_t> mask(static_cast<std::size_t>(d.cellCount()), 0);
    const auto solid = static_cast<std::uint8_t>(CellFlag::Solid);
    auto idx = [&d](int x, int y, int z) {
        return static_cast<std::size_t>(x)
             + static_cast<std::size_t>(d.nx) * (y + static_cast<long long>(d.ny) * z);
    };
    int marked = 0;
    for (int x = std::max(1, x0); x <= std::min(d.nx - 2, x1); ++x) {
        // Topmost solid in this column (suction surface) at mid-span; the foil
        // is span-extruded so the row is the same on every z plane.
        int topSolid = -1;
        for (int y = d.ny - 2; y >= 1; --y)
            if (app.cleanFlags[idx(x, y, d.nz / 2)] == solid) { topSolid = y; break; }
        if (topSolid < 0) continue; // no surface in this column
        // Mark the band of fluid cells just above the surface, full span.
        for (int z = 0; z < d.nz; ++z)
            for (int y = topSolid + 1; y <= topSolid + bandCells && y < d.ny - 1;
                 ++y) {
                const std::size_t c = idx(x, y, z);
                if (app.cleanFlags[c] != solid) { mask[c] = 1; ++marked; }
            }
    }
    if (marked == 0) {
        app.solver.setTransitionTrip({}, 0.0f);
        app.readouts.tripActive = false;
        app.readouts.tripCells = 0;
        setStatus(app, "transition strip: no suction surface in the x-band");
        return;
    }
    std::string terr;
    if (app.solver.setTransitionTrip(mask, t.intensityFrac * app.readouts.scaling.u_lat,
                                     &terr)) {
        app.readouts.tripActive = true;
        app.readouts.tripCells = marked;
        char msg[160];
        std::snprintf(msg, sizeof msg,
                      "transition strip: x/c %.2f +/- %.2f, %d cells, "
                      "intensity %.0f%% u_lat",
                      t.xc, 0.5f * t.widthC, marked, 100.0f * t.intensityFrac);
        logLine(msg);
    } else {
        app.readouts.tripActive = false;
        app.readouts.tripCells = 0;
        setStatus(app, "transition strip failed: " + terr);
    }
}

/// Build the q-LIBB sub-cell vane field for one refinement level and hand it to
/// the solver. Ray-casts every vane link against the analytic slabs
/// (buildVaneQLinks), densifies to the per-cell device layout (densifyQLinks),
/// counts resolved vs fallen-back links, and uploads via the level's setter.
/// The solver only keeps it when q-LIBB is enabled. level: 0 = coarse/base
/// (the ISLBM stretch grid uses this — its fluid refines but the vane SOLID
/// stays a base-resolution voxel staircase, so q-LIBB is what actually gives the
/// vane a sub-cell surface there), 1 = fine, 2 = finer.
/// @param fallbackOut Receives the count of vane links left at half-way (links
///                    with no analytic hit / outside the clamp) for the readout.
void uploadQLinksForLevel(App& app, const GridDims& dims,
                          const std::vector<std::uint8_t>& active,
                          const std::vector<std::uint8_t>& clean,
                          const std::vector<VaneSlab>& slabs, int level) {
    const long long ncells = dims.cellCount();
    std::vector<std::uint8_t>  qFrac;
    std::vector<std::uint32_t> ffMask;
    int links = 0, fallback = 0;
    if (!slabs.empty()) {
        const QLinkList ql = buildVaneQLinks(dims, active, clean, slabs);
        densifyQLinks(ql, ncells, qFrac, ffMask);
        links = static_cast<int>(ql.size());
        // Of the resolved links, those whose two-node branch is unusable (thin
        // vane: x_ff solid) AND whose cut is < 1/2 silently keep the half-way
        // value in the kernel — surface that as the fallback count.
        for (std::size_t i = 0; i < ql.size(); ++i)
            if (!ql.ffFluid[i] && ql.q[i] < 0.5f) ++fallback;
    }
    if (level == 0)      app.solver.setCoarseQLinks(qFrac, ffMask, links, fallback);
    else if (level == 1) app.solver.setFineQLinks(qFrac, ffMask, links, fallback);
    else                 app.solver.setFinerQLinks(qFrac, ffMask, links, fallback);
}

/// Build a coarse flag field holding ONLY the VG solids (no foil) at the live
/// layout — every VG voxelized into an all-Fluid field. The ISLBM sub-base path
/// seeds its wall-distance field from this so the FINEST stretched cells land on
/// the vanes (the foil wall, whose distance to a VG is large, coarsens back to
/// base). Mirrors deriveVGPatchBoxFine's VG-only voxelization, at the coarse
/// layout, with the custom-STL meshes threaded through.
std::vector<std::uint8_t> buildVGOnlyFlags(const App& app) {
    std::vector<std::uint8_t> vgOnly(
        static_cast<std::size_t>(app.layout.dims.cellCount()),
        static_cast<std::uint8_t>(CellFlag::Fluid));
    for (const VGParams& vg : app.params.vgs)
        voxelizeVG(vg, app.airfoil, app.params.aoaDeg, app.layout, vgOnly,
                   nullptr, &app.vgMeshes);
    return vgOnly;
}

/// Build a coarse flag field with a single solid marker column at the LEADING
/// EDGE of the foil. When there are no VGs, the LE is where the action is —
/// stagnation point, suction peak, the boundary layer's birth — so anchoring the
/// continuous-refine field there puts the finest stretched cells at the LE and
/// coarsens smoothly downstream. The marker is a few cells stamped at the LE
/// surface point (transformed through the same AoA + layout anchor the airfoil
/// voxelizer uses) so the wall-distance field's global minimum lands there.
std::vector<std::uint8_t> buildLEAnchorFlags(const App& app) {
    std::vector<std::uint8_t> mark(
        static_cast<std::size_t>(app.layout.dims.cellCount()),
        static_cast<std::uint8_t>(CellFlag::Fluid));
    if (!app.airfoil.isValid()) return mark;
    // LE surface point at x/c ~ 0 on the upper surface; transform like stampVane.
    const SurfaceFrame le = surfaceFrameAt(app.airfoil, 0.0f, /*upper=*/true);
    if (!le.valid) return mark;
    const float aoaRad = app.params.aoaDeg * 3.14159265358979f / 180.0f;
    const float chord  = static_cast<float>(app.layout.chordCells);
    const Vec2f pr = rotated(le.point - Vec2f(0.25f, 0.0f), -aoaRad);
    const int lx = static_cast<int>(std::lround(app.layout.anchorX() + pr.x * chord));
    const int ly = static_cast<int>(std::lround(app.layout.anchorY() + pr.y * chord));
    const GridDims& d = app.layout.dims;
    const auto solid = static_cast<std::uint8_t>(CellFlag::Solid);
    // A small 3x3 marker across the full span at the LE so the distance field
    // anchors there (span-extruded like the foil, so every z plane is finest).
    for (int z = 0; z < d.nz; ++z)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                const int x = lx + dx, y = ly + dy;
                if (x < 1 || x >= d.nx - 1 || y < 1 || y >= d.ny - 1) continue;
                mark[static_cast<std::size_t>(x)
                     + static_cast<std::size_t>(d.nx)
                       * (y + static_cast<long long>(d.ny) * z)] = solid;
            }
    return mark;
}

/// (Re)build the two-level refinement patch (plan M-refine) for the current
/// geometry: derive the box from the VG-merged flags (vanes must sit inside),
/// voxelize foil + VGs at 2x into the fine flag field, stamp the Interface
/// shell, and hand it to the solver. Disabled/STL/derivation-failure paths
/// tear the fine level down — the coarse sim continues either way.
void applyRefinement(App& app) {
    // Effective factor: the user's Mesh-panel setting, raised by the VG
    // resolution target when it is on — a vane below the target sheds a
    // systematically weak vortex, and the patch is the cheap fix because it
    // multiplies resolution only around the geometry. The target can pull the
    // patch into existence from factor 1 for the same reason.
    int factor = std::clamp(app.params.refine.factor, 1, kMaxRefineFactor);
    const bool vgTargetDrives =
        app.params.refine.vgTargetAuto && !app.params.vgs.empty()
        && !app.stlActive;
    if (vgTargetDrives) {
        // What the cascade must reach to seat every vane at the target height.
        // The nested VG box (finerVGPatch) already doubles the fine factor, so
        // when it is on the FINE patch only needs to supply HALF the target
        // factor — keeping the expensive whole-foil patch as small as possible
        // and letting the tiny nested box carry the rest (4x fine x 2x nested =
        // 8x effective = kMaxVGTargetFactor). Without the nested box the fine
        // patch must carry the whole target itself.
        const int needed = recommendedRefineFactorForVGs(
            app.params.vgs, app.layout.chordCells,
            app.params.refine.vgTargetCells, kMaxVGTargetFactor);
        // When the nested box (finerVGPatch) is on it doubles the fine factor,
        // so the FINE patch supplies ceil(needed/2). But the cascade has no
        // standalone-2x mode: the nested box only exists ON TOP of a >= 2x fine
        // patch, so once any refinement is needed the fine patch must be at
        // least 2x — otherwise factor < 2 below tears the whole cascade (and the
        // nested box) down and the vane gets 1x. Floor at 2 whenever needed >= 2.
        int fineNeeded;
        if (needed <= 1) {
            fineNeeded = needed;
        } else if (app.params.refine.finerVGPatch) {
            fineNeeded = std::max(2, (needed + 1) / 2); // ceil(needed/2), >= 2
        } else {
            fineNeeded = needed;
        }
        // The single fine patch never exceeds kMaxRefineFactor (4x); the nested
        // box supplies any remaining factor up to the 8x ceiling. A target that
        // still can't be met keeps the honest under-resolution warning.
        factor = std::clamp(std::max(factor, fineNeeded), 1, kMaxRefineFactor);
    } else if (app.params.refine.autoVGFactor && !app.params.vgs.empty()
               && !app.stlActive) {
        // Legacy whole-patch auto-raise (kept for back-compat; the target
        // driver above is the default path now).
        factor = std::max(factor, recommendedRefineFactorForVGs(
                                      app.params.vgs, app.layout.chordCells));
    }

    // Keep the solver's q-LIBB switch in sync with the UI before any q-link
    // upload below — uploadQLinksForLevel only takes effect when it is on.
    app.solver.setQLIBBEnabled(app.params.refine.qlibbVanes);

    // Tearing down the fine level always tears down the nested VG level too
    // (the solver enforces the same invariant; this keeps the App-side mirror
    // and the shutdown call in lock-step on every early-return path).
    auto clearFiner = [&app]() {
        app.solver.shutdownFinerRefinement();
        app.finerBox = PatchBox{};
        app.finerFlags.clear();
    };

    // ---- ISLBM stretched-mesh mode (continuous-gradient refinement) -------
    // One smoothly-stretched grid replaces the discrete levels: build the wall-
    // distance field over the live solids (foil + VGs, so the mesh concentrates
    // around the vanes too) and hand it to the solver, which sizes dx finest at
    // the wall. Mutually exclusive with the cascade — tear any levels down first.
    // Falls back to a uniform grid if the build fails (OOM) or no solids exist.
    if (app.params.refine.meshMode == UIParams::MeshMode::Stretch
        && !app.stlActive) {
        app.solver.shutdownRefinement(); // drop any cascade levels
        app.patchBox = PatchBox{};
        app.fineFlags.clear();
        clearFiner();
        // Continuous near-wall refinement (acoustic re-anchor): drive the finest
        // stretch cell BELOW base dx so a band gets finer-than-base cells (true
        // "resolve the curve"), with the field smoothly coarsening outward.
        // WHERE the finest cells land is set by the wall-distance field's anchor,
        // and Stretch ALWAYS refines something now (so there is a real gradient):
        //   - VGs present + target on  -> anchor at the VANES (VG-only field), k
        //     from the VG resolution target (but never below the LE default),
        //   - else                     -> anchor at the LEADING EDGE (where the
        //     stagnation / suction peak / BL birth lives), at k = the manual
        //     slider OR kIslbmDefaultLEFactor, whichever is larger.
        // k = 1 (no refinement at all) is no longer a default outcome — ISLBM's
        // whole point is a finer-near-the-action grid.
        float nearWallK = static_cast<float>(kIslbmDefaultLEFactor);
        std::vector<float> wallDist;
        const int manualK = std::clamp(app.params.refine.islbmNearWallK, 1, 4);
        if (app.params.refine.vgTargetAuto && !app.params.vgs.empty()) {
            const int vgK = recommendedRefineFactorForVGs(
                app.params.vgs, app.layout.chordCells,
                app.params.refine.vgTargetCells, kMaxVGTargetFactor);
            // Never coarser than the LE default even if a big vane needs no extra
            // refinement — the near-wall field always concentrates at the vanes.
            nearWallK = static_cast<float>(std::max(vgK, kIslbmDefaultLEFactor));
            // VG-only distance field: only cells near the vanes become finest.
            wallDist = buildWallDistanceField(app.layout.dims, buildVGOnlyFlags(app));
        } else {
            // No VGs (or VG-auto off): anchor the finest cells at the leading
            // edge, at the manual k or the default, whichever is larger.
            nearWallK = static_cast<float>(std::max(manualK, kIslbmDefaultLEFactor));
            wallDist = buildWallDistanceField(app.layout.dims, buildLEAnchorFlags(app));
        }
        std::string serr;
        if (app.solver.initStretchMode(wallDist, nearWallK, &serr)) {
            app.solver.setStretchFastGather(app.params.refine.islbmFastGather);
            const StretchInfo si = app.solver.stretchInfo();
            char msg[220];
            std::snprintf(msg, sizeof msg,
                          "stretched mesh (ISLBM): dx %.3g->%.3g mm (growth "
                          "x%.4f/y%.4f), tau wall %.3f -> far %.3f%s, near-wall "
                          "k=%.0f",
                          si.dxMin * 1e3f, si.dxMax * 1e3f, si.growthX,
                          si.growthY, si.tauWall, si.tauFar,
                          si.tauFloorClamped ? " [tau-floor clamped]" : "",
                          si.nearWallFactor);
            logLine(msg);

            // q-LIBB sub-cell vane surfaces on the BASE (stretch) grid. ISLBM
            // refines the FLUID dx but keeps the same lattice cell count, so the
            // VG SOLID stays a base-resolution voxel staircase — q-LIBB is what
            // gives the vane a TRUE sub-cell surface there (the finer near-wall
            // dx then actually resolves the vane curvature instead of a blocky
            // edge). Build the analytic vane slabs at the base layout and upload
            // them as the coarse-level q-links (the solver keeps them only when
            // q-LIBB is on). No VGs -> no slabs -> a no-op clear.
            app.solver.setQLIBBEnabled(app.params.refine.qlibbVanes);
            std::vector<VaneSlab> baseSlabs;
            std::vector<std::uint8_t> baseActive = app.cleanFlags;
            for (const VGParams& vg : app.params.vgs)
                voxelizeVG(vg, app.airfoil, app.params.aoaDeg, app.layout,
                           baseActive, &baseSlabs, &app.vgMeshes);
            closeTrailingEdgeGaps(app.layout.dims, baseActive);
            uploadQLinksForLevel(app, app.layout.dims, baseActive, app.cleanFlags,
                                 baseSlabs, /*level=*/0);
        } else {
            setStatus(app, "stretched mesh unavailable: " + serr
                               + " — running uniform grid");
            app.solver.shutdownStretchMode();
        }
        applyWallModelPolicy(app); // the (single) finest level changed
        return;
    }
    // Leaving stretch mode (or never in it): make sure the solver isn't holding
    // a stale stretched mesh before the cascade path below allocates levels.
    app.solver.shutdownStretchMode();

    if (factor < 2 || app.stlActive) {
        app.solver.shutdownRefinement();
        app.patchBox = PatchBox{};
        app.fineFlags.clear();
        clearFiner();
        applyWallModelPolicy(app); // finest level is now the coarse grid
        return;
    }

    // Margins from chord fractions to coarse cells (>= 2 so the restriction
    // band never touches a solid).
    const int nc = app.layout.chordCells;
    auto cellsOf = [nc](float chords) {
        return std::max(2, static_cast<int>(std::lround(
                               chords * static_cast<float>(nc))));
    };
    const PatchBox box = derivePatchBox(
        app.layout.dims, app.activeFlags,
        cellsOf(app.params.refine.upstreamC), cellsOf(app.params.refine.wakeC),
        cellsOf(app.params.refine.belowC), cellsOf(app.params.refine.aboveC));
    if (!box.valid()) {
        app.solver.shutdownRefinement();
        app.patchBox = PatchBox{};
        app.fineFlags.clear();
        clearFiner();
        applyWallModelPolicy(app);
        return;
    }

    // Fine flag field: foil + every VG voxelized at the factor's resolution,
    // the same TE-closure pass, then the coupling shell. Same machinery as
    // the coarse build — only the layout (scaled chord cells, patch-local
    // anchor) differs, so the two levels always agree on where geometry is.
    const DomainLayout fineLayout = makeFineLayout(app.layout, box, factor);
    std::vector<std::uint8_t> fine(
        static_cast<std::size_t>(fineLayout.dims.cellCount()),
        static_cast<std::uint8_t>(CellFlag::Fluid));
    voxelizeAirfoil(app.airfoil, app.params.aoaDeg, fineLayout, fine);
    // Capture the analytic vane slabs for q-LIBB sub-cell vane surfaces.
    std::vector<VaneSlab> fineSlabs;
    for (const VGParams& vg : app.params.vgs)
        voxelizeVG(vg, app.airfoil, app.params.aoaDeg, fineLayout, fine,
                   &fineSlabs, &app.vgMeshes);
    closeTrailingEdgeGaps(fineLayout.dims, fine);
    stampInterfaceShell(fineLayout.dims, fine);

    std::string err;
    if (!app.solver.initRefinement(box, factor, fine, &err)) {
        setStatus(app, "refinement patch unavailable: " + err
                           + " — running base grid only");
        app.patchBox = PatchBox{};
        app.fineFlags.clear();
        clearFiner();
        applyWallModelPolicy(app);
        return;
    }

    // Fine CLEAN reference for the wall model: the foil alone at fine
    // resolution (no VGs, no shell) — the fine wall list keys its normals
    // and VG-vane exclusion off this exactly like the coarse level keys off
    // cleanFlags (sim-module contract: setRefinedSurfaceReference). Kept here
    // because q-LIBB's vane-provenance test needs the same clean field.
    std::vector<std::uint8_t> fineClean(
        static_cast<std::size_t>(fineLayout.dims.cellCount()),
        static_cast<std::uint8_t>(CellFlag::Fluid));
    voxelizeAirfoil(app.airfoil, app.params.aoaDeg, fineLayout, fineClean);
    closeTrailingEdgeGaps(fineLayout.dims, fineClean);
    app.solver.setRefinedSurfaceReference(fineClean);

    // q-LIBB sub-cell vane surfaces on the fine level: ray-cast the vane links
    // against the analytic slabs, densify, and hand the field to the solver
    // (it uploads only when q-LIBB is enabled).
    uploadQLinksForLevel(app, fineLayout.dims, fine, fineClean, fineSlabs,
                         /*level=*/1);

    app.patchBox  = box;
    app.fineFlags = std::move(fine);
    char msg[160];
    std::snprintf(msg, sizeof msg,
                  "refinement patch (%dx): coarse box [%d,%d)x[%d,%d) -> fine "
                  "%d x %d x %d",
                  factor, box.x0, box.x1, box.y0, box.y1, fineLayout.dims.nx,
                  fineLayout.dims.ny, fineLayout.dims.nz);
    logLine(msg);

    // ---- nested VG patch (third level): a tiny box hugging just the vanes,
    // run at 2x the fine factor (effective m*2 vs coarse). Built only when the
    // user keeps it on AND there are VGs to wrap. Any failure tears down only
    // level 2 — the 2x patch + coarse sim keep running. ---------------------
    clearFiner();
    if (app.params.refine.finerVGPatch && !app.params.vgs.empty()) {
        constexpr int kFinerFactor = 2; // 2x the fine grid

        // Margins ~half the vane extent on each side -> the box is roughly 2x
        // the vane envelope with the vanes centered. Scaled off the FINE chord
        // resolution so it tracks grid size. The floor is the coupling buffer
        // that MUST sit between a vane and the box edge: the 2-cell Interface
        // shell + the 2-cell restriction band + 2 cells of trilinear-stencil
        // reach, so the shell is filled from clean fluid and the vane lives
        // fully inside the restricted (fine-overwrites-coarse) interior. The
        // solid-aware fill tolerates a vane grazing the stencil, but keeping
        // this margin means it almost never has to.
        constexpr int kVGMarginFloor =
            kInterfaceShellFine + kRestrictBandCoarse + 2; // = 6 fine cells
        const int fnc = fineLayout.chordCells;
        auto vgCellsOf = [fnc](float chords) {
            return std::max(kVGMarginFloor, static_cast<int>(std::lround(
                                   chords * static_cast<float>(fnc))));
        };
        const int mx = vgCellsOf(0.05f); // streamwise pad (fine cells)
        const int my = vgCellsOf(0.03f); // wall-normal pad
        const PatchBox box2 = deriveVGPatchBoxFine(
            fineLayout, app.params.vgs, app.airfoil, app.params.aoaDeg,
            mx, mx, my, my, &app.vgMeshes);

        if (box2.valid()) {
            // Level-2 flag field: foil (so the vane roots have a wall + the
            // wall model finds the surface) + every VG, at the finer layout.
            const DomainLayout finerLayout =
                makeFineLayout(fineLayout, box2, kFinerFactor);
            std::vector<std::uint8_t> finer(
                static_cast<std::size_t>(finerLayout.dims.cellCount()),
                static_cast<std::uint8_t>(CellFlag::Fluid));
            voxelizeAirfoil(app.airfoil, app.params.aoaDeg, finerLayout, finer);
            // Capture vane slabs at the FINEST resolution — this is the level
            // where q-LIBB matters most (vanes are best resolved here).
            std::vector<VaneSlab> finerSlabs;
            for (const VGParams& vg : app.params.vgs)
                voxelizeVG(vg, app.airfoil, app.params.aoaDeg, finerLayout, finer,
                           &finerSlabs, &app.vgMeshes);
            closeTrailingEdgeGaps(finerLayout.dims, finer);
            stampInterfaceShell(finerLayout.dims, finer);

            std::string err2;
            if (app.solver.initFinerRefinement(box2, kFinerFactor, finer,
                                               &err2)) {
                // Clean (VG-free) finer reference for the wall model + q-LIBB.
                std::vector<std::uint8_t> finerClean(
                    static_cast<std::size_t>(finerLayout.dims.cellCount()),
                    static_cast<std::uint8_t>(CellFlag::Fluid));
                voxelizeAirfoil(app.airfoil, app.params.aoaDeg, finerLayout,
                                finerClean);
                closeTrailingEdgeGaps(finerLayout.dims, finerClean);
                app.solver.setRefinedFinerSurfaceReference(finerClean);

                // q-LIBB on the nested level (the headline win for VG fidelity).
                uploadQLinksForLevel(app, finerLayout.dims, finer, finerClean,
                                     finerSlabs, /*level=*/2);

                app.finerBox   = box2;
                app.finerFlags = std::move(finer);
                char msg2[180];
                std::snprintf(msg2, sizeof msg2,
                              "nested VG patch (%dx of fine = %dx): fine box "
                              "[%d,%d)x[%d,%d) -> finer %d x %d x %d",
                              kFinerFactor, factor * kFinerFactor,
                              box2.x0, box2.x1, box2.y0, box2.y1,
                              finerLayout.dims.nx, finerLayout.dims.ny,
                              finerLayout.dims.nz);
                logLine(msg2);
            } else {
                setStatus(app, "nested VG patch unavailable: " + err2
                                   + " — running 2x patch only");
                clearFiner();
            }
        }
    }
    // The finest level changed; the Auto wall-model verdict may flip with it.
    applyWallModelPolicy(app);
}

/// Mesh-sequencing pre-convergence (plan M-refine part 2): converge a 4x-
/// coarser companion sim (~1/64 the cells, seconds of wall time), upsample
/// its macroscopic field onto the full grid, and continue from a developed
/// flow. Synchronous by design — the chunked loop syncs the stream between
/// TDR-safe batches, and the whole pass is shorter than the wake transit it
/// replaces. Sections only in v1 (STL re-voxelization at 4x is future work).
void runPreconverge(App& app) {
    if (!app.params.preconvergeCoarse || app.stlActive || app.cudaFailure)
        return;

    const int nc = std::max(48, currentChordCells(app.params) / 4);
    // Same margins AND span as the live domain so the companion field upsamples
    // cleanly (VG-aware nz must match the full grid's snapped span).
    const DomainLayout preLayout = defaultLayout(nc, app.params.domain,
                                                 &app.params.vgs);

    // Voxelize the SAME geometry (foil + VGs) at presolve resolution.
    std::vector<std::uint8_t> flags =
        buildCleanFoilFlags(app.airfoil, app.params.aoaDeg, preLayout);
    if (!app.params.vgs.empty()) {
        flags = buildFlagsWithVGs(app.params.vgs, app.airfoil,
                                  app.params.aoaDeg, preLayout, flags,
                                  /*slabsOut=*/nullptr, &app.vgMeshes);
    }

    PhysicalParams phys;
    phys.chord_m     = app.params.chordM;
    phys.airspeed_ms = app.params.airspeedMs;
    const LatticeScaling preScaling =
        computeScaling(phys, nc, currentULat(app.params));

    LBMSolver pre;
    std::string err;
    if (!pre.init(preLayout.dims, preScaling, flags, app.stream, &err)) {
        logLine("preconverge skipped (init failed: " + err + ")");
        return;
    }

    // Two flow-throughs — the force-gate convergence criterion — in chunks
    // small enough that each batch stays far under the TDR budget.
    const long long totalSteps = static_cast<long long>(
        kForceGateFlowThroughs * preScaling.flowThroughSteps(preLayout.dims.nx));
    const double t0 = platform::timerSeconds();
    long long done = 0;
    while (done < totalSteps) {
        const int n = static_cast<int>(
            std::min<long long>(1024, totalSteps - done));
        if (pre.stepN(n) != cudaSuccess) {
            logLine("preconverge aborted (step failed)");
            return;
        }
        cudaStreamSynchronize(app.stream);
        if (pre.nanDetected()) {
            logLine("preconverge aborted (coarse companion diverged) — "
                    "starting cold instead");
            return;
        }
        done += n;
    }

    if (!app.solver.seedFromCoarse(pre, &err)) {
        logLine("preconverge seed failed: " + err + " — starting cold");
        return;
    }
    char msg[160];
    std::snprintf(msg, sizeof msg,
                  "pre-converged at %d cells/chord (%lld steps, %.1f s) — "
                  "seeded full grid",
                  nc, totalSteps, platform::timerSeconds() - t0);
    setStatus(app, msg);
}

// ===========================================================================
// Voxel view (View panel): render the flag fields' Solid cells as exposed-
// face cube soup — the stair-step walls the lattice actually bounces off,
// with half-size cubes inside the refinement patch. Pushed through the
// existing STL mesh pipeline, so the renderer needs zero changes.
// ===========================================================================

/// Append one cube face (two flat-shaded triangles) to a triangle soup.
void pushQuad(std::vector<StlTriangle>& out, const Vec3f& a, const Vec3f& b,
              const Vec3f& c, const Vec3f& d, const Vec3f& n) {
    out.push_back(StlTriangle{a, b, c, n});
    out.push_back(StlTriangle{a, c, d, n});
}

/// Emit the exposed faces of every Solid cell in a flag field. A face is
/// exposed when the neighbor across it is not Solid (out-of-range counts as
/// exposed — the spanwise end caps just look like the extruded prism's).
/// @param dims     Grid the flags cover.
/// @param flags    Unpadded flag field.
/// @param cellSize Cube edge length in COARSE lattice units (1.0 coarse,
///                 0.5 for the 2x fine patch).
/// @param origin   Lattice-space position of cell (0,0,0)'s low corner.
/// @param skipXY   Optional box (in THIS grid's own cells) to omit, because a
///                 finer pass renders that region instead — the coarse pass
///                 skips the fine-patch box, the fine pass skips the nested
///                 VG box.
void emitVoxelFaces(const GridDims& dims, const std::vector<std::uint8_t>& flags,
                    float cellSize, const Vec3f& origin, const PatchBox* skipXY,
                    std::vector<StlTriangle>& out) {
    const auto solidFlag = static_cast<std::uint8_t>(CellFlag::Solid);
    const long long nxny = static_cast<long long>(dims.nx) * dims.ny;
    auto solidAt = [&](int x, int y, int z) {
        if (x < 0 || x >= dims.nx || y < 0 || y >= dims.ny || z < 0
            || z >= dims.nz)
            return false;
        return flags[static_cast<std::size_t>(
                   x + static_cast<long long>(dims.nx) * y + nxny * z)]
               == solidFlag;
    };

    for (int z = 0; z < dims.nz; ++z) {
        for (int y = 0; y < dims.ny; ++y) {
            for (int x = 0; x < dims.nx; ++x) {
                if (!solidAt(x, y, z)) continue;
                if (skipXY && x >= skipXY->x0 && x < skipXY->x1
                    && y >= skipXY->y0 && y < skipXY->y1)
                    continue; // fine pass owns this region

                // Cube corner span in lattice space.
                const float x0 = origin.x + cellSize * static_cast<float>(x);
                const float y0 = origin.y + cellSize * static_cast<float>(y);
                const float z0 = origin.z + cellSize * static_cast<float>(z);
                const float x1 = x0 + cellSize, y1 = y0 + cellSize,
                            z1 = z0 + cellSize;

                if (!solidAt(x - 1, y, z))
                    pushQuad(out, {x0, y0, z0}, {x0, y0, z1}, {x0, y1, z1},
                             {x0, y1, z0}, {-1, 0, 0});
                if (!solidAt(x + 1, y, z))
                    pushQuad(out, {x1, y0, z0}, {x1, y1, z0}, {x1, y1, z1},
                             {x1, y0, z1}, {1, 0, 0});
                if (!solidAt(x, y - 1, z))
                    pushQuad(out, {x0, y0, z0}, {x1, y0, z0}, {x1, y0, z1},
                             {x0, y0, z1}, {0, -1, 0});
                if (!solidAt(x, y + 1, z))
                    pushQuad(out, {x0, y1, z0}, {x0, y1, z1}, {x1, y1, z1},
                             {x1, y1, z0}, {0, 1, 0});
                if (!solidAt(x, y, z - 1))
                    pushQuad(out, {x0, y0, z0}, {x0, y1, z0}, {x1, y1, z0},
                             {x1, y0, z0}, {0, 0, -1});
                if (!solidAt(x, y, z + 1))
                    pushQuad(out, {x0, y0, z1}, {x1, y0, z1}, {x1, y1, z1},
                             {x0, y1, z1}, {0, 0, 1});
            }
        }
    }
}

/// Build the voxel-view triangle soup from the live flag fields: coarse cells
/// everywhere except the refinement patch, which contributes its fine cells
/// at half size — exactly the composite geometry the two-level solver sees.
/// Toggle-rate cost (one linear scan per level, ~100s of ms at Ultra): never
/// runs per frame.
StlMesh buildVoxelMesh(const App& app) {
    StlMesh m;
    m.name = "voxel view";
    const RefinementInfo ri = app.solver.refinementInfo();
    const bool fineActive = ri.active && !app.fineFlags.empty();
    emitVoxelFaces(app.layout.dims, app.activeFlags, 1.0f, Vec3f(0, 0, 0),
                   fineActive ? &app.patchBox : nullptr, m.triangles);
    if (fineActive) {
        const GridDims fineDims =
            fineDimsFor(app.patchBox, app.layout.dims, ri.factor);
        const float fineCell = 1.0f / static_cast<float>(ri.factor);
        // Fine-patch origin in coarse units.
        const Vec3f fineOrigin(static_cast<float>(app.patchBox.x0),
                               static_cast<float>(app.patchBox.y0), 0.0f);
        // The nested VG level renders its own region at 1/(m*m2); the fine pass
        // skips that region (finerBox is already in fine cells, which is the
        // grid the fine pass iterates).
        const bool finerActive = ri.finerActive && !app.finerFlags.empty();
        emitVoxelFaces(fineDims, app.fineFlags, fineCell, fineOrigin,
                       finerActive ? &app.finerBox : nullptr, m.triangles);
        if (finerActive) {
            const GridDims finerDims =
                fineDimsFor(app.finerBox, fineDims, ri.finerFactor);
            // Finer cell size in coarse units, and its origin = fine origin
            // shifted by the finer box (in fine cells) scaled to coarse units.
            const float finerCell =
                fineCell / static_cast<float>(ri.finerFactor);
            const Vec3f finerOrigin(
                fineOrigin.x + fineCell * static_cast<float>(app.finerBox.x0),
                fineOrigin.y + fineCell * static_cast<float>(app.finerBox.y0),
                0.0f);
            emitVoxelFaces(finerDims, app.finerFlags, finerCell, finerOrigin,
                           nullptr, m.triangles);
        }
    }
    return m;
}

/// Upload whichever render mesh the view settings ask for: the voxel soup
/// (solver's-eye view), the normalized STL solid, or the smooth extruded
/// foil + VG prism. The ONE place that decides, so every geometry-changing
/// path stays consistent with the voxel-view toggle.
void uploadRenderGeometry(App& app) {
    if (app.params.voxelView) {
        // Voxel view still simulates the same airfoil section — keep the tuft
        // anchors tracking the live AoA/section even though the smooth prism
        // isn't drawn (uploadGeometry, which normally stashes it, is skipped
        // on this branch). STL solids have no 2D section to anchor on.
        if (!app.stlActive) {
            app.viz.setTuftGeometry(app.airfoil, app.params.aoaDeg, app.layout);
        }
        app.viz.uploadStlMesh(buildVoxelMesh(app));
        return;
    }
    if (app.stlActive) {
        // STL mode has no 2D section for the tuft seeder to anchor on — clear
        // the stash (an invalid section makes the seeder retire the strands)
        // so tufts from the previous airfoil don't float inside the solid.
        app.viz.setTuftGeometry(AirfoilGeometry{}, 0.0f, app.layout);
        // Re-derive the normalized STL render mesh from the kept raw import
        // (the normalized copy is transient — this path re-creates it the
        // same way initSimulation does).
        const StlNormalization norm = computeAutoNormalization(
            app.stlMeshRaw, app.params.stlImport.axisPreset,
            app.params.stlImport.chordCells, app.layout.anchorX(),
            app.layout.anchorY(), app.layout.dims.nz);
        StlMesh mesh = app.stlMeshRaw;
        applyNormalization(mesh, norm);
        app.viz.uploadStlMesh(mesh);
        return;
    }
    app.viz.uploadGeometry(app.airfoil, app.params.vgs, app.params.aoaDeg,
                           app.layout);
}

/// Geometry changed at fixed grid dims (airfoil swap or AoA edit): rebuild
/// flags and cold-restart the solver — geometry edits always restart the sim
/// (the warm-start cache was removed; see the App-struct note).
void applyGeometryCold(App& app) {
    rebuildFlagFields(app);
    app.solver.setFlags(app.activeFlags);
    // The live flags may include VG voxels; the delta99/separation-onset
    // extraction must always measure from the CLEAN foil surface, or a vane
    // crossing mid-span would read as the surface (false separation pinned
    // at the VG station — the mission's core readout corrupted).
    app.solver.setSurfaceReference(app.cleanFlags);
    // Fine level first: the voxel view (if active) composites its flags.
    applyRefinement(app);
    uploadRenderGeometry(app);
    runPreconverge(app);
}

/// Stamp free-slip walls on the z faces of a flag field (STL mode, plan 7.4).
/// The fused kernel treats SlipFront/SlipBack as z-specular walls (kMirZ).
void stampZFreeSlipWalls(const GridDims& dims, std::vector<std::uint8_t>& flags) {
    const long long planeCells = static_cast<long long>(dims.nx) * dims.ny;
    for (long long i = 0; i < planeCells; ++i) {
        // Front plane z = 0 occupies the first nx*ny cells; back plane the
        // last. Only Fluid cells convert — inlet/outlet/slip stamps win.
        std::uint8_t& front = flags[static_cast<size_t>(i)];
        if (front == static_cast<std::uint8_t>(CellFlag::Fluid))
            front = kCellFlagSlipFront;
        std::uint8_t& back = flags[static_cast<size_t>(
            (static_cast<long long>(dims.nz) - 1) * planeCells + i)];
        if (back == static_cast<std::uint8_t>(CellFlag::Fluid))
            back = kCellFlagSlipBack;
    }
}

/// Voxelize the pending STL with the user's modal choices and make it the
/// active geometry (plan 7.2/7.3/7.4 import flow).
void applyStlImport(App& app) {
    const StlImportUI& ui = app.params.stlImport;
    const GridDims& dims = app.layout.dims;

    // Normalize a working copy: axis remap -> scale-to-chord -> recenter.
    const StlNormalization norm = computeAutoNormalization(
        app.stlMeshRaw, ui.axisPreset, ui.chordCells, app.layout.anchorX(),
        app.layout.anchorY(), dims.nz);
    StlMesh mesh = app.stlMeshRaw;
    applyNormalization(mesh, norm);

    // Boundary flags first, optional z walls, then the GPU ray-parity fill.
    std::vector<std::uint8_t> flags = buildBoundaryFlags(dims);
    if (ui.zFreeSlipWalls) stampZFreeSlipWalls(dims, flags);
    const StlVoxelizeResult vr = voxelizeStl(mesh, dims, flags);
    if (!vr.ok) {
        setStatus(app, "STL voxelization failed: " + vr.error);
        return;
    }
    // Non-watertight detection (plan 7.3): warn, don't crash.
    const float badFrac = vr.badRowFraction(dims);
    if (badFrac > 0.001f) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "warning: mesh not watertight — %.2f%% of rows had odd "
                      "ray parity (%d rows); voxels may leak",
                      badFrac * 100.0f, vr.oddParityRows);
        setStatus(app, buf);
    } else {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "imported %u triangles -> %d solid cells",
                      static_cast<unsigned>(mesh.triangles.size()),
                      vr.solidCells);
        setStatus(app, buf);
    }

    // Commit: STL becomes the active geometry. VGs clear (no parametric
    // surface to seat them on, plan 7.4) and the snapshot key switches to the
    // content hash so the cache survives reimports of the same file. The
    // normalization choices are PART of the identity: the same bytes with a
    // different axis remap / chord scale / z-boundary mode produce a
    // different solid, and keying on the hash alone would make those
    // variants evict each other's warm-cache entries (the flagsHash check
    // keeps that correct, but every toggle would silently cache-miss).
    char idBuf[64];
    std::snprintf(idBuf, sizeof(idBuf), "stl:%016llx:a%d:c%d:%s",
                  static_cast<unsigned long long>(app.stlMeshRaw.contentHash),
                  static_cast<int>(ui.axisPreset), ui.chordCells,
                  ui.zFreeSlipWalls ? "w" : "p");
    app.geometryId = idBuf;
    app.stlActive  = true;
    app.params.source = AirfoilSource::StlImport;
    app.params.vgs.clear();
    app.params.selectedVG = -1;
    app.readouts.loadedAirfoilName = app.stlMeshRaw.name;
    app.cleanFlags  = flags;
    app.activeFlags = std::move(flags);
    app.solver.setFlags(app.activeFlags);
    app.solver.setSurfaceReference(app.cleanFlags); // STL: clean == active
    // STL mode is base-grid only in v1: the previous section's fine level
    // (and its patch box) would be nonsense over the new solid.
    applyRefinement(app); // stlActive is set -> tears the fine level down
    // Render mesh through the central dispatcher: the normalized triangle
    // soup normally, the voxel soup when the solver's-eye view is on.
    uploadRenderGeometry(app);
    app.params.viz.showFoilMesh = true;
}

/// Route a dropped/dialog-selected file by extension: .dat -> airfoil loader,
/// .stl -> import modal (plan 7.1 + the section-9 drop contract).
void handleIncomingFile(App& app, const std::filesystem::path& path) {
    // All display/ID conversions go through pathToUtf8 — path::string() on
    // MSVC converts via the ANSI code page and can THROW std::system_error
    // for unrepresentable characters (CJK filename on a Western locale),
    // which would propagate straight out of the frame loop.
    const std::string fileNameUtf8 = platform::pathToUtf8(path.filename());
    std::string ext = platform::pathToUtf8(path.extension());
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (ext == ".dat") {
        AirfoilLoadResult res = loadAirfoilDat(path);
        if (!res.ok) {
            setStatus(app, fileNameUtf8 + " rejected: " + res.rejectionReason);
            return;
        }
        app.airfoil    = res.airfoil;
        app.geometryId = datGeometryId(path);
        app.stlActive  = false;
        app.params.source = AirfoilSource::DatFile;
        app.params.selectedDatIndex = -1; // external file, not in the catalog
        app.readouts.loadedAirfoilName = res.airfoil.name;
        applyGeometryCold(app);
        setStatus(app, "loaded " + res.airfoil.name);
    } else if (ext == ".stl") {
        StlLoadResult res = loadStl(path);
        if (!res.ok) {
            setStatus(app, fileNameUtf8 + " rejected: " + res.rejectionReason);
            return;
        }
        // Stash the mesh; the modal collects normalization choices before
        // any voxelization happens (plan 7.2).
        app.stlMeshRaw = std::move(res.mesh);
        app.vgStlPending = false; // a dropped .stl is a FOIL import, not a VG
        StlImportUI& ui = app.params.stlImport;
        ui.open          = true;
        ui.forVg         = false; // clear any stale VG-import flag
        ui.fileName      = fileNameUtf8;
        ui.solidName     = app.stlMeshRaw.name;
        ui.triangleCount = static_cast<std::uint32_t>(app.stlMeshRaw.triangles.size());
        ui.bounds        = app.stlMeshRaw.bounds;
        ui.wasBinary     = app.stlMeshRaw.wasBinary;
        ui.chordCells    = app.layout.chordCells;
    } else {
        setStatus(app, "unsupported file type: " + ext);
    }
}

/// Load the program's default geometry: the Glasair III's airfoil, the
/// NASA/LANGLEY LS(1)-0413MOD section from the bundled catalog. Must run
/// BEFORE initSimulation so the first voxelization already uses it (the init
/// path only generates the NACA fallback when no valid airfoil is loaded).
/// A missing/rejected dat is not fatal — the app boots on the NACA section.
void tryLoadDefaultAirfoil(App& app) {
    const std::filesystem::path dat = app.airfoilsDir / "uiuc" / "ls413mod.dat";
    AirfoilLoadResult res = loadAirfoilDat(dat);
    if (!res.ok) {
        setStatus(app, "default airfoil (ls413mod.dat) unavailable: "
                           + res.rejectionReason + " — using NACA 2412");
        return;
    }
    app.airfoil    = res.airfoil;
    app.geometryId = datGeometryId(dat);
    app.stlActive  = false;
    app.params.source = AirfoilSource::DatFile;
    app.readouts.loadedAirfoilName = res.airfoil.name;
    // Highlight the entry in the catalog list so the UI reflects the choice.
    for (std::size_t i = 0; i < app.catalog.size(); ++i) {
        if (app.catalog[i].path == dat) {
            app.params.selectedDatIndex = static_cast<int>(i);
            break;
        }
    }
    setStatus(app, "loaded default airfoil: " + res.airfoil.name);
}

// ===========================================================================
// VG geometry application.
// ===========================================================================

/// VG edit: rebuild the flag field and restart the sim. The default is a
/// full cold restart so the force history is clean and not contaminated by
/// the pre-edit flow state. When the user opts into warm restarts (the VG
/// editor's "Keep flow between edits" checkbox), the developed field is kept
/// instead: the solver diffs old vs new flags in place (newly solid cells
/// zeroed, newly fluid cells equilibrium-filled from neighbors) and the force
/// gate reopens after a short re-settle rather than a from-scratch run.
/// @return True when the warm path was taken (for the status log line).
bool applyVGEdit(App& app) {
    app.activeFlags = app.params.vgs.empty()
        ? app.cleanFlags
        : buildFlagsWithVGs(app.params.vgs, app.airfoil, app.params.aoaDeg,
                            app.layout, app.cleanFlags, /*slabsOut=*/nullptr,
                            &app.vgMeshes);
    warnIfVgMeshMissing(app);
    // Warm continuation only makes sense from a healthy field — after a NaN
    // trip there is nothing worth keeping, so fall back to the cold path.
    const bool warm = app.params.vgWarmRestart && !app.solver.nanDetected();
    if (warm) {
        // In-place flag swap: keeps the developed flow, skips the viscosity
        // ramp, reopens the force gate after the warm re-settle window.
        app.solver.applyEditedFlags(app.activeFlags);
    } else {
        // Cold restart — VG geometry changes the flow field enough that
        // continuing from the old state produces misleading force transients.
        app.solver.setFlags(app.activeFlags);
    }
    app.solver.setSurfaceReference(app.cleanFlags);
    // The patch box tracks the VG bbox (a vane near the placement limit can
    // move it), so the fine level rebuilds with the new vanes BEFORE the
    // render mesh (the voxel view composites the fine flags). initRefinement
    // seeds the fine state from the LIVE coarse field, so the rebuild is
    // equally valid after the warm in-place swap.
    applyRefinement(app);
    uploadRenderGeometry(app);
    // The presolve refills the developed flow a cold restart just discarded;
    // the warm path already HAS that flow — re-seeding would throw it away.
    if (!warm) runPreconverge(app);
    return warm;
}

/// Commit a pending custom-VG-STL import: normalize the stashed raw mesh to
/// unit space under the chosen axis preset, store it, attach it to the selected
/// VG (creating a CustomStl entry if the selection isn't already one), and run
/// the normal VG rebuild. The mesh rides the parametric foil — stlActive stays
/// false, so the foil + cascade keep working.
void applyVgStlImport(App& app) {
    if (!app.vgStlPending || app.stlMeshRaw.triangles.empty()) {
        app.vgStlPending = false;
        return;
    }
    // Unit-normalize a working copy under the modal's axis preset.
    StlMesh mesh = app.stlMeshRaw;
    normalizeVgMeshUnit(mesh, app.params.stlImport.axisPreset);

    const int newId = static_cast<int>(app.vgMeshes.size());
    std::string name = app.stlMeshRaw.name.empty()
                           ? app.params.stlImport.fileName
                           : app.stlMeshRaw.name;
    if (name.empty()) name = "VG mesh " + std::to_string(newId + 1);
    app.vgMeshes.push_back(std::move(mesh));
    app.vgMeshNames.push_back(name);

    // Attach to the selected VG: set its PROFILE to CustomStl (the arrangement —
    // single / pair / array — is preserved, so the mesh inherits whatever
    // arrangement the entry already had). Append a fresh entry when nothing is
    // selected. Clear any legacy type value so effectiveProfile resolves cleanly.
    const int sel = app.params.selectedVG;
    VGParams* target = nullptr;
    if (sel >= 0 && sel < static_cast<int>(app.params.vgs.size())) {
        target = &app.params.vgs[sel];
    } else {
        app.params.vgs.push_back(defaultVGParams());
        app.params.selectedVG = static_cast<int>(app.params.vgs.size()) - 1;
        target = &app.params.vgs.back();
    }
    target->type      = effectiveArrangement(*target); // drop legacy type value
    target->profile   = VGProfile::CustomStl;
    target->stlMeshId = newId;
    target->stlAxis   = app.params.stlImport.axisPreset;

    app.vgStlPending = false;
    app.params.stlImport.forVg = false;
    app.params.source = AirfoilSource::DatFile; // unchanged; VG rides the foil
    setStatus(app, "loaded VG mesh: " + name);

    // Rebuild with the new VG via the standard edit path (cold/warm per the UI).
    applyVGEdit(app);
}

// ===========================================================================
// (Re)initialization.
// ===========================================================================

/// Point the camera at the foil itself (not the whole wind tunnel): quarter-
/// chord anchor + a radius generous enough to keep the near wake in frame.
void focusCameraOnFoil(App& app, bool snap) {
    const float c = static_cast<float>(app.layout.chordCells);
    const Vec3f center(app.layout.anchorX() + 0.25f * c, app.layout.anchorY(),
                       static_cast<float>(app.layout.dims.nz) * 0.5f);
    app.camera.frameRegion(center, 0.95f * c);
    if (snap) app.camera.snapToGoals();
}

/// Default slice positions: mid-domain planes so enabling a slice always
/// shows something sensible without hunting for the position slider.
void resetSliceDefaults(App& app) {
    app.params.viz.slices[0].axis = SliceAxis::X;
    app.params.viz.slices[0].cell = app.layout.dims.nx / 2;
    app.params.viz.slices[1].axis = SliceAxis::Y;
    app.params.viz.slices[1].cell = app.layout.dims.ny / 2;
    app.params.viz.slices[2].axis = SliceAxis::Z;
    app.params.viz.slices[2].cell = app.layout.dims.nz / 2;
}

/// Build the default NACA 2412 domain and bring the solver + renderer up.
/// Also the full-reinit path for resolution/HiFi changes when @p reinit.
bool initSimulation(App& app, bool reinit, std::string* error) {
    // Grid invalidates everything VRAM-resident.
    if (reinit) {
        app.solver.shutdown();
        app.viz.shutdown();
    }

    const int chordCells = currentChordCells(app.params);
    // VG-aware span: when parametric VGs are present, snap nz to a whole number
    // of VG periods so the periodic spanwise tiling has no clipped unit at the
    // seam. STL-foil mode has no parametric VGs, so it keeps the default span.
    app.layout = defaultLayout(chordCells, app.params.domain,
                               app.stlActive ? nullptr : &app.params.vgs);

    // Geometry + flags: STL keeps its imported mesh through grid changes;
    // sections re-voxelize from their polygon.
    StlMesh stlMeshNormalized; // kept past the branch: the renderer needs it
    if (app.stlActive) {
        const StlNormalization norm = computeAutoNormalization(
            app.stlMeshRaw, app.params.stlImport.axisPreset,
            app.params.stlImport.chordCells, app.layout.anchorX(),
            app.layout.anchorY(), app.layout.dims.nz);
        stlMeshNormalized = app.stlMeshRaw;
        applyNormalization(stlMeshNormalized, norm);
        std::vector<std::uint8_t> flags = buildBoundaryFlags(app.layout.dims);
        if (app.params.stlImport.zFreeSlipWalls)
            stampZFreeSlipWalls(app.layout.dims, flags);
        const StlVoxelizeResult vr =
            voxelizeStl(stlMeshNormalized, app.layout.dims, flags);
        if (!vr.ok) {
            *error = "STL re-voxelization failed: " + vr.error;
            return false;
        }
        app.cleanFlags  = flags;
        app.activeFlags = std::move(flags);
    } else {
        AirfoilLoadResult gen;
        if (!app.airfoil.isValid()) {
            // First boot: the default section from the UI params.
            gen = generateNACA4(app.params.nacaDigits);
            if (!gen.ok) {
                *error = "airfoil generation failed: " + gen.rejectionReason;
                return false;
            }
            app.airfoil    = gen.airfoil;
            app.geometryId = "naca:" + app.params.nacaDigits;
            app.readouts.loadedAirfoilName = gen.airfoil.name;
        }
        rebuildFlagFields(app);
    }

    // Units: scaling derived ONCE here, displayed by the UI (effective vs
    // target Re), consumed by the solver.
    const LatticeScaling scaling = currentScaling(app.params);
    app.readouts.scaling = scaling;
    app.readouts.dims    = app.layout.dims;

    if (!app.solver.init(app.layout.dims, scaling, app.activeFlags, app.stream,
                         error)) {
        return false;
    }
    // The solver received the MERGED flags (VG voxels included when VGs
    // exist); the suction-surface extraction must measure from the clean
    // foil, never from a vane crest (Lin-2002 guidance contract).
    app.solver.setSurfaceReference(app.cleanFlags);
    // High Fidelity widens the force EMA to many flow-throughs (plan Mission /
    // 4.4); set right after init since a fresh solver resets to the standard
    // 1.0 window. Sim-module contract: notes/integration_sim.md.
    app.solver.setForceEmaWindow(app.params.highFidelity.enabled
                                     ? app.params.highFidelity.forceEmaFlowThroughs
                                     : StandardPreset{}.forceEmaFlowThroughs);
    if (!app.viz.init(app.layout.dims, app.params.viz.particleCount, app.stream,
                      error)) {
        return false;
    }
    app.camera.frameDomain(app.layout.dims.nx, app.layout.dims.ny,
                           app.layout.dims.nz);
    focusCameraOnFoil(app, /*snap=*/true);
    resetSliceDefaults(app);
    // Fine level + developed-flow seed close out every (re)init. Both are
    // no-ops/graceful when disabled or in STL mode. The render mesh uploads
    // AFTER the refinement so the voxel view can composite the fine flags
    // (uploadRenderGeometry re-derives the normalized STL mesh when needed,
    // so the stlMeshNormalized local above stays voxelization-only).
    applyRefinement(app);
    applyTransitionTrip(app); // arm/refresh the trip band for the new geometry
    uploadRenderGeometry(app);
    runPreconverge(app);
    return true;
}

// ===========================================================================
// Per-frame readouts + guidance refresh.
// ===========================================================================

/// Refresh the read-only numbers the panels display.
void updateReadouts(App& app) {
    app.readouts.perf = app.solver.perfStats();
    app.readouts.stepCount = app.solver.stepCount();
    app.readouts.flowThroughs = app.solver.flowThroughsCompleted();
    app.readouts.currentTau = app.solver.currentTau();
    app.readouts.preStepCurrent =
        app.solver.preStepProgress(app.readouts.preStepTotal);
    app.readouts.nanTripped = app.solver.nanDetected();
    app.readouts.nanDiagnosis = app.solver.nanDiagnosis();
    // A watchdog trip silently freezes stepping (stepN no-ops while latched),
    // and the red diagnosis box lives in the Sim panel which may be hidden
    // behind another tab — so the trip ALSO lands in the status line and the
    // session log, exactly once per latch.
    if (app.readouts.nanTripped && !app.nanLogged) {
        app.nanLogged = true;
        setStatus(app, "SIMULATION DIVERGED — sim paused. "
                       + app.readouts.nanDiagnosis);
        // Full reproduction context for the log: the diagnosis alone says
        // WHAT tripped; this says what the user was simulating at the time.
        const std::string summary = configSummary(app);
        logLine("DIVERGENCE at step " + std::to_string(app.readouts.stepCount)
                + " (" + app.readouts.nanDiagnosis + ")");
        logLine(summary);
        // Also append a dedicated, easy-to-find incident file so the settings
        // survive the session log's heartbeat noise — the user pastes this back.
        writeDivergenceReport(app, summary);
    } else if (!app.readouts.nanTripped) {
        app.nanLogged = false; // re-arm after a cold reset clears the latch
    }

    // Telemetry heartbeat: one log line every ~10 s while running, so an
    // incident report shows the performance/progress trajectory leading up
    // to it (and a frozen heartbeat is itself a diagnostic).
    const double nowT = platform::timerSeconds();
    if (app.params.running && nowT - app.lastHeartbeatT > 10.0) {
        app.lastHeartbeatT = nowT;
        char hb[192];
        std::snprintf(hb, sizeof hb,
                      "telemetry: step %lld | %.2f flow-throughs | %.2f ms sim"
                      " | %.0f MLUPS | GPU %d%% | VRAM %.0f%%",
                      app.readouts.stepCount, app.readouts.flowThroughs,
                      app.readouts.simElapsedMs, app.readouts.perf.mlups,
                      app.readouts.gpuUtilPercent,
                      app.readouts.vramUsedFraction * 100.0f);
        logLineToFileOnly(hb); // log only — no status-line noise every 10 s
    }
    app.readouts.cudaErrorTripped = app.cudaFailure;
    app.readouts.cudaErrorDiagnosis = app.cudaFailureMsg;
    app.readouts.forces = app.solver.forces();

    // Refinement-patch status for the Mesh panel (plan M-refine).
    {
        const RefinementInfo ri = app.solver.refinementInfo();
        auto& rr = app.readouts.refine;
        rr.active          = ri.active;
        rr.factor          = ri.factor;
        rr.fineDims        = ri.fineDims;
        rr.fineVramGB      = ri.vramBytes / (1024.0 * 1024.0 * 1024.0);
        rr.fineReEffective = ri.fineScaling.reEffective;
        rr.forcesFromFine  = ri.forcesFromFine;
        rr.patchX0 = ri.box.x0; rr.patchX1 = ri.box.x1;
        rr.patchY0 = ri.box.y0; rr.patchY1 = ri.box.y1;

        rr.finerActive = ri.finerActive;
        rr.finerFactor = ri.finerFactor;
        rr.finerEffectiveFactor = ri.finerActive ? ri.factor * ri.finerFactor : 0;
        rr.finerDims   = ri.finerDims;
        rr.finerVramGB = ri.finerVramBytes / (1024.0 * 1024.0 * 1024.0);
        if (ri.finerActive && ri.factor > 0) {
            // finerBox is in FINE cells; map to coarse lattice space for the
            // 3D overlay: coarse = patch origin + fineCell / factor.
            const float inv = 1.0f / static_cast<float>(ri.factor);
            rr.finerCX0 = ri.box.x0 + inv * static_cast<float>(ri.finerBox.x0);
            rr.finerCX1 = ri.box.x0 + inv * static_cast<float>(ri.finerBox.x1);
            rr.finerCY0 = ri.box.y0 + inv * static_cast<float>(ri.finerBox.y0);
            rr.finerCY1 = ri.box.y0 + inv * static_cast<float>(ri.finerBox.y1);
        }
        // q-LIBB sub-cell vane surfaces.
        const QLIBBReadout qr = app.solver.qlibbReadout();
        rr.qlibbActive   = qr.enabled && qr.links > 0;
        rr.qlibbLinks    = qr.links;
        rr.qlibbFallback = qr.fallback;

        // VG resolution target readout: the resolved height the selected (or
        // first enabled) vane ACTUALLY gets at the live cascade resolution, so
        // the panel can show "current X cells (target Y)" and colour it by
        // whether the target was met. The effective chord resolution is the
        // base chord times the finest level the vanes live in: the nested VG
        // box when active (factor * finerFactor), else the fine patch, else 1.
        rr.vgHeightCellsLive = 0.0f;
        rr.vgTargetCells     = 0;
        // Effective near-wall resolution factor: the cascade's finest level
        // (nested VG box, else fine patch), OR the ISLBM stretch near-wall
        // re-anchor k when the stretched mesh owns the resolution. The vane (and
        // the wall in general) lives in cells `effFactor` times finer than the
        // base grid. Computed ALWAYS so the under-resolved warnings judge against
        // the true effective resolution and clear when genuinely refined.
        float effFactor = 1.0f;
        {
            const StretchInfo sinfo = app.solver.stretchInfo();
            if (sinfo.active) effFactor = std::max(1.0f, sinfo.nearWallFactor);
            else if (ri.finerActive) effFactor = static_cast<float>(ri.factor * ri.finerFactor);
            else if (ri.active) effFactor = static_cast<float>(ri.factor);
        }
        const int effChord = static_cast<int>(std::lround(
            app.layout.chordCells * std::max(1.0f, effFactor)));
        rr.effChordCells = effChord;
        if (!app.params.vgs.empty() && !app.stlActive) {
            // Pick the entry the panel's guidance follows: the selected VG, or
            // the first enabled one when nothing is selected.
            const VGParams* probe = nullptr;
            const int sel = app.params.selectedVG;
            if (sel >= 0 && sel < static_cast<int>(app.params.vgs.size())
                && app.params.vgs[sel].enabled) {
                probe = &app.params.vgs[sel];
            } else {
                for (const VGParams& vg : app.params.vgs)
                    if (vg.enabled) { probe = &vg; break; }
            }
            if (probe) rr.vgHeightCellsLive = vgHeightCells(*probe, effChord);
            rr.vgTargetCells = app.params.refine.vgTargetCells;
        }
    }

    // ISLBM stretched-mesh status (mutually exclusive with the cascade above;
    // its readout is zeroed whenever the mesh is off).
    {
        const StretchInfo si = app.solver.stretchInfo();
        auto& sr = app.readouts.stretch;
        sr.active          = si.active;
        sr.dxMin           = si.dxMin;
        sr.dxMax           = si.dxMax;
        sr.growthX         = si.growthX;
        sr.growthY         = si.growthY;
        sr.tauWall         = si.tauWall;
        sr.tauFar          = si.tauFar;
        sr.tauFloorClamped = si.tauFloorClamped;
        sr.fluidCellSaving = si.fluidCellSaving;
        sr.vramGB          = si.vramBytes / (1024.0 * 1024.0 * 1024.0);
    }
    // The presolve runs synchronously inside the apply functions, so there is
    // never a mid-flight progress value to report between frames.
    app.readouts.preconvergeProgress = -1.0f;

    // Testing-suite snapshot: a frozen copy so the panel renders a stable table
    // (the live results vector grows under it as cases finish). Only refreshed
    // while a sweep is active or finished, so an idle session pays nothing.
    {
        auto& sr = app.readouts.sweep;
        const App::SweepRun& run = app.sweep;
        sr.running = (run.phase == App::SweepPhase::Converging
                      || run.phase == App::SweepPhase::Recording);
        sr.paused  = run.paused;
        sr.current = run.current + 1; // 1-based for display
        sr.total   = static_cast<int>(run.cases.size());
        sr.results = run.results;     // small (tens of rows) — cheap to copy
        sr.bestIndex = run.results.empty()
                           ? -1
                           : rankSweepResults(run.results).front();
    }

    // VRAM fraction (plan 4.6 warn-above-80%) + GPU load: driver queries, so
    // poll at ~2 Hz instead of every frame.
    if (app.vramPollCounter-- <= 0) {
        app.vramPollCounter = 30;
        size_t freeB = 0, totalB = 0;
        if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess && totalB > 0) {
            app.readouts.vramUsedFraction =
                1.0f - static_cast<float>(static_cast<double>(freeB)
                                          / static_cast<double>(totalB));
        }
        app.readouts.gpuUtilPercent = platform::gpuUtilizationPercent();
    }

    // Physical time simulated since the last cold start: steps * dt. dt is
    // tiny (microseconds at default scaling), so milliseconds is the natural
    // display unit.
    app.readouts.simElapsedMs = static_cast<double>(app.solver.stepCount())
                              * static_cast<double>(app.readouts.scaling.dt)
                              * 1000.0;

    // ETA until the force gate opens (the moment Cl/Cd readouts go live):
    // remaining flow-through steps over the measured per-step wall time. An
    // estimate of GPU time, so it reads slightly optimistic — fine for a
    // progress hint. Hidden while paused or before the first timing sample.
    if (app.readouts.forces.valid) {
        app.readouts.etaSeconds = 0.0;
    } else if (app.params.running && app.readouts.perf.lastStepMs > 0.0) {
        const float ftSteps =
            app.readouts.scaling.flowThroughSteps(app.readouts.dims.nx);
        const double remainingSteps = std::max(
            0.0, static_cast<double>(kForceGateFlowThroughs
                                     - app.readouts.forces.flowThroughs)
                     * static_cast<double>(ftSteps));
        app.readouts.etaSeconds =
            remainingSteps * app.readouts.perf.lastStepMs / 1000.0;
    } else {
        app.readouts.etaSeconds = -1.0;
    }
}

/// Guidance-overlay refresh (Mission statement): delta99 profile + Lin-2002
/// bands at UI rate. extractSuctionDelta99 downloads the velocity field, so
/// this is rate-limited to kGuidancePeriod, never per frame.
void refreshGuidance(App& app) {
    const double now = platform::timerSeconds();
    if (now - app.lastGuidanceT < kGuidancePeriod) return;
    app.lastGuidanceT = now;
    // Wall-model telemetry rides the same UI-rate throttle (its readout does
    // a small synchronous device read — sim-module contract).
    app.readouts.wallModel = app.solver.wallModelReadout();
    if (app.stlActive || app.solver.stepCount() <= 0) {
        app.readouts.delta99Profile.clear();
        app.readouts.separationXc = -1.0f;
        app.readouts.heightBand = GuidanceBand{};
        app.readouts.stationBand = GuidanceBand{};
        return;
    }

    // Even spread of suction-surface stations; the panel snaps the selected
    // VG to the nearest sample.
    std::vector<float> stations;
    stations.reserve(kGuidanceStations);
    for (int i = 0; i < kGuidanceStations; ++i) {
        stations.push_back(0.02f + 0.93f * static_cast<float>(i)
                                         / static_cast<float>(kGuidanceStations - 1));
    }
    app.readouts.delta99Profile = app.solver.extractSuctionDelta99(stations);
    app.readouts.separationXc   = app.solver.separationOnsetXc();

    // Anchor station/height: the selected VG, or the Strausak defaults so the
    // panel shows guidance before the first VG exists.
    float stationXc = 0.07f, heightC = 0.01f;
    if (app.params.selectedVG >= 0
        && app.params.selectedVG < static_cast<int>(app.params.vgs.size())) {
        const VGParams& vg = app.params.vgs[static_cast<size_t>(app.params.selectedVG)];
        stationXc = vg.x_c;
        heightC   = vg.height_c;
    }
    // Height band from the delta99 sample nearest the anchor station.
    const Delta99Sample* nearest = nullptr;
    for (const Delta99Sample& s : app.readouts.delta99Profile) {
        if (!s.valid) continue;
        if (!nearest
            || std::fabs(s.x_c - stationXc) < std::fabs(nearest->x_c - stationXc)) {
            nearest = &s;
        }
    }
    app.readouts.heightBand = nearest
        ? recommendedHeightBand(nearest->delta99_c) : GuidanceBand{};
    app.readouts.stationBand =
        recommendedStationBand(app.readouts.separationXc, heightC);

    // ---- vortex-strength audit (Mission statement honesty meter): two
    // small plane downloads, only when a VG exists, the flow has settled
    // past the force gate, and the BL sample at the vane station is live ----
    app.readouts.vgAudit = VGAuditReadout{};
    if (app.params.selectedVG >= 0
        && app.params.selectedVG < static_cast<int>(app.params.vgs.size())
        && app.readouts.forces.valid && nearest) {
        const VGParams& vg =
            app.params.vgs[static_cast<size_t>(app.params.selectedVG)];
        app.readouts.vgAudit = auditVGVortexStrength(
            app.solver, vg, app.layout.chordCells, nearest->ueEdge,
            nearest->delta99_c * static_cast<float>(app.layout.chordCells));
    }
}

// ===========================================================================
// Testing suite: in-frame sweep state machine (2026-06-19).
// ===========================================================================

/// Run the Wendt vortex-strength audit for a converged VG case, reusing the
/// live delta99 profile (refreshGuidance keeps it current) to find the
/// boundary-layer sample nearest the first vane's station. Mirrors the
/// refreshGuidance audit, but anchored on the case's own VGs rather than the
/// UI selection. Returns an invalid readout when there are no VGs or no usable
/// BL sample (ranking tolerates that).
VGAuditReadout runAuditForCase(App& app, const SweepCase& c) {
    if (!c.hasVgs || c.vgs.empty()) return VGAuditReadout{};
    const float stationXc = c.vgs.front().x_c;
    const Delta99Sample* nearest = nullptr;
    for (const Delta99Sample& s : app.readouts.delta99Profile) {
        if (!s.valid) continue;
        if (!nearest
            || std::fabs(s.x_c - stationXc) < std::fabs(nearest->x_c - stationXc))
            nearest = &s;
    }
    if (!nearest) return VGAuditReadout{};
    return auditVGVortexStrength(
        app.solver, c.vgs.front(), app.layout.chordCells, nearest->ueEdge,
        nearest->delta99_c * static_cast<float>(app.layout.chordCells));
}

/// Configure the solver for sweep case @p idx and cold-restart it. Reuses the
/// interactive per-case primitive applyGeometryCold (which rebuilds flags,
/// applies the VG-resolution refinement, and clears the NaN latch on cold
/// restart), so a sweep case is set up exactly like a hand edit.
void setupSweepCase(App& app, int idx) {
    const SweepCase& c = app.sweep.cases[static_cast<std::size_t>(idx)];
    app.params.aoaDeg = c.aoaDeg;
    app.params.vgs    = c.vgs;             // empty for a clean-airfoil case
    app.params.selectedVG = c.vgs.empty() ? -1 : 0;
    app.params.running = true;             // make sure the sim steps
    applyGeometryCold(app);                // rebuild + cold restart + preconverge
    app.sweep.caseStartStep = app.solver.stepCount();
    app.sweep.caseStartWall = platform::timerSeconds();
}

/// Advance the machine to the next case, or finalize when the list is done.
void advanceSweep(App& app);

/// Snapshot the current (converged or failed) case into the results list.
void recordSweepCase(App& app, bool diverged, bool converged) {
    auto& run = app.sweep;
    SweepResult r;
    r.config      = run.cases[static_cast<std::size_t>(run.current)];
    r.forces      = app.solver.forces();
    r.diverged    = diverged;
    r.converged   = converged;
    r.steps       = app.solver.stepCount() - run.caseStartStep;
    r.wallSeconds = platform::timerSeconds() - run.caseStartWall;
    if (converged && !diverged) r.audit = runAuditForCase(app, r.config);
    run.results.push_back(std::move(r));
}

/// Write @p text to @p path (binary, so the LF line endings survive verbatim),
/// returning false on open failure. Modeled on logOpen's UTF-8-safe fopen; used
/// by the sweep CSV export.
bool writeTextFile(const std::filesystem::path& path, const std::string& text) {
    std::FILE* f = nullptr;
#ifdef _WIN32
    _wfopen_s(&f, path.wstring().c_str(), L"wb");
#else
    f = std::fopen(path.string().c_str(), "wb");
#endif
    if (!f) return false;
    std::fwrite(text.data(), 1, text.size(), f);
    std::fclose(f);
    return true;
}

/// Finalize a completed sweep: log a one-line summary pointing at the best case.
void finalizeSweep(App& app) {
    auto& run = app.sweep;
    run.phase = App::SweepPhase::Done;
    app.params.running = false; // stop on the last case so the user can inspect
    if (run.results.empty()) { setStatus(app, "sweep finished: no results"); return; }
    const std::vector<int> order = rankSweepResults(run.results);
    const SweepResult& best = run.results[static_cast<std::size_t>(order.front())];
    char msg[200];
    std::snprintf(msg, sizeof msg,
                  "sweep finished: %zu cases. Best AoA %.1f deg -> L/D %.2f "
                  "(Cl %.3f, Cd %.4f)",
                  run.results.size(), best.config.aoaDeg,
                  best.converged ? best.forces.ldMedian : best.forces.liftToDrag,
                  best.converged ? best.forces.clAvg : best.forces.cl,
                  best.converged ? best.forces.cdAvg : best.forces.cd);
    logLine(msg);
    setStatus(app, msg);
}

void advanceSweep(App& app) {
    auto& run = app.sweep;
    ++run.current;
    if (run.current >= static_cast<int>(run.cases.size())) {
        finalizeSweep(app);
        return;
    }
    setupSweepCase(app, run.current);
    run.phase = App::SweepPhase::Converging;
}

/// Tick the sweep state machine once per frame (after updateReadouts, so
/// forces() is fresh). Non-blocking: it observes convergence and advances at
/// most one phase per frame, letting the normal frame loop step the solver and
/// render each case live.
void tickSweep(App& app) {
    auto& run = app.sweep;
    if (run.phase != App::SweepPhase::Converging
        && run.phase != App::SweepPhase::Recording)
        return;
    if (run.paused) return;

    if (run.phase == App::SweepPhase::Converging) {
        // A latched CUDA failure is session-fatal — abandon the sweep.
        if (app.cudaFailure) {
            recordSweepCase(app, /*diverged=*/true, /*converged=*/false);
            run.phase = App::SweepPhase::Cancelled;
            app.params.running = false;
            setStatus(app, "sweep cancelled: CUDA failure");
            return;
        }
        // NaN divergence: record and move on (the next cold restart clears it).
        if (app.solver.nanDetected()) {
            recordSweepCase(app, /*diverged=*/true, /*converged=*/false);
            advanceSweep(app);
            return;
        }
        // Safety cap: a case that never opens the gate is recorded non-converged.
        const long long ran = app.solver.stepCount() - run.caseStartStep;
        if (ran >= run.params.maxStepsPerCase) {
            recordSweepCase(app, /*diverged=*/false, /*converged=*/false);
            advanceSweep(app);
            return;
        }
        // Converged when the force gate has been open long enough to also have
        // collected the requested extra averaging window.
        const ForceReadout f = app.solver.forces();
        if (f.valid
            && f.flowThroughs
                   >= kForceGateFlowThroughs + run.params.averagingFlowThroughs) {
            run.phase = App::SweepPhase::Recording;
        }
        return;
    }

    // Recording (one frame): snapshot the converged case, then advance.
    recordSweepCase(app, /*diverged=*/false, /*converged=*/true);
    advanceSweep(app);
}

// ===========================================================================
// UI event application — the ONE place edit ordering lives
// (voxelize -> snapshot -> solver, per ui.h's contract).
// ===========================================================================

void applyEvents(App& app) {
    UIEvents& ev = app.events;
    std::string err;

    // ---- mid-sweep gating: while a case is in flight the sweep machine OWNS
    // the geometry/AoA/refinement, so swallow any panel edits that would
    // restart or re-voxelize the running case out from under it. The sweep's
    // own controls (start/pause/cancel/export/scrub) are honored below. ----
    if (app.sweep.phase == App::SweepPhase::Converging
        || app.sweep.phase == App::SweepPhase::Recording) {
        ev.aoaChanged = ev.vgEdited = ev.airspeedChanged = false;
        ev.resolutionChanged = ev.meshRefinementChanged = false;
        ev.reloadAirfoil = ev.highFidelityToggled = false;
    }

    // ---- catalog + aircraft manifest (plan 15.5: re-read on refresh) ----
    if (ev.refreshAirfoils) {
        app.catalog = scanAirfoilDirectory(app.airfoilsDir);
        reloadAircraftManifest(app);
        setStatus(app, "scanned " + std::to_string(app.catalog.size())
                           + " .dat files, "
                           + std::to_string(app.aircraft.entries.size())
                           + " aircraft");
    }

    // ---- full re-init (grid dims change): resolution preset, HiFi, chord ----
    if (ev.resolutionChanged || ev.highFidelityToggled) {
        if (!initSimulation(app, /*reinit=*/true, &err)) {
            setStatus(app, "re-init failed: " + err
                               + " — try a smaller preset");
        } else {
            setStatus(app, ev.highFidelityToggled
                               ? (app.params.highFidelity.enabled
                                      ? "High Fidelity ON: Ultra grid, u_lat 0.05"
                                      : "High Fidelity off")
                               : "grid rebuilt");
        }
    }

    // ---- geometry changes at fixed grid ----
    if (ev.reloadAirfoil) {
        if (resolveAirfoil(app, &err)) {
            applyGeometryCold(app);
            setStatus(app, "loaded " + app.readouts.loadedAirfoilName);
        } else {
            setStatus(app, "airfoil rejected: " + err);
        }
    }
    if (ev.aoaChanged && !app.stlActive) {
        // AoA rotates the geometry: re-voxelize + clean restart.
        applyGeometryCold(app);
        char msg[64];
        std::snprintf(msg, sizeof msg, "AoA -> %.1f deg (sim restarted)",
                      app.params.aoaDeg);
        logLine(msg);
    }

    // ---- airspeed OR chord: units rescale (new dt/tau/Re-target at the same
    // grid and u_lat) followed by a clean restart of the flow. Predictable
    // by design — every edit restarts the sim (cache-removal note, App). ----
    if (ev.airspeedChanged) {
        const LatticeScaling scaling = currentScaling(app.params);
        app.readouts.scaling = scaling;
        app.solver.shutdown();
        if (!app.solver.init(app.layout.dims, scaling, app.activeFlags,
                             app.stream, &err)) {
            setStatus(app, "units-rescale re-init failed: " + err);
        } else {
            // Fresh solver: restore the clean-foil surface reference so the
            // guidance extraction never measures from VG voxels.
            app.solver.setSurfaceReference(app.cleanFlags);
            // A fresh solver resets the force EMA window to the standard 1.0;
            // re-apply the High Fidelity widening (sim-module contract).
            app.solver.setForceEmaWindow(
                app.params.highFidelity.enabled
                    ? app.params.highFidelity.forceEmaFlowThroughs
                    : StandardPreset{}.forceEmaFlowThroughs);
            // solver.init() freed the fine level with everything else; the
            // new scaling also changes the fine tau relation — full rebuild,
            // then re-seed the developed flow.
            applyRefinement(app);
            runPreconverge(app);
            // The flow restarts from rest — stand the tufts back up so they
            // comb over with the fresh ramp instead of lingering mid-bend.
            app.viz.reseedTufts();
            char msg[96];
            std::snprintf(msg, sizeof msg,
                          "airspeed/chord -> %.2f m/s / %.2f m (sim restarted)",
                          app.params.airspeedMs, app.params.chordM);
            logLine(msg);
        }
    }

    // ---- VG edits: cold restart by default; warm in-place continuation when
    // the VG editor's "Keep flow between edits" option is checked ----
    if (ev.vgEdited && !app.stlActive) {
        // A VG count/pitch change can change the VG-aware spanwise period, which
        // changes nz — that's a GRID resize the in-place flag swap can't do, so
        // route those through a full re-init. Other VG edits keep the fast path.
        const int wantNz = defaultLayout(currentChordCells(app.params),
                                         app.params.domain, &app.params.vgs)
                               .dims.nz;
        if (wantNz != app.layout.dims.nz) {
            std::string e;
            if (!initSimulation(app, /*reinit=*/true, &e))
                setStatus(app, "VG span re-init failed: " + e);
            else
                logLine("VG edit — grid span resized to " + std::to_string(wantNz)
                        + " (whole VG periods), full re-init");
        } else {
            const bool warm = applyVGEdit(app);
            char msg[64];
            std::snprintf(msg, sizeof msg, "VG edit — %s (%d VG entries)",
                          warm ? "warm continue" : "cold restart",
                          static_cast<int>(app.params.vgs.size()));
            logLine(msg);
        }
    }

    // ---- Refinement patch toggled or margins changed: rebuild the fine
    // level IN PLACE. The coarse flow is untouched — initRefinement seeds the
    // new fine state from the live coarse field, so no restart and no
    // presolve are needed; the sim keeps running through the change.
    if (ev.meshRefinementChanged && !app.stlActive) {
        applyRefinement(app);
        // A changed patch changes the composite voxel geometry on screen.
        if (app.params.voxelView) uploadRenderGeometry(app);
        logLine(app.params.refine.enabled()
                    ? "refinement patch rebuilt (live)"
                    : "refinement patch disabled");
    }

    // ---- wall-model combo: re-apply the Auto/On/Off policy in place. The
    // solver rebuilds its wall lists internally; the flow keeps running ----
    if (ev.wallModelChanged) {
        applyWallModelPolicy(app);
        logLine(app.solver.wallModelEnabled() ? "wall model ON"
                                              : "wall model off");
    }

    // ---- transition strip: rebuild the trip band live; the flow keeps running.
    if (ev.tripChanged)
        applyTransitionTrip(app);

    // ---- voxel view toggle: swap the render mesh, nothing else changes ----
    if (ev.voxelViewToggled) {
        uploadRenderGeometry(app);
        logLine(app.params.voxelView ? "voxel view ON (solver cells)"
                                     : "voxel view off (smooth mesh)");
    }

    // ---- STL import flow ----
    if (ev.loadStlRequested) {
        const auto path = platform::openFileDialog(
            "Load STL solid", {{"STL mesh", "*.stl"}});
        if (path) handleIncomingFile(app, *path);
    }
    if (ev.stlImportConfirmed) applyStlImport(app);
    if (ev.stlImportCancelled) {
        app.stlMeshRaw = StlMesh{};
        app.vgStlPending = false;
        setStatus(app, "STL import cancelled");
    }

    // ---- custom-VG-STL import flow (rides the parametric foil) ----
    if (ev.loadVgStlRequested) {
        const auto path = platform::openFileDialog(
            "Load VG mesh", {{"STL mesh", "*.stl"}});
        if (path) {
            // Load directly (NOT handleIncomingFile, which routes .stl to the
            // foil modal). Stash the raw mesh and open the shared import modal
            // in forVg mode to collect the axis preset.
            StlLoadResult res = loadStl(*path);
            if (!res.ok) {
                setStatus(app, platform::pathToUtf8(path->filename())
                                   + " rejected: " + res.rejectionReason);
            } else {
                app.stlMeshRaw = std::move(res.mesh);
                app.vgStlPending = true;
                StlImportUI& ui = app.params.stlImport;
                ui.open          = true;
                ui.forVg         = true;
                ui.fileName      = platform::pathToUtf8(path->filename());
                ui.solidName     = app.stlMeshRaw.name;
                ui.triangleCount =
                    static_cast<std::uint32_t>(app.stlMeshRaw.triangles.size());
                ui.bounds        = app.stlMeshRaw.bounds;
                ui.wasBinary     = app.stlMeshRaw.wasBinary;
            }
        }
    }
    if (ev.vgStlImportConfirmed) applyVgStlImport(app);

    // ---- testing suite (sweep) controls ----
    if (ev.sweepStart) {
        auto& run = app.sweep;
        run.params = app.params.sweepParams;
        run.cases  = buildSweepCases(run.params, app.params.vgs);
        run.results.clear();
        run.current = -1;
        run.paused = false;
        run.scrubLoadedCase = -1;
        if (run.cases.empty()) {
            setStatus(app, "sweep: no cases (add at least one AoA)");
            run.phase = App::SweepPhase::Idle;
        } else {
            char msg[120];
            std::snprintf(msg, sizeof msg, "sweep started: %zu cases",
                          run.cases.size());
            setStatus(app, msg);
            advanceSweep(app); // sets up case 0 and enters Converging
        }
    }
    if (ev.sweepPause) {
        app.sweep.paused = !app.sweep.paused;
        // Freeze/resume the sim along with the machine so the case visibly halts.
        app.params.running = !app.sweep.paused;
        setStatus(app, app.sweep.paused ? "sweep paused" : "sweep resumed");
    }
    if (ev.sweepCancel) {
        app.sweep.phase = App::SweepPhase::Cancelled;
        app.params.running = false;
        setStatus(app, "sweep cancelled (partial results kept)");
    }
    if (ev.sweepExportCsv && !app.sweep.results.empty()) {
        const auto path = platform::saveFileDialog(
            "Export sweep CSV", {{"CSV", "*.csv"}}, "foilcfd_sweep.csv");
        if (path) {
            if (writeTextFile(*path, sweepResultsToCsv(app.sweep.results)))
                setStatus(app, "exported " + platform::pathToUtf8(path->filename()));
            else
                setStatus(app, "CSV export failed (could not open file)");
        }
    }
    if (ev.sweepScrubTo) {
        const int sel = app.params.sweepSelectedResult;
        if (sel >= 0 && sel < static_cast<int>(app.sweep.results.size())) {
            // Load this result's case into the interactive view WITHOUT starting
            // the machine, so the user can watch it run live and tweak from there.
            const SweepCase& c = app.sweep.results[static_cast<std::size_t>(sel)].config;
            app.params.aoaDeg = c.aoaDeg;
            app.params.vgs    = c.vgs;
            app.params.selectedVG = c.vgs.empty() ? -1 : 0;
            applyGeometryCold(app);
            app.sweep.scrubLoadedCase = sel;
            setStatus(app, "loaded sweep case into view");
        }
    }

    // ---- sim transport ----
    if (ev.resetCold) {
        app.solver.reset();
        // Flow restarts from rest: stand the tufts back upright with it.
        app.viz.reseedTufts();
        setStatus(app, "cold reset");
    }
    if (ev.singleStep) {
        app.params.running = false;
        if (!app.cudaFailure) {
            if (const cudaError_t serr = app.solver.stepN(1);
                serr != cudaSuccess) {
                latchCudaFailure(app, "single step", serr);
            }
        }
    }
    // ---- view ----
    if (ev.particleCountChanged) {
        if (!app.viz.resizeParticlePool(app.params.viz.particleCount, &err)) {
            setStatus(app, "particle pool resize failed: " + err);
        }
    }
    if (ev.frameFoilView) {
        focusCameraOnFoil(app, /*snap=*/false);
    }
    if (ev.screenshot) {
        // Timestamped name so successive screenshots never clobber each other.
        char name[64];
        std::snprintf(name, sizeof(name), "foilcfd_%lld.png",
                      app.solver.stepCount());
        const auto path = platform::executableDirectory() / "screenshots" / name;
        if (!app.viz.screenshotPNG(path, &err)) {
            setStatus(app, "screenshot failed: " + err);
        } else {
            // UTF-8 conversion: the exe may live under a non-ASCII profile
            // path, where path::string() would throw or mojibake.
            setStatus(app, "saved " + platform::pathToUtf8(path));
        }
    }

    ev.reset();
}

// ===========================================================================
// Camera input: orbit on LMB drag, pan on RMB/MMB/shift+LMB, zoom on scroll,
// all suppressed while ImGui owns the mouse, all spring-smoothed.
// ===========================================================================

void updateCameraInput(App& app, float dt) {
    double cx = 0.0, cy = 0.0;
    glfwGetCursorPos(app.window, &cx, &cy);
    const float dx = static_cast<float>(cx - app.lastCursorX);
    const float dy = static_cast<float>(cy - app.lastCursorY);
    app.lastCursorX = cx;
    app.lastCursorY = cy;

    const bool uiOwns = app.uiActive && uiWantsInput();
    if (!uiOwns) {
        const bool lmb = glfwGetMouseButton(app.window, GLFW_MOUSE_BUTTON_LEFT)
                         == GLFW_PRESS;
        const bool rmb = glfwGetMouseButton(app.window, GLFW_MOUSE_BUTTON_RIGHT)
                         == GLFW_PRESS;
        const bool mmb = glfwGetMouseButton(app.window, GLFW_MOUSE_BUTTON_MIDDLE)
                         == GLFW_PRESS;
        const bool shift =
            glfwGetKey(app.window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS
            || glfwGetKey(app.window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
        if (lmb && !shift) {
            app.camera.rotate(dx, dy);
        } else if (rmb || mmb || (lmb && shift)) {
            app.camera.pan(dx, dy);
        }
        if (app.scrollPending != 0.0f) {
            app.camera.zoom(app.scrollPending);
        }
    }
    app.scrollPending = 0.0f;
    app.camera.update(dt); // springs run even while UI owns input
}

// ===========================================================================
// Main loops.
// ===========================================================================

/// The plan 9.3 frame loop.
int runInteractive(App& app) {
    if (!uiInit(app.window)) {
        std::fprintf(stderr, "ImGui init failed\n");
        return 1;
    }
    app.uiActive = true;
    glfwGetCursorPos(app.window, &app.lastCursorX, &app.lastCursorY);
    app.lastFrameT = platform::timerSeconds();

    while (!glfwWindowShouldClose(app.window)) {
        glfwPollEvents();

        // -- frame timing for the camera springs --
        const double now = platform::timerSeconds();
        const float dt = static_cast<float>(now - app.lastFrameT);
        app.lastFrameT = now;

        // -- dropped files queued by the GLFW callback. GLFW documents drop
        // paths as UTF-8; constructing filesystem::path from the narrow
        // string directly would decode them via the ANSI code page on MSVC
        // and mangle any non-ASCII path (plan 7.1 drag-and-drop). --
        for (const std::string& f : app.droppedFiles) {
            handleIncomingFile(app, platform::pathFromUtf8(f));
        }
        app.droppedFiles.clear();

        // -- camera: input mapping + spring smoothing --
        updateCameraInput(app, dt);

        // -- sim: adaptive N steps under the TDR-safe budget. The returned
        // cudaError_t is the END of the whole launch-wrapper error chain —
        // dropping it here would leave a TDR'd session spinning silently on
        // a frozen field, so any failure latches and stops the sim. --
        int stepsThisFrame = 0;
        if (app.params.running && !app.solver.nanDetected() && !app.cudaFailure) {
            stepsThisFrame = app.solver.adaptiveStepsForBudget();
            if (stepsThisFrame > 0) {
                if (const cudaError_t err = app.solver.stepN(stepsThisFrame);
                    err != cudaSuccess) {
                    latchCudaFailure(app, "sim step", err);
                }
            }
        }

        // -- particles + slices: interop map/unmap inside updateFields.
        // Skipped entirely once a CUDA failure is latched: the interop
        // map/unmap calls would fail the same way every frame. --
        // Keep the volume's freestream baseline in lockstep with the active
        // inflow (HiFi lowers u_lat), so the "calm air" haze threshold tracks
        // the real freestream rather than a stale default.
        app.params.viz.freestreamLatticeSpeed = currentULat(app.params);
        if (!app.cudaFailure) {
            const LBMSolver::GridResolutionField grf =
                app.solver.gridResolutionField();
            const GridResolution gridRes{grf.tauField, grf.nuWall, grf.dxRatioMax};
            if (const cudaError_t err = app.viz.updateFields(
                    app.solver.velocityField(), app.solver.deviceRho(),
                    app.solver.deviceFlags(),
                    static_cast<float>(stepsThisFrame), app.params.viz, gridRes);
                err != cudaSuccess) {
                latchCudaFailure(app, "render field update", err);
            }
        }

        // -- draw scene, then UI on top (pure GL — still fine after a CUDA
        // failure; the UI must keep rendering to show the diagnosis) --
        int fbw = 0, fbh = 0;
        glfwGetFramebufferSize(app.window, &fbw, &fbh);
        app.viz.drawFrame(app.camera, app.params.viz, fbw, fbh);

        updateReadouts(app);
        // Guidance extraction and snapshot capture both issue device work —
        // pointless (and noisy) once the context is broken.
        if (!app.cudaFailure) {
            refreshGuidance(app);
        }
        // Tick the testing-suite sweep AFTER readouts/guidance so it sees fresh
        // forces() and a current delta99 profile (the per-case audit reuses it),
        // and may cold-restart the next case for the following frame to render.
        tickSweep(app);

        // Status messages decay so stale errors don't linger for the session.
        if (!app.status.empty() && now - app.statusT > kStatusLifetime) {
            app.status.clear();
        }

        uiBeginFrame();
        UIContext ctx;
        ctx.params = &app.params;
        ctx.readouts = &app.readouts;
        ctx.events = &app.events;
        ctx.airfoilCatalog = &app.catalog;
        ctx.aircraftManifest = &app.aircraft.entries;
        ctx.vgMeshNames = &app.vgMeshNames;
        ctx.statusMessage = app.status;
        // Camera matrix for world-space line overlays (the patch box): same
        // aspect as drawFrame so the projected lines land exactly on the GL
        // scene underneath.
        const Mat4f viewProj =
            app.camera.projMatrix(fbh > 0 ? static_cast<float>(fbw) / fbh
                                          : 1.0f)
            * app.camera.viewMatrix();
        ctx.viewProj = &viewProj;
        drawUI(ctx);
        uiEndFrame();

        applyEvents(app);
        glfwSwapBuffers(app.window);
    }
    uiShutdown();
    app.uiActive = false;
    return 0;
}

/// --selftest: init everything headless-ish (hidden window), run 200 steps,
/// render one frame, write screenshots/selftest.png, print PASS. Exit code 0
/// only if every stage succeeded — this is the build-verification hook.
int runSelftest(App& app) {
    // First half absorbs one-time launch latency (module load, allocator
    // growth, ramp shock) so the timed half reports steady-state throughput.
    if (cudaError_t err = app.solver.stepN(100); err != cudaSuccess) {
        std::fprintf(stderr, "selftest: stepN failed: %s\n",
                     cudaGetErrorString(err));
        return 1;
    }
    if (cudaError_t err = cudaStreamSynchronize(app.stream); err != cudaSuccess) {
        std::fprintf(stderr, "selftest: stream sync failed: %s\n",
                     cudaGetErrorString(err));
        return 1;
    }

    // Timed half: wall-clock MLUPS over 100 fused steps, sync-bracketed so the
    // measurement covers exactly the lattice work (perf sanity, plan 4.6/11).
    const double t0 = platform::timerSeconds();
    if (cudaError_t err = app.solver.stepN(100); err != cudaSuccess) {
        std::fprintf(stderr, "selftest: stepN failed: %s\n",
                     cudaGetErrorString(err));
        return 1;
    }
    if (cudaError_t err = cudaStreamSynchronize(app.stream); err != cudaSuccess) {
        std::fprintf(stderr, "selftest: stream sync failed: %s\n",
                     cudaGetErrorString(err));
        return 1;
    }
    const double elapsed = platform::timerSeconds() - t0;
    const double cells = static_cast<double>(app.layout.dims.cellCount());
    const double mlups = (elapsed > 0.0) ? cells * 100.0 / (elapsed * 1e6) : 0.0;
    std::printf("selftest: %.0f MLUPS (100 timed steps, %.3f s, grid %dx%dx%d)\n",
                mlups, elapsed, app.layout.dims.nx, app.layout.dims.ny,
                app.layout.dims.nz);

    // One full render pass so the screenshot has defined contents. The
    // selftest exists to prove the GL + CUDA + interop plumbing — a dropped
    // updateFields failure here would defeat its whole purpose.
    app.params.viz.freestreamLatticeSpeed = currentULat(app.params);
    if (cudaError_t err = app.viz.updateFields(
            app.solver.velocityField(), app.solver.deviceRho(),
            app.solver.deviceFlags(), 200.0f, app.params.viz);
        err != cudaSuccess) {
        std::fprintf(stderr, "selftest: updateFields failed: %s\n",
                     cudaGetErrorString(err));
        return 1;
    }
    int fbw = 0, fbh = 0;
    glfwGetFramebufferSize(app.window, &fbw, &fbh);
    app.viz.drawFrame(app.camera, app.params.viz, fbw, fbh);

    std::string err;
    const auto shotPath = std::filesystem::path("screenshots") / "selftest.png";
    if (!app.viz.screenshotPNG(shotPath, &err)) {
        std::fprintf(stderr, "selftest: screenshot failed: %s\n", err.c_str());
        return 1;
    }
    std::printf("selftest: %lld steps total, screenshot at %s\n",
                app.solver.stepCount(),
                platform::pathToUtf8(shotPath).c_str());

    // Re-init survival check: a resolution/HiFi/airspeed change rebuilds the
    // solver AND the renderer in place, and a latent stale-latch class of bug
    // (pending async readbacks surviving event destruction) once poisoned the
    // first post-rebuild launch with cudaErrorInvalidResourceHandle. Exercise
    // the heaviest interop combination through a full re-init — velocity
    // volume and Q isosurface both live, so their 3D textures are released
    // and lazily recreated — then render and step again. Runs AFTER the
    // screenshot so the PNG keeps the developed-flow content.
    {
        std::string rerr;
        app.params.viz.showVelocityVolume = true;
        app.params.viz.showQRaycast = true;
        if (!initSimulation(app, /*reinit=*/true, &rerr)) {
            std::fprintf(stderr, "selftest: re-init failed: %s\n", rerr.c_str());
            return 1;
        }
        app.params.viz.freestreamLatticeSpeed = currentULat(app.params);
        if (cudaError_t e = app.viz.updateFields(
                app.solver.velocityField(), app.solver.deviceRho(),
                app.solver.deviceFlags(), 1.0f, app.params.viz);
            e != cudaSuccess) {
            std::fprintf(stderr, "selftest: post-reinit updateFields: %s\n",
                         cudaGetErrorString(e));
            return 1;
        }
        app.viz.drawFrame(app.camera, app.params.viz, fbw, fbh);
        if (cudaError_t e = app.solver.stepN(20); e != cudaSuccess) {
            std::fprintf(stderr, "selftest: post-reinit stepN: %s\n",
                         cudaGetErrorString(e));
            return 1;
        }
        cudaStreamSynchronize(app.stream);
        std::printf("selftest: re-init cycle OK\n");
    }

    std::printf("PASS\n");
    return 0;
}

// ===========================================================================
// --vgtest: headless VG-stability harness. Reproduces the reported divergence
// configs (high-factor VG cascade box + ISLBM, from rest, at high AoA) and
// reports STABLE/DIVERGED per case. Hard-capped on both step count AND wall
// time so it never monopolizes the GPU. Exists so the from-rest VG shockwave
// fix can be verified without the interactive UI.
// ===========================================================================
int runVgTest(App& app) {
    using MeshMode = UIParams::MeshMode;

    // One scenario = a named configuration to drive to convergence-or-blowup.
    struct Scenario {
        const char* name;
        float aoaDeg;
        MeshMode mode;
        int   vgTargetCells; // drives the cascade depth (8x path) when high
        bool  finerVGPatch;
    };
    // These mirror the two pasted crash reports: a 6x-ish cascade VG box at the
    // working AoA, and the ISLBM/stretch case at the stall AoA.
    // target=8 with finerVGPatch gives a 6x cascade box (3x fine + 2x nested) —
    // exactly the shape the crash reports showed diverging (nested VG patch,
    // from rest, ~step 200, AoA 4 and 18). The pathological 8x/16x boxes are too
    // heavy to test interactively (tens of millions of cells) and are not a
    // config to default to, so the harness verifies the realistic crash shapes.
    const Scenario scenarios[] = {
        {"cascade-VG-6x  AoA4",  4.0f,  MeshMode::Cascade, 8, true},
        {"cascade-VG-6x  AoA18", 18.0f, MeshMode::Cascade, 8, true},
        {"stretch-VG     AoA18", 18.0f, MeshMode::Stretch, 8, true},
        // ISLBM continuous-refine: dense at the VG, coarse elsewhere, ONE grid
        // (no full-span box) — the blazing-real-time high-res VG path. Reports
        // its MLUPS + effective near-wall k so we see it's both stable AND fast.
        {"stretch-VG-refine AoA18", 18.0f, MeshMode::Stretch, 16, true},
    };

    // Budget guards: never exceed these regardless of scenario count. The
    // reported divergence hit at step ~200-400, so 500 steps past the pre-steps
    // window is plenty to prove the from-rest pulse no longer forms. Small
    // batches so the per-case wall cap can actually interrupt a slow box (the
    // cap is only checked BETWEEN batches).
    constexpr int    kStepsPerCase   = 500;
    constexpr int    kBatch          = 25;
    constexpr double kWallCapSeconds = 360.0; // 6 min hard cap for the WHOLE run
    constexpr double kPerCaseCap     = 90.0;  // and per-case, so one slow deep
                                              // cascade can't eat the whole run
    const double t0 = platform::timerSeconds();

    int diverged = 0, ran = 0;
    for (const Scenario& sc : scenarios) {
        if (platform::timerSeconds() - t0 > kWallCapSeconds) {
            std::printf("vgtest: WALL-TIME CAP hit — stopping early "
                        "(%d/%zu cases run)\n", ran,
                        sizeof scenarios / sizeof scenarios[0]);
            break;
        }
        ++ran;

        // Configure: one Strausak-style counter-rotating pair on the suction
        // surface, the scenario's AoA + mesh mode + VG-resolution target.
        app.params.aoaDeg = sc.aoaDeg;
        app.params.refine.meshMode      = sc.mode;
        app.params.refine.vgTargetAuto  = true;
        app.params.refine.vgTargetCells = sc.vgTargetCells;
        app.params.refine.finerVGPatch  = sc.finerVGPatch;
        app.params.vgs.clear();
        app.params.vgs.push_back(defaultVGParams()); // counter-rotating pair
        app.params.selectedVG = 0;
        applyGeometryCold(app); // rebuild + cold restart (re-arms the ramp)

        // Step in capped batches, watching the NaN watchdog.
        bool blew = false, timedOut = false;
        const double caseT0 = platform::timerSeconds();
        long long startStep = app.solver.stepCount();
        for (int s = 0; s < kStepsPerCase; s += kBatch) {
            if (platform::timerSeconds() - t0 > kWallCapSeconds) { timedOut = true; break; }
            if (platform::timerSeconds() - caseT0 > kPerCaseCap) { timedOut = true; break; }
            if (app.solver.stepN(kBatch) != cudaSuccess) { blew = true; break; }
            cudaStreamSynchronize(app.stream);
            if (app.solver.nanDetected()) { blew = true; break; }
        }
        const long long did = app.solver.stepCount() - startStep;
        const double caseSec = platform::timerSeconds() - caseT0;
        // Real-time proof: MLUPS over this case + the effective near-wall
        // resolution the VG actually got (stretch k, or cascade factor).
        const double cells = static_cast<double>(app.layout.dims.cellCount());
        const double mlups = (caseSec > 1e-6)
            ? (cells * static_cast<double>(did)) / caseSec / 1e6 : 0.0;
        const StretchInfo si = app.solver.stretchInfo();
        const RefinementInfo ri = app.solver.refinementInfo();
        float vgK = 1.0f;
        if (si.active) vgK = si.nearWallFactor;
        else if (ri.finerActive) vgK = static_cast<float>(ri.factor * ri.finerFactor);
        else if (ri.active) vgK = static_cast<float>(ri.factor);
        const float vgCells = defaultVGParams().height_c
                            * static_cast<float>(app.layout.chordCells) * vgK;
        if (blew) {
            ++diverged;
            std::printf("vgtest: [DIVERGED] %-22s at step %lld — %s\n", sc.name,
                        did, app.solver.nanDiagnosis().c_str());
        } else {
            std::printf("vgtest: [stable%s]  %-22s %lld steps  %.0f MLUPS  "
                        "VG k=%.1f (~%.1f cells)\n",
                        timedOut ? "*" : " ", sc.name, did, mlups, vgK, vgCells);
        }
        std::fflush(stdout); // live progress even when piped/redirected
    }

    std::printf("vgtest: %d/%d cases ran, %d diverged. %s\n", ran,
                static_cast<int>(sizeof scenarios / sizeof scenarios[0]),
                diverged, diverged == 0 ? "PASS" : "FAIL");
    return diverged == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    bool vgtest   = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        if (std::strcmp(argv[i], "--vgtest") == 0)   vgtest   = true;
    }
    // The VG stability harness runs headless like selftest (hidden window).
    const bool headless = selftest || vgtest;

    // Session log first: everything from here on (statuses, errors, NaN
    // trips) lands in logs/foilcfd.log so issues are diagnosable after the
    // fact even when the on-screen status line has already decayed.
    logOpen();
    logLine(std::string("FoilCFD starting (")
            + (selftest ? "selftest" : vgtest ? "vgtest" : "interactive") + ")");

    App app;
    std::string error;
    // Init order matters: GL context first (interop registration needs it
    // current), then CUDA, then sim + renderer.
    if (!initWindowAndGL(app, /*visible=*/!headless, &error) ||
        !initCuda(app, &error)) {
        logLine("FATAL: init failed: " + error);
        glfwTerminate();
        logClose();
        return 1;
    }

    // Data directories before sim init: the airfoil catalog (recursive,
    // includes uiuc/).
    app.airfoilsDir = findAirfoilsDirectory();
    app.catalog = scanAirfoilDirectory(app.airfoilsDir);
    reloadAircraftManifest(app); // plan 15.5: aircraft rows link into the catalog

    // Default geometry: the Glasair III's airfoil (NASA LS(1)-0413MOD) from
    // the bundled catalog. Selftest keeps the deterministic NACA section so
    // its golden screenshot/forces stay reproducible; vgtest wants the real
    // LS(1)-0413 to reproduce the reported VG divergence faithfully.
    if (!selftest) tryLoadDefaultAirfoil(app);

    if (!initSimulation(app, /*reinit=*/false, &error)) {
        logLine("FATAL: init failed: " + error);
        // initSimulation can fail PARTWAY with live GPU state (e.g. solver up,
        // viz failed at slice-texture registration after interop buffers were
        // already registered). Tear those down NOW, while the GL context and
        // window still exist — App's destructor runs after glfwTerminate, and
        // unregistering interop resources whose GL objects are gone is
        // driver-dependent at best, a crash at worst. Mirrors the success-
        // path teardown order below.
        app.viz.shutdown();
        app.solver.shutdown();
        if (app.stream) cudaStreamDestroy(app.stream);
        glfwDestroyWindow(app.window);
        glfwTerminate();
        logClose();
        return 1;
    }

    // Baseline configuration line: incident reports begin with a known state.
    logLine(configSummary(app));

    const int rc = selftest ? runSelftest(app)
                 : vgtest   ? runVgTest(app)
                            : runInteractive(app);

    // Orderly teardown: GPU resources before the GL context dies.
    app.viz.shutdown();
    app.solver.shutdown();
    if (app.stream) cudaStreamDestroy(app.stream);
    glfwDestroyWindow(app.window);
    glfwTerminate();
    logLine(rc == 0 ? "clean exit" : "exit with error code");
    logClose();
    return rc;
}
