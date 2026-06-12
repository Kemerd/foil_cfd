// Application entry point: owns the plan 9.3 frame loop
// (poll input -> N sim steps -> particle update -> map interop -> draw ->
// ImGui -> present), the UI-event application in its defined order
// (voxelize -> snapshot -> solver), the plan-8 warm-start cache flow, the
// GLFW drop routing (.dat -> airfoil loader, .stl -> import modal), and the
// --selftest mode that proves the GL + CUDA + sim plumbing end-to-end.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include <glad/gl.h>
// GLFW must come after the loader so it does not inject its own GL header.
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "app/aircraft_manifest.h"
#include "app/camera.h"
#include "app/ui.h"
#include "geom/airfoil.h"
#include "geom/stl.h"
#include "geom/vg.h"
#include "geom/voxelizer.h"
#include "geom/voxelizer_stl.cuh"
#include "platform/platform.h"
#include "render/viz.h"
#include "sim/lbm_solver.h"
#include "sim/snapshot.h"
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

// z-face free-slip flags for STL mode (plan 7.4): CellFlag::SlipFront/SlipBack
// landed in lbm_core.cuh and the fused kernel mirrors them via kMirZ, so the
// stamped planes behave as proper specular walls.
constexpr std::uint8_t kCellFlagSlipFront =
    static_cast<std::uint8_t>(CellFlag::SlipFront); ///< z = 0 plane.
constexpr std::uint8_t kCellFlagSlipBack =
    static_cast<std::uint8_t>(CellFlag::SlipBack);  ///< z = nz-1 plane.

/// Build the default domain layout for a chord resolution: the plan 4.6
/// proportions (nx = 3*N_c, ny = 1.25*N_c, nz = 0.375*N_c) reproduce
/// 768 x 320 x 96 at the default 256-cell chord.
DomainLayout defaultLayout(int chordCells) {
    DomainLayout layout;
    layout.chordCells = chordCells;
    layout.dims.nx = 3 * chordCells;
    layout.dims.ny = (5 * chordCells) / 4;
    layout.dims.nz = (3 * chordCells) / 8;
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

    // -- geometry/flag state the cache flow keys against (plan 8) --
    std::string geometryId;              ///< "naca:..", "dat:..", "stl:<hash>".
    std::vector<std::uint8_t> cleanFlags;  ///< Clean foil (no VGs) at current AoA.
    std::vector<std::uint8_t> activeFlags; ///< cleanFlags OR VG voxels = live flags.

    // -- warm-start cache (plan 8) --
    VramSnapshot cleanSnap;              ///< The ONE VRAM-resident clean slot.
    std::unique_ptr<DiskSnapshotCache> diskCache; ///< Owns its store worker
                                         ///< thread (storeAsync) — saves never
                                         ///< block the frame loop.

    // -- STL import state (plan 7) --
    StlMesh stlMeshRaw;                  ///< As loaded, pre-normalization (kept
                                         ///< so resolution changes re-voxelize).
    bool stlActive = false;

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
/// timestamp so the overlay clears itself.
void setStatus(App& app, std::string msg) {
    std::fprintf(stderr, "[foilcfd] %s\n", msg.c_str());
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

/// The exact plan-8 snapshot key for the current state: geometry id, 0.5-deg
/// AoA bucket, u_lat, grid dims, solver version. Airspeed deliberately absent.
SnapshotKey currentKey(const App& app) {
    SnapshotKey key;
    key.geometryId = app.geometryId;
    key.aoaBucket  = SnapshotKey::bucketForAoA(
        app.stlActive ? 0.0f : app.params.aoaDeg);
    key.u_lat = currentULat(app.params);
    key.dims  = app.layout.dims;
    return key;
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

/// Rebuild clean + active flag fields for the current airfoil/AoA/VG state.
/// The clean field is kept separately so VG slider ticks never repeat the
/// O(nx*ny) airfoil parity tests (plan 6.2).
void rebuildFlagFields(App& app) {
    app.cleanFlags = buildCleanFoilFlags(app.airfoil, app.params.aoaDeg,
                                         app.layout);
    app.activeFlags = app.params.vgs.empty()
        ? app.cleanFlags
        : buildFlagsWithVGs(app.params.vgs, app.airfoil, app.params.aoaDeg,
                            app.layout, app.cleanFlags);
}

/// Try to skip the cold-start transient by restoring a disk-cached compact
/// snapshot for the current key (plan 8 disk cache). Only sound for the
/// clean-foil state — the cache stores clean flow, so the flags must match.
void tryDiskWarmRestore(App& app) {
    if (!app.diskCache || !app.params.vgs.empty()) return;
    const SnapshotKey key = currentKey(app);
    const auto path = app.diskCache->find(key);
    if (!path) return;
    CompactSnapshot snap;
    std::string err;
    if (!snap.loadFromFile(*path, &err)) {
        setStatus(app, "cache load failed: " + err);
        return;
    }
    // Exact-match validation: key AND geometry hash, never fuzzy (plan 8).
    if (!(snap.key() == key)) return;
    if (snap.flagsHash() != hashFlags(app.activeFlags.data(),
                                      app.activeFlags.size())) return;
    if (snap.restore(app.solver, app.stream)) {
        setStatus(app, "warm-started from disk cache (" + key.toString() + ")");
    }
}

/// Geometry changed at fixed grid dims (airfoil swap or AoA edit): rebuild
/// flags, cold-restart the solver (plan 13: AoA invalidates the warm cache),
/// then try the disk cache to soften the cold start.
void applyGeometryCold(App& app) {
    rebuildFlagFields(app);
    app.solver.setFlags(app.activeFlags);
    // The live flags may include VG voxels; the delta99/separation-onset
    // extraction must always measure from the CLEAN foil surface, or a vane
    // crossing mid-span would read as the surface (false separation pinned
    // at the VG station — the mission's core readout corrupted).
    app.solver.setSurfaceReference(app.cleanFlags);
    tryDiskWarmRestore(app);
    app.viz.uploadGeometry(app.airfoil, app.params.vgs, app.params.aoaDeg,
                           app.layout);
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
    tryDiskWarmRestore(app);
    // Hand the normalized triangle soup to the renderer: it replaces the
    // extruded-foil prism in the shared mesh VBO, so the drawn solid matches
    // the voxelized one exactly (both consumed the same normalized mesh).
    app.viz.uploadStlMesh(mesh);
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
        StlImportUI& ui = app.params.stlImport;
        ui.open          = true;
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

/// Locate the bundled default STL (the Glasair III) if it ships with this
/// build. Mirrors findAirfoilsDirectory's search so it works in both the dev
/// layout (exe two levels deep under build/) and an installed layout (assets
/// next to the exe). Returns an empty path when no default STL is present —
/// the program then simply boots on the NACA section instead.
std::filesystem::path findDefaultStl() {
    namespace fs = std::filesystem;
    const fs::path exeDir = platform::executableDirectory();
    // Accept a couple of conventional names so dropping a file in assets/stl/
    // "just works" without a rebuild.
    const char* names[] = {"glasair_iii.stl", "glasair3.stl", "default.stl"};
    const fs::path roots[] = {
        fs::current_path() / "assets" / "stl",
        exeDir / "assets" / "stl",
        exeDir / ".." / ".." / "assets" / "stl",
    };
    for (const fs::path& root : roots) {
        for (const char* n : names) {
            std::error_code ec;
            const fs::path c = root / n;
            if (fs::is_regular_file(c, ec)) return fs::weakly_canonical(c, ec);
        }
    }
    return {};
}

/// Load the bundled Glasair III STL as the program's default geometry, if it
/// exists. Reuses the real import path (loadStl -> normalization defaults ->
/// applyStlImport) so the boot geometry is identical to a manual import. A
/// missing file is not an error: the app keeps the NACA default and notes it.
void tryLoadDefaultStl(App& app) {
    const std::filesystem::path stl = findDefaultStl();
    if (stl.empty()) {
        setStatus(app, "default Glasair III STL not bundled — starting on the "
                       "NACA section (drop an .stl to import one)");
        return;
    }
    StlLoadResult res = loadStl(stl);
    if (!res.ok) {
        setStatus(app, "default STL rejected: " + res.rejectionReason);
        return;
    }
    app.stlMeshRaw = std::move(res.mesh);

    // Seed the import modal's choices with sensible defaults (same as a manual
    // import would start from), then voxelize straight away — no modal, this
    // is the startup default.
    StlImportUI& ui = app.params.stlImport;
    ui.open          = false;
    ui.fileName      = platform::pathToUtf8(stl.filename());
    ui.solidName     = app.stlMeshRaw.name;
    ui.triangleCount = static_cast<std::uint32_t>(app.stlMeshRaw.triangles.size());
    ui.bounds        = app.stlMeshRaw.bounds;
    ui.wasBinary     = app.stlMeshRaw.wasBinary;
    ui.chordCells    = app.layout.chordCells;
    applyStlImport(app);
}

// ===========================================================================
// Snapshot capture (plan 8): the VRAM clean slot for instant VG restarts and
// the compact disk variant persisted on a worker thread.
// ===========================================================================

/// Capture the current (clean!) flow into the VRAM slot, and optionally into
/// the LRU disk cache. Refuses while VGs are stamped — the whole point of the
/// clean slot is restoring the un-VG'd wing.
void captureCleanState(App& app, bool toDisk) {
    if (!app.params.vgs.empty()) {
        setStatus(app, "clean-state capture skipped: VGs are present");
        return;
    }
    const SnapshotKey key = currentKey(app);
    std::string err;
    if (!app.cleanSnap.capture(app.solver, key, app.stream, &err)) {
        setStatus(app, "VRAM snapshot failed: " + err);
        return;
    }
    if (toDisk && app.diskCache) {
        auto compact = std::make_shared<CompactSnapshot>();
        if (compact->capture(app.solver, key, app.stream, &err)) {
            // The cache's own worker thread streams the (hundreds of MB)
            // write to disk (plan 8: "saved on a worker thread"); the
            // shared_ptr keeps the snapshot alive until then. Crucially this
            // ENQUEUES and returns — a bespoke join-then-spawn here used to
            // stall the render thread for seconds whenever a previous save
            // was still in flight on a slow disk.
            app.diskCache->storeAsync(std::move(compact));
        } else {
            setStatus(app, "compact snapshot failed: " + err);
        }
    }
    setStatus(app, "clean flow cached (" + key.toString() + ")");
}

/// Auto-cache policy (plan 9.2 toggle): once the clean-foil flow converges
/// (force gate open) and the VRAM slot doesn't already hold this key, capture
/// it so the user's first VG edit is instant.
void maybeAutoCache(App& app) {
    if (!app.params.autoCacheClean || !app.params.vgs.empty()) return;
    if (!app.readouts.forces.valid) return;
    const SnapshotKey key = currentKey(app);
    if (app.cleanSnap.hasData() && app.cleanSnap.key() == key) return;
    captureCleanState(app, /*toDisk=*/true);
}

/// VG edit flow (plan 6.2/8): restore the clean snapshot when it matches the
/// current clean-foil state, then applyEditedFlags — NEVER a cold restart.
/// Without a matching snapshot the new flags still apply warm onto the live
/// field (correct everywhere except near the edited vanes, which is exactly
/// where the solver re-converges first).
void applyVGEdit(App& app) {
    app.activeFlags = app.params.vgs.empty()
        ? app.cleanFlags
        : buildFlagsWithVGs(app.params.vgs, app.airfoil, app.params.aoaDeg,
                            app.layout, app.cleanFlags);
    const bool snapUsable =
        app.cleanSnap.hasData() && app.cleanSnap.key() == currentKey(app)
        && app.cleanSnap.flagsHash() == hashFlags(app.cleanFlags.data(),
                                                  app.cleanFlags.size());
    if (snapUsable) {
        app.cleanSnap.restore(app.solver, app.stream);
    }
    app.solver.applyEditedFlags(app.activeFlags);
    app.viz.uploadGeometry(app.airfoil, app.params.vgs, app.params.aoaDeg,
                           app.layout);
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
    // Grid invalidates everything VRAM-resident, including the clean slot.
    if (reinit) {
        app.cleanSnap.release();
        app.solver.shutdown();
        app.viz.shutdown();
    }

    const int chordCells = currentChordCells(app.params);
    app.layout = defaultLayout(chordCells);

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
    // STL mode renders the imported triangles; section mode the foil prism.
    if (app.stlActive) {
        app.viz.uploadStlMesh(stlMeshNormalized);
    } else {
        app.viz.uploadGeometry(app.airfoil, app.params.vgs, app.params.aoaDeg,
                               app.layout);
    }
    app.camera.frameDomain(app.layout.dims.nx, app.layout.dims.ny,
                           app.layout.dims.nz);
    focusCameraOnFoil(app, /*snap=*/true);
    resetSliceDefaults(app);
    tryDiskWarmRestore(app);
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
    app.readouts.nanTripped = app.solver.nanDetected();
    app.readouts.nanDiagnosis = app.solver.nanDiagnosis();
    app.readouts.cudaErrorTripped = app.cudaFailure;
    app.readouts.cudaErrorDiagnosis = app.cudaFailureMsg;
    app.readouts.forces = app.solver.forces();

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
}

// ===========================================================================
// UI event application — the ONE place edit ordering lives
// (voxelize -> snapshot -> solver, per ui.h's contract).
// ===========================================================================

void applyEvents(App& app) {
    UIEvents& ev = app.events;
    std::string err;

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
        // Plan 13: AoA rotates the geometry -> warm cache key changes ->
        // cold restart (softened by the per-AoA-bucket disk cache).
        applyGeometryCold(app);
    }

    // ---- airspeed OR chord: pure units rescale, flow field stays valid
    // (plan 13). Grid dims, u_lat, and the flag field are untouched — only
    // dx/dt/tau/Re-target change — so this must NEVER cold-start or drop
    // the warm cache (a chord sweep to study Re trends rides this path). ----
    if (ev.airspeedChanged) {
        const LatticeScaling scaling = currentScaling(app.params);
        app.readouts.scaling = scaling;
        // Workaround for the proposed LBMSolver::updateScaling (see
        // notes/integration_app.md): u_lat is unchanged, so the lattice state
        // is exactly reusable — re-init with the new tau/dt, then restore the
        // clean snapshot to skip the cold transient when we have one.
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
            if (app.cleanSnap.hasData()
                && app.cleanSnap.key() == currentKey(app)) {
                app.cleanSnap.restore(app.solver, app.stream);
                // Re-apply the live flags: restores the VG'd state warm when
                // VGs are stamped, and is a harmless no-op refresh when clean.
                app.solver.applyEditedFlags(app.activeFlags);
            }
        }
    }

    // ---- VG edits: ALWAYS warm (plan 6.2) ----
    if (ev.vgEdited && !app.stlActive) {
        applyVGEdit(app);
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
        setStatus(app, "STL import cancelled");
    }

    // ---- sim transport ----
    if (ev.resetCold) {
        app.solver.reset();
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
    if (ev.saveCleanState) {
        captureCleanState(app, /*toDisk=*/true);
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
            if (const cudaError_t err = app.viz.updateFields(
                    app.solver.velocityField(), app.solver.deviceRho(),
                    app.solver.deviceFlags(),
                    static_cast<float>(stepsThisFrame), app.params.viz);
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
            maybeAutoCache(app);
        }

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
        ctx.statusMessage = app.status;
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

} // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
    }

    App app;
    std::string error;
    // Init order matters: GL context first (interop registration needs it
    // current), then CUDA, then sim + renderer.
    if (!initWindowAndGL(app, /*visible=*/!selftest, &error) ||
        !initCuda(app, &error)) {
        std::fprintf(stderr, "FoilCFD init failed: %s\n", error.c_str());
        glfwTerminate();
        return 1;
    }

    // Data directories before sim init: the airfoil catalog (recursive,
    // includes uiuc/) and the LRU snapshot disk cache next to the exe
    // (plan 8) — created first so even the first cold start can warm-restore.
    app.airfoilsDir = findAirfoilsDirectory();
    app.catalog = scanAirfoilDirectory(app.airfoilsDir);
    reloadAircraftManifest(app); // plan 15.5: aircraft rows link into the catalog
    app.diskCache = std::make_unique<DiskSnapshotCache>(
        platform::executableDirectory() / "cache");

    if (!initSimulation(app, /*reinit=*/false, &error)) {
        std::fprintf(stderr, "FoilCFD init failed: %s\n", error.c_str());
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
        return 1;
    }

    // Default geometry: load the bundled Glasair III STL for the interactive
    // session (if present). Selftest keeps the deterministic NACA section so
    // its golden screenshot/forces stay reproducible and don't depend on an
    // optional asset.
    if (!selftest) {
        tryLoadDefaultStl(app);
        focusCameraOnFoil(app, /*snap=*/true); // re-frame on the new solid
    }

    const int rc = selftest ? runSelftest(app) : runInteractive(app);

    // Orderly teardown: drain the snapshot disk cache's writer queue first
    // (host/disk state only — destroying the cache joins its worker), then
    // GPU resources before the GL context dies.
    app.diskCache.reset();
    app.viz.shutdown();
    app.solver.shutdown();
    if (app.stream) cudaStreamDestroy(app.stream);
    glfwDestroyWindow(app.window);
    glfwTerminate();
    return rc;
}
