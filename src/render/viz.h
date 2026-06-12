// Renderer front-end (plan section 9.1): owns all GL resources and the
// CUDA<->GL interop registrations (registered ONCE at init — re-registering
// per frame silently tanks performance, plan section 13). Particle hero mode
// (1M GL_POINTS), slice planes via surface-object writes into GL textures,
// extruded foil/VG mesh, viridis/coolwarm colormaps, and PNG screenshots.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "../app/camera.h"
#include "../core/vec.h"
#include "../geom/airfoil.h"
#include "../geom/vg.h"
#include "../sim/lbm_core.cuh"

namespace foilcfd {

// Defined in geom/stl.h; forward-declared so the renderer header stays light
// (only uploadStlMesh takes it, by const reference).
struct StlMesh;

/// Hero-mode particle count (plan 9.1: 1,000,000 GL_POINTS).
inline constexpr int kDefaultParticleCount = 1'000'000;

/// @brief Colormaps offered for field rendering. The quantitative views keep
/// the plan-9.1 pairing — sequential viridis for speed-like scalars, diverging
/// coolwarm for signed quantities (vorticity-z, pressure deviation) — plus
/// two hero palettes for the 3D modes: inferno (perceptually uniform heat)
/// and rainbow (full HSV sweep, the classic wind-tunnel speed-band look that
/// the Q-isosurface defaults to). Numeric order matches the shader/kernel
/// palette selector (0/1/2/3).
enum class Colormap { Viridis, Coolwarm, Inferno, Rainbow };

/// @brief What scalar drives particle color (plan 9.1).
enum class ParticleColorBy { Speed, VorticityMag };

/// @brief Scalar shown on a slice plane (plan 9.1 mode 2).
enum class SliceField { SpeedMag, VorticityZ, Pressure };

/// @brief Which axis a slice plane is perpendicular to.
enum class SliceAxis { X, Y, Z };

/// @brief One movable axis-aligned slice plane.
///
/// The renderer keeps ONE CUDA-registered GL texture per axis (sized for that
/// axis's cross-section and registered exactly once at init — plan section 13
/// forbids re-registration, so textures are never reallocated on axis moves).
/// Consequently at most one ENABLED slice per axis is drawn; if two enabled
/// slots share an axis the lower-indexed slot wins and the other is skipped.
struct SliceConfig {
    bool      enabled = false;
    SliceAxis axis    = SliceAxis::Z;
    int       cell    = -1;       ///< Plane position along `axis`, in cells;
                                  ///< negative = domain center (the default).
    SliceField field  = SliceField::VorticityZ;
    Colormap  colormap = Colormap::Coolwarm;
    float     rangeScale = 1.0f;  ///< Multiplier on the field's default
                                  ///< normalization range (UI contrast knob).
};

/// @brief All user-facing view settings (the UI's View panel binds here).
/// Modes can overlay freely (plan 9.1: hotkeys 1/2/3, all can overlay).
struct VizSettings {
    // -- mode toggles --
    bool showParticles = false;   ///< Mode 1 (off by default: the velocity
                                  ///< volume below is the new hero mode).
    bool showSlices    = false;   ///< Mode 2.
    bool showFoilMesh  = true;    ///< Extruded foil + VG geometry.
    bool foilWireframe = false;   ///< Draw the foil/VG mesh as wireframe lines
                                  ///< instead of shaded solid (view option).

    // -- velocity volume (mode 5, the translucent "smoke" option) --
    bool  showVelocityVolume = false; ///< Off by default — the opaque Q
                                      ///< isosurface below is the hero mode;
                                      ///< this fog view layers on top when
                                      ///< wanted.
    Colormap volumeColormap  = Colormap::Inferno; ///< Hot map by default.
    float velocitySpeedScale = 0.16f; ///< Speed (lattice units) mapped to the
                                      ///< palette top — ~2x freestream u_lat so
                                      ///< the wake/suction peak spans the ramp.
    float freestreamLatticeSpeed = 0.08f; ///< Inlet speed u_lat in lattice
                                      ///< units. Main.cpp keeps this in sync
                                      ///< with the active scaling (HiFi lowers
                                      ///< it); the volume shader measures "calm
                                      ///< air" disturbance from this baseline.
    float slowAirOpacity     = 0.05f; ///< Floor alpha for quiet/freestream air
                                      ///< (NOT a hard cull: slow air stays a
                                      ///< faint haze so structure underneath
                                      ///< it is still visible).
    float volumeDensity      = 0.85f; ///< Opacity gain for disturbed (fast/wake)
                                      ///< air at the top of the ramp.
    int   volumeUpdateEveryNFrames = 2; ///< Recompute cadence (cheap, but the
                                      ///< field barely moves frame-to-frame).

    // -- particle appearance --
    int             particleCount = kDefaultParticleCount;
    ParticleColorBy particleColorBy = ParticleColorBy::VorticityMag;
    Colormap        particleColormap = Colormap::Viridis;
    float           particlePointSize = 1.5f;
    // Normalization scales for the particle color scalar (full-palette value).
    // Speed is in lattice units (u_lat ~ 0.05-0.12, wake peaks higher);
    // vorticity magnitude is in lattice 1/step units (shear-layer scale).
    float particleSpeedColorScale     = 0.15f;
    float particleVorticityColorScale = 0.05f;

    // -- slices (up to one per axis is plenty for v1; see SliceConfig note) --
    // Defaults: one slot pre-armed per axis so toggling a slice on shows
    // something sensible immediately (spanwise vorticity at mid-span is the
    // classic airfoil view, hence slot 2 enables with showSlices).
    SliceConfig slices[3] = {
        {false, SliceAxis::X, -1, SliceField::SpeedMag,   Colormap::Viridis,  1.0f},
        {false, SliceAxis::Y, -1, SliceField::SpeedMag,   Colormap::Viridis,  1.0f},
        {true,  SliceAxis::Z, -1, SliceField::VorticityZ, Colormap::Coolwarm, 1.0f},
    };

    // -- Q-criterion isosurface (mode 3 — the default hero view) --
    bool  showQRaycast = true;    ///< Default ON: opaque first-hit raycast of
                                  ///< the Q-criterion vortex skins (hotkey 3).
    float qThreshold   = 0.25f;   ///< Iso threshold on the NORMALIZED Q in [0,1].
    float qScale       = 5e-4f;   ///< Q value mapped to 1.0 in the volume
                                  ///< texture (lattice 1/step^2 units). Sized
                                  ///< to shear-layer cores (Q ~ 0.5*omega^2 at
                                  ///< omega ~ 0.03-0.05/step) so the wake uses
                                  ///< the full [0,1] range while freestream
                                  ///< micro-turbulence (Q ~ 1e-5) normalizes
                                  ///< to ~0.02 — far below any threshold. A
                                  ///< too-small scale clamps BOTH to 1.0 and
                                  ///< no threshold can separate them.
    int   qUpdateEveryNFrames = 4;///< Recompute cadence (plan 9.1: every N frames).
    bool  qColorByVelocity = true;///< Paint the isosurface with the local air
                                  ///< speed (shares the velocity volume) vs
                                  ///< the fixed pale-cyan vortex-core tint.
    Colormap qColormap = Colormap::Rainbow; ///< Speed palette for the skins:
                                  ///< blue slow -> green freestream -> red fast.
};

/// @brief Renderer + interop owner. Construction order contract: a current
/// GL 4.6 context and the CUDA runtime context must both exist before init()
/// — main.cpp guarantees this in its startup sequence (plan 9.3).
class Visualizer {
public:
    Visualizer();
    ~Visualizer();
    Visualizer(const Visualizer&) = delete;
    Visualizer& operator=(const Visualizer&) = delete;

    /// @brief Create GL programs/buffers/textures, seed the particle VBO, and
    /// register every CUDA-mapped GL resource exactly once (particle VBO +
    /// slice textures). Re-registration after this point is a bug.
    /// @param dims          Lattice dims (slice texture sizes, particle spawn box).
    /// @param particleCount Initial particle pool size.
    /// @param stream        The app's single CUDA stream.
    /// @param error         On failure, a human-readable reason.
    /// @return True on success.
    bool init(const GridDims& dims, int particleCount, cudaStream_t stream,
              std::string* error);

    /// @brief Destroy all GL/CUDA-interop resources (also from destructor).
    void shutdown();

    /// @brief Upload (or replace) the extruded foil + VG render mesh. Called
    /// on airfoil/AoA/VG edits; builds a prism mesh by extruding the rotated
    /// outline polygon across the span plus simple vane boxes (plan 9.1: no
    /// marching cubes on voxels).
    /// @param airfoil Section outline.
    /// @param vgs     Current VG configuration (rendered as placed boxes).
    /// @param aoa_deg Angle of attack (same rotation as the voxelizer).
    /// @param layout  Foil placement/scale inside the grid.
    void uploadGeometry(const AirfoilGeometry& airfoil,
                        const std::vector<VGParams>& vgs, float aoa_deg,
                        const DomainLayout& layout);

    /// @brief Upload (or replace) an imported STL render mesh, replacing the
    /// extruded-foil prism geometry while STL mode is active. Vertices must
    /// already be in lattice-cell coordinates (post applyNormalization, plan
    /// 7.2) — the same space the voxelizer consumed, so mesh and voxel solid
    /// coincide. Triangles are flat-shaded from their winding (the mesh shader
    /// flips normals toward the viewer, so file winding direction is
    /// irrelevant). Passing an empty mesh clears the STL geometry; call
    /// uploadGeometry() afterwards to restore the airfoil prism.
    /// @param mesh Normalized mesh from the import flow (plan 7.2).
    void uploadStlMesh(const StlMesh& mesh);

    /// @brief Per-frame GPU update: map the interop resources (map/unmap
    /// brackets ONLY the GL-touching kernels, plan 9.3), run particle
    /// advection over the solver's velocity field, fill enabled slice
    /// textures via surface-object writes, unmap.
    /// @param vel      Solver velocity field view.
    /// @param rho      Solver density array (pressure slices), may be null
    ///                 if no pressure slice is enabled.
    /// @param flags    Cell flags (particle respawn inside solids).
    /// @param dtSteps  Sim steps advanced this frame (advection time).
    /// @param settings Active view settings.
    /// @return CUDA error from the interop/kernel sequence (cudaSuccess normally).
    cudaError_t updateFields(DeviceVelocityField vel, const float* rho,
                             const std::uint8_t* flags, float dtSteps,
                             const VizSettings& settings);

    /// @brief Draw the scene with current settings: foil/VG mesh (depth-
    /// tested), slice planes, then particles as additive-blended GL_POINTS
    /// depth-tested against the mesh (plan 9.1). ImGui draws after this.
    /// @param camera         Orbit camera supplying view/projection.
    /// @param settings       Active view settings.
    /// @param viewportWidth  Framebuffer width in pixels.
    /// @param viewportHeight Framebuffer height in pixels.
    void drawFrame(const OrbitCamera& camera, const VizSettings& settings,
                   int viewportWidth, int viewportHeight);

    /// @brief Resize the particle pool (UI slider). Re-registers the ONE
    /// particle VBO with CUDA — the only sanctioned re-registration point,
    /// because the buffer object itself is recreated.
    /// @return False on GL/CUDA failure (old pool stays active).
    bool resizeParticlePool(int newCount, std::string* error);

    /// @brief Read back the current framebuffer and write a PNG (plan 9.2
    /// screenshot button; also the --selftest output). Creates parent
    /// directories as needed.
    /// @param path  Destination .png path.
    /// @param error On failure, a reason.
    /// @return True on success.
    bool screenshotPNG(const std::filesystem::path& path, std::string* error);

private:
    struct Impl; // GL handles + cudaGraphicsResource pointers live here, off the header.
    std::unique_ptr<Impl> impl_;
};

} // namespace foilcfd
