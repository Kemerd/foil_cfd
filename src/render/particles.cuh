// CUDA render-kernel API (plan 9.1): RK2 advection of 1M tracer particles
// through the trilinearly-sampled lattice velocity field (inlet-rate-balanced
// respawn, solid-entry respawn, age fade), slice-plane field rasterization via
// surface-object writes into CUDA-registered GL textures, and the Q-criterion
// volume fill for the raycast mode. The Visualizer maps the GL resources and
// calls these wrappers; the solver-facing declaration in sim/lbm_core.cuh
// (launchAdvectParticles) forwards to the same advection kernel.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

#include "../sim/lbm_core.cuh"

namespace foilcfd {

// ===========================================================================
// Particle buffer layout (binding contract with the GL side):
//   positions : float4 per particle — xyz = position in LATTICE CELL space
//               (so the velocity sample needs no rescale), w = age in [0,1]
//               (1 = newborn, fades to 0; shader maps it to alpha).
//   colorKey  : float per particle — the colormap input in [0,1], already
//               normalized by ParticleAdvectParams::colorScale. Stored
//               separately so the position VBO stays a tight 16-byte stride.
// ===========================================================================

/// @brief Tunables for one advection launch.
struct ParticleAdvectParams {
    float dtSteps    = 1.0f;  ///< Advection time in lattice steps this frame.
    float ageRate    = 0.002f;///< Age decrement per lattice step (lifetime ~1/ageRate).
    float inletSpawnFraction = 0.3f; ///< Probability that a LIFETIME or SOLID
                              ///< death respawns at the inlet plane instead of
                              ///< re-seeding the fluid volume. Outflow exits
                              ///< ALWAYS respawn at the inlet 1:1, which is
                              ///< what keeps inlet seeding rate-balanced with
                              ///< outflow loss (plan 9.1); this knob only
                              ///< biases the remaining deaths so the wake and
                              ///< the inlet sheet both stay populated.
    unsigned int seed = 0;    ///< Per-frame RNG seed (respawn jitter).
    int  colorBy     = 0;     ///< 0 = speed, 1 = vorticity magnitude
                              ///< (mirrors ParticleColorBy; int to keep this
                              ///< header free of render-enum dependencies).
    float colorScale = 0.15f; ///< Scalar value mapped to colorKey = 1.0
                              ///< (speed or |vorticity|, lattice units).
};

/// @brief RK2 (midpoint) advection over the trilinear velocity field:
///   u1 = sample(x);  xm = x + 0.5*dt*u1;  x' = x + dt*sample(xm)
/// Particles that leave the domain through the outlet, exceed their lifetime,
/// or step into a SOLID cell respawn — outflow losses at the inlet plane
/// (rate-balanced), the rest split between inlet and fluid-volume jitter so
/// the wake stays seeded. Also writes the per-particle color scalar.
/// Defensive against a not-yet-converged solver: non-finite positions or
/// velocities trigger a respawn instead of propagating NaNs into the VBO.
/// @param positions Mapped GL VBO: float4 per particle (see layout above).
/// @param colorKeys Mapped GL VBO: float per particle (colormap input);
///                  nullptr skips color writes (solver-side forwarder).
/// @param count     Particle count.
/// @param vel       Velocity field view (lattice units).
/// @param flags     Cell flags for solid/outlet detection (cellCount bytes).
/// @param params    Launch tunables.
/// @param stream    Stream to run on (the app's single stream).
/// @return cudaGetLastError() from the launch.
cudaError_t launchParticleAdvectRK2(float4* positions, float* colorKeys,
                                    int count, DeviceVelocityField vel,
                                    const std::uint8_t* flags,
                                    const ParticleAdvectParams& params,
                                    cudaStream_t stream);

/// @brief Seed a fresh particle pool: uniform spawn inside the domain box
/// with randomized ages (so the pool doesn't fade in lockstep). Runs once at
/// init and after pool resizes.
/// @param positions Mapped GL VBO to fill.
/// @param colorKeys Mapped GL VBO to zero-fill (may be nullptr).
/// @param count     Particle count.
/// @param dims      Domain dimensions (spawn box).
/// @param seed      RNG seed.
/// @param stream    Stream to run on.
cudaError_t launchParticleSeed(float4* positions, float* colorKeys, int count,
                               GridDims dims, unsigned int seed,
                               cudaStream_t stream);

// ===========================================================================
// Slice planes (plan 9.1 mode 2): a CUDA kernel rasterizes one axis-aligned
// cross-section of a derived field straight into the GL texture's cudaArray
// through a surface object — colormapped on the GPU, zero host round trips.
// ===========================================================================

/// @brief Parameters for one slice-texture fill.
///
/// Texel -> cell mapping (must match the textured quad the renderer draws):
///   axis 0 (X plane): texel (s,t) -> cell (cell, t, s); width = nz, height = ny
///   axis 1 (Y plane): texel (s,t) -> cell (s, cell, t); width = nx, height = nz
///   axis 2 (Z plane): texel (s,t) -> cell (s, t, cell); width = nx, height = ny
struct SliceFillParams {
    int   axis     = 2;    ///< 0 = X, 1 = Y, 2 = Z (plane is perpendicular to it).
    int   cell     = 0;    ///< Plane position along `axis` (caller clamps).
    int   field    = 0;    ///< 0 = |u|, 1 = vorticity-z, 2 = pressure (cs^2*(rho-1)).
    int   colormap = 0;    ///< 0 = viridis (sequential), 1 = coolwarm (diverging).
    float scale    = 0.15f;///< Field value mapped to the palette end (signed
                           ///< fields use +/-scale around the palette center).
    int   width    = 0;    ///< Texture width in texels (see mapping above).
    int   height   = 0;    ///< Texture height in texels.
};

/// @brief Fill one RGBA8 slice texture: derive the requested scalar at every
/// texel's cell (central-difference vorticity, cs^2*(rho-1) pressure), push it
/// through the colormap, and surf2Dwrite the result. SOLID cells render as a
/// neutral dark gray so the foil silhouette reads inside the slice.
/// @param surface RGBA8 surface object over the mapped GL texture array.
/// @param vel     Velocity field view.
/// @param rho     Density array (pressure field only; may be nullptr, which
///                renders the pressure slice as the palette midpoint).
/// @param flags   Cell flags (solid masking).
/// @param params  Slice description (see SliceFillParams).
/// @param stream  Stream to run on.
/// @return cudaGetLastError() from the launch.
cudaError_t launchSliceFill(cudaSurfaceObject_t surface, DeviceVelocityField vel,
                            const float* rho, const std::uint8_t* flags,
                            const SliceFillParams& params, cudaStream_t stream);

// ===========================================================================
// Q-criterion volume (plan 9.1 mode 3, stretch): Q = 0.5*(|Omega|^2 - |S|^2)
// from central-difference velocity gradients, normalized by qScale and stored
// as R8 into the registered GL 3D texture the fragment raymarcher samples.
// ===========================================================================

/// @brief Compute the Q-criterion field over the whole grid and write the
/// clamped normalized value (Q / qScale into [0,1]) into an R8 3D surface.
/// Solid and domain-boundary cells write 0 so the raymarcher never lights
/// the foil interior or the box faces.
/// @param volumeSurface R8 surface object over the mapped GL_TEXTURE_3D array.
/// @param vel           Velocity field view.
/// @param flags         Cell flags (solid masking).
/// @param qScale        Q value that maps to texel value 1.0.
/// @param stream        Stream to run on.
/// @return cudaGetLastError() from the launch.
cudaError_t launchQCriterionVolume(cudaSurfaceObject_t volumeSurface,
                                   DeviceVelocityField vel,
                                   const std::uint8_t* flags, float qScale,
                                   cudaStream_t stream);

// ===========================================================================
// Velocity volume (the hero "wind-tunnel smoke" mode): the per-cell speed
// magnitude normalized into [0,1], stored as R16 (GL_R16) into a registered
// GL 3D texture. The raymarch shader colormaps it on the GPU and culls the
// quiet freestream so only accelerated / wake air lights up (the wind-tunnel
// smoke look). Storing the SCALAR (not a baked color) keeps the palette + the
// slow-air opacity entirely in the shader, so both update with zero re-fills.
// ===========================================================================

/// @brief Parameters for one velocity-volume fill.
struct VelocityVolumeParams {
    float speedScale = 0.15f; ///< Speed magnitude (lattice units) mapped to 1.0.
    int   width      = 0;     ///< Volume texture extent (= grid nx).
    int   height     = 0;     ///< (= grid ny).
    int   depth      = 0;     ///< (= grid nz).
};

/// @brief Fill the velocity-volume 3D texture: write normalized |u| (in
/// [0,1], clamped) to every cell as a single-channel float, so the raymarch
/// shader can colormap + threshold it. Solid and domain-boundary cells write
/// 0 (the shader treats 0 as fully transparent void), keeping the foil
/// interior and the box shell dark.
/// @param volumeSurface R16/float surface object over the mapped GL 3D array.
/// @param vel           Velocity field view.
/// @param flags         Cell flags (solid masking).
/// @param params        Volume description (speed normalization + extents).
/// @param stream        Stream to run on.
/// @return cudaGetLastError() from the launch.
cudaError_t launchVelocityVolume(cudaSurfaceObject_t volumeSurface,
                                 DeviceVelocityField vel,
                                 const std::uint8_t* flags,
                                 const VelocityVolumeParams& params,
                                 cudaStream_t stream);

} // namespace foilcfd
