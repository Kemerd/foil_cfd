// ISLBM (interpolation-supplemented LBM) stretched-mesh mode: a single Cartesian
// grid whose physical cell spacing dx varies continuously (finest at the wall,
// smoothly coarsening outward) instead of the discrete x2/x4 refinement levels.
// This is the "true gradient" refinement — no levels, no seams — auto-built from
// a signed-distance-to-surface field so it works for any geometry (foil, VG, STL).
//
// PHYSICS: the D3Q19 TRT-Smagorinsky COLLIDE half is byte-identical to the
// uniform kernel. Only the streaming PULL changes, and only in BULK cells (a
// cell whose whole interpolation stencil is pure fluid): there the characteristic
// foot lands off-node on the stretched mesh, so each population is recovered by a
// separable 3-point quadratic Lagrange interpolation (He-Luo 1996 ISLBM family;
// Xu et al. 2025 GPU realization). Within a thin COLLAR around walls/slip planes
// the EXACT integer flag-predicated pull (bounce-back, slip mirror, q-LIBB, Ladd
// wall slip) is retained verbatim, so the Cd/delta99 the project depends on stay
// trustworthy — an off-node gather is ill-defined at those links.
//
// This is lattice-METRIC stretching (only the grid dx varies; the geometry and
// its voxelization are untouched), NOT the geometry-distorting coordinate stretch
// rejected earlier.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

namespace foilcfd {

// Forward declarations keep this header light enough to include from the kernel
// TUs (lbm_core.cuh pulls it in only for the StretchView used in StepParams /
// launchStreamCollide). The host builder below needs the full GridDims /
// LatticeScaling definitions; lbm_stretch.cpp includes their headers directly.
struct GridDims;
struct LatticeScaling;

// ===========================================================================
// Tunables.
// ===========================================================================

/// Maximum per-cell spacing growth ratio dx(i+1)/dx(i). The single most
/// important accuracy knob: a tiny growth keeps every link's interpolation foot
/// within a fraction of a cell of an integer node, so the quadratic gather stays
/// a small correction and the scheme stays close to Galilean-invariant
/// (Xu 2025 runs stable at <=1.0044). 0.5%/cell is comfortably inside that.
inline constexpr float kMaxStretchGrowth = 1.005f;

/// Chebyshev margin (in cells) the bulk predicate keeps between a gathered cell
/// and any non-fluid cell or domain face. Sized to the FULL interpolation
/// stencil reach so the gather can never read a non-fluid cell: the rounded foot
/// base is clamped to at most 2 cells out and the 3-tap Lagrange stencil adds
/// +/-1, so the worst-case reach is 3 cells. A 3-cell collar covers it exactly
/// (we do not rely on the wall-side frac->0 cancellation for safety).
inline constexpr int kStretchCollarCells = 3;

/// Above this local stretch ratio the bulk gather regularizes its fneq (project
/// onto the Hermite stress subspace, mass/momentum preserved) to drain the
/// ghost-mode aliasing the off-node interpolation injects where stretch is
/// steepest — the continuous-mesh analog of the refinement-interface fix. Sits
/// in the upper part of the <=kMaxStretchGrowth band so the gentle bulk pays
/// nothing.
inline constexpr float kStretchRegularizeRatio = 1.003f;

// ===========================================================================
// Device view of the stretch mesh (uploaded once per geometry edit). The maps
// are 1-D per axis (the mesh is a Cartesian product of independent X/Y
// stretchings; Z stays uniform/periodic, so it is never stored here). The
// per-cell tau IS a full 3-D field because the local viscosity nu_lat ~ 1/dx^2
// depends on both dxX(x) and dxY(y).
// ===========================================================================

/// @brief Per-axis interpolation foot maps + per-cell relaxation field. An empty
/// (inactive) view selects the uniform/cascade kernel instantiation — bit-
/// identical to the pre-ISLBM path, exactly like the WallSlipView/QLinkView tags.
struct StretchView {
    // Foot maps in INDEX space, indexed [signBit*n + i]. signBit 0 = the link
    // travels -1 in index (pull from the -x/-y side), 1 = +1. footFrac is the
    // fractional offset of the characteristic foot from its rounded node;
    // footBase is the rounded integer offset (typically -/+1, occasionally 2).
    const float*       footFracX = nullptr;
    const std::int8_t* footBaseX = nullptr;
    const float*       footFracY = nullptr;
    const std::int8_t* footBaseY = nullptr;

    /// Per-cell relaxation time tau (UNPADDED, ncells entries, indexed by cell):
    /// tau(i) = 3 * nu_phys * dt / dx(i)^2 + 1/2, clamped >= kMinTau. The wall
    /// (finest dx) carries the reference tau; tau falls toward the floor in the
    /// coarse far field. The kernel reads this only under the ISLBM template tag.
    const float*       tauField  = nullptr;

    int nx = 0; ///< Grid x extent (foot-map stride).
    int ny = 0; ///< Grid y extent.

    /// Gather quality. FAST (default): interpolate raw populations — cheap (one
    /// load per tap), ~full uniform throughput, great for interactive preview,
    /// but it blends incompatibly-scaled near-wall stresses so lift/drag are
    /// only qualitative. ACCURATE (false): reconstruct each tap into
    /// feq + tau-rescaled fneq — physically correct Cl/Cd, but ~4-5x the gather
    /// cost (per-tap moments + equilibrium). Toggle per the UI "fast gather" box.
    bool fastGather = true;

    /// @brief True when a stretch mesh is supplied this step.
    bool active() const {
        return footFracX && footBaseX && footFracY && footBaseY && tauField;
    }
};

// ===========================================================================
// Host-owning stretch mesh: builds the per-axis spacing profiles + foot LUTs +
// per-cell tau from a wall-distance field and a scaling, owns the device
// mirror, and hands out a StretchView for the kernel. Mirrors the lifecycle of
// the other solver device buffers (build once per geometry edit, free on
// teardown / mode switch).
// ===========================================================================

struct StretchMesh {
    // Host-side diagnostics the UI reads (no device round-trip).
    float dxMin   = 0.0f;   ///< Finest physical spacing (wall) [m].
    float dxMax   = 0.0f;   ///< Coarsest physical spacing (far field) [m].
    float growthX = 1.0f;   ///< Achieved per-cell growth ratio, X axis.
    float growthY = 1.0f;   ///< Achieved per-cell growth ratio, Y axis.
    float tauWall = 0.0f;   ///< tau at the finest cell (the reference tau).
    float tauFar  = 0.0f;   ///< tau at the coarsest cell (>= kMinTau).
    bool  tauFloorClamped = false; ///< dxMax was reduced to keep tauFar>=kMinTau.
    double fluidCellSaving = 0.0;  ///< Fraction of cells coarser than dxMin
                                   ///< (rough "what the gradient buys" readout).
    float nearWallFactor = 1.0f;   ///< Sub-base refinement k: the finest cell is
                                   ///< k-times finer than the base grid dx (1.0 =
                                   ///< legacy, wall == base). dxMin = base/k.

    int nx = 0, ny = 0, nz = 0;
    bool active = false;
    bool fastGather = true; ///< Mirrors StretchView::fastGather (UI toggle).

    // Device arrays (owned). Foot maps are [2*n] (sign-major); tau is [ncells].
    float*       dFootFracX = nullptr;
    std::int8_t* dFootBaseX = nullptr;
    float*       dFootFracY = nullptr;
    std::int8_t* dFootBaseY = nullptr;
    float*       dTauField  = nullptr;

    /// @brief Device view for the kernel (inactive when unbuilt).
    StretchView view() const {
        if (!active) return StretchView{};
        StretchView v;
        v.footFracX = dFootFracX; v.footBaseX = dFootBaseX;
        v.footFracY = dFootFracY; v.footBaseY = dFootBaseY;
        v.tauField  = dTauField;  v.nx = nx;   v.ny = ny;
        v.fastGather = fastGather;
        return v;
    }

    /// @brief Release every device allocation; safe to call repeatedly.
    void free() {
        cudaFree(dFootFracX); cudaFree(dFootBaseX);
        cudaFree(dFootFracY); cudaFree(dFootBaseY);
        cudaFree(dTauField);
        *this = StretchMesh{};
    }
};

/// @brief Build the stretched mesh from a wall-distance field and the base
/// scaling, and upload it to the device. By default the finest spacing is the
/// base grid's dx (the wall stays as well-resolved as the uniform grid); dx
/// grows smoothly outward at <= kMaxStretchGrowth per cell, saturating at a
/// far-field dxMax chosen so the far-field tau stays >= kMinTau (clamped +
/// reported if not). Per-cell tau follows nu_lat ~ 1/dx^2.
///
/// SUB-BASE REFINEMENT (@p nearWallFactor k > 1): the finest spacing becomes
/// base/k, so the wall band gets cells FINER than the base grid — true local
/// super-refinement inside the smooth-gradient framework. The whole scheme is
/// ratio-based (foot = dxMin/dxi, tau = nuWall*(dxMin/dxEff)^2), so this is a
/// pure RE-ANCHOR: dxMin shrinks and the wall viscosity reference nuWall scales
/// by k^2 in lockstep, so the finest cell self-collapses to the exact integer
/// pull (frac 0) and base-spacing cells correctly relax to the original base
/// tau. No kernel / foot-map / collar change is needed — dxMin simply remains
/// the global minimum spacing. dt is NOT referenced in the time integration, so
/// no sub-cycling: the physics rides entirely in the per-cell tau field.
///
/// Frees any previous mesh first; leaves @p mesh inactive and returns an error
/// on OOM (caller falls back to uniform). All work is on @p stream; the upload
/// is synchronized before return.
/// @param mesh           Mesh to (re)build in place.
/// @param dims           Grid dims (must match the wall-distance field).
/// @param scaling        Base (finest-cell) scaling — its dx/tau anchor the wall.
/// @param wallDist       buildWallDistanceField output (dims.cellCount() floats).
/// @param stream         CUDA stream for the upload.
/// @param nearWallFactor Sub-base refinement k (>= 1; 1 = legacy base-wall).
/// @param error          On failure, receives a human-readable reason.
/// @return True on success (mesh.active == true), false on failure.
bool buildStretchMesh(StretchMesh& mesh, const GridDims& dims,
                      const LatticeScaling& scaling,
                      const std::vector<float>& wallDist, cudaStream_t stream,
                      float nearWallFactor, std::string* error);

} // namespace foilcfd
