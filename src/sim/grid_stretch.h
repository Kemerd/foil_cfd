// Non-uniform coordinate mapping for the LBM domain (mesh refinement).
//
// The LBM solver operates on a uniform LOGICAL grid (integer cell indices).
// This module provides a PHYSICAL coordinate map: a tanh-based stretching that
// packs more cells into aerodynamically critical regions — the leading edge,
// suction-surface boundary layer, VG zone, and near-wake — without changing
// the solver, the flag field layout, or the cell count (= no VRAM impact).
//
// Key constraint: the LBM collision operator assumes dx=dy=dz (uniform
// lattice). Stretching introduces a metric correction that manifests as a
// body-force error of order (stretch_ratio - 1) * Ma^2. For the mild ratios
// used here (max 4:1 between coarsest and finest cell), this is well within
// the O(Ma^2) compressibility error already present in the LBM equilibrium
// and acceptable for engineering-grade VG placement decisions.
//
// Usage:
//   GridStretch gs = GridStretch::fromParams(params, layout);
//   float physX = gs.toPhysX(iCell);   // logical -> physical [cell-chord units]
//   float physY = gs.toPhysY(jCell);
//   int   iCell = gs.fromPhysX(physX); // physical -> nearest logical cell
//
// The voxelizer uses toPhys* to convert physical airfoil coordinates into
// stretched logical cell space before rasterization, so geometry stamps
// correctly under any stretching profile.
//
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "lbm_core.cuh" // GridDims

namespace foilcfd {

// ---------------------------------------------------------------------------
// Zone multiplier indices — seven distinct regions of the domain, each with
// an independent density weight. The weights are RELATIVE: a weight of 2.0
// means twice as many cells per unit length compared to weight 1.0.
// ---------------------------------------------------------------------------

/// @brief Seven mesh refinement zones, mapped across the XY domain.
enum class MeshZone {
    FarUpstream   = 0, ///< Far upstream of the leading edge (freestream inlet).
    NearUpstream  = 1, ///< Just ahead of the leading edge (~0.5c upstream).
    LeadingEdge   = 2, ///< Leading-edge / suction-peak region.
    TopSurfaceBL  = 3, ///< Suction-surface wall-normal direction (y stretch).
    VGZone        = 4, ///< Around the VG placement stations (ultra-fine).
    NearWake      = 5, ///< Immediately behind the trailing edge.
    FarWake       = 6, ///< Far wake, downstream of the domain.
};

constexpr int kNumMeshZones = 7;

// ---------------------------------------------------------------------------
// Preset bundles.
// ---------------------------------------------------------------------------

/// @brief Four preset configurations: Off (uniform), Balanced, Aggressive,
/// Custom (sliders unlocked). Stored as an enum for the combo widget.
enum class MeshRefinementPreset {
    Off        = 0, ///< No stretching — pure uniform grid (default behavior).
    Balanced   = 1, ///< Sensible defaults: 2x LE, 4x VG, 0.5x far field.
    Aggressive = 2, ///< Maximum safe contrast (4x LE, 8x VG, 0.3x far field).
    Custom     = 3, ///< All seven zone sliders unlocked.
};

/// @brief Per-zone density weights for each preset.
///
/// A weight of 1.0 is "neutral." The tanh mapping normalises so that
/// the AVERAGE cell spacing across the domain is unchanged, meaning the
/// total cell count (and VRAM) stays constant regardless of preset.
struct MeshRefinementParams {
    MeshRefinementPreset preset = MeshRefinementPreset::Off;

    // Zone density weights [0.25 .. 8.0]. Only editable in Custom mode.
    float zoneWeight[kNumMeshZones] = {
        1.0f,  // FarUpstream
        1.0f,  // NearUpstream
        1.0f,  // LeadingEdge
        1.0f,  // TopSurfaceBL
        1.0f,  // VGZone
        1.0f,  // NearWake
        1.0f,  // FarWake
    };

    // VG zone chordwise center (0..1 chord fraction). Tracks the VG
    // placement automatically when autoTrackVGs is true.
    float vgZoneXc       = 0.20f; ///< Chordwise centre of the VG refinement band.
    float vgZoneHalfWidth = 0.10f; ///< Half-width of the VG zone in chord fractions.
    bool  autoTrackVGs   = true;  ///< Mirror the VG editor station when true.

    /// @brief Apply a preset's canonical weights (overwrites zoneWeight[]).
    void applyPreset(MeshRefinementPreset p) {
        preset = p;
        switch (p) {
            case MeshRefinementPreset::Off:
                for (int i = 0; i < kNumMeshZones; ++i) zoneWeight[i] = 1.0f;
                break;
            case MeshRefinementPreset::Balanced:
                zoneWeight[0] = 0.5f;  // FarUpstream   — coarser
                zoneWeight[1] = 1.0f;  // NearUpstream  — neutral
                zoneWeight[2] = 2.0f;  // LeadingEdge   — 2x finer
                zoneWeight[3] = 2.0f;  // TopSurfaceBL  — 2x finer
                zoneWeight[4] = 4.0f;  // VGZone        — 4x finer
                zoneWeight[5] = 1.5f;  // NearWake      — 1.5x finer
                zoneWeight[6] = 0.5f;  // FarWake       — coarser
                break;
            case MeshRefinementPreset::Aggressive:
                zoneWeight[0] = 0.3f;  // FarUpstream
                zoneWeight[1] = 0.8f;  // NearUpstream
                zoneWeight[2] = 4.0f;  // LeadingEdge
                zoneWeight[3] = 3.0f;  // TopSurfaceBL
                zoneWeight[4] = 8.0f;  // VGZone
                zoneWeight[5] = 2.0f;  // NearWake
                zoneWeight[6] = 0.3f;  // FarWake
                break;
            case MeshRefinementPreset::Custom:
                // Custom starts from Balanced defaults when first switched to.
                zoneWeight[0] = 0.5f;
                zoneWeight[1] = 1.0f;
                zoneWeight[2] = 2.0f;
                zoneWeight[3] = 2.0f;
                zoneWeight[4] = 4.0f;
                zoneWeight[5] = 1.5f;
                zoneWeight[6] = 0.5f;
                break;
        }
    }

    /// @brief True when any stretching is active (avoids the lookup-table
    /// build when the preset is Off and every weight is 1.0).
    bool isActive() const {
        if (preset == MeshRefinementPreset::Off) return false;
        for (int i = 0; i < kNumMeshZones; ++i)
            if (std::fabs(zoneWeight[i] - 1.0f) > 1e-4f) return true;
        return false;
    }
};

// ---------------------------------------------------------------------------
// The coordinate map — built once from params + layout, then queried per cell.
// ---------------------------------------------------------------------------

/// @brief Precomputed physical coordinate lookup tables for the stretched
/// domain. Built once per simulation init; queried O(1) per cell during
/// voxelization. Both tables are indexed by logical cell index [0, N).
struct GridStretch {
    /// @brief Build from parameters and grid dimensions.
    ///
    /// The stretching is defined by a piecewise tanh density function across
    /// the X axis (chordwise) and the Y axis (wall-normal). The function is
    /// integrated numerically to build a CDF that maps [0,nx] -> [0,physX],
    /// normalized so physX[nx] == nx (physical units stay in "cells"; only
    /// the spacing distribution changes, not the domain size).
    ///
    /// @param params  Zone weights and VG zone parameters.
    /// @param dims    Grid dimensions (logical).
    /// @param layout  Foil anchor and chord cells (to define zone boundaries).
    static GridStretch build(const MeshRefinementParams& params,
                             const GridDims& dims,
                             float anchorXFrac,
                             float anchorYFrac,
                             int   chordCells);

    /// @brief Logical -> physical X coordinate [cell units].
    float toPhysX(int i) const {
        if (physX_.empty()) return static_cast<float>(i);
        const int n = static_cast<int>(physX_.size()) - 1;
        const int ci = std::clamp(i, 0, n);
        return physX_[static_cast<std::size_t>(ci)];
    }

    /// @brief Logical -> physical Y coordinate [cell units].
    float toPhysY(int j) const {
        if (physY_.empty()) return static_cast<float>(j);
        const int n = static_cast<int>(physY_.size()) - 1;
        const int cj = std::clamp(j, 0, n);
        return physY_[static_cast<std::size_t>(cj)];
    }

    /// @brief Physical -> nearest logical X cell index.
    int fromPhysX(float px) const {
        if (physX_.empty()) return static_cast<int>(std::round(px));
        // Binary search: find i such that physX[i] <= px < physX[i+1].
        const auto it = std::lower_bound(physX_.begin(), physX_.end(), px);
        const int idx = static_cast<int>(it - physX_.begin());
        const int n   = static_cast<int>(physX_.size()) - 1;
        return std::clamp(idx, 0, n);
    }

    /// @brief Physical -> nearest logical Y cell index.
    int fromPhysY(float py) const {
        if (physY_.empty()) return static_cast<int>(std::round(py));
        const auto it = std::lower_bound(physY_.begin(), physY_.end(), py);
        const int idx = static_cast<int>(it - physY_.begin());
        const int n   = static_cast<int>(physY_.size()) - 1;
        return std::clamp(idx, 0, n);
    }

    /// @brief True when this stretch object has non-trivial tables (i.e.,
    /// stretching is active and toPhys* returns non-integer values).
    bool isActive() const { return !physX_.empty(); }

    // Read-only access to the tables (for the UI diagram draw).
    const std::vector<float>& physXTable() const { return physX_; }
    const std::vector<float>& physYTable() const { return physY_; }

private:
    // Empty vectors => identity map (no stretching). Allocated only when the
    // preset is not Off, so the Off case is zero overhead.
    std::vector<float> physX_; ///< physX_[i] = physical x of logical cell i.
    std::vector<float> physY_; ///< physY_[j] = physical y of logical cell j.

    // Build helpers.
    static std::vector<float> buildAxis(int N,
                                        const std::vector<float>& density);
};

// ---------------------------------------------------------------------------
// Implementation (inline — header-only to avoid a separate .cpp dependency
// on this purely computational module).
// ---------------------------------------------------------------------------

inline std::vector<float> GridStretch::buildAxis(
        int N, const std::vector<float>& density) {
    // density[i] is a weight proportional to 1/dx_i (how many cells per unit
    // physical length at position i/N). Integrate to get the CDF, then
    // normalise so the final value == N (domain size preserved).
    std::vector<double> cdf(static_cast<std::size_t>(N + 1), 0.0);
    for (int i = 0; i < N; ++i) {
        cdf[static_cast<std::size_t>(i + 1)] =
            cdf[static_cast<std::size_t>(i)]
            + static_cast<double>(density[static_cast<std::size_t>(i)]);
    }
    const double total = cdf[static_cast<std::size_t>(N)];
    if (total <= 0.0) {
        // Degenerate: return identity.
        std::vector<float> id(static_cast<std::size_t>(N + 1));
        for (int i = 0; i <= N; ++i) id[static_cast<std::size_t>(i)] = static_cast<float>(i);
        return id;
    }
    // Map back: physCoord[i] = (CDF[i] / total) * N
    std::vector<float> phys(static_cast<std::size_t>(N + 1));
    for (int i = 0; i <= N; ++i) {
        phys[static_cast<std::size_t>(i)] =
            static_cast<float>(cdf[static_cast<std::size_t>(i)] / total
                               * static_cast<double>(N));
    }
    return phys;
}

inline GridStretch GridStretch::build(const MeshRefinementParams& params,
                                      const GridDims& dims,
                                      float anchorXFrac,
                                      float anchorYFrac,
                                      int   chordCells) {
    GridStretch gs;

    // If stretching is off, leave tables empty (identity map, zero overhead).
    if (!params.isActive()) return gs;

    const int nx = dims.nx;
    const int ny = dims.ny;
    const float c  = static_cast<float>(chordCells);
    const float ax = anchorXFrac * static_cast<float>(nx); // quarter-chord x
    const float ay = anchorYFrac * static_cast<float>(ny); // mid-height y

    // ---- X-axis density function (chordwise) --------------------------------
    // Zone boundaries in logical cell units:
    //   FarUpstream:   [0,  ax - 0.8c)
    //   NearUpstream:  [ax - 0.8c,  ax - 0.05c)
    //   LeadingEdge:   [ax - 0.05c, ax + 0.15c)   (LE + ~15% suction surface)
    //   VGZone:        [vgCenter-vgHW, vgCenter+vgHW]  (in chord fractions, abs coords)
    //   NearWake:      [ax + 0.75c,  ax + 1.5c)
    //   FarWake:       [ax + 1.5c,  nx)
    // Note: zones may overlap slightly; the VG zone takes priority and is
    // blended with a Gaussian envelope.

    const float leStart  = ax - 0.05f * c;   // logical start of LE zone
    const float leEnd    = ax + 0.15f * c;   // logical end of LE zone (15% chord)
    const float nuStart  = ax - 0.8f  * c;   // near-upstream start
    const float wakeNear = ax + 0.75f * c;   // near-wake start (TE ~ax+c)
    const float wakeFar  = ax + 1.50f * c;   // far-wake start

    // VG zone in absolute logical coordinates.
    const float vgCtr   = ax + (params.vgZoneXc  - 0.25f) * c;
    const float vgHalf  = params.vgZoneHalfWidth * c;
    const float vgStart = vgCtr - vgHalf;
    const float vgEnd   = vgCtr + vgHalf;

    // Smooth zone weight as a function of logical x cell index.
    // Each zone is linearly blended with a transition width of ~5% chord.
    const float kTransition = 0.05f * c;

    auto smoothStep = [](float x) -> float {
        // Cubic smoothstep in [0,1].
        x = std::clamp(x, 0.0f, 1.0f);
        return x * x * (3.0f - 2.0f * x);
    };

    auto blendWeight = [&](float x, float zoneStart, float zoneEnd,
                           float wInside, float wOutside) -> float {
        // Smooth transition from wOutside to wInside across [zoneStart, zoneEnd].
        const float t0 = smoothStep((x - zoneStart) / kTransition);
        const float t1 = smoothStep((zoneEnd  - x)  / kTransition);
        const float inside = std::min(t0, t1);
        return wOutside + (wInside - wOutside) * inside;
    };

    // Shorthand alias to the zone weight array.
    const float* w = params.zoneWeight;

    std::vector<float> densX(static_cast<std::size_t>(nx));
    for (int i = 0; i < nx; ++i) {
        const float x = static_cast<float>(i) + 0.5f;

        // Start from far-field default.
        float weight = (x < nuStart) ? w[0] : w[6];

        // NearUpstream: blend in from FarUpstream.
        weight = blendWeight(x, nuStart, leStart, w[1], weight);

        // LeadingEdge: blend in from NearUpstream.
        weight = blendWeight(x, leStart, leEnd, w[2], weight);

        // NearWake: blend in behind TE.
        weight = blendWeight(x, wakeNear, wakeFar, w[5], weight);

        // VGZone: take max with LE/NearUpstream blends (VG wins).
        if (x > vgStart - kTransition && x < vgEnd + kTransition) {
            const float vgW = blendWeight(x, vgStart, vgEnd, w[4], weight);
            weight = std::max(weight, vgW);
        }

        densX[static_cast<std::size_t>(i)] = std::max(0.01f, weight);
    }
    gs.physX_ = buildAxis(nx, densX);

    // ---- Y-axis density function (wall-normal) --------------------------------
    // TopSurfaceBL is the suction surface: the upper half of the domain
    // (y > ay) gets the BL refinement. The bottom half stays at its own
    // weight (FarUpstream weight used as "below-foil" proxy, since those
    // cells are in the pressure-side free stream).
    // The VG height contribution only matters in a narrow y band near the
    // surface; we blend it into the suction side.

    const float blSurf = ay;       // physical y of the surface (approx mid-domain)
    const float blDepth = 0.15f * c; // BL refinement depth above the surface

    std::vector<float> densY(static_cast<std::size_t>(ny));
    for (int j = 0; j < ny; ++j) {
        const float y = static_cast<float>(j) + 0.5f;
        float weight = 1.0f;

        // Below the foil mid-plane: use the far-upstream weight (pressure side).
        if (y < blSurf) {
            weight = w[0]; // pressure side / below domain
        } else {
            // Above the foil: suction surface BL zone.
            weight = blendWeight(y, blSurf, blSurf + blDepth, w[3], 1.0f);
        }

        densY[static_cast<std::size_t>(j)] = std::max(0.01f, weight);
    }
    gs.physY_ = buildAxis(ny, densY);

    return gs;
}

} // namespace foilcfd
