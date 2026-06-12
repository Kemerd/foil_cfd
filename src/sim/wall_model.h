// Host-side construction of the wall-model cell list: the set of fluid cells
// adjacent to the FOIL surface (never VG vanes), each with an estimated wall
// normal, a velocity-sampling cell along that normal, and the bounce-back
// link mask the device correction applies to. The list feeds the iMEM-style
// slip-velocity wall function (Asmuth et al., Phys. Fluids 33:105111) whose
// device half lives in lbm_wallmodel.cuh — this module is pure host code so
// the geometry tests can exercise it without a GPU.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <cstdint>
#include <vector>

#include "lbm_core.cuh"

namespace foilcfd {

// ===========================================================================
// Tunables. The neighborhood radius and sampling offsets are fixed by the
// scheme, not user-facing: changing them re-calibrates the wall model.
// ===========================================================================

/// Half-width of the occupancy-gradient stencil used for wall normals: the
/// normal at a wall-adjacent cell is the (negated, normalized) first moment
/// of the solid mask over a (2r+1)^3 box. r = 2 (a 5^3 box) smooths the
/// stair-step enough that a 45-degree facet reads as a 45-degree normal
/// while staying local enough for the curved nose (test: synthetic sphere
/// max error < 15 degrees).
inline constexpr int kWallNormalRadius = 2;

/// Default velocity-sampling distance in cells along the wall normal, per
/// level. WMLES practice avoids matching the law of the wall at the first
/// off-wall node (its LES content is the most under-resolved in the domain,
/// Kawai & Larsson 2012); two cells out on the coarse grid / three on the
/// fine grid sits in the log layer at the resolutions this solver runs.
inline constexpr int kWallSampleCellsCoarse = 2;
inline constexpr int kWallSampleCellsFine   = 3;

// ===========================================================================
// Build output. SoA vectors sized size() so the solver can cudaMemcpy each
// array straight into its device mirror without repacking.
// ===========================================================================

/// @brief One entry per FOIL-adjacent fluid cell selected for wall modeling.
/// Parallel arrays (SoA); index i across all vectors describes one cell.
struct WallCellList {
    /// UNPADDED linear index (x + nx*(y + ny*z)) of the wall-adjacent fluid
    /// cell. Device code derives the padded index by adding nx*ny.
    std::vector<long long> cellIdx;

    /// UNPADDED linear index of the velocity-sampling cell: the nearest
    /// fluid cell to (cell center + sampleCells * normal), after the
    /// fallback chain k -> k-1 -> ... -> 1 -> the wall cell itself.
    std::vector<long long> sampleIdx;

    /// Unit wall normal pointing INTO the fluid, one (x, y, z) triple per
    /// entry, from the occupancy gradient of the CLEAN-FOIL mask.
    std::vector<float> normalX;
    std::vector<float> normalY;
    std::vector<float> normalZ;

    /// Wall-normal distance of the sampling cell's center from the wall
    /// plane, in cells of this level: 0.5 (half-way bounce-back puts the
    /// wall half a cell below the first fluid center) plus the projection
    /// of the cell-to-sample offset onto the normal.
    std::vector<float> sampleDist;

    /// Bit q set (q = 1..18) when the pull of direction q at this cell
    /// bounces off a solid: flags[cell - c_q] == Solid in the ACTIVE field.
    /// After VG exclusion every masked link is a foil link, so the device
    /// correction may apply uniformly across the mask.
    std::vector<std::uint32_t> linkMask;

    /// @brief Number of listed wall cells.
    std::size_t size() const { return cellIdx.size(); }
};

/// @brief Build diagnostics, surfaced in the UI wall-model readout so a
/// configuration that silently degrades (many excluded or degenerate cells)
/// is visible rather than mysterious.
struct WallListStats {
    int listed     = 0; ///< Cells in the final list.
    int excludedVG = 0; ///< Candidates dropped for touching a VG-only solid.
    int degenerate = 0; ///< Candidates dropped for an unresolvable normal.
};

/// @brief Scan a flag field and build the wall-model cell list.
///
/// Selection: a cell qualifies when it is Fluid in @p activeFlags and at
/// least one of its 18 pull neighbors (cell - c_q, z wrapped periodically)
/// is Solid. It is then EXCLUDED when any such solid neighbor is VG-only —
/// Solid in @p activeFlags but not in @p cleanFlags — because the wall-
/// function assumptions (attached log-layer flow) are meaningless on a
/// vane 1-2 cells thick; those cells keep exact plain bounce-back. Passing
/// the same vector for both fields (no VGs configured) disables exclusion.
///
/// Normals come from the clean-foil mask only, so a vane rooted on the
/// surface cannot bend the foil normal under neighboring listed cells.
///
/// @param dims        Grid dimensions of both flag fields.
/// @param activeFlags Live flag field (foil + VGs), dims.cellCount() bytes.
/// @param cleanFlags  Clean-foil flag field (VG-free), same size.
/// @param sampleCells Sampling distance k along the normal, in cells
///                    (kWallSampleCellsCoarse / kWallSampleCellsFine).
/// @param stats       Optional build diagnostics (may be nullptr).
/// @return The wall cell list; empty when no foil surface exists.
WallCellList buildWallCellList(const GridDims& dims,
                               const std::vector<std::uint8_t>& activeFlags,
                               const std::vector<std::uint8_t>& cleanFlags,
                               int sampleCells,
                               WallListStats* stats = nullptr);

} // namespace foilcfd
