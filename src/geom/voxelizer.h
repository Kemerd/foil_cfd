// Fast-path voxelization (plan section 5.3): rotate the airfoil polygon about
// quarter-chord by AoA, point-in-polygon test every (x,y) lattice column,
// extrude across z, close single-cell TE gaps, and stamp the domain boundary
// flags. Produces the host flag field the solver and snapshots key against.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <cstdint>
#include <vector>

#include "../sim/grid_stretch.h"
#include "../sim/lbm_core.cuh"
#include "airfoil.h"

namespace foilcfd {

/// @brief Where the foil sits inside the lattice and how big it is. The
/// defaults reproduce the plan 4.6 layout: chord = N_c cells, quarter-chord
/// anchored at x = 0.3*nx, mid-height. Snapshot keys depend on grid dims and
/// AoA only, so this struct stays out of the key — it is derived layout.
struct DomainLayout {
    GridDims dims;            ///< Lattice dimensions (e.g. 768 x 320 x 96).
    int   chordCells = 256;   ///< N_c — chord length in cells.
    float foilAnchorXFrac = 0.3f; ///< Quarter-chord x position as fraction of nx.
    float foilAnchorYFrac = 0.5f; ///< Quarter-chord y position as fraction of ny.

    /// @brief Lattice x coordinate of the quarter-chord anchor (cells).
    float anchorX() const { return foilAnchorXFrac * static_cast<float>(dims.nx); }
    /// @brief Lattice y coordinate of the quarter-chord anchor (cells).
    float anchorY() const { return foilAnchorYFrac * static_cast<float>(dims.ny); }
};

/// @brief Build a fresh flag field with ONLY the domain boundary conditions
/// stamped (plan 4.2): Inlet at x=0, Outlet at x=nx-1, SlipBottom at y=0,
/// SlipTop at y=ny-1, Fluid everywhere else. z is periodic, so z faces stay
/// Fluid. Corner precedence: the inlet wins over the slip planes (its
/// storage refreshes every step), but the outlet YIELDS to them — an outlet
/// corner cell would zero-gradient-copy from a slip marker whose population
/// storage is never written, freezing cold-start equilibrium into the
/// interior corner links. Voxelizers OR solids into the result afterward.
/// @param dims Grid dimensions.
/// @return dims.cellCount() CellFlag bytes, index = x + nx*(y + ny*z).
std::vector<std::uint8_t> buildBoundaryFlags(const GridDims& dims);

/// @brief Voxelize an airfoil into an existing flag field (plan 5.3).
///
/// Pipeline: (1) rotate the outline polygon about its quarter-chord point
/// (0.25, 0) by -aoa_deg (positive AoA pitches the nose up; rotating the
/// GEOMETRY keeps the inflow axis-aligned for cheap cache keying, plan 4.2);
/// (2) scale by chordCells and translate to the layout anchor; (3) for each
/// (x,y) column, parity-test the cell center against the polygon and mark
/// SOLID; (4) extrude the identical 2D mask across all z. O(nx*ny) tests.
///
/// When @p stretch is non-null the polygon is transformed into LOGICAL cell
/// space via the inverse coordinate map (GridStretch::fromPhys*) before
/// rasterization, so geometry stamps at the correct physical location under
/// any stretching profile.
///
/// Does NOT run TE closure — call closeTrailingEdgeGaps() after all solid
/// stamping (foil + VGs) so vane roots also benefit from the pass.
/// @param airfoil  Normalized section to voxelize.
/// @param aoa_deg  Angle of attack in degrees (UI range -5..20).
/// @param layout   Foil placement and scale inside the grid.
/// @param flags    Flag field to mark (from buildBoundaryFlags or a copy of a
///                 cached clean-foil mask). Only writes CellFlag::Solid.
/// @param stretch  Optional coordinate map (null = uniform, no change).
void voxelizeAirfoil(const AirfoilGeometry& airfoil, float aoa_deg,
                     const DomainLayout& layout, std::vector<std::uint8_t>& flags,
                     const GridStretch* stretch = nullptr);

/// @brief Trailing-edge single-cell-gap closure (plan section 13): voxelized
/// TEs thinner than one cell leave leaky one-cell channels; one pass converts
/// any Fluid cell whose +y AND -y neighbors are both Solid into Solid.
/// Run once after ALL solid geometry (foil + VGs) is stamped.
/// @param dims  Grid dimensions of @p flags.
/// @param flags Flag field, modified in place.
/// @return Number of cells closed (useful for logging/tests).
int closeTrailingEdgeGaps(const GridDims& dims, std::vector<std::uint8_t>& flags);

/// @brief Convenience: full clean-foil flag build = buildBoundaryFlags +
/// voxelizeAirfoil + closeTrailingEdgeGaps. VG editing re-runs only the VG
/// stamping on a kept copy of this result (plan 6.2), so the airfoil parity
/// tests don't repeat per VG slider tick.
/// @param airfoil  Normalized section.
/// @param aoa_deg  Angle of attack in degrees.
/// @param layout   Placement/scale description.
/// @param stretch  Optional coordinate map (null = uniform).
/// @return Complete flag field ready for LBMSolver::init / setFlags.
std::vector<std::uint8_t> buildCleanFoilFlags(const AirfoilGeometry& airfoil,
                                              float aoa_deg,
                                              const DomainLayout& layout,
                                              const GridStretch* stretch = nullptr);

} // namespace foilcfd
