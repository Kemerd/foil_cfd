// Fast-path voxelization (plan section 5.3): rotate the airfoil polygon about
// quarter-chord by AoA, point-in-polygon test every (x,y) lattice column,
// extrude across z, close single-cell TE gaps, and stamp the domain boundary
// flags. Produces the host flag field the solver and snapshots key against.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <cstdint>
#include <vector>

#include "../sim/lbm_core.cuh"
#include "../sim/lbm_refine.cuh" // PatchBox + shell constants (sim layer owns them)
#include "airfoil.h"

namespace foilcfd {

// VG model lives in vg.h, which includes THIS header — forward-declare to keep
// the dependency one-directional (deriveVGPatchBoxFine takes VGParams by ref).
struct VGParams;

/// @brief Where the foil sits inside the lattice and how big it is. The
/// defaults reproduce the plan 4.6 layout: chord = N_c cells, quarter-chord
/// anchored at x = 0.3*nx, mid-height. Snapshot keys depend on grid dims and
/// AoA only, so this struct stays out of the key — it is derived layout.
///
/// The optional ABSOLUTE anchor override (anchorXCells/anchorYCells >= 0)
/// exists for the refinement patch (plan M-refine): the fine level is a
/// sub-box of the domain, so the foil's quarter-chord must land at an
/// explicit patch-local cell coordinate rather than a fraction of the
/// (patch-sized) grid. Negative values (the default) keep the fractional
/// behavior, so every existing call site is untouched.
struct DomainLayout {
    GridDims dims;            ///< Lattice dimensions (e.g. 768 x 320 x 96).
    int   chordCells = 256;   ///< N_c — chord length in cells.
    float foilAnchorXFrac = 0.3f; ///< Quarter-chord x position as fraction of nx.
    float foilAnchorYFrac = 0.5f; ///< Quarter-chord y position as fraction of ny.
    float anchorXCells = -1.0f;   ///< Absolute x anchor [cells]; < 0 = use frac.
    float anchorYCells = -1.0f;   ///< Absolute y anchor [cells]; < 0 = use frac.

    /// @brief Lattice x coordinate of the quarter-chord anchor (cells).
    float anchorX() const {
        return anchorXCells >= 0.0f
                   ? anchorXCells
                   : foilAnchorXFrac * static_cast<float>(dims.nx);
    }
    /// @brief Lattice y coordinate of the quarter-chord anchor (cells).
    float anchorY() const {
        return anchorYCells >= 0.0f
                   ? anchorYCells
                   : foilAnchorYFrac * static_cast<float>(dims.ny);
    }
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
/// Does NOT run TE closure — call closeTrailingEdgeGaps() after all solid
/// stamping (foil + VGs) so vane roots also benefit from the pass.
/// @param airfoil Normalized section to voxelize.
/// @param aoa_deg Angle of attack in degrees (UI range -5..25).
/// @param layout  Foil placement and scale inside the grid.
/// @param flags   Flag field to mark (from buildBoundaryFlags or a copy of a
///                cached clean-foil mask). Only writes CellFlag::Solid.
void voxelizeAirfoil(const AirfoilGeometry& airfoil, float aoa_deg,
                     const DomainLayout& layout, std::vector<std::uint8_t>& flags);

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
/// @param airfoil Normalized section.
/// @param aoa_deg Angle of attack in degrees.
/// @param layout  Placement/scale description.
/// @return Complete flag field ready for LBMSolver::init / setFlags.
std::vector<std::uint8_t> buildCleanFoilFlags(const AirfoilGeometry& airfoil,
                                              float aoa_deg,
                                              const DomainLayout& layout);

// ===========================================================================
// Refinement-patch flag construction (two-level ML-LBM). The fine level is a
// 2x-finer sub-box of the coarse domain spanning the full z extent; its flag
// field carries Fluid/Solid plus a 2-cell CellFlag::Interface shell on the
// x/y faces, where the coarse-to-fine fill writes interpolated populations
// each fine step. No inlet/outlet/slip flags exist at the fine level — all
// outer boundary information arrives through the interface shell.
// ===========================================================================

/// @brief Build the fine-level layout for a refinement patch: factor-scaled
/// dims and chord cells, and the foil anchor translated into patch-local
/// fine cells (anchor_f = m * (anchor_c - box.x0)).
/// @param coarse Coarse domain layout (anchor source).
/// @param box    Patch box in coarse cells.
/// @param factor Refinement factor m (2..kMaxRefineFactor).
/// @return Fine DomainLayout ready for voxelizeAirfoil / VG stamping.
DomainLayout makeFineLayout(const DomainLayout& coarse, const PatchBox& box,
                            int factor = kRefineFactor);

/// @brief Build the complete fine-level flag field for a refinement patch:
/// all-Fluid base, foil voxelized at factor-times resolution, TE closure, and
/// the 2-fine-cell Interface shell stamped on the x/y faces (z stays
/// periodic). VG stamping happens at the call site between voxelize and
/// shell stamping when VGs exist — see main.cpp.
/// @param airfoil Normalized section.
/// @param aoa_deg Angle of attack in degrees.
/// @param coarse  Coarse domain layout.
/// @param box     Patch box in coarse cells.
/// @param factor  Refinement factor m (2..kMaxRefineFactor).
/// @return Fine flag field of makeFineLayout(...).dims.cellCount() bytes.
std::vector<std::uint8_t> buildFinePatchFlags(const AirfoilGeometry& airfoil,
                                              float aoa_deg,
                                              const DomainLayout& coarse,
                                              const PatchBox& box,
                                              int factor = kRefineFactor);

/// @brief Stamp the 2-fine-cell Interface shell onto a fine flag field's
/// x/y faces (idempotent; overwrites whatever is there — the shell must win
/// over any solid that grazes the patch edge, which the margin clamps are
/// supposed to prevent anyway).
/// @param dims  Fine grid dimensions.
/// @param flags Fine flag field, modified in place.
void stampInterfaceShell(const GridDims& dims, std::vector<std::uint8_t>& flags);

/// @brief Derive the refinement patch box from the solid content of a flag
/// field: the tight bbox of all Solid cells (any z), padded by the given
/// margins (in cells), clamped to keep at least @p clearance cells of coarse
/// fluid between the shell and every domain face.
/// @param dims      Coarse grid dimensions.
/// @param flags     Coarse flag field to scan (clean or VG-merged).
/// @param marginX0  Cells added upstream  (-x) of the solid bbox.
/// @param marginX1  Cells added downstream (+x).
/// @param marginY0  Cells added below (-y).
/// @param marginY1  Cells added above (+y).
/// @param clearance Minimum gap to the domain faces (>= 3 for the
///                  interpolation stencil).
/// @return Patch box; .valid() is false when no solids exist.
PatchBox derivePatchBox(const GridDims& dims,
                        const std::vector<std::uint8_t>& flags,
                        int marginX0, int marginX1, int marginY0, int marginY1,
                        int clearance = 3);

// ===========================================================================
// Nested VG patch (third level, plan "nested-4x"). The finer level is a tiny
// sub-box of the FINE patch, run at 2x the fine factor, hugging only the VG
// vanes. Its box is expressed in FINE cells (it couples to the fine grid the
// same way the fine grid couples to the coarse grid), so the same derivePatchBox
// machinery applies — only the source flag field (VGs alone) and the larger
// clearance (the fine Interface shell is an invalid interpolation source) differ.
// ===========================================================================

/// @brief Derive the nested VG refinement box, in FINE cells, from the VG
/// vanes alone: voxelize every VG (NO foil, NO TE closure, NO shell) into an
/// all-Fluid field at the fine layout, take the tight Solid bbox via
/// derivePatchBox, and pad it by the given margins so the box is roughly 2x the
/// vane envelope with the vanes centered. The clearance is forced to at least
/// kInterfaceShellFine + 3 fine cells so the box never reaches the fine level's
/// own Interface shell (whose populations are one-sub-step-old boundary data,
/// not evolved fluid — an invalid source for the level-2 interpolation stencil).
/// @param fineLayout The FINE level's layout (makeFineLayout output).
/// @param vgs        Configured VG entries (empty -> invalid box).
/// @param airfoil    Section the VGs mount on (placement frame source).
/// @param aoa_deg    Current angle of attack (vanes rotate with the foil).
/// @param marginX0   Fine cells added upstream  (-x) of the vane bbox.
/// @param marginX1   Fine cells added downstream (+x).
/// @param marginY0   Fine cells added below (-y).
/// @param marginY1   Fine cells added above (+y).
/// @return Box in FINE cells; .valid() is false when there are no VGs or the
///         padded box is below the coupling minimum (caller skips level 2).
PatchBox deriveVGPatchBoxFine(const DomainLayout& fineLayout,
                              const std::vector<VGParams>& vgs,
                              const AirfoilGeometry& airfoil, float aoa_deg,
                              int marginX0, int marginX1,
                              int marginY0, int marginY1);

} // namespace foilcfd
