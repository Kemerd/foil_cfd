// Fast-path voxelization (plan section 5.3): domain boundary flag stamping,
// quarter-chord AoA rotation of the airfoil outline, scanline parity
// rasterization of the polygon per (x,y) cell center with extrusion across z,
// and the trailing-edge single-cell gap closure pass (plan section 13).
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include "voxelizer.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "vg.h" // VGParams + voxelizeVG, for deriveVGPatchBoxFine (cycle-free:
                // vg.h includes voxelizer.h, this .cpp includes vg.h)

namespace foilcfd {
namespace {

constexpr float kPi = 3.14159265358979f;

// Raw flag bytes used in the hot loops below (the field stores uint8, not the
// enum, so the comparisons stay branch-cheap and readable).
constexpr std::uint8_t kFluid = static_cast<std::uint8_t>(CellFlag::Fluid);
constexpr std::uint8_t kSolid = static_cast<std::uint8_t>(CellFlag::Solid);

} // namespace

std::vector<std::uint8_t> buildBoundaryFlags(const GridDims& dims) {
    // Layout reminder: index = x + nx*(y + ny*z), x fastest.
    std::vector<std::uint8_t> flags(
        static_cast<std::size_t>(dims.cellCount()),
        static_cast<std::uint8_t>(CellFlag::Fluid));
    for (int z = 0; z < dims.nz; ++z) {
        for (int y = 0; y < dims.ny; ++y) {
            const std::size_t row = static_cast<std::size_t>(dims.nx)
                                  * (static_cast<std::size_t>(y)
                                     + static_cast<std::size_t>(dims.ny) * z);
            // y faces first; the inlet stamp below wins at the x=0 edges
            // (the inlet column rewrites its storage every step, so corner
            // reads always see fresh equilibrium). The OUTLET column instead
            // yields to the slip planes at its two corners: an outlet corner
            // cell would zero-gradient-copy from (nx-2, 0/ny-1, z) — a slip
            // MARKER whose f storage is never written — and feed that frozen
            // cold-start equilibrium into the interior corner links forever.
            // With slip winning, the corner markers are never read (slip
            // pulls redirect to the adjacent outlet cells, rewritten every
            // step) and every outlet cell copies from a live fluid column.
            if (y == 0)
                for (int x = 0; x < dims.nx; ++x)
                    flags[row + x] = static_cast<std::uint8_t>(CellFlag::SlipBottom);
            if (y == dims.ny - 1)
                for (int x = 0; x < dims.nx; ++x)
                    flags[row + x] = static_cast<std::uint8_t>(CellFlag::SlipTop);
            flags[row + 0] = static_cast<std::uint8_t>(CellFlag::Inlet);
            if (y != 0 && y != dims.ny - 1)
                flags[row + dims.nx - 1] = static_cast<std::uint8_t>(CellFlag::Outlet);
        }
    }
    return flags;
}

void voxelizeAirfoil(const AirfoilGeometry& airfoil, float aoa_deg,
                     const DomainLayout& layout,
                     std::vector<std::uint8_t>& flags) {
    if (!airfoil.isValid()) return;
    const GridDims& dims = layout.dims;
    if (dims.nx < 3 || dims.ny < 3 || dims.nz < 1) return;

    // ---- Step 1: transform the outline into lattice coordinates. ----------
    // Rotate about the quarter-chord point (0.25, 0) by -AoA: a positive AoA
    // pitches the nose UP while the inflow stays axis-aligned (plan 4.2 —
    // rotating geometry instead of inflow keeps warm-cache keying simple).
    // Then scale chord units to cells and translate to the layout anchor.
    const float aoaRad = aoa_deg * kPi / 180.0f;
    const float chord = static_cast<float>(layout.chordCells);
    const float ax = layout.anchorX();
    const float ay = layout.anchorY();
    const Vec2f quarterChord(0.25f, 0.0f);

    const std::size_t n = airfoil.points.size();
    std::vector<Vec2f> poly;
    poly.reserve(n);
    float minY = std::numeric_limits<float>::max();
    float maxY = std::numeric_limits<float>::lowest();
    for (const Vec2f& p : airfoil.points) {
        const Vec2f q = rotated(p - quarterChord, -aoaRad);
        const Vec2f lat(ax + q.x * chord, ay + q.y * chord);
        minY = std::min(minY, lat.y);
        maxY = std::max(maxY, lat.y);
        poly.push_back(lat);
    }

    // ---- Step 2: scanline parity rasterization of one (x,y) slice. --------
    // For each lattice row, intersect the row's cell-center line y = y+0.5
    // with every polygon edge; the strict half-open rule (a.y > yc) !=
    // (b.y > yc) counts each vertex in exactly one of its two edges, so a
    // closed loop always yields an even crossing count, and zero-length edges
    // (duplicated closing TE point) contribute nothing. Cells whose centers
    // fall between odd/even crossing pairs are inside — the parity test of
    // plan 5.3, organized per row (O(rows * edges) instead of per cell).
    const int y0 = std::clamp(static_cast<int>(std::floor(minY - 0.5f)), 1,
                              dims.ny - 2);
    const int y1 = std::clamp(static_cast<int>(std::ceil(maxY - 0.5f)), 1,
                              dims.ny - 2);
    std::vector<std::uint8_t> mask(
        static_cast<std::size_t>(dims.nx) * static_cast<std::size_t>(dims.ny), 0);
    std::vector<float> crossings;
    crossings.reserve(8); // a simple foil section crosses each row 2-4 times

    for (int y = y0; y <= y1; ++y) {
        const float yc = static_cast<float>(y) + 0.5f;
        crossings.clear();
        for (std::size_t i = 0; i < n; ++i) {
            const Vec2f& pa = poly[i];
            const Vec2f& pb = poly[(i + 1) % n];
            if ((pa.y > yc) != (pb.y > yc)) {
                // Edge straddles the row: record the x of the intersection.
                const float t = (yc - pa.y) / (pb.y - pa.y);
                crossings.push_back(pa.x + t * (pb.x - pa.x));
            }
        }
        std::sort(crossings.begin(), crossings.end());
        // Fill between successive crossing pairs: cell centers x+0.5 strictly
        // inside (cross[k], cross[k+1]) are interior by the parity rule.
        const std::size_t rowBase = static_cast<std::size_t>(y)
                                  * static_cast<std::size_t>(dims.nx);
        for (std::size_t k = 0; k + 1 < crossings.size(); k += 2) {
            const int xa = std::max(1, static_cast<int>(
                                           std::ceil(crossings[k] - 0.5f)));
            const int xb = std::min(dims.nx - 2, static_cast<int>(
                                           std::floor(crossings[k + 1] - 0.5f)));
            for (int x = xa; x <= xb; ++x) mask[rowBase + x] = 1;
        }
    }

    // ---- Step 3: extrude the 2D mask identically across all z slices. -----
    // Only Fluid cells become Solid: the domain-face flags (inlet/outlet/
    // slip) stamped by buildBoundaryFlags stay untouched even if a huge foil
    // bounding box brushes them.
    const std::size_t sliceSize = static_cast<std::size_t>(dims.nx)
                                * static_cast<std::size_t>(dims.ny);
    for (int z = 0; z < dims.nz; ++z) {
        const std::size_t zBase = sliceSize * static_cast<std::size_t>(z);
        for (int y = y0; y <= y1; ++y) {
            const std::size_t rowBase = static_cast<std::size_t>(y)
                                      * static_cast<std::size_t>(dims.nx);
            for (int x = 1; x < dims.nx - 1; ++x) {
                if (mask[rowBase + x] && flags[zBase + rowBase + x] == kFluid)
                    flags[zBase + rowBase + x] = kSolid;
            }
        }
    }
}

int closeTrailingEdgeGaps(const GridDims& dims,
                          std::vector<std::uint8_t>& flags) {
    if (dims.nx < 3 || dims.ny < 3 || dims.nz < 1) return 0;
    // Single deterministic pass (plan 13): read neighbor states from an
    // unmodified snapshot so closures made this pass cannot cascade into
    // chains — only ORIGINAL single-cell gaps (Solid / Fluid / Solid in y)
    // are sealed, which is exactly the leaky-TE failure mode.
    const std::vector<std::uint8_t> before = flags;
    const std::size_t nx = static_cast<std::size_t>(dims.nx);
    const std::size_t sliceSize = nx * static_cast<std::size_t>(dims.ny);
    int closed = 0;
    for (int z = 0; z < dims.nz; ++z) {
        for (int y = 1; y < dims.ny - 1; ++y) {
            const std::size_t base = sliceSize * static_cast<std::size_t>(z)
                                   + nx * static_cast<std::size_t>(y);
            for (int x = 0; x < dims.nx; ++x) {
                const std::size_t i = base + x;
                // A fluid cell pinched between solids at both +y and -y is a
                // one-cell leak channel through the thin TE wedge: seal it.
                if (before[i] == kFluid && before[i - nx] == kSolid
                    && before[i + nx] == kSolid) {
                    flags[i] = kSolid;
                    ++closed;
                }
            }
        }
    }
    return closed;
}

std::vector<std::uint8_t> buildCleanFoilFlags(const AirfoilGeometry& airfoil,
                                              float aoa_deg,
                                              const DomainLayout& layout) {
    // Composition order matters: boundary faces first, foil solids OR'd over
    // fluid, TE closure last so it sees the final solid set (plan 5.3 / 13).
    std::vector<std::uint8_t> flags = buildBoundaryFlags(layout.dims);
    voxelizeAirfoil(airfoil, aoa_deg, layout, flags);
    closeTrailingEdgeGaps(layout.dims, flags);
    return flags;
}

// ===========================================================================
// Refinement-patch flag construction (plan M-refine). The fine level reuses
// the SAME voxelization machinery as the coarse grid — the only differences
// are the doubled chord cells, the patch-local anchor, and the Interface
// shell replacing the domain boundary flags.
// ===========================================================================

DomainLayout makeFineLayout(const DomainLayout& coarse, const PatchBox& box,
                            int factor) {
    DomainLayout fine;
    fine.dims       = fineDimsFor(box, coarse.dims, factor);
    fine.chordCells = factor * coarse.chordCells;
    // Anchor in patch-local FINE cells: the coarse anchor shifted to the
    // patch origin, scaled up. Fractions are bypassed via the absolute
    // override so the foil lands at the exact same physical spot.
    fine.anchorXCells = static_cast<float>(factor)
                      * (coarse.anchorX() - static_cast<float>(box.x0));
    fine.anchorYCells = static_cast<float>(factor)
                      * (coarse.anchorY() - static_cast<float>(box.y0));
    return fine;
}

void stampInterfaceShell(const GridDims& dims,
                         std::vector<std::uint8_t>& flags) {
    const auto iface = static_cast<std::uint8_t>(CellFlag::Interface);
    const std::size_t nx = static_cast<std::size_t>(dims.nx);
    const std::size_t sliceSize = nx * static_cast<std::size_t>(dims.ny);
    const int s = kInterfaceShellFine;
    for (int z = 0; z < dims.nz; ++z) {
        const std::size_t zBase = sliceSize * static_cast<std::size_t>(z);
        for (int y = 0; y < dims.ny; ++y) {
            const std::size_t rowBase = zBase + nx * static_cast<std::size_t>(y);
            const bool yShell = (y < s) || (y >= dims.ny - s);
            if (yShell) {
                // Whole row is shell on the top/bottom bands.
                for (int x = 0; x < dims.nx; ++x) flags[rowBase + x] = iface;
            } else {
                // Left/right bands only.
                for (int x = 0; x < s; ++x) flags[rowBase + x] = iface;
                for (int x = dims.nx - s; x < dims.nx; ++x)
                    flags[rowBase + x] = iface;
            }
        }
    }
}

std::vector<std::uint8_t> buildFinePatchFlags(const AirfoilGeometry& airfoil,
                                              float aoa_deg,
                                              const DomainLayout& coarse,
                                              const PatchBox& box,
                                              int factor) {
    const DomainLayout fine = makeFineLayout(coarse, box, factor);
    // All-Fluid base: the fine level has no inlet/outlet/slip faces — every
    // outer boundary condition arrives through the Interface shell, and the
    // z faces stay periodic exactly like the coarse grid.
    std::vector<std::uint8_t> flags(
        static_cast<std::size_t>(fine.dims.cellCount()),
        static_cast<std::uint8_t>(CellFlag::Fluid));
    voxelizeAirfoil(airfoil, aoa_deg, fine, flags);
    closeTrailingEdgeGaps(fine.dims, flags);
    stampInterfaceShell(fine.dims, flags);
    return flags;
}

PatchBox derivePatchBox(const GridDims& dims,
                        const std::vector<std::uint8_t>& flags,
                        int marginX0, int marginX1, int marginY0, int marginY1,
                        int clearance) {
    // Tight bbox of every Solid cell across all z (the patch always spans the
    // full z extent, so only x/y matter here). One linear pass — runs only on
    // geometry edits.
    int sx0 = dims.nx, sx1 = -1, sy0 = dims.ny, sy1 = -1;
    const std::size_t nx = static_cast<std::size_t>(dims.nx);
    const std::size_t sliceSize = nx * static_cast<std::size_t>(dims.ny);
    for (int z = 0; z < dims.nz; ++z) {
        const std::size_t zBase = sliceSize * static_cast<std::size_t>(z);
        for (int y = 0; y < dims.ny; ++y) {
            const std::size_t rowBase = zBase + nx * static_cast<std::size_t>(y);
            for (int x = 0; x < dims.nx; ++x) {
                if (flags[rowBase + x] == kSolid) {
                    sx0 = std::min(sx0, x);
                    sx1 = std::max(sx1, x);
                    sy0 = std::min(sy0, y);
                    sy1 = std::max(sy1, y);
                }
            }
        }
    }

    PatchBox box; // default-invalid when no solids exist
    if (sx1 < sx0) return box;

    // Pad by the margins, clamp to keep `clearance` coarse cells between the
    // patch and every domain face (the interpolation stencil + the boundary
    // flag columns must never enter the shell's source region).
    box.x0 = std::max(clearance, sx0 - marginX0);
    box.x1 = std::min(dims.nx - clearance, sx1 + 1 + marginX1);
    box.y0 = std::max(clearance, sy0 - marginY0);
    box.y1 = std::min(dims.ny - clearance, sy1 + 1 + marginY1);
    return box;
}

PatchBox deriveVGPatchBoxFine(const DomainLayout& fineLayout,
                              const std::vector<VGParams>& vgs,
                              const AirfoilGeometry& airfoil, float aoa_deg,
                              int marginX0, int marginX1,
                              int marginY0, int marginY1) {
    // No vanes -> no nested level. Return the default-invalid box so the caller
    // tears down (or never builds) level 2.
    if (vgs.empty()) return PatchBox{};

    // VG-only field at the FINE resolution: start all-Fluid, stamp ONLY the
    // vanes. We deliberately skip the airfoil, TE closure, and Interface shell —
    // derivePatchBox keys off Solid cells, and the only Solid we want measured
    // here is the vane envelope (the foil under the vanes is covered separately
    // in the level-2 flag field the solver actually steps).
    std::vector<std::uint8_t> vgOnly(
        static_cast<std::size_t>(fineLayout.dims.cellCount()), kFluid);
    for (const VGParams& vg : vgs)
        voxelizeVG(vg, airfoil, aoa_deg, fineLayout, vgOnly);

    // Keep at least kInterfaceShellFine + 3 fine cells between the box and every
    // fine-domain face: the fine Interface shell carries one-sub-step-old
    // boundary populations, not evolved fluid, so the level-2 interpolation
    // stencil (and the restriction band) must never reach it.
    const int clearance = kInterfaceShellFine + 3;
    return derivePatchBox(fineLayout.dims, vgOnly, marginX0, marginX1, marginY0,
                          marginY1, clearance);
}

} // namespace foilcfd
