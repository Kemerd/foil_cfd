// Fast-path voxelization (plan section 5.3): domain boundary flag stamping,
// quarter-chord AoA rotation of the airfoil outline, scanline parity
// rasterization of the polygon per (x,y) cell center with extrusion across z,
// and the trailing-edge single-cell gap closure pass (plan section 13).
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include "voxelizer.h"

#include <algorithm>
#include <cmath>
#include <limits>

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
            // y faces first so the x-face stamps below win at the edges —
            // inlet/outlet columns must be inlet/outlet all the way up.
            if (y == 0)
                for (int x = 0; x < dims.nx; ++x)
                    flags[row + x] = static_cast<std::uint8_t>(CellFlag::SlipBottom);
            if (y == dims.ny - 1)
                for (int x = 0; x < dims.nx; ++x)
                    flags[row + x] = static_cast<std::uint8_t>(CellFlag::SlipTop);
            flags[row + 0] = static_cast<std::uint8_t>(CellFlag::Inlet);
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

} // namespace foilcfd
