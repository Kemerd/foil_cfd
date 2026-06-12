// Fast-path voxelization (plan section 5.3): domain boundary flag stamping,
// quarter-chord AoA rotation of the airfoil outline, scanline parity
// rasterization of the polygon per (x,y) cell center with extrusion across z,
// and the trailing-edge single-cell gap closure pass (plan section 13).
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include "voxelizer.h"

#include <algorithm>
#include <cmath>
#include <limits>

// grid_stretch.h is included via voxelizer.h (it pulls in GridStretch).

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
                     std::vector<std::uint8_t>& flags,
                     const GridStretch* stretch) {
    if (!airfoil.isValid()) return;
    const GridDims& dims = layout.dims;
    if (dims.nx < 3 || dims.ny < 3 || dims.nz < 1) return;

    // ---- Step 1: transform the outline into lattice coordinates. ----------
    // Rotate about the quarter-chord point (0.25, 0) by -AoA: a positive AoA
    // pitches the nose UP while the inflow stays axis-aligned (plan 4.2 —
    // rotating geometry instead of inflow keeps warm-cache keying simple).
    // Then scale chord units to cells and translate to the layout anchor.
    //
    // When a GridStretch is active, the anchor (ax, ay) is expressed in
    // PHYSICAL cell units (the stretched coordinate). The rasterization
    // scanline loop iterates over LOGICAL cell indices, so after computing
    // the physical polygon we also need the physical -> logical inverse for
    // the scanline Y loop bounds. The polygon crossing math stays in physical
    // coordinates; the final (x,y) comparison is against the cell centre's
    // PHYSICAL position (stretch->toPhysX(i)+0.5 etc.), which keeps the
    // parity rule correct under stretching.
    const float aoaRad = aoa_deg * kPi / 180.0f;
    const float chord = static_cast<float>(layout.chordCells);
    const float ax = layout.anchorX();
    const float ay = layout.anchorY();
    const Vec2f quarterChord(0.25f, 0.0f);

    const std::size_t n = airfoil.points.size();
    std::vector<Vec2f> poly;
    poly.reserve(n);
    float minYPhys = std::numeric_limits<float>::max();
    float maxYPhys = std::numeric_limits<float>::lowest();
    for (const Vec2f& p : airfoil.points) {
        const Vec2f q = rotated(p - quarterChord, -aoaRad);
        // Physical lattice coordinate: same formula as the uniform case.
        // Under stretching, ax/ay are physical positions (fromPhys* inverts
        // them to logical later); the polygon itself is in physical coords.
        const Vec2f lat(ax + q.x * chord, ay + q.y * chord);
        minYPhys = std::min(minYPhys, lat.y);
        maxYPhys = std::max(maxYPhys, lat.y);
        poly.push_back(lat);
    }

    // ---- Step 2: scanline parity rasterization of one (x,y) slice. --------
    // For each lattice row, intersect the row's cell-center line (in PHYSICAL
    // coordinates) with every polygon edge; the strict half-open rule
    // (a.y > yc) != (b.y > yc) counts each vertex in exactly one edge, so a
    // closed loop yields an even crossing count.
    //
    // When a GridStretch is active, the polygon is in PHYSICAL space. For
    // each logical row j we compute its physical centre (stretch->toPhysY(j)
    // + half_cell_phys) and intersect; x crossings (physical) are inverted
    // back to logical with fromPhysX. Without stretching both toPhys*
    // return identity (float(index)), so the code path is identical.
    //
    // Y loop bounds derived from the physical polygon bbox mapped back to
    // logical via fromPhysY (conservative: +/-1 cell margin).
    int y0, y1;
    if (stretch && stretch->isActive()) {
        y0 = std::clamp(stretch->fromPhysY(minYPhys) - 1, 1, dims.ny - 2);
        y1 = std::clamp(stretch->fromPhysY(maxYPhys) + 1, 1, dims.ny - 2);
    } else {
        y0 = std::clamp(static_cast<int>(std::floor(minYPhys - 0.5f)), 1, dims.ny - 2);
        y1 = std::clamp(static_cast<int>(std::ceil (maxYPhys - 0.5f)), 1, dims.ny - 2);
    }

    std::vector<std::uint8_t> mask(
        static_cast<std::size_t>(dims.nx) * static_cast<std::size_t>(dims.ny), 0);
    std::vector<float> crossings;
    crossings.reserve(8); // a simple foil section crosses each row 2-4 times

    for (int y = y0; y <= y1; ++y) {
        // Physical y of this logical row's cell centre.
        float yc;
        if (stretch && stretch->isActive()) {
            // Physical centre = midpoint between physY[y] and physY[y+1].
            yc = (stretch->toPhysY(y) + stretch->toPhysY(y + 1)) * 0.5f;
        } else {
            yc = static_cast<float>(y) + 0.5f;
        }

        crossings.clear();
        for (std::size_t i = 0; i < n; ++i) {
            const Vec2f& pa = poly[i];
            const Vec2f& pb = poly[(i + 1) % n];
            if ((pa.y > yc) != (pb.y > yc)) {
                // Edge straddles the row: record the physical x of the crossing.
                const float t = (yc - pa.y) / (pb.y - pa.y);
                crossings.push_back(pa.x + t * (pb.x - pa.x));
            }
        }
        std::sort(crossings.begin(), crossings.end());

        // Fill between successive crossing pairs. With stretching, convert
        // physical x crossings to logical cell indices via fromPhysX.
        const std::size_t rowBase = static_cast<std::size_t>(y)
                                  * static_cast<std::size_t>(dims.nx);
        for (std::size_t k = 0; k + 1 < crossings.size(); k += 2) {
            int xa, xb;
            if (stretch && stretch->isActive()) {
                // Convert physical crossing coords to logical cell indices.
                xa = std::max(1, stretch->fromPhysX(crossings[k]));
                xb = std::min(dims.nx - 2, stretch->fromPhysX(crossings[k + 1]));
            } else {
                xa = std::max(1, static_cast<int>(
                                     std::ceil(crossings[k] - 0.5f)));
                xb = std::min(dims.nx - 2, static_cast<int>(
                                     std::floor(crossings[k + 1] - 0.5f)));
            }
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
                                              const DomainLayout& layout,
                                              const GridStretch* stretch) {
    // Composition order matters: boundary faces first, foil solids OR'd over
    // fluid, TE closure last so it sees the final solid set (plan 5.3 / 13).
    std::vector<std::uint8_t> flags = buildBoundaryFlags(layout.dims);
    voxelizeAirfoil(airfoil, aoa_deg, layout, flags, stretch);
    closeTrailingEdgeGaps(layout.dims, flags);
    return flags;
}

} // namespace foilcfd
