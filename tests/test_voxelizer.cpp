// geom/voxelizer checks (plan 5.3 + section 13): boundary-flag stamping
// contract, parity voxelization of a synthetic diamond polygon against a
// hand-computed inside/outside classification, identical z-extrusion, and the
// trailing-edge single-cell-gap closure pass on a deliberately leaky pattern.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include <cmath>
#include <cstdio>
#include <vector>

#include "geom/voxelizer.h"
#include "test_util.h"

using namespace foilcfd;
using namespace foilcfd::testutil;

// Helper-scope fatal check (returns out of the calling void function when the
// condition fails — later checks in that helper would crash or be meaningless).
#define TREQUIRE_VOID(expr)                                                    \
    do {                                                                       \
        const bool ok_ = static_cast<bool>(expr);                              \
        ::foilcfd::testutil::recordCheck(ok_, #expr, __FILE__, __LINE__);      \
        if (!ok_) return;                                                      \
    } while (0)

namespace {

// Linear cell index for the frozen layout (x fastest — lbm_core.cuh).
std::size_t idx(const GridDims& d, int x, int y, int z) {
    return static_cast<std::size_t>(x)
         + static_cast<std::size_t>(d.nx)
               * (static_cast<std::size_t>(y)
                  + static_cast<std::size_t>(d.ny) * static_cast<std::size_t>(z));
}

std::uint8_t flagOf(CellFlag f) { return static_cast<std::uint8_t>(f); }

// -----------------------------------------------------------------------
// buildBoundaryFlags: Inlet plane at x=0, Outlet at x=nx-1, free-slip y
// faces, Fluid bulk, NO z-face flags (z is periodic). The inlet wins at
// its y-face corners (its storage is rewritten every step, so corner reads
// stay fresh); the OUTLET yields to the slip planes at its corners — an
// outlet corner would zero-gradient-copy from a slip marker cell whose f
// storage is never written, leaking frozen cold-start equilibrium into the
// interior corner links forever.
// -----------------------------------------------------------------------
void checkBoundaryFlags() {
    const GridDims dims{8, 6, 3};
    const std::vector<std::uint8_t> flags = buildBoundaryFlags(dims);
    TREQUIRE_VOID(flags.size() == static_cast<std::size_t>(dims.cellCount()));

    bool ok = true;
    for (int z = 0; z < dims.nz && ok; ++z) {
        for (int y = 0; y < dims.ny && ok; ++y) {
            for (int x = 0; x < dims.nx && ok; ++x) {
                std::uint8_t expect = flagOf(CellFlag::Fluid);
                if (y == 0) expect = flagOf(CellFlag::SlipBottom);
                if (y == dims.ny - 1) expect = flagOf(CellFlag::SlipTop);
                if (x == 0) expect = flagOf(CellFlag::Inlet);   // inlet wins
                if (x == dims.nx - 1 && y != 0 && y != dims.ny - 1)
                    expect = flagOf(CellFlag::Outlet);          // slip wins at corners
                if (flags[idx(dims, x, y, z)] != expect) {
                    std::printf("  flag mismatch at (%d,%d,%d): got %d want %d\n",
                                x, y, z, flags[idx(dims, x, y, z)], expect);
                    ok = false;
                }
            }
        }
    }
    TCHECK(ok);
}

// -----------------------------------------------------------------------
// Synthetic diamond voxelization. Outline: a diamond with chord [0,1] and
// half-thickness 0.25, expressed in the canonical Selig loop ordering
// (TE -> upper -> LE -> lower -> TE) and subdivided so it passes the >= 20
// point validity bar. With aoa = 0, chordCells = 16 and the default anchors
// on a 32 x 24 grid, the quarter-chord (0.25, 0) lands at lattice
// (0.3*32, 0.5*24) = (9.6, 12), so in lattice space the diamond is
//   |px - 13.6| / 8 + |py - 12| / 4 <= 1     (center 13.6,12; semi-axes 8,4).
//
// The hand-computed expectation is evaluated with a sub-cell safety margin
// at BOTH plausible cell-center conventions (p = (i,j) and p = (i+.5,j+.5)),
// and only enforced where the two agree — that pins the geometry while
// staying robust to the implementation's exact center convention. Cells
// within half a cell of the outline are left unasserted by construction.
// -----------------------------------------------------------------------
AirfoilGeometry makeDiamond() {
    AirfoilGeometry foil;
    foil.name = "synthetic diamond";
    // Four corners of the loop in Selig order; LE is the third corner.
    const Vec2f corners[5] = {
        {1.0f, 0.0f}, {0.5f, 0.25f}, {0.0f, 0.0f}, {0.5f, -0.25f}, {1.0f, 0.0f}};
    constexpr int kSub = 6; // 6 segments per edge -> 24 points, >= 20 minimum
    for (int e = 0; e < 4; ++e) {
        for (int s = 0; s < kSub; ++s) {
            const float t = static_cast<float>(s) / static_cast<float>(kSub);
            foil.points.push_back(corners[e] + (corners[e + 1] - corners[e]) * t);
        }
    }
    foil.points.push_back(corners[4]); // close the loop explicitly at the TE
    foil.leIndex = 2 * kSub;           // index of the (0,0) corner
    return foil;
}

// Signed "diamond coordinate": s <= 1 inside, > 1 outside. Its gradient
// magnitude is sqrt((1/8)^2 + (1/4)^2) ~ 0.28 per cell, so an s-margin of
// 0.25 corresponds to ~0.9 cells of clearance from the outline.
float diamondS(float px, float py) {
    return std::fabs(px - 13.6f) / 8.0f + std::fabs(py - 12.0f) / 4.0f;
}

void checkDiamondParity() {
    const GridDims dims{32, 24, 3};
    DomainLayout layout;
    layout.dims = dims;
    layout.chordCells = 16; // diamond spans lattice x in [5.6, 21.6]

    std::vector<std::uint8_t> flags = buildBoundaryFlags(dims);
    voxelizeAirfoil(makeDiamond(), /*aoa_deg=*/0.0f, layout, flags);

    int solidCount = 0;        // solids in the z=0 layer for the area check
    bool classifierOk = true;  // every enforced cell matched the prediction
    for (int j = 1; j < dims.ny - 1; ++j) {
        for (int i = 1; i < dims.nx - 1; ++i) {
            const bool solid =
                flags[idx(dims, i, j, 0)] == flagOf(CellFlag::Solid);
            if (solid) ++solidCount;

            // Hand-computed classification at both center conventions.
            const float sA = diamondS(static_cast<float>(i), static_cast<float>(j));
            const float sB = diamondS(static_cast<float>(i) + 0.5f,
                                      static_cast<float>(j) + 0.5f);
            if (sA <= 0.75f && sB <= 0.75f && !solid) {
                std::printf("  cell (%d,%d) is well inside (s=%.2f/%.2f) but not Solid\n",
                            i, j, sA, sB);
                classifierOk = false;
            }
            if (sA >= 1.25f && sB >= 1.25f && solid) {
                std::printf("  cell (%d,%d) is well outside (s=%.2f/%.2f) but Solid\n",
                            i, j, sA, sB);
                classifierOk = false;
            }
        }
    }
    TCHECK(classifierOk);

    // Area sanity: the diamond covers 0.5 * 16 * 8 = 64 cell^2; stair-step
    // quantization keeps the voxel count near that.
    TCHECK_MSG(solidCount >= 40 && solidCount <= 90,
               "z=0 solid count = %d (expected ~64)", solidCount);

    // Extrusion: the 2D mask must be bit-identical across every z layer.
    bool extrusionOk = true;
    for (int z = 1; z < dims.nz && extrusionOk; ++z)
        for (int j = 0; j < dims.ny && extrusionOk; ++j)
            for (int i = 0; i < dims.nx && extrusionOk; ++i)
                if (flags[idx(dims, i, j, z)] != flags[idx(dims, i, j, 0)])
                    extrusionOk = false;
    TCHECK(extrusionOk);

    // Boundary flags must survive voxelization untouched on the inlet column
    // (the diamond never reaches x=0).
    TCHECK(flags[idx(dims, 0, 12, 0)] == flagOf(CellFlag::Inlet));
}

// -----------------------------------------------------------------------
// TE gap closure (plan section 13): a Fluid cell whose +y AND -y neighbors
// are both Solid is a leaky single-cell channel and must become Solid.
// Build the pattern by hand and verify the pass closes exactly it.
// -----------------------------------------------------------------------
void checkTrailingEdgeClosure() {
    const GridDims dims{8, 8, 2};
    std::vector<std::uint8_t> flags(
        static_cast<std::size_t>(dims.cellCount()), flagOf(CellFlag::Fluid));

    // Deliberate 1-cell gap at (3,3,*): Solid above and below, Fluid between.
    for (int z = 0; z < dims.nz; ++z) {
        flags[idx(dims, 3, 2, z)] = flagOf(CellFlag::Solid);
        flags[idx(dims, 3, 4, z)] = flagOf(CellFlag::Solid);
        // Control cell: a lone Solid at (6,2,*) must NOT convert its
        // single-sided neighbors.
        flags[idx(dims, 6, 2, z)] = flagOf(CellFlag::Solid);
    }

    const int closed = closeTrailingEdgeGaps(dims, flags);

    // Exactly the two gap cells (one per z layer) close; nothing else does.
    TCHECK_MSG(closed == 2, "closed = %d (expected 2)", closed);
    for (int z = 0; z < dims.nz; ++z) {
        TCHECK(flags[idx(dims, 3, 3, z)] == flagOf(CellFlag::Solid));
        TCHECK(flags[idx(dims, 6, 1, z)] == flagOf(CellFlag::Fluid));
        TCHECK(flags[idx(dims, 6, 3, z)] == flagOf(CellFlag::Fluid));
        // The pre-existing solids are untouched.
        TCHECK(flags[idx(dims, 3, 2, z)] == flagOf(CellFlag::Solid));
        TCHECK(flags[idx(dims, 3, 4, z)] == flagOf(CellFlag::Solid));
    }
}

} // namespace

int main() {
    checkBoundaryFlags();
    checkDiamondParity();
    checkTrailingEdgeClosure();
    return finish("test_voxelizer");
}
