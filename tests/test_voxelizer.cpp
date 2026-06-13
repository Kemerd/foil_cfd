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

// -----------------------------------------------------------------------
// AoA 25-degree margin lock (high-AoA support): at the slider's new extreme
// the rotated geometry must keep enough clearance to every domain face for
// the slip walls, the patch-derivation clearance (3 cells), and the
// restriction band (2 cells). The diamond's 50% total thickness makes it a
// strictly harsher case than any real section, so a pass here covers the
// whole catalog. The layout reproduces the plan-4.6 proportions
// (3 Nc x 1.25 Nc, quarter-chord anchored at 0.3 nx / 0.5 ny).
// -----------------------------------------------------------------------
void checkAoA25Margins() {
    const int nc = 128;
    DomainLayout layout;
    layout.dims = GridDims{3 * nc, nc + nc / 4, 8};
    layout.chordCells = nc;

    for (const float aoa : {25.0f, -5.0f}) {
        std::vector<std::uint8_t> flags = buildBoundaryFlags(layout.dims);
        voxelizeAirfoil(makeDiamond(), aoa, layout, flags);

        // Tight bbox of all Solid cells in the z=0 layer (extrusion makes
        // every layer identical — checked by checkDiamondParity).
        int x0 = layout.dims.nx, x1 = -1, y0 = layout.dims.ny, y1 = -1;
        for (int j = 0; j < layout.dims.ny; ++j) {
            for (int i = 0; i < layout.dims.nx; ++i) {
                if (flags[idx(layout.dims, i, j, 0)] != flagOf(CellFlag::Solid))
                    continue;
                x0 = std::min(x0, i); x1 = std::max(x1, i);
                y0 = std::min(y0, j); y1 = std::max(y1, j);
            }
        }
        TREQUIRE_VOID(x1 >= 0); // something voxelized at all

        // 8 cells of clearance: 1 boundary plane + 3 patch clearance + 2
        // restriction band + 2 of slack. The y faces are the tight ones
        // (the TE swings down 0.75c * sin 25 ~ 0.32c at +25 deg).
        constexpr int kClear = 8;
        TCHECK_MSG(y0 >= kClear, "aoa %.0f: bottom clearance %d", aoa, y0);
        TCHECK_MSG(y1 < layout.dims.ny - kClear,
                   "aoa %.0f: top clearance %d", aoa, layout.dims.ny - 1 - y1);
        TCHECK_MSG(x0 >= kClear, "aoa %.0f: inlet clearance %d", aoa, x0);
        TCHECK_MSG(x1 < layout.dims.nx - kClear,
                   "aoa %.0f: outlet clearance %d", aoa,
                   layout.dims.nx - 1 - x1);

        // The default-margin patch derivation must still produce a valid box
        // (margins in cells at this chord: 0.2c/0.5c/0.1c/0.2c).
        const PatchBox box = derivePatchBox(layout.dims, flags,
                                            nc / 5, nc / 2, nc / 10, nc / 5);
        TCHECK_MSG(box.valid(), "aoa %.0f: patch box invalid", aoa);
    }
}

// -----------------------------------------------------------------------
// Patch-local anchor regression (the nested VG box bug): makeFineLayout must
// keep a NEGATIVE patch-local anchor authoritative. A VG-hugging refinement
// box sits on the suction surface ABOVE the quarter-chord line (and can start
// downstream of it), so anchor - box origin is legitimately negative on
// either axis. The old ">= 0 means absolute" sentinel silently fell back to
// the mid-box FRACTION there, voxelizing a misplaced section into the level
// (the user-visible "mesh flipped inside the x4 box"). Pin both the accessor
// values and the voxelized geometry against the same analytic diamond used
// by checkDiamondParity, expressed in patch-local fine coordinates.
// -----------------------------------------------------------------------
void checkPatchLocalAnchorAboveChord() {
    // Same coarse setup as checkDiamondParity: anchor (9.6, 12), diamond
    // center (13.6, 12), semi-axes (8, 4) in coarse cells.
    const GridDims dims{32, 24, 3};
    DomainLayout layout;
    layout.dims = dims;
    layout.chordCells = 16;

    // Patch origin above AND downstream of the anchor -> both patch-local
    // anchor coordinates go negative: x: 2*(9.6-10) = -0.8, y: 2*(12-13) = -2.
    PatchBox box;
    box.x0 = 10; box.x1 = 20; // coarse cells, exclusive upper
    box.y0 = 13; box.y1 = 22;
    const int factor = 2;
    const DomainLayout fine = makeFineLayout(layout, box, factor);

    TCHECK_MSG(std::fabs(fine.anchorX() - (-0.8f)) < 1e-4f,
               "fine anchorX = %.3f (want -0.8; fraction fallback?)",
               fine.anchorX());
    TCHECK_MSG(std::fabs(fine.anchorY() - (-2.0f)) < 1e-4f,
               "fine anchorY = %.3f (want -2.0; fraction fallback?)",
               fine.anchorY());

    // Voxelize the diamond into a Fluid-initialized fine field (the real
    // refinement pipeline starts from Fluid too — no boundary stamps).
    std::vector<std::uint8_t> flags(
        static_cast<std::size_t>(fine.dims.cellCount()), flagOf(CellFlag::Fluid));
    voxelizeAirfoil(makeDiamond(), /*aoa_deg=*/0.0f, fine, flags);

    // Analytic diamond in patch-local FINE coordinates: center
    // ((13.6-10)*2, (12-13)*2) = (7.2, -2), semi-axes (16, 8) fine cells.
    const auto fineS = [](float px, float py) {
        return std::fabs(px - 7.2f) / 16.0f + std::fabs(py + 2.0f) / 8.0f;
    };

    // Dual cell-center-convention enforcement, same scheme as
    // checkDiamondParity: only cells where both conventions agree with margin
    // are asserted, so the test pins placement without pinning the convention.
    bool classifierOk = true;
    int enforcedInside = 0;
    // Skip the outermost ring like checkDiamondParity does: the voxelizer
    // leaves domain-boundary planes to their boundary flags, and the real
    // pipeline stamps the patch rim as Interface shell anyway.
    for (int j = 1; j < fine.dims.ny - 1; ++j) {
        for (int i = 1; i < fine.dims.nx - 1; ++i) {
            const bool solid =
                flags[idx(fine.dims, i, j, 0)] == flagOf(CellFlag::Solid);
            const float sA = fineS(static_cast<float>(i), static_cast<float>(j));
            const float sB = fineS(static_cast<float>(i) + 0.5f,
                                   static_cast<float>(j) + 0.5f);
            if (sA <= 0.75f && sB <= 0.75f) {
                ++enforcedInside;
                if (!solid) {
                    std::printf("  fine cell (%d,%d) is well inside "
                                "(s=%.2f/%.2f) but not Solid\n", i, j, sA, sB);
                    classifierOk = false;
                }
            }
            if (sA >= 1.25f && sB >= 1.25f && solid) {
                std::printf("  fine cell (%d,%d) is well outside "
                            "(s=%.2f/%.2f) but Solid\n", i, j, sA, sB);
                classifierOk = false;
            }
        }
    }
    TCHECK(classifierOk);
    // The box overlaps the diamond's upper half — the inside assertion must
    // have actually fired (a misplaced section would also tend to zero this).
    TCHECK_MSG(enforcedInside > 10, "only %d enforced-inside cells",
               enforcedInside);
}

} // namespace

int main() {
    checkBoundaryFlags();
    checkDiamondParity();
    checkTrailingEdgeClosure();
    checkAoA25Margins();
    checkPatchLocalAnchorAboveChord();
    return finish("test_voxelizer");
}
