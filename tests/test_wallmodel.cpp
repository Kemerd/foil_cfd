// Wall-model cell list tests (wall_model.h): selection and link masks on a
// flat slab, 45-degree facet normals on a stair-stepped wedge, normal quality
// on a synthetic sphere, periodic-z handling, build determinism, VG-vane
// exclusion, and the sample-cell fallback chain in a low channel.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#include <cmath>
#include <cstdint>
#include <vector>

#include "sim/wall_model.h"
#include "test_util.h"

using namespace foilcfd;
using namespace foilcfd::testutil;

namespace {

constexpr std::uint8_t kFluid = static_cast<std::uint8_t>(CellFlag::Fluid);
constexpr std::uint8_t kSolid = static_cast<std::uint8_t>(CellFlag::Solid);

/// All-fluid flag field of the given dims.
std::vector<std::uint8_t> fluidField(const GridDims& d) {
    return std::vector<std::uint8_t>(static_cast<std::size_t>(d.cellCount()),
                                     kFluid);
}

long long idx(int x, int y, int z, const GridDims& d) {
    return static_cast<long long>(x)
         + static_cast<long long>(d.nx)
               * (static_cast<long long>(y)
                  + static_cast<long long>(d.ny) * static_cast<long long>(z));
}

/// Angle in degrees between a list entry's normal and an expected direction.
float normalErrorDeg(const WallCellList& list, std::size_t i, float ex,
                     float ey, float ez) {
    const float emag = std::sqrt(ex * ex + ey * ey + ez * ez);
    const float dot = (list.normalX[i] * ex + list.normalY[i] * ey
                       + list.normalZ[i] * ez) / emag;
    const float c = std::fmax(-1.0f, std::fmin(1.0f, dot));
    return std::acos(c) * 180.0f / 3.14159265358979f;
}

void testFlatSlab() {
    const GridDims d{32, 32, 8};
    auto flags = fluidField(d);
    // Solid floor: every cell with y < 16, all x and z.
    for (int z = 0; z < d.nz; ++z)
        for (int y = 0; y < 16; ++y)
            for (int x = 0; x < d.nx; ++x)
                flags[idx(x, y, z, d)] = kSolid;

    WallListStats stats{};
    const WallCellList list =
        buildWallCellList(d, flags, flags, kWallSampleCellsCoarse, &stats);

    // Exactly the y = 16 row qualifies: y = 17 cells reach at most y = 16
    // (fluid) through any pull link.
    TCHECK(stats.listed == d.nx * d.nz);
    TCHECK(stats.excludedVG == 0);
    TCHECK(stats.degenerate == 0);

    // The pull of direction q bounces iff (y - cy) == 15, i.e. cy == +1:
    // q = 3, 7, 11, 14, 17 in the frozen D3Q19 table.
    const std::uint32_t expectMask = (1u << 3) | (1u << 7) | (1u << 11)
                                   | (1u << 14) | (1u << 17);
    // Per-cell checks skip the two columns nearest the x faces: the real
    // domain flags those Inlet/Outlet (never Fluid), so the clipped-window
    // normals the builder produces there are unreachable in practice.
    bool masksOk = true, normalsOk = true, samplesOk = true;
    for (std::size_t i = 0; i < list.size(); ++i) {
        const int cellX = static_cast<int>(list.cellIdx[i] % d.nx);
        if (cellX < 2 || cellX >= d.nx - 2) continue;
        masksOk &= (list.linkMask[i] == expectMask);
        normalsOk &= (normalErrorDeg(list, i, 0.0f, 1.0f, 0.0f) < 0.5f);
        // k = 2 along +y from y = 16 lands at y = 18 (fluid): distance is
        // the half-cell wall offset plus two cells of projection.
        samplesOk &= approxEq(list.sampleDist[i], 2.5, 1e-5);
        samplesOk &= (list.sampleIdx[i] == list.cellIdx[i]
                                              + 2ll * d.nx);
    }
    TCHECK_MSG(masksOk, "flat-slab link masks wrong");
    TCHECK_MSG(normalsOk, "flat-slab normals deviate from +y");
    TCHECK_MSG(samplesOk, "flat-slab sample cells/distances wrong");
}

void testWedge45() {
    // Stair-stepped 45-degree wedge: solid where x + y < 32. The occupancy
    // gradient must read the staircase as the diagonal facet it represents.
    const GridDims d{64, 64, 8};
    auto flags = fluidField(d);
    for (int z = 0; z < d.nz; ++z)
        for (int y = 0; y < d.ny; ++y)
            for (int x = 0; x < d.nx; ++x)
                if (x + y < 32) flags[idx(x, y, z, d)] = kSolid;

    const WallCellList list =
        buildWallCellList(d, flags, flags, kWallSampleCellsCoarse, nullptr);
    TCHECK(list.size() > 0);
    if (list.size() == 0) return; // nothing further to inspect

    float maxErr = 0.0f;
    for (std::size_t i = 0; i < list.size(); ++i) {
        // Skip entries near the open wedge ends (x or y < 4) where the facet
        // meets the domain edge and the half-space assumption breaks.
        const long long c = list.cellIdx[i];
        const int x = static_cast<int>(c % d.nx);
        const int y = static_cast<int>((c / d.nx) % d.ny);
        if (x < 4 || y < 4 || x >= 30 || y >= 30) continue;
        maxErr = std::fmax(maxErr,
                           normalErrorDeg(list, i, 1.0f, 1.0f, 0.0f));
    }
    TCHECK_MSG(maxErr < 5.0f, "45-degree facet normal error %.2f deg", maxErr);
}

void testSphereNormals() {
    // Synthetic sphere: the curved-nose proxy. Every listed normal is
    // compared against the exact radial direction.
    const GridDims d{64, 64, 64};
    auto flags = fluidField(d);
    const float cx = 32.0f, cy = 32.0f, cz = 32.0f, r = 14.0f;
    for (int z = 0; z < d.nz; ++z)
        for (int y = 0; y < d.ny; ++y)
            for (int x = 0; x < d.nx; ++x) {
                const float dx = static_cast<float>(x) - cx;
                const float dy = static_cast<float>(y) - cy;
                const float dz = static_cast<float>(z) - cz;
                if (dx * dx + dy * dy + dz * dz < r * r)
                    flags[idx(x, y, z, d)] = kSolid;
            }

    const WallCellList list =
        buildWallCellList(d, flags, flags, kWallSampleCellsCoarse, nullptr);
    TCHECK(list.size() > 0);
    if (list.size() == 0) return; // nothing further to inspect

    float maxErr = 0.0f;
    double sumErr = 0.0;
    for (std::size_t i = 0; i < list.size(); ++i) {
        const long long c = list.cellIdx[i];
        const int x = static_cast<int>(c % d.nx);
        const int y = static_cast<int>((c / d.nx) % d.ny);
        const int z = static_cast<int>(c / (d.nx * d.ny));
        const float err = normalErrorDeg(list, i, static_cast<float>(x) - cx,
                                         static_cast<float>(y) - cy,
                                         static_cast<float>(z) - cz);
        maxErr = std::fmax(maxErr, err);
        sumErr += err;
    }
    const double meanErr = sumErr / static_cast<double>(list.size());
    TCHECK_MSG(maxErr < 15.0f, "sphere max normal error %.2f deg", maxErr);
    TCHECK_MSG(meanErr < 8.0, "sphere mean normal error %.2f deg", meanErr);
}

void testPeriodicZ() {
    // One solid plane at z = 0: cells at z = 1 see it below (+z normal) and
    // cells at z = nz-1 see it through the periodic wrap (-z normal).
    const GridDims d{16, 16, 8};
    auto flags = fluidField(d);
    for (int y = 0; y < d.ny; ++y)
        for (int x = 0; x < d.nx; ++x)
            flags[idx(x, y, 0, d)] = kSolid;

    const WallCellList list =
        buildWallCellList(d, flags, flags, kWallSampleCellsCoarse, nullptr);
    TCHECK(list.size() == static_cast<std::size_t>(2 * d.nx * d.ny));

    bool ok = true;
    for (std::size_t i = 0; i < list.size(); ++i) {
        const long long c = list.cellIdx[i];
        const int x = static_cast<int>(c % d.nx);
        const int y = static_cast<int>((c / d.nx) % d.ny);
        const int z = static_cast<int>(c / (d.nx * d.ny));
        // Membership is checked everywhere; normal direction only away from
        // the x/y faces, where the clipped stencil tilts it (see flat slab).
        if (z != 1 && z != d.nz - 1) { ok = false; continue; }
        if (x < 2 || x >= d.nx - 2 || y < 2 || y >= d.ny - 2) continue;
        if (z == 1) ok &= (normalErrorDeg(list, i, 0, 0, 1) < 0.5f);
        else ok &= (normalErrorDeg(list, i, 0, 0, -1) < 0.5f);
    }
    TCHECK_MSG(ok, "periodic-z wall cells/normals wrong");
}

void testDeterminism() {
    // Identical inputs must produce bit-identical lists regardless of the
    // thread count the builder picked.
    const GridDims d{48, 48, 16};
    auto flags = fluidField(d);
    const float cx = 24.0f, cy = 24.0f, cz = 8.0f, r = 9.0f;
    for (int z = 0; z < d.nz; ++z)
        for (int y = 0; y < d.ny; ++y)
            for (int x = 0; x < d.nx; ++x) {
                const float dx = static_cast<float>(x) - cx;
                const float dy = static_cast<float>(y) - cy;
                const float dz = static_cast<float>(z) - cz;
                if (dx * dx + dy * dy + dz * dz < r * r)
                    flags[idx(x, y, z, d)] = kSolid;
            }
    const WallCellList a =
        buildWallCellList(d, flags, flags, kWallSampleCellsCoarse, nullptr);
    const WallCellList b =
        buildWallCellList(d, flags, flags, kWallSampleCellsCoarse, nullptr);
    TCHECK(a.cellIdx == b.cellIdx);
    TCHECK(a.sampleIdx == b.sampleIdx);
    TCHECK(a.normalX == b.normalX);
    TCHECK(a.normalY == b.normalY);
    TCHECK(a.normalZ == b.normalZ);
    TCHECK(a.sampleDist == b.sampleDist);
    TCHECK(a.linkMask == b.linkMask);
}

void testVGExclusion() {
    // Clean field: solid floor (y < 16). Active field: floor plus a thin
    // vane (1 cell thick in x, 8 tall, 2 deep in z). Cells whose pull links
    // touch vane material must be excluded wholesale.
    const GridDims d{64, 32, 8};
    auto clean = fluidField(d);
    for (int z = 0; z < d.nz; ++z)
        for (int y = 0; y < 16; ++y)
            for (int x = 0; x < d.nx; ++x)
                clean[idx(x, y, z, d)] = kSolid;
    auto active = clean;
    for (int z = 3; z < 5; ++z)
        for (int y = 16; y < 24; ++y)
            active[idx(32, y, z, d)] = kSolid;

    WallListStats stats{};
    const WallCellList list =
        buildWallCellList(d, active, clean, kWallSampleCellsCoarse, &stats);

    TCHECK_MSG(stats.excludedVG > 0, "vane produced no exclusions");
    // No listed cell may have any pull link onto vane-only solid, and no
    // listed cell may BE one of the vane-adjacent floor cells next to the
    // root (e.g. (31..33, 16, 3) all touch the vane through some link).
    bool ok = true;
    for (std::size_t i = 0; i < list.size(); ++i) {
        for (int q = 1; q < kQ; ++q) {
            if (!(list.linkMask[i] & (1u << q))) continue;
            const long long c = list.cellIdx[i];
            const int x = static_cast<int>(c % d.nx) - kCx[q];
            const int y = static_cast<int>((c / d.nx) % d.ny) - kCy[q];
            int z = static_cast<int>(c / (d.nx * d.ny)) - kCz[q];
            z = (z + d.nz) % d.nz;
            const long long n = idx(x, y, z, d);
            // Masked link must land on CLEAN solid (foil), never vane-only.
            ok &= (clean[n] == kSolid);
        }
    }
    TCHECK_MSG(ok, "a listed cell still links onto vane material");
    // Far from the vane the floor must be listed normally.
    TCHECK(stats.listed > 0);
}

void testSampleFallback() {
    // Low channel: floor (y < 16) and ceiling (18 <= y < 20) leave two fluid
    // rows. Floor wall cells at y = 16 want their sample at y = 18 (k = 2)
    // which is solid, so the chain must fall back to y = 17 with the
    // correspondingly shorter wall distance.
    const GridDims d{32, 32, 8};
    auto flags = fluidField(d);
    for (int z = 0; z < d.nz; ++z)
        for (int x = 0; x < d.nx; ++x) {
            for (int y = 0; y < 16; ++y) flags[idx(x, y, z, d)] = kSolid;
            for (int y = 18; y < 20; ++y) flags[idx(x, y, z, d)] = kSolid;
        }

    const WallCellList list =
        buildWallCellList(d, flags, flags, kWallSampleCellsCoarse, nullptr);
    TCHECK(list.size() > 0);
    if (list.size() == 0) return; // nothing further to inspect

    bool ok = true;
    for (std::size_t i = 0; i < list.size(); ++i) {
        const long long c = list.cellIdx[i];
        const int x = static_cast<int>(c % d.nx);
        const int y = static_cast<int>((c / d.nx) % d.ny);
        if (y != 16) continue; // ceiling underside cells checked implicitly
        if (x < 2 || x >= d.nx - 2) continue; // clipped-stencil edge columns
        ok &= (list.sampleIdx[i] == c + d.nx);        // one row up, not two
        ok &= approxEq(list.sampleDist[i], 1.5, 1e-5); // 0.5 + 1 cell
    }
    TCHECK_MSG(ok, "floor sample fallback wrong in low channel");
}

} // namespace

int main() {
    testFlatSlab();
    testWedge45();
    testSphereNormals();
    testPeriodicZ();
    testDeterminism();
    testVGExclusion();
    testSampleFallback();
    return finish("test_wallmodel");
}
