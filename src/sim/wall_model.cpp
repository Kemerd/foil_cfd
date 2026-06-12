// Wall-model cell list construction (see wall_model.h for the contract).
// Pure host code, deliberately CUDA-free: the heavy lifting is one scan of
// the flag field, parallelized over contiguous z slabs with a deterministic
// in-order merge so repeated builds of the same geometry are bit-identical.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#include "wall_model.h"

#include <algorithm>
#include <cmath>
#include <thread>

namespace foilcfd {

namespace {

/// Occupancy-gradient magnitudes below this are "no usable direction": the
/// cell sits in a pocket where solids surround it near-symmetrically. Such
/// cells keep plain bounce-back rather than receiving a junk normal.
constexpr float kDegenerateNormalEps = 1e-3f;

/// @brief Periodic wrap of a z coordinate into [0, nz). The spanwise axis is
/// periodic everywhere in the solver, so the host scan must match.
inline int wrapZ(int z, int nz) {
    if (z < 0) return z + nz;
    if (z >= nz) return z - nz;
    return z;
}

/// @brief Unpadded linear index for in-range x/y and a wrapped z.
inline long long cellIndex(int x, int y, int z, const GridDims& dims) {
    return static_cast<long long>(x)
         + static_cast<long long>(dims.nx)
               * (static_cast<long long>(y)
                  + static_cast<long long>(dims.ny) * static_cast<long long>(z));
}

/// @brief Solid test with x/y bounds guarding (out-of-range counts as not
/// solid — the foil never touches the domain faces) and periodic z.
inline bool isSolidAt(const std::vector<std::uint8_t>& flags,
                      const GridDims& dims, int x, int y, int z) {
    if (x < 0 || x >= dims.nx || y < 0 || y >= dims.ny) return false;
    return flags[cellIndex(x, y, wrapZ(z, dims.nz), dims)]
           == static_cast<std::uint8_t>(CellFlag::Solid);
}

/// @brief Per-slab output: a private WallCellList plus the slab's share of
/// the diagnostics, merged in slab order after the parallel phase.
struct SlabResult {
    WallCellList  list;
    WallListStats stats;
};

/// @brief Scan one contiguous z range [z0, z1) and emit its wall cells.
/// Shared inputs are read-only; everything written goes to @p out.
void scanSlab(const GridDims& dims,
              const std::vector<std::uint8_t>& activeFlags,
              const std::vector<std::uint8_t>& cleanFlags,
              int sampleCells, int z0, int z1, SlabResult& out) {
    const std::uint8_t kFluid = static_cast<std::uint8_t>(CellFlag::Fluid);
    const std::uint8_t kSolid = static_cast<std::uint8_t>(CellFlag::Solid);

    for (int z = z0; z < z1; ++z) {
        for (int y = 0; y < dims.ny; ++y) {
            for (int x = 0; x < dims.nx; ++x) {
                const long long cell = cellIndex(x, y, z, dims);
                if (activeFlags[cell] != kFluid) continue;

                // Pull-link scan: bit q marks "the stream-collide pull of
                // direction q at this cell bounces off a solid", i.e. the
                // upstream cell (x - c_q) is Solid — the exact condition the
                // device kernel tests (lbm_core.cu pull loop).
                std::uint32_t mask = 0;
                bool touchesVGOnly = false;
                for (int q = 1; q < kQ; ++q) {
                    const int nxp = x - kCx[q];
                    const int nyp = y - kCy[q];
                    const int nzp = z - kCz[q];
                    if (!isSolidAt(activeFlags, dims, nxp, nyp, nzp)) continue;
                    mask |= (1u << q);
                    // VG provenance: solid in the live field but fluid in the
                    // clean-foil field means the neighbor is vane material.
                    if (!isSolidAt(cleanFlags, dims, nxp, nyp, nzp)) {
                        touchesVGOnly = true;
                    }
                }
                if (mask == 0) continue; // interior fluid: not a wall cell
                if (touchesVGOnly) {
                    // Wall-function assumptions are meaningless beside a vane
                    // 1-2 cells thick; the whole cell keeps plain bounce-back.
                    ++out.stats.excludedVG;
                    continue;
                }

                // Wall normal: negated first moment of the CLEAN solid mask
                // over the (2r+1)^3 box. The clean field is used so vanes
                // rooted nearby cannot tilt the foil normal; stair steps
                // average out to the wanted 45-degree facets.
                float gx = 0.0f, gy = 0.0f, gz = 0.0f;
                for (int dz = -kWallNormalRadius; dz <= kWallNormalRadius; ++dz) {
                    for (int dy = -kWallNormalRadius; dy <= kWallNormalRadius; ++dy) {
                        for (int dx = -kWallNormalRadius; dx <= kWallNormalRadius; ++dx) {
                            if (isSolidAt(cleanFlags, dims, x + dx, y + dy, z + dz)) {
                                gx += static_cast<float>(dx);
                                gy += static_cast<float>(dy);
                                gz += static_cast<float>(dz);
                            }
                        }
                    }
                }
                const float gmag = std::sqrt(gx * gx + gy * gy + gz * gz);
                if (gmag < kDegenerateNormalEps) {
                    ++out.stats.degenerate;
                    continue;
                }
                // The moment points INTO the solid; the fluid normal is its
                // negation.
                const float nX = -gx / gmag;
                const float nY = -gy / gmag;
                const float nZ = -gz / gmag;

                // Velocity-sampling cell: nearest cell to k normals out from
                // this center, falling back toward the wall (k-1, ..., 1)
                // when the rounded target is not plain fluid — thin-BL nose
                // regions and patch-shell proximity both land here. The wall
                // cell itself is the terminal fallback (it IS fluid).
                long long sample = cell;
                int sx = x, sy = y, sz = z;
                for (int k = sampleCells; k >= 1; --k) {
                    const int cx = x + static_cast<int>(std::lround(nX * k));
                    const int cy = y + static_cast<int>(std::lround(nY * k));
                    const int cz = wrapZ(z + static_cast<int>(std::lround(nZ * k)),
                                         dims.nz);
                    if (cx < 0 || cx >= dims.nx || cy < 0 || cy >= dims.ny) continue;
                    const long long cand = cellIndex(cx, cy, cz, dims);
                    if (activeFlags[cand] != kFluid) continue;
                    sample = cand;
                    sx = cx; sy = cy; sz = cz;
                    break;
                }

                // Wall distance of the sample center: half a cell (half-way
                // bounce-back wall plane) plus the offset's projection onto
                // the normal, with the z offset taken through the periodic
                // minimal image. Floors at 0.5 — a fallback that went purely
                // tangential still sits half a cell off the wall.
                int dzWrap = sz - z;
                if (dzWrap > dims.nz / 2) dzWrap -= dims.nz;
                if (dzWrap < -dims.nz / 2) dzWrap += dims.nz;
                const float proj = static_cast<float>(sx - x) * nX
                                 + static_cast<float>(sy - y) * nY
                                 + static_cast<float>(dzWrap) * nZ;
                const float ys = 0.5f + std::max(proj, 0.0f);

                out.list.cellIdx.push_back(cell);
                out.list.sampleIdx.push_back(sample);
                out.list.normalX.push_back(nX);
                out.list.normalY.push_back(nY);
                out.list.normalZ.push_back(nZ);
                out.list.sampleDist.push_back(ys);
                out.list.linkMask.push_back(mask);
                ++out.stats.listed;
            }
        }
    }
}

} // namespace

WallCellList buildWallCellList(const GridDims& dims,
                               const std::vector<std::uint8_t>& activeFlags,
                               const std::vector<std::uint8_t>& cleanFlags,
                               int sampleCells, WallListStats* stats) {
    WallCellList merged;
    WallListStats tally;
    if (dims.cellCount() <= 0
        || activeFlags.size() != static_cast<std::size_t>(dims.cellCount())
        || cleanFlags.size() != static_cast<std::size_t>(dims.cellCount())) {
        if (stats) *stats = tally;
        return merged;
    }

    // Slab the z axis across hardware threads. Each slab fills a private
    // list; merging in slab order afterwards reproduces the sequential
    // x-fastest iteration order exactly, so the build is deterministic
    // regardless of thread count or scheduling.
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    const int slabCount = std::clamp(std::min(hw, dims.nz), 1, 32);
    std::vector<SlabResult> results(static_cast<std::size_t>(slabCount));
    {
        std::vector<std::thread> pool;
        pool.reserve(static_cast<std::size_t>(slabCount));
        for (int s = 0; s < slabCount; ++s) {
            const int z0 = static_cast<int>(
                static_cast<long long>(dims.nz) * s / slabCount);
            const int z1 = static_cast<int>(
                static_cast<long long>(dims.nz) * (s + 1) / slabCount);
            pool.emplace_back(scanSlab, std::cref(dims), std::cref(activeFlags),
                              std::cref(cleanFlags), sampleCells, z0, z1,
                              std::ref(results[static_cast<std::size_t>(s)]));
        }
        for (auto& t : pool) t.join();
    }

    // In-order merge: reserve once, then append slab by slab.
    std::size_t total = 0;
    for (const auto& r : results) total += r.list.size();
    merged.cellIdx.reserve(total);
    merged.sampleIdx.reserve(total);
    merged.normalX.reserve(total);
    merged.normalY.reserve(total);
    merged.normalZ.reserve(total);
    merged.sampleDist.reserve(total);
    merged.linkMask.reserve(total);
    for (const auto& r : results) {
        merged.cellIdx.insert(merged.cellIdx.end(), r.list.cellIdx.begin(),
                              r.list.cellIdx.end());
        merged.sampleIdx.insert(merged.sampleIdx.end(), r.list.sampleIdx.begin(),
                                r.list.sampleIdx.end());
        merged.normalX.insert(merged.normalX.end(), r.list.normalX.begin(),
                              r.list.normalX.end());
        merged.normalY.insert(merged.normalY.end(), r.list.normalY.begin(),
                              r.list.normalY.end());
        merged.normalZ.insert(merged.normalZ.end(), r.list.normalZ.begin(),
                              r.list.normalZ.end());
        merged.sampleDist.insert(merged.sampleDist.end(),
                                 r.list.sampleDist.begin(),
                                 r.list.sampleDist.end());
        merged.linkMask.insert(merged.linkMask.end(), r.list.linkMask.begin(),
                               r.list.linkMask.end());
        tally.listed += r.stats.listed;
        tally.excludedVG += r.stats.excludedVG;
        tally.degenerate += r.stats.degenerate;
    }
    if (stats) *stats = tally;
    return merged;
}

} // namespace foilcfd
