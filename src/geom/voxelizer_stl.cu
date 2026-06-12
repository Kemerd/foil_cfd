// GPU ray-parity STL voxelizer (plan section 7.3): triangle upload, one
// thread per (y,z) row casting a +x ray through the brute-force triangle
// list (shared-memory tiled), in-thread crossing sort, odd/even span fill,
// and atomic counting of bad-parity rows so non-watertight input becomes a
// UI warning instead of garbage flags.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include "voxelizer_stl.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace foilcfd {
namespace {

// ===========================================================================
// Tuning constants.
// ===========================================================================

/// Threads per block. Rows are independent and the kernel is bound by the
/// triangle loop, not occupancy games — 128 keeps shared memory per SM low.
constexpr int kRowBlockSize = 128;

/// Triangles staged per shared-memory tile. 128 * 9 floats = 4.5 KB shared;
/// every thread in the block tests the same staged triangle, so the global
/// triangle buffer is read once per BLOCK instead of once per THREAD —
/// a ~kRowBlockSize-fold cut in global traffic for the brute-force loop.
constexpr int kTriangleTile = 128;

/// Per-row crossing capacity. A +x ray through any sane closed part crosses
/// a handful of surfaces; 64 allows extremely convoluted geometry (32 solid
/// spans on one row). Rows that exceed it are reported like odd-parity rows
/// and left fluid — never trusted, never overrun.
constexpr int kMaxCrossingsPerRow = 64;

/// Sample-point jitter in cells. Rays are cast at cell centers (y+0.5,
/// z+0.5); meshes scaled/translated by the import modal routinely land
/// vertices and axis-aligned edges EXACTLY on those half-integer lines
/// (e.g. a box face after scale-to-chord), and a ray through a shared edge
/// either double-counts or zero-counts with a strict inside test. Nudging
/// the sample point breaks every such tie at a geometric cost (~0.02% of a
/// cell) far below stair-step voxelization error. The two axes get
/// DIFFERENT offsets on purpose: quad-meshed faces carry 45-degree
/// diagonals through lattice-aligned vertices, and an equal jitter would
/// keep every y==z sample exactly on those shared diagonals (verified the
/// hard way on a cube — entire diagonal rows came back with 0 crossings).
constexpr float kRaySampleJitterY = 1.0f / 4096.0f;
constexpr float kRaySampleJitterZ = 1.0f / 8192.0f;

// ===========================================================================
// Ray/triangle crossing test.
//
// The ray is the line { (t, py, pz) : t in R } traversed toward +x, so the
// 3D intersection collapses to 2D: project the triangle onto the (y,z)
// plane, test the point (py, pz) with barycentric edge functions, and if it
// is strictly inside, interpolate the crossing's x from the vertices with
// the same barycentric weights. Triangles parallel to the ray project to
// zero area, make the all-same-sign test fail, and are skipped — their
// boundary contribution belongs to the adjacent non-parallel facets.
//
// Edge functions (u = y, v = z), one per triangle edge, each measuring the
// signed parallelogram area between the edge and the sample point:
//   w0 = e(b, c, p),  w1 = e(c, a, p),  w2 = e(a, b, p)
// All three strictly positive (CCW projection) or strictly negative (CW)
// means strictly inside; the crossing is x = (w0*ax + w1*bx + w2*cx) / sum.
// ===========================================================================

/// @brief Strict-interior ray-parity crossing test, +x ray at (py, pz).
/// constexpr + __host__ __device__ so the compile-time sanity checks below
/// exercise the EXACT code the kernel runs.
/// @return True if the ray crosses the triangle's interior; *xOut = crossing x.
__host__ __device__ constexpr bool rayCrossingX(float ax, float ay, float az,
                                                float bx, float by, float bz,
                                                float cx, float cy, float cz,
                                                float py, float pz,
                                                float* xOut) {
    // Edge functions in the (y,z) projection. Sign convention: w0 pairs with
    // vertex a (edge b->c), w1 with b (edge c->a), w2 with c (edge a->b).
    const float w0 = (cy - by) * (pz - bz) - (cz - bz) * (py - by);
    const float w1 = (ay - cy) * (pz - cz) - (az - cz) * (py - cy);
    const float w2 = (by - ay) * (pz - az) - (bz - az) * (py - ay);

    // Strict interior only: on-edge hits (all w >= 0 with one == 0) are
    // EXCLUDED, because a shared edge would then be counted by both facets
    // and flip parity. The host-side jitter makes exact zeros measure-zero.
    const bool allPos = (w0 > 0.0f) && (w1 > 0.0f) && (w2 > 0.0f);
    const bool allNeg = (w0 < 0.0f) && (w1 < 0.0f) && (w2 < 0.0f);
    if (!(allPos || allNeg)) return false;

    // Barycentric interpolation of x. sum != 0 is guaranteed here: all three
    // weights share one sign, so their sum is bounded away from zero.
    const float sum = w0 + w1 + w2;
    *xOut = (w0 * ax + w1 * bx + w2 * cx) / sum;
    return true;
}

// ===========================================================================
// Compile-time sanity of the intersector against a hand-computed cube wall.
//
// Take the -x face of an axis-aligned box, lying in the plane x = 2, split
// into the usual two triangles; plus one slanted triangle in the plane
// x = y for the interpolation path. All arithmetic below is exact in FP32
// (small integers), so the asserts compare with == legitimately.
//
// Case 1: wall triangle a=(2,0,0) b=(2,4,0) c=(2,0,4), ray at (py,pz)=(1,1).
//   w0 = (0-4)(1-0) - (4-0)(1-4) = -4 + 12 = 8
//   w1 = (0-0)(1-4) - (0-4)(1-0) =  0 +  4 = 4
//   w2 = (4-0)(1-0) - (0-0)(1-0) =  4 -  0 = 4   -> all positive, inside
//   x  = (8*2 + 4*2 + 4*2) / 16 = 32/16 = 2      -> crossing at the wall. OK
// Case 2: the face's OTHER triangle a=(2,4,4) b=(2,0,4) c=(2,4,0), same ray:
//   w0 = (4-0)(1-4) - (0-4)(1-0) = -12 + 4 = -8
//   w1 = (4-4)(1-0) - (4-0)(1-0) =   0 - 4 = -4
//   w2 = (0-4)(1-4) - (4-4)(1-4) =  12 - 0 = 12  -> mixed signs, miss. OK
//   (Together cases 1+2 prove a cube face yields exactly ONE crossing.)
// Case 3: slanted triangle a=(0,0,0) b=(4,4,0) c=(0,0,4) (plane x = y):
//   w0 = (0-4)(1-0) - (4-0)(1-4) = 8, w1 = 4, w2 = 4 -> inside
//   x  = (8*0 + 4*4 + 4*0) / 16 = 1 = py            -> x == y as required. OK
// Case 4: ray (3,3) against case-1 wall: 3+3 > 4 puts it outside the
//   triangle's projected hypotenuse -> miss.
// ===========================================================================

/// Constexpr harness: returns the crossing x, or -1e30f on a miss.
constexpr float testCrossingX(float ax, float ay, float az, float bx, float by,
                              float bz, float cx, float cy, float cz, float py,
                              float pz) {
    float x = -1e30f;
    const bool hit = rayCrossingX(ax, ay, az, bx, by, bz, cx, cy, cz, py, pz, &x);
    return hit ? x : -1e30f;
}

static_assert(testCrossingX(2, 0, 0, 2, 4, 0, 2, 0, 4, 1, 1) == 2.0f,
              "cube wall: +x ray at (1,1) must cross x=2 plane at x=2");
static_assert(testCrossingX(2, 4, 4, 2, 0, 4, 2, 4, 0, 1, 1) == -1e30f,
              "cube wall: second face triangle must NOT also count (1,1)");
static_assert(testCrossingX(0, 0, 0, 4, 4, 0, 0, 0, 4, 1, 1) == 1.0f,
              "slanted plane x=y: crossing at y=1 must interpolate to x=1");
static_assert(testCrossingX(2, 0, 0, 2, 4, 0, 2, 0, 4, 3, 3) == -1e30f,
              "cube wall: ray outside the projected triangle must miss");

// ===========================================================================
// The kernel. One thread per (y,z) row; triangle loop tiled through shared
// memory; crossings sorted in-thread (insertion sort — the list is tiny);
// SOLID written only over Fluid so Inlet/Outlet/Slip stamps survive.
// ===========================================================================

__global__ void rayParityFillKernel(const float* __restrict__ tris,
                                    std::uint32_t triCount, int nx, int ny,
                                    int nz, std::uint8_t* __restrict__ flags,
                                    int* __restrict__ badRowCounter,
                                    int* __restrict__ solidCellCounter) {
    __shared__ float tile[kTriangleTile * 9];

    const long long rowCount = static_cast<long long>(ny) * nz;
    const long long row =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    // Out-of-range threads stay alive to help with the cooperative shared-
    // memory loads (every __syncthreads() must be reached by the whole block).
    const bool active = row < rowCount;
    const int y = active ? static_cast<int>(row % ny) : 0;
    const int z = active ? static_cast<int>(row / ny) : 0;

    // Cell-center sample line, jittered off exact half-integer alignment
    // with per-axis offsets (see the kRaySampleJitter* rationale above).
    const float py = static_cast<float>(y) + 0.5f + kRaySampleJitterY;
    const float pz = static_cast<float>(z) + 0.5f + kRaySampleJitterZ;

    float xs[kMaxCrossingsPerRow];
    int nCross = 0;
    bool overflow = false;

    // Brute force over all triangles (plan 7.3: acceptable to ~200k), staged
    // through shared memory in kTriangleTile chunks.
    for (std::uint32_t base = 0; base < triCount; base += kTriangleTile) {
        const std::uint32_t tileCount =
            (triCount - base < static_cast<std::uint32_t>(kTriangleTile))
                ? triCount - base
                : static_cast<std::uint32_t>(kTriangleTile);
        // Cooperative, coalesced load of the tile's 9*tileCount floats.
        for (std::uint32_t i = threadIdx.x; i < tileCount * 9u; i += blockDim.x) {
            tile[i] = tris[static_cast<std::size_t>(base) * 9u + i];
        }
        __syncthreads();

        if (active && !overflow) {
            for (std::uint32_t t = 0; t < tileCount; ++t) {
                const float* v = &tile[t * 9u];
                float xHit = 0.0f;
                if (rayCrossingX(v[0], v[1], v[2], v[3], v[4], v[5], v[6],
                                 v[7], v[8], py, pz, &xHit)) {
                    if (nCross < kMaxCrossingsPerRow) xs[nCross++] = xHit;
                    else overflow = true; // keep counting? no — row is condemned
                }
            }
        }
        __syncthreads(); // tile must not be overwritten while others still read
    }

    if (!active) return;

    // Parity verdict. Odd crossing count = the ray entered the solid and
    // never left (or vice versa) — the mesh is not watertight along this
    // row. Per plan 7.3 the row is left FLUID and counted for the warning;
    // overflow rows get the same conservative treatment.
    if (overflow || (nCross & 1)) {
        atomicAdd(badRowCounter, 1);
        return;
    }
    if (nCross == 0) return; // empty row, nothing to fill

    // In-thread insertion sort: nCross <= 64 and is typically 2-6, where
    // insertion sort beats anything with setup cost.
    for (int i = 1; i < nCross; ++i) {
        const float key = xs[i];
        int j = i - 1;
        while (j >= 0 && xs[j] > key) {
            xs[j + 1] = xs[j];
            --j;
        }
        xs[j + 1] = key;
    }

    // Fill alternate spans: cells whose center x+0.5 lies strictly between
    // crossing 2k (entry) and 2k+1 (exit) are inside the solid.
    //   first cell: smallest integer x with x + 0.5 > xEnter
    //   last  cell: largest  integer x with x + 0.5 < xExit
    // floor/ceil +-1 give the STRICT bounds even when the crossing sits
    // exactly on a cell center (surface-grazing cells stay fluid — the
    // half-way bounce-back wall lands at the face either way).
    int marked = 0;
    const long long rowBase =
        static_cast<long long>(nx) * (y + static_cast<long long>(ny) * z);
    for (int i = 0; i + 1 < nCross; i += 2) {
        int x0 = static_cast<int>(floorf(xs[i] - 0.5f)) + 1;
        int x1 = static_cast<int>(ceilf(xs[i + 1] - 0.5f)) - 1;
        // Geometry may stick out of the domain; clamp the span, never the ray.
        if (x0 < 0) x0 = 0;
        if (x1 > nx - 1) x1 = nx - 1;
        for (int x = x0; x <= x1; ++x) {
            std::uint8_t* cell = flags + rowBase + x;
            // SOLID is OR'd in semantically: only plain Fluid cells convert,
            // so the boundary stamps (Inlet/Outlet/Slip*) and any previously
            // stamped solids are preserved exactly as the header promises.
            if (*cell == static_cast<std::uint8_t>(CellFlag::Fluid)) {
                *cell = static_cast<std::uint8_t>(CellFlag::Solid);
                ++marked;
            }
        }
    }
    if (marked > 0) atomicAdd(solidCellCounter, marked);
}

} // namespace

// ===========================================================================
// Host-side entry points.
// ===========================================================================

bool uploadTriangles(const StlMesh& mesh, DeviceTriangleBuffer* out,
                     std::string* error) {
    if (!out) return false;
    *out = DeviceTriangleBuffer{};
    if (mesh.triangles.empty()) {
        if (error) *error = "mesh has no triangles";
        return false;
    }

    // Flatten host triangles into the 9-floats-per-triangle device layout
    // the kernel's tiled loads expect. The file normals are deliberately
    // dropped — parity voxelization never consults them (plan 7.3).
    std::vector<float> flat;
    flat.reserve(mesh.triangles.size() * 9);
    for (const StlTriangle& t : mesh.triangles) {
        for (const Vec3f* v : {&t.v0, &t.v1, &t.v2}) {
            flat.push_back(v->x);
            flat.push_back(v->y);
            flat.push_back(v->z);
        }
    }

    float* d = nullptr;
    if (auto err = cudaMalloc(&d, flat.size() * sizeof(float)); err != cudaSuccess) {
        if (error) *error = cudaGetErrorString(err);
        return false;
    }
    if (auto err = cudaMemcpy(d, flat.data(), flat.size() * sizeof(float),
                              cudaMemcpyHostToDevice);
        err != cudaSuccess) {
        cudaFree(d);
        if (error) *error = cudaGetErrorString(err);
        return false;
    }
    out->vertices = d;
    out->count = static_cast<std::uint32_t>(mesh.triangles.size());
    return true;
}

void freeDeviceTriangles(DeviceTriangleBuffer* buf) {
    if (!buf || !buf->vertices) return;
    cudaFree(const_cast<float*>(buf->vertices));
    *buf = DeviceTriangleBuffer{};
}

StlVoxelizeResult voxelizeStlOnDevice(DeviceTriangleBuffer tris,
                                      const GridDims& dims,
                                      std::uint8_t* d_flags,
                                      cudaStream_t stream) {
    StlVoxelizeResult result;

    // Input validation up front — a null launch would "succeed" silently.
    if (!tris.vertices || tris.count == 0) {
        result.error = "no triangles uploaded";
        return result;
    }
    if (dims.nx <= 0 || dims.ny <= 0 || dims.nz <= 0 || !d_flags) {
        result.error = "invalid grid dimensions or null device flag buffer";
        return result;
    }

    // Two device counters: [0] bad-parity rows (odd crossings or overflow),
    // [1] cells newly marked SOLID. Allocated per call — voxelization is an
    // import/modal-time event, not a hot path.
    int* d_counters = nullptr;
    if (auto err = cudaMalloc(&d_counters, 2 * sizeof(int)); err != cudaSuccess) {
        result.error = cudaGetErrorString(err);
        return result;
    }
    auto failWith = [&](cudaError_t err) {
        result.error = cudaGetErrorString(err);
        cudaFree(d_counters);
        return result;
    };
    if (auto err = cudaMemsetAsync(d_counters, 0, 2 * sizeof(int), stream);
        err != cudaSuccess) {
        return failWith(err);
    }

    // One thread per (y,z) row; the kernel handles tail threads internally
    // so the launch just rounds up.
    const long long rows = static_cast<long long>(dims.ny) * dims.nz;
    const int blocks = static_cast<int>((rows + kRowBlockSize - 1) / kRowBlockSize);
    rayParityFillKernel<<<blocks, kRowBlockSize, 0, stream>>>(
        tris.vertices, tris.count, dims.nx, dims.ny, dims.nz, d_flags,
        d_counters, d_counters + 1);
    if (auto err = cudaGetLastError(); err != cudaSuccess) {
        return failWith(err);
    }

    // Read the diagnostics back. The pageable-host copy synchronizes the
    // transfer; the explicit stream sync afterwards also surfaces any kernel
    // runtime fault as a checkable error instead of leaking it downstream.
    int counters[2] = {0, 0};
    if (auto err = cudaMemcpyAsync(counters, d_counters, sizeof(counters),
                                   cudaMemcpyDeviceToHost, stream);
        err != cudaSuccess) {
        return failWith(err);
    }
    if (auto err = cudaStreamSynchronize(stream); err != cudaSuccess) {
        return failWith(err);
    }
    cudaFree(d_counters);

    result.oddParityRows = counters[0];
    result.solidCells = counters[1];
    result.ok = true;
    return result;
}

StlVoxelizeResult voxelizeStl(const StlMesh& mesh, const GridDims& dims,
                              std::vector<std::uint8_t>& flags) {
    StlVoxelizeResult result;

    // The host field must already be the full boundary-stamped grid; a size
    // mismatch means the caller passed the wrong grid and the download below
    // would corrupt memory.
    const long long cellCount = dims.cellCount();
    if (cellCount <= 0 ||
        flags.size() != static_cast<std::size_t>(cellCount)) {
        result.error = "flag field size does not match grid dimensions";
        return result;
    }

    // Upload the triangle soup (one-shot path: tests and single imports; the
    // interactive modal keeps its own persistent DeviceTriangleBuffer).
    DeviceTriangleBuffer tris;
    std::string err;
    if (!uploadTriangles(mesh, &tris, &err)) {
        result.error = err;
        return result;
    }

    // Stage the host flags on device, voxelize, and pull the result back.
    std::uint8_t* d_flags = nullptr;
    auto cleanup = [&]() {
        if (d_flags) cudaFree(d_flags);
        freeDeviceTriangles(&tris);
    };
    if (auto cuerr = cudaMalloc(&d_flags, static_cast<std::size_t>(cellCount));
        cuerr != cudaSuccess) {
        result.error = cudaGetErrorString(cuerr);
        cleanup();
        return result;
    }
    if (auto cuerr = cudaMemcpy(d_flags, flags.data(),
                                static_cast<std::size_t>(cellCount),
                                cudaMemcpyHostToDevice);
        cuerr != cudaSuccess) {
        result.error = cudaGetErrorString(cuerr);
        cleanup();
        return result;
    }

    result = voxelizeStlOnDevice(tris, dims, d_flags, /*stream=*/nullptr);

    // Only a successful pass overwrites the caller's flags — on failure the
    // input field stays untouched (header contract: flags valid only when ok).
    if (result.ok) {
        if (auto cuerr = cudaMemcpy(flags.data(), d_flags,
                                    static_cast<std::size_t>(cellCount),
                                    cudaMemcpyDeviceToHost);
            cuerr != cudaSuccess) {
            result.ok = false;
            result.error = cudaGetErrorString(cuerr);
        }
    }
    cleanup();
    return result;
}

} // namespace foilcfd
