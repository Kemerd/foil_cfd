// GPU ray-parity voxelizer for imported STL solids (plan section 7.3): one
// thread per (y,z) row casts a +x ray, sorts triangle crossings, fills SOLID
// between odd/even pairs, and counts odd-parity rows so non-watertight input
// is detected and reported instead of producing silent garbage.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../sim/lbm_core.cuh"
#include "stl.h"

namespace foilcfd {

/// @brief Flat GPU-friendly triangle buffer: 9 floats per triangle
/// (v0.xyz, v1.xyz, v2.xyz), vertices already in lattice-cell coordinates
/// (i.e. after applyNormalization). File normals are not needed for parity.
struct DeviceTriangleBuffer {
    const float* vertices = nullptr; ///< Device array, 9 * count floats.
    std::uint32_t count = 0;         ///< Triangle count.
};

/// @brief Outcome of a GPU voxelization pass.
struct StlVoxelizeResult {
    bool ok = false;            ///< False on CUDA failure or hopeless geometry.
    int  oddParityRows = 0;     ///< Rows whose +x ray crossed an odd number of
                                ///< triangles — nonzero means the mesh is not
                                ///< watertight along those rows (plan 7.3).
    int  solidCells = 0;        ///< Total cells marked SOLID (sanity readout).
    std::string error;          ///< CUDA / allocation failure description.

    /// @brief Convenience: fraction of rows with bad parity. The UI warns the
    /// user above ~0.1% instead of crashing (plan 7.3 requirement).
    float badRowFraction(const GridDims& dims) const {
        const float rows = static_cast<float>(dims.ny) * static_cast<float>(dims.nz);
        return rows > 0.0f ? static_cast<float>(oddParityRows) / rows : 0.0f;
    }
};

/// @brief Upload a normalized host mesh into a flat device triangle buffer.
/// Caller owns the returned device memory; free with freeDeviceTriangles().
/// (Kept as an explicit step so repeated re-voxelizations — e.g. while the
/// user scrubs the normalization modal — skip the host->device copy.)
/// @param mesh   Normalized mesh (lattice-cell coordinates).
/// @param out    Receives the device pointer + count.
/// @param error  On failure (OOM), a description.
/// @return True on success.
bool uploadTriangles(const StlMesh& mesh, DeviceTriangleBuffer* out,
                     std::string* error);

/// @brief Free a buffer produced by uploadTriangles (safe on empty buffers).
void freeDeviceTriangles(DeviceTriangleBuffer* buf);

/// @brief Run the GPU ray-parity voxelization (plan 7.3).
///
/// One thread per (y,z) row: walk ALL triangles (brute force — acceptable to
/// ~200k triangles; uniform-grid binning is a stretch optimization), collect
/// +x ray crossings at the row's cell-center (y+0.5, z+0.5) line, sort the
/// small crossing list in-thread, then mark SOLID between alternate pairs.
/// Rows with an odd crossing count are left untouched and counted into
/// StlVoxelizeResult::oddParityRows.
/// @param tris        Device triangle buffer (lattice coordinates).
/// @param dims        Grid dimensions.
/// @param d_flags     DEVICE flag field (cellCount bytes); SOLID is OR'd in,
///                    existing boundary flags are preserved.
/// @param stream      CUDA stream to run on.
/// @return Result with parity diagnostics; flags are only valid when ok.
StlVoxelizeResult voxelizeStlOnDevice(DeviceTriangleBuffer tris,
                                      const GridDims& dims,
                                      std::uint8_t* d_flags,
                                      cudaStream_t stream);

/// @brief Host-convenience wrapper: upload triangles, allocate a device flag
/// field seeded from @p flags, voxelize, download the result back into
/// @p flags, and free the temporaries. The interactive path keeps its own
/// persistent device buffers; this exists for tests and one-shot imports.
/// @param mesh  Normalized mesh.
/// @param dims  Grid dimensions.
/// @param flags Host flag field (boundary flags already stamped); SOLID OR'd in.
StlVoxelizeResult voxelizeStl(const StlMesh& mesh, const GridDims& dims,
                              std::vector<std::uint8_t>& flags);

} // namespace foilcfd
