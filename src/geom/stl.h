// STL import (plan section 7, secondary feature): binary + ASCII loader with
// triangle-count guardrails, and the normalization transform (axis remap,
// uniform scale to a target chord, recentering) applied before voxelization.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "../core/vec.h"

namespace foilcfd {

/// v1 refuses meshes above this triangle count (plan 7.1) — the brute-force
/// GPU ray-parity voxelizer stays interactive only up to a few hundred k.
inline constexpr std::uint32_t kMaxStlTriangles = 2'000'000;

/// @brief One STL facet. The file normal is kept for rendering; voxelization
/// recomputes geometric normals from the winding when the file lies (many do).
struct StlTriangle {
    Vec3f v0, v1, v2; ///< Vertices in file coordinates (pre-normalization).
    Vec3f normal;     ///< Facet normal as stored in the file (unit-ish, untrusted).
};

/// @brief Axis-aligned bounding box of a mesh.
struct Aabb {
    Vec3f min{0, 0, 0};
    Vec3f max{0, 0, 0};

    /// @brief Extent along each axis (max - min).
    Vec3f size() const { return max - min; }
    /// @brief Geometric center.
    Vec3f center() const { return (min + max) * 0.5f; }
};

/// @brief Loaded STL mesh plus the metadata the normalization modal shows.
struct StlMesh {
    std::string name;                 ///< Solid name (ASCII) or filename stem.
    std::vector<StlTriangle> triangles;
    Aabb bounds;                      ///< Bounding box in file coordinates.
    std::uint64_t contentHash = 0;    ///< FNV-1a of the raw file bytes — the
                                      ///< "stl:<hash>" snapshot cache key (plan 7.4).
    bool wasBinary = false;           ///< Detected format (informational).
};

/// @brief Load result with rejection reporting (same contract as the airfoil
/// loader: the UI shows `rejectionReason` verbatim on failure).
struct StlLoadResult {
    bool ok = false;
    StlMesh mesh;
    std::string rejectionReason; ///< e.g. "binary header declares 3.1M triangles
                                 ///< (limit 2.0M)" or "ASCII parse error line 88".
};

/// @brief Load an STL file, auto-detecting binary vs ASCII (binary iff the
/// 84-byte-header triangle count matches the file size; the "solid" prefix
/// alone is unreliable — binary exporters write it too).
/// @param path File to load (from the GLFW drop callback or the file dialog).
/// @return Mesh with populated bounds/hash, or a rejection reason.
StlLoadResult loadStl(const std::filesystem::path& path);

// ===========================================================================
// Watertightness pre-check (plan 7.3). The GPU voxelizer detects bad parity
// per row at voxelization time; this host-side edge-pairing census runs once
// at import so the modal can warn BEFORE the user commits to a 23M-cell
// voxelization of a leaky mesh. A closed 2-manifold has every undirected
// edge shared by exactly two facets — anything else will leak rays.
// ===========================================================================

/// @brief Edge-pairing census of a triangle mesh. Vertices are matched by
/// exact float bit pattern (STL exporters duplicate shared vertices verbatim,
/// so this is reliable in practice; meshes "welded" only to within an epsilon
/// will conservatively report boundary edges — which is exactly the honest
/// answer, since the ray-parity voxelizer sees the same cracks).
struct WatertightReport {
    std::uint64_t uniqueEdges = 0;      ///< Distinct undirected edges seen.
    std::uint64_t boundaryEdges = 0;    ///< Edges used by exactly ONE facet (holes/cracks).
    std::uint64_t nonManifoldEdges = 0; ///< Edges used by 3+ facets (self-intersecting shells).

    /// @brief True when every edge pairs cleanly — the mesh is a closed
    /// 2-manifold and ray-parity voxelization is trustworthy.
    bool watertight() const {
        return uniqueEdges > 0 && boundaryEdges == 0 && nonManifoldEdges == 0;
    }
};

/// @brief Run the edge-pairing watertightness census. O(triangles) with two
/// hash maps; ~2M-triangle worst case completes well under a second, fine
/// for the import modal (run it once per load, not per slider tick).
/// @param mesh Loaded mesh (pre- or post-normalization — the census only
///             compares vertex identity, which both states preserve).
WatertightReport checkWatertight(const StlMesh& mesh);

// ===========================================================================
// Normalization (plan 7.2). STLs arrive in random units and orientation; the
// import modal builds one of these from user choices and bakes it into the
// triangle soup before voxelization. Chord-extent semantics: after transform,
// the mesh's x-extent equals `chordCells` lattice cells and the mesh is
// centered at the standard foil position.
// ===========================================================================

/// @brief Axis remap presets offered in the import modal.
enum class StlAxisPreset {
    XYZ,      ///< File axes already match FoilCFD (x = chord, y = up, z = span).
    ZUpToYUp, ///< CAD-style Z-up: rotate so file Z becomes lattice Y.
    YUpToZUp, ///< Inverse remap for span-along-Y exports.
};

/// @brief Apply an axis-remap preset to a single vector. The remaps are
/// proper rotations (signed axis permutations with determinant +1), so
/// triangle winding and normals stay right-handed:
///   ZUpToYUp: (x, y, z) -> (x,  z, -y)   (file Z up lands on lattice +y)
///   YUpToZUp: (x, y, z) -> (x, -z,  y)   (file Y span lands on lattice +z)
Vec3f remapAxes(const Vec3f& v, StlAxisPreset preset);

/// @brief Axis-remapped copy of a bounding box (exact — the remaps are axis
/// permutations, so transforming the two extreme corners and re-sorting per
/// component reproduces the true AABB). The import modal shows this so the
/// user sees the post-remap extents before committing.
Aabb remappedBounds(const Aabb& bounds, StlAxisPreset preset);

/// @brief The full import transform, in application order:
/// axis remap -> uniform scale -> translation.
struct StlNormalization {
    StlAxisPreset axisPreset = StlAxisPreset::XYZ;
    float uniformScale = 1.0f; ///< Computed so the remapped x-extent maps to
                               ///< the chosen chord length in cells.
    Vec3f translation{0, 0, 0};///< Recenter at the standard foil position (cells).
};

/// @brief Compute the normalization that maps @p mesh so its (post-remap)
/// x-extent spans @p chordCells cells, centered at (anchorX, anchorY, nz/2).
/// This is the modal's "auto" suggestion; the user can tweak fields after.
/// @param mesh       Loaded mesh (bounds must be populated).
/// @param axisPreset Chosen axis remap.
/// @param chordCells Target chord length in lattice cells.
/// @param anchorX    Lattice x of the foil anchor (DomainLayout::anchorX()).
/// @param anchorY    Lattice y of the foil anchor.
/// @param nz         Spanwise cell count (mesh is centered at nz/2).
StlNormalization computeAutoNormalization(const StlMesh& mesh,
                                          StlAxisPreset axisPreset,
                                          int chordCells, float anchorX,
                                          float anchorY, int nz);

/// @brief Apply a normalization to a mesh in place: vertices end up in
/// lattice-cell coordinates ready for the GPU ray-parity voxelizer
/// (voxelizer_stl.cuh). Normals are remapped/renormalized, bounds recomputed.
void applyNormalization(StlMesh& mesh, const StlNormalization& norm);

} // namespace foilcfd
