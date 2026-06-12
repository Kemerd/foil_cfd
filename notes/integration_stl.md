# Integration notes — STL import + GPU ray-parity voxelizer

Module agent for: `src/geom/stl.cpp`, `src/geom/stl.h`,
`src/geom/voxelizer_stl.cu`, `src/geom/voxelizer_stl.cuh`. All four are
fully implemented and verified (see "Verification" below).

## Interface additions in `src/geom/stl.h` (files I own — informational)

Downstream consumers (UI modal agent, tests agent) get three new public
entry points beyond the scaffold header:

```cpp
// Edge-pairing watertightness pre-check for the import modal (plan 7.3):
// run once per load, warn BEFORE committing to a full-grid voxelization.
struct WatertightReport {
    std::uint64_t uniqueEdges;      // distinct undirected edges
    std::uint64_t boundaryEdges;    // used by exactly 1 facet (holes/cracks)
    std::uint64_t nonManifoldEdges; // used by 3+ facets
    bool watertight() const;        // uniqueEdges > 0 && both counts == 0
};
WatertightReport checkWatertight(const StlMesh& mesh);

// Axis-remap helpers so the modal can preview post-remap extents without
// touching the triangle soup:
Vec3f remapAxes(const Vec3f& v, StlAxisPreset preset);
Aabb  remappedBounds(const Aabb& bounds, StlAxisPreset preset);
```

Vertex matching in `checkWatertight` is exact-bit (STL exporters duplicate
shared vertices verbatim); epsilon-welded meshes conservatively report
boundary edges — same cracks the ray-parity pass would see.

## Semantics decisions (already coded, relevant to UI / solver agents)

- **Axis remaps are proper rotations** (det +1), not swaps:
  `ZUpToYUp: (x,y,z) -> (x, z, -y)`, `YUpToZUp: (x,y,z) -> (x, -z, y)`.
  Winding/normals stay right-handed through normalization.
- **`StlVoxelizeResult.oddParityRows` also includes overflow rows** (rows
  with > 64 ray crossings — pathological geometry). Both kinds are left
  FLUID and warrant the same non-watertight UI warning. `ok` stays true;
  `ok == false` only on CUDA failures / invalid inputs.
- **SOLID is written only over `CellFlag::Fluid`** — Inlet/Outlet/Slip
  stamps from `buildBoundaryFlags()` and previously stamped solids survive,
  per the `voxelizeStlOnDevice` header contract.
- **`voxelizeStl` (host convenience) leaves the caller's flag vector
  untouched on failure**; it is only overwritten after a successful pass.
- Ray sample points are cell centers jittered by 1/4096 (y) and 1/8192 (z)
  cells — *different per axis*, because quad-face 45° diagonals otherwise
  pass exactly through equal-jitter samples and drop crossings.
- Loader drops zero-area facets with bitwise-identical vertices silently;
  rejects non-finite vertices, >2M triangles (binary: before allocation,
  from the header count; ASCII: mid-parse), and reports ASCII errors with
  line numbers. ASCII tolerances: case-insensitive keywords, CRLF, leading
  `+` on floats, missing `endsolid`, multiple concatenated `solid` sections
  (merged into one mesh).
- `StlMesh::contentHash` is FNV-1a-64 over the raw file bytes, computed
  before any parsing/normalization — feed it to `SnapshotKey::geometryId`
  as `"stl:<16-hex>"` (snapshot.h contract).

## No changes requested to interfaces owned by others

Coded strictly against the existing `GridDims`/`CellFlag` (lbm_core.cuh)
and `DomainLayout` anchor convention (voxelizer.h). No edits needed in
CMakeLists (both sources were already listed).

## Verification

- Both TUs compile standalone: MSVC 19.44 `/std:c++20 /W4` (stl.cpp) and
  nvcc 12.9 `-std=c++20 -arch=sm_120` (voxelizer_stl.cu), zero warnings.
- The ray-triangle intersector is proven at COMPILE TIME: four
  `static_assert`s in voxelizer_stl.cu check a hand-computed cube wall
  (hit at x=2, the face's second triangle correctly missing, an outside
  miss) and a slanted x=y plane interpolation — derivations documented in
  the comment block above them.
- Runtime GPU harness (temp dir, not committed): binary cube with "solid"
  header prefix -> detected binary, 12 tris, watertight (18 edges);
  hole-punched cube -> 3 boundary edges + 28 odd-parity GPU rows, no crash;
  auto-normalization scale/translation exact; voxelization fills exactly
  8x8x8 = 512 cells with 0 odd rows; downloaded flags match the device
  counter; ASCII tetra roundtrip + line-numbered rejection both pass.

Status: COMPLETE
