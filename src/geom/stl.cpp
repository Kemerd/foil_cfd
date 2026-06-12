// STL import implementation (plan section 7): binary + ASCII parsing with
// defensive rejection reporting, FNV-1a content hashing for warm-cache keys,
// the import normalization transform (axis remap -> uniform scale ->
// translation), and the edge-pairing watertightness pre-check the import
// modal shows before voxelization.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include "stl.h"

// UTF-8 path conversion: path::string() decodes via the ANSI code page on
// MSVC and can throw for the non-ASCII paths drag-and-drop now delivers.
#include "../platform/platform.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string_view>
#include <unordered_map>

namespace foilcfd {
namespace {

// ===========================================================================
// File reading + content hashing.
// ===========================================================================

// FNV-1a 64-bit constants (Fowler–Noll–Vo). Matches the hashing family the
// snapshot module uses for flag fields; chosen because it is dependency-free
// and streams over bytes in one pass.
constexpr std::uint64_t kFnvOffsetBasis = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

/// FNV-1a over raw bytes. Cheap, deterministic, and good enough as a cache
/// identity: the snapshot key also carries grid dims, AoA bucket and u_lat,
/// so an astronomically unlikely hash collision risks at worst a stale warm
/// start (which converges anyway), never wrong geometry in the flags.
std::uint64_t fnv1a64(const char* data, std::size_t n) {
    std::uint64_t h = kFnvOffsetBasis;
    for (std::size_t i = 0; i < n; ++i) {
        h ^= static_cast<unsigned char>(data[i]);
        h *= kFnvPrime;
    }
    return h;
}

/// Hard ceiling on the raw STL file size we are willing to slurp. Any valid
/// mesh under the 2M-triangle cap fits easily: binary is 84 + 50 B/facet
/// (~100 MB), ASCII runs ~250-350 B/facet (~700 MB worst case). Anything
/// larger is either over the triangle cap anyway or not an STL at all (e.g.
/// a multi-GB file renamed .stl) — refuse with a readable reason instead of
/// letting a multi-GB host allocation throw std::bad_alloc through the
/// frame loop (plan 7.1 refusal requirement).
constexpr std::uint64_t kMaxStlFileBytes = 1ull << 30; // 1 GiB

/// Slurp the whole file into memory. STL files are read in full anyway for
/// hashing, and the size cap above bounds the allocation — fine to hold
/// while parsing, and it lets the ASCII parser run over a flat buffer with
/// proper line tracking instead of fragile line-by-line stream reads.
bool readWholeFile(const std::filesystem::path& path, std::vector<char>& out,
                   std::string& err) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        err = "cannot open file";
        return false;
    }
    const std::streamsize size = in.tellg();
    if (size < 0) {
        err = "cannot determine file size";
        return false;
    }
    // Size cap BEFORE any allocation: every valid <= 2M-triangle STL (binary
    // or ASCII) is far under this; only pathological/garbage files exceed it.
    if (static_cast<std::uint64_t>(size) > kMaxStlFileBytes) {
        char buf[96];
        std::snprintf(buf, sizeof(buf),
                      "file is %.1f GB — no STL under the 2M-triangle limit "
                      "is this large",
                      static_cast<double>(size) / (1024.0 * 1024.0 * 1024.0));
        err = buf;
        return false;
    }
    in.seekg(0, std::ios::beg);
    out.resize(static_cast<std::size_t>(size));
    if (size > 0 && !in.read(out.data(), size)) {
        err = "read failed (file truncated while reading?)";
        return false;
    }
    return true;
}

/// Human-friendly triangle counts for rejection messages ("3.1M", "742").
std::string humanCount(std::uint64_t n) {
    if (n >= 1'000'000ull) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.1fM", static_cast<double>(n) / 1e6);
        return buf;
    }
    return std::to_string(n);
}

// ===========================================================================
// Shared triangle hygiene. Both parsers reject non-finite vertices (NaN/Inf
// would poison the bounds, the normalization scale, and ultimately the flag
// field) and silently drop facets with bitwise-identical vertices (zero-area
// slivers some exporters emit; they contribute nothing to ray parity).
// ===========================================================================

bool isFiniteVec(const Vec3f& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

bool verticesFinite(const StlTriangle& t) {
    return isFiniteVec(t.v0) && isFiniteVec(t.v1) && isFiniteVec(t.v2);
}

bool hasIdenticalVertices(const StlTriangle& t) {
    // Bitwise comparison on purpose: exporters that emit degenerate facets
    // duplicate coordinates verbatim, and bitwise-equal is the only notion of
    // "same vertex" that cannot misfire on legitimately tiny triangles.
    auto same = [](const Vec3f& a, const Vec3f& b) {
        return std::memcmp(&a, &b, sizeof(Vec3f)) == 0;
    };
    return same(t.v0, t.v1) || same(t.v1, t.v2) || same(t.v0, t.v2);
}

/// Recompute the AABB from scratch over all vertices. Called after load and
/// after applyNormalization (a remap can swap which corner is min/max, so
/// transforming the old bounds is not a substitute here — we want truth).
Aabb computeBounds(const std::vector<StlTriangle>& tris) {
    Aabb b;
    if (tris.empty()) return b;
    b.min = b.max = tris.front().v0;
    auto grow = [&b](const Vec3f& v) {
        b.min.x = std::min(b.min.x, v.x);
        b.min.y = std::min(b.min.y, v.y);
        b.min.z = std::min(b.min.z, v.z);
        b.max.x = std::max(b.max.x, v.x);
        b.max.y = std::max(b.max.y, v.y);
        b.max.z = std::max(b.max.z, v.z);
    };
    for (const StlTriangle& t : tris) {
        grow(t.v0);
        grow(t.v1);
        grow(t.v2);
    }
    return b;
}

// ===========================================================================
// Binary parser. Layout (little-endian, packed, no alignment):
//   80-byte header | uint32 triangle count | count x 50-byte records
//   record = float normal[3] | float v0[3] | float v1[3] | float v2[3]
//            | uint16 attribute byte count (ignored)
// The caller has already verified size == 84 + 50*count and count <= cap.
// ===========================================================================

bool parseBinary(const std::vector<char>& bytes, std::uint32_t count,
                 StlMesh& mesh, std::string& reason) {
    mesh.triangles.reserve(count);
    const char* rec = bytes.data() + 84;
    for (std::uint32_t i = 0; i < count; ++i, rec += 50) {
        // memcpy because the 50-byte records leave every float after the
        // first record misaligned — a reinterpret_cast load would be UB and
        // can fault on alignment-checking configurations.
        float v[12];
        std::memcpy(v, rec, sizeof(v));
        StlTriangle t;
        t.normal = {v[0], v[1], v[2]};
        t.v0 = {v[3], v[4], v[5]};
        t.v1 = {v[6], v[7], v[8]};
        t.v2 = {v[9], v[10], v[11]};
        if (!verticesFinite(t)) {
            reason = "facet " + std::to_string(i) + " has a non-finite vertex";
            return false;
        }
        if (hasIdenticalVertices(t)) continue; // zero-area sliver, drop
        mesh.triangles.push_back(t);
    }
    if (mesh.triangles.empty()) {
        reason = "all " + humanCount(count) + " facets are degenerate";
        return false;
    }
    return true;
}

// ===========================================================================
// ASCII parser. Token-stream grammar with line tracking so rejections read
// like compiler errors ("ASCII parse error line 88: expected 'vertex'").
// Tolerances chosen from the same "the zoo is messy" stance as the .dat
// loader: case-insensitive keywords, CRLF, arbitrary whitespace, a missing
// final 'endsolid', and multiple concatenated 'solid' sections in one file.
// ===========================================================================

struct AsciiCursor {
    const char* p;
    const char* end;
    int line = 1;

    /// Skip whitespace, counting newlines so error messages carry a line.
    void skipSpace() {
        while (p < end && std::isspace(static_cast<unsigned char>(*p))) {
            if (*p == '\n') ++line;
            ++p;
        }
    }

    /// Next whitespace-delimited token (empty view at EOF).
    std::string_view next() {
        skipSpace();
        const char* start = p;
        while (p < end && !std::isspace(static_cast<unsigned char>(*p))) ++p;
        return {start, static_cast<std::size_t>(p - start)};
    }

    /// Remainder of the current line, trimmed — solid/endsolid names may
    /// contain spaces, so they cannot be read as a single token.
    std::string restOfLine() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\r')) ++p;
        const char* start = p;
        while (p < end && *p != '\n') ++p;
        const char* stop = p;
        while (stop > start &&
               (stop[-1] == ' ' || stop[-1] == '\t' || stop[-1] == '\r')) {
            --stop;
        }
        return std::string(start, stop);
    }
};

/// Case-insensitive keyword comparison (the spec says lowercase; the wild
/// files say otherwise).
bool ieq(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i])) !=
            std::tolower(static_cast<unsigned char>(b[i]))) {
            return false;
        }
    }
    return true;
}

/// Parse one float token. std::from_chars handles scientific notation but
/// not a leading '+' on the mantissa (some exporters write "+1.0e+00"), so
/// that sign is skipped manually. Non-finite values ("nan", "inf") and
/// partially-consumed tokens are treated as parse failures.
bool parseFloatToken(AsciiCursor& c, float* out) {
    const std::string_view tok = c.next();
    if (tok.empty()) return false;
    const char* first = tok.data();
    const char* last = tok.data() + tok.size();
    if (first < last && *first == '+') ++first;
    const auto res = std::from_chars(first, last, *out);
    return res.ec == std::errc{} && res.ptr == last && std::isfinite(*out);
}

bool parseAscii(const std::vector<char>& bytes, StlMesh& mesh,
                std::string& reason) {
    AsciiCursor c{bytes.data(), bytes.data() + bytes.size()};
    auto fail = [&](const std::string& what) {
        reason = "ASCII parse error line " + std::to_string(c.line) + ": " + what;
        return false;
    };

    // Header: "solid <optional name>". The name (when present) is what the
    // import modal titles itself with.
    if (!ieq(c.next(), "solid")) return fail("expected 'solid'");
    mesh.name = c.restOfLine();

    std::uint64_t degenerate = 0;
    for (;;) {
        const std::string_view tok = c.next();
        // EOF without 'endsolid' — tolerated; plenty of generators truncate.
        if (tok.empty()) break;

        if (ieq(tok, "endsolid")) {
            c.restOfLine(); // optional repeated name
            // Some exporters glue several 'solid' sections into one file
            // (multi-body parts). Treat them as one mesh — the voxelizer's
            // parity test is indifferent to how shells are grouped.
            const std::string_view nxt = c.next();
            if (nxt.empty()) break;
            if (!ieq(nxt, "solid")) {
                return fail("unexpected content after 'endsolid'");
            }
            c.restOfLine();
            continue;
        }

        if (!ieq(tok, "facet")) {
            return fail("unexpected token '" + std::string(tok) + "'");
        }
        // Cap enforcement mid-parse: bail out before allocating gigabytes
        // for a pathological file (plan 7.1 refusal requirement).
        if (mesh.triangles.size() >= kMaxStlTriangles) {
            reason = "more than " + humanCount(kMaxStlTriangles) +
                     " triangles (limit " + humanCount(kMaxStlTriangles) + ")";
            return false;
        }

        StlTriangle t;
        if (!ieq(c.next(), "normal")) return fail("expected 'normal'");
        if (!parseFloatToken(c, &t.normal.x) ||
            !parseFloatToken(c, &t.normal.y) ||
            !parseFloatToken(c, &t.normal.z)) {
            return fail("bad facet normal value");
        }
        if (!ieq(c.next(), "outer")) return fail("expected 'outer'");
        if (!ieq(c.next(), "loop")) return fail("expected 'loop'");

        // Exactly three vertices: STL has no n-gon form, so a fourth
        // 'vertex' (or a premature 'endloop') is a structural error worth
        // reporting precisely rather than guessing around.
        Vec3f* verts[3] = {&t.v0, &t.v1, &t.v2};
        for (Vec3f* vp : verts) {
            if (!ieq(c.next(), "vertex")) {
                return fail("expected 'vertex' (facets need exactly 3)");
            }
            if (!parseFloatToken(c, &vp->x) || !parseFloatToken(c, &vp->y) ||
                !parseFloatToken(c, &vp->z)) {
                return fail("bad vertex coordinate");
            }
        }
        if (!ieq(c.next(), "endloop")) {
            return fail("expected 'endloop' (facets need exactly 3 vertices)");
        }
        if (!ieq(c.next(), "endfacet")) return fail("expected 'endfacet'");

        if (hasIdenticalVertices(t)) {
            ++degenerate; // zero-area sliver, drop silently like the binary path
            continue;
        }
        mesh.triangles.push_back(t);
    }

    if (mesh.triangles.empty()) {
        reason = degenerate > 0 ? "all facets are degenerate"
                                : "no facets found";
        return false;
    }
    return true;
}

} // namespace

// ===========================================================================
// Public loader.
// ===========================================================================

StlLoadResult loadStl(const std::filesystem::path& path) {
    StlLoadResult result;

    std::vector<char> bytes;
    std::string err;
    if (!readWholeFile(path, bytes, err)) {
        result.rejectionReason = err;
        return result;
    }
    // Smaller than the shortest conceivable ASCII STL ("solid\nendsolid") and
    // far below the 84-byte binary minimum — reject before parsing.
    if (bytes.size() < 15) {
        result.rejectionReason =
            "file too small to be an STL (" + std::to_string(bytes.size()) +
            " bytes)";
        return result;
    }

    StlMesh mesh;
    // Hash the raw bytes FIRST, before any parsing/normalization mutates
    // anything — this is the "stl:<hash>" warm-cache identity (plan 7.4) and
    // must depend only on file content.
    mesh.contentHash = fnv1a64(bytes.data(), bytes.size());
    // UTF-8 stem: drag-and-drop now delivers correctly-decoded non-ASCII
    // paths, for which stem().string() (ANSI code page on MSVC) would throw.
    const std::string stem = platform::pathToUtf8(path.stem());
    mesh.name = stem;

    // Binary detection (plan 7.1): trust the 84-byte header + exact size
    // arithmetic, never the "solid" prefix — binary exporters write that
    // prefix into the 80-byte comment header all the time.
    std::uint32_t declared = 0;
    bool binarySizeMatch = false;
    if (bytes.size() >= 84) {
        std::memcpy(&declared, bytes.data() + 80, sizeof(declared));
        binarySizeMatch =
            bytes.size() == 84ull + 50ull * static_cast<std::uint64_t>(declared);
    }

    if (binarySizeMatch) {
        if (declared == 0) {
            result.rejectionReason = "binary STL declares zero triangles";
            return result;
        }
        // Refusal above the v1 cap (plan 7.1) BEFORE allocating anything:
        // the brute-force GPU voxelizer stops being interactive long before
        // 2M, and the parse alone would eat ~100 MB.
        if (declared > kMaxStlTriangles) {
            result.rejectionReason =
                "binary header declares " + humanCount(declared) +
                " triangles (limit " + humanCount(kMaxStlTriangles) + ")";
            return result;
        }
        if (!parseBinary(bytes, declared, mesh, err)) {
            result.rejectionReason = err;
            return result;
        }
        mesh.wasBinary = true;
    } else {
        // ASCII fallback. If this also fails, compose a message that covers
        // the truncated-binary case too — "why was my file rejected" must be
        // answerable from the message alone (same stance as the .dat loader).
        if (!parseAscii(bytes, mesh, err)) {
            if (bytes.size() >= 84 && declared > 0) {
                const std::uint64_t expected =
                    84ull + 50ull * static_cast<std::uint64_t>(declared);
                err += " (also not binary: header declares " +
                       humanCount(declared) + " facets -> expected " +
                       std::to_string(expected) + " bytes, file has " +
                       std::to_string(bytes.size()) + ")";
            }
            result.rejectionReason = err;
            return result;
        }
        mesh.wasBinary = false;
    }

    // ASCII solids may carry an empty name; fall back to the filename stem
    // so the UI dropdown / modal title never shows a blank.
    if (mesh.name.empty()) mesh.name = stem;
    mesh.bounds = computeBounds(mesh.triangles);

    result.ok = true;
    result.mesh = std::move(mesh);
    return result;
}

// ===========================================================================
// Normalization (plan 7.2): axis remap -> uniform scale -> translation.
// Pure math — the modal UI in app/ui drives these and renders the numbers.
// ===========================================================================

Vec3f remapAxes(const Vec3f& v, StlAxisPreset preset) {
    switch (preset) {
        // CAD-style Z-up to lattice Y-up is a -90 deg rotation about +x:
        // file Z (up) lands on lattice +y, file Y lands on -z. The spanwise
        // sign is irrelevant to the solver (the z BC is symmetric), but using
        // a proper rotation (det +1) instead of a swap keeps the mesh
        // right-handed so winding and normals stay consistent.
        case StlAxisPreset::ZUpToYUp: return {v.x, v.z, -v.y};
        // Inverse rotation (+90 deg about +x) for span-along-Y exports.
        case StlAxisPreset::YUpToZUp: return {v.x, -v.z, v.y};
        case StlAxisPreset::XYZ:
        default: return v;
    }
}

Aabb remappedBounds(const Aabb& bounds, StlAxisPreset preset) {
    // The remaps are signed axis permutations, so the two extreme corners
    // map to two (different) extreme corners; per-component min/max of the
    // transformed pair reconstructs the exact AABB. No need to touch the
    // triangle soup for the modal's preview numbers.
    const Vec3f a = remapAxes(bounds.min, preset);
    const Vec3f b = remapAxes(bounds.max, preset);
    Aabb out;
    out.min = {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
    out.max = {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
    return out;
}

StlNormalization computeAutoNormalization(const StlMesh& mesh,
                                          StlAxisPreset axisPreset,
                                          int chordCells, float anchorX,
                                          float anchorY, int nz) {
    StlNormalization norm;
    norm.axisPreset = axisPreset;

    // Scale is decided AFTER the remap: "chord" means the streamwise (x)
    // extent in lattice space, and the remap is what decides which file axis
    // ends up streamwise.
    const Aabb rb = remappedBounds(mesh.bounds, axisPreset);
    const float xExtent = rb.size().x;
    norm.uniformScale = (xExtent > 1e-9f)
                            ? static_cast<float>(chordCells) / xExtent
                            : 1.0f; // degenerate flat mesh: leave scale alone

    // Center the scaled mesh at the standard foil position: the layout
    // anchor in (x, y) and mid-span in z, matching where the airfoil fast
    // path puts its geometry so camera/cache layout assumptions carry over.
    const Vec3f scaledCenter = rb.center() * norm.uniformScale;
    norm.translation =
        Vec3f(anchorX, anchorY, static_cast<float>(nz) * 0.5f) - scaledCenter;
    return norm;
}

void applyNormalization(StlMesh& mesh, const StlNormalization& norm) {
    for (StlTriangle& t : mesh.triangles) {
        // Application order is the struct contract: remap, then scale, then
        // translate. Vertices end up in lattice-cell coordinates ready for
        // the GPU ray-parity voxelizer.
        t.v0 = remapAxes(t.v0, norm.axisPreset) * norm.uniformScale + norm.translation;
        t.v1 = remapAxes(t.v1, norm.axisPreset) * norm.uniformScale + norm.translation;
        t.v2 = remapAxes(t.v2, norm.axisPreset) * norm.uniformScale + norm.translation;
        // Normals: rotation only (uniform scale and translation do not change
        // direction). Renormalize to wash out file sloppiness — many STLs
        // store unnormalized or outright wrong normals, which is why parity
        // voxelization never consults them; rendering still wants them sane.
        t.normal = normalized(remapAxes(t.normal, norm.axisPreset));
    }
    mesh.bounds = computeBounds(mesh.triangles);
}

// ===========================================================================
// Watertightness pre-check: edge-pairing census (plan 7.3 import modal).
// ===========================================================================

namespace {

/// Vertex identity key: the three coordinate bit patterns. -0.0f is folded
/// onto +0.0f so the two encodings of zero key identically (they compare
/// equal as floats and exporters mix them freely).
struct VertexKey {
    std::uint32_t x = 0, y = 0, z = 0;
    bool operator==(const VertexKey&) const = default;
};

struct VertexKeyHash {
    std::size_t operator()(const VertexKey& k) const noexcept {
        // FNV-1a over the 12 key bytes — same family as the content hash.
        std::uint64_t h = kFnvOffsetBasis;
        for (std::uint32_t word : {k.x, k.y, k.z}) {
            for (int i = 0; i < 4; ++i) {
                h ^= (word >> (8 * i)) & 0xffu;
                h *= kFnvPrime;
            }
        }
        return static_cast<std::size_t>(h);
    }
};

std::uint32_t coordBits(float f) {
    if (f == 0.0f) f = 0.0f; // collapse -0.0f onto +0.0f
    return std::bit_cast<std::uint32_t>(f);
}

} // namespace

WatertightReport checkWatertight(const StlMesh& mesh) {
    WatertightReport report;
    if (mesh.triangles.empty()) return report;

    // Pass 1 (fused with pass 2 below): assign a dense integer id to every
    // distinct vertex position. For a closed mesh Euler's formula gives
    // V ~ T/2, so reserving T entries avoids rehashing without overshooting.
    std::unordered_map<VertexKey, std::uint32_t, VertexKeyHash> vertexIds;
    vertexIds.reserve(mesh.triangles.size());
    auto idOf = [&vertexIds](const Vec3f& v) {
        const VertexKey key{coordBits(v.x), coordBits(v.y), coordBits(v.z)};
        // try_emplace: existing vertices return their id, new ones get the
        // next dense index. Single hash lookup either way.
        return vertexIds.try_emplace(key, static_cast<std::uint32_t>(vertexIds.size()))
            .first->second;
    };

    // Count facet uses of every undirected edge. Ids are < 3*2M < 2^32, so a
    // packed (lo << 32 | hi) uint64 key is collision-free by construction.
    std::unordered_map<std::uint64_t, std::uint32_t> edgeUse;
    edgeUse.reserve(mesh.triangles.size() * 3 / 2 + 16); // E = 3T/2 closed

    for (const StlTriangle& t : mesh.triangles) {
        const std::uint32_t i0 = idOf(t.v0);
        const std::uint32_t i1 = idOf(t.v1);
        const std::uint32_t i2 = idOf(t.v2);
        const std::uint32_t edges[3][2] = {{i0, i1}, {i1, i2}, {i2, i0}};
        for (const auto& e : edges) {
            const std::uint64_t lo = std::min(e[0], e[1]);
            const std::uint64_t hi = std::max(e[0], e[1]);
            ++edgeUse[(lo << 32) | hi];
        }
    }

    // A closed 2-manifold uses every edge exactly twice. Once = an open
    // boundary (hole/crack -> rays leak); three+ = non-manifold junctions
    // (typically self-intersecting shells -> parity is ill-defined there).
    report.uniqueEdges = edgeUse.size();
    for (const auto& [key, uses] : edgeUse) {
        (void)key;
        if (uses == 1) ++report.boundaryEdges;
        else if (uses > 2) ++report.nonManifoldEdges;
    }
    return report;
}

} // namespace foilcfd
