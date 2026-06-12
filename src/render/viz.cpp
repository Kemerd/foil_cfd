// Renderer implementation: GL programs/buffers/textures, the CUDA<->GL
// interop registrations (each resource registered exactly ONCE for its
// lifetime — plan section 13), particle hero mode, slice planes, the extruded
// foil + VG mesh, the Q-criterion raycast, and the dependency-free PNG
// screenshot encoder (stored-block zlib — valid PNG, no compression library).
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include "viz.h"

#include <glad/gl.h>
// cuda_gl_interop.h must see the GL typedefs first; glad defines the __gl_h_
// guard so the system <GL/gl.h> never collides with the loader.
#include <cuda_gl_interop.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../geom/stl.h"
#include "../platform/platform.h"
#include "particles.cuh"

namespace foilcfd {

namespace {

constexpr float kPiF = 3.14159265358979f;

// ---------------------------------------------------------------------------
// Minimal PNG encoding: CRC-32 (PNG chunk checksums) + Adler-32 (zlib) +
// stored-mode deflate. Output is a perfectly valid, merely uncompressed PNG.
// ---------------------------------------------------------------------------

std::uint32_t crc32Update(std::uint32_t crc, const std::uint8_t* p, std::size_t n) {
    // Lazily-built standard CRC-32 (reflected, poly 0xEDB88320) table.
    static std::uint32_t table[256];
    static bool built = false;
    if (!built) {
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        built = true;
    }
    crc = ~crc;
    for (std::size_t i = 0; i < n; ++i) crc = table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

void putU32BE(std::vector<std::uint8_t>& out, std::uint32_t v) {
    out.push_back(static_cast<std::uint8_t>(v >> 24));
    out.push_back(static_cast<std::uint8_t>(v >> 16));
    out.push_back(static_cast<std::uint8_t>(v >> 8));
    out.push_back(static_cast<std::uint8_t>(v));
}

void writeChunk(std::vector<std::uint8_t>& out, const char type[4],
                const std::uint8_t* data, std::size_t n) {
    putU32BE(out, static_cast<std::uint32_t>(n));
    const std::size_t typeAt = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), data, data + n);
    // CRC covers the chunk type and data, not the length field.
    const std::uint32_t crc = crc32Update(0, out.data() + typeAt, n + 4);
    putU32BE(out, crc);
}

/// Encode tightly-packed RGB8 rows (top-down) into a PNG byte stream.
std::vector<std::uint8_t> encodePNG(const std::uint8_t* rgb, int w, int h) {
    // Raw scanline stream: every row gets a leading filter-type-0 byte.
    std::vector<std::uint8_t> raw;
    raw.reserve(static_cast<std::size_t>(h) * (1 + 3 * static_cast<std::size_t>(w)));
    for (int y = 0; y < h; ++y) {
        raw.push_back(0); // filter: None
        const std::uint8_t* row = rgb + static_cast<std::size_t>(y) * 3 * w;
        raw.insert(raw.end(), row, row + 3 * static_cast<std::size_t>(w));
    }

    // zlib wrapper + stored deflate blocks (max 65535 bytes per block).
    std::vector<std::uint8_t> idat;
    idat.push_back(0x78); // CMF: deflate, 32K window
    idat.push_back(0x01); // FLG: no preset dict, fastest (checksum-valid pair)
    std::size_t off = 0;
    while (off < raw.size()) {
        const std::size_t len = std::min<std::size_t>(65535, raw.size() - off);
        const bool last = (off + len == raw.size());
        idat.push_back(last ? 1 : 0); // BFINAL + BTYPE=00 (stored)
        idat.push_back(static_cast<std::uint8_t>(len & 0xFF));
        idat.push_back(static_cast<std::uint8_t>(len >> 8));
        idat.push_back(static_cast<std::uint8_t>(~len & 0xFF));
        idat.push_back(static_cast<std::uint8_t>((~len >> 8) & 0xFF));
        idat.insert(idat.end(), raw.begin() + off, raw.begin() + off + len);
        off += len;
    }
    // Adler-32 of the UNCOMPRESSED data closes the zlib stream.
    std::uint32_t a = 1, b = 0;
    for (std::uint8_t byte : raw) {
        a = (a + byte) % 65521;
        b = (b + a) % 65521;
    }
    putU32BE(idat, (b << 16) | a);

    std::vector<std::uint8_t> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    std::uint8_t ihdr[13];
    ihdr[0] = static_cast<std::uint8_t>(w >> 24); ihdr[1] = static_cast<std::uint8_t>(w >> 16);
    ihdr[2] = static_cast<std::uint8_t>(w >> 8);  ihdr[3] = static_cast<std::uint8_t>(w);
    ihdr[4] = static_cast<std::uint8_t>(h >> 24); ihdr[5] = static_cast<std::uint8_t>(h >> 16);
    ihdr[6] = static_cast<std::uint8_t>(h >> 8);  ihdr[7] = static_cast<std::uint8_t>(h);
    ihdr[8] = 8;  // bit depth
    ihdr[9] = 2;  // color type: truecolor RGB
    ihdr[10] = 0; // compression
    ihdr[11] = 0; // filter
    ihdr[12] = 0; // interlace
    writeChunk(png, "IHDR", ihdr, sizeof ihdr);
    writeChunk(png, "IDAT", idat.data(), idat.size());
    writeChunk(png, "IEND", nullptr, 0);
    return png;
}

// ---------------------------------------------------------------------------
// Shader loading. GLSL sources live in assets/shaders (plan section 3); the
// exe usually runs from build/<config>/, so search upward from the exe
// directory as well as the CWD before giving up with a useful error.
// ---------------------------------------------------------------------------

std::filesystem::path findShaderDirectory() {
    std::vector<std::filesystem::path> candidates;
    // Walk up from the exe: covers an installed layout (assets next to the
    // exe) and the dev layout (exe two levels deep in build/Release).
    std::filesystem::path base = platform::executableDirectory();
    for (int up = 0; up <= 4; ++up) {
        candidates.push_back(base / "assets" / "shaders");
        if (!base.has_parent_path() || base.parent_path() == base) break;
        base = base.parent_path();
    }
    candidates.push_back(std::filesystem::current_path() / "assets" / "shaders");

    for (const auto& c : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(c / "particles.vert", ec)) return c;
    }
    return {};
}

bool readTextFile(const std::filesystem::path& path, std::string* out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    *out = ss.str();
    return !out->empty();
}

/// Compile one shader stage; on failure, append the GLSL info log to *error.
GLuint compileStage(GLenum type, const std::string& src, const char* label,
                    std::string* error) {
    const GLuint sh = glCreateShader(type);
    const char* p = src.c_str();
    glShaderSource(sh, 1, &p, nullptr);
    glCompileShader(sh);
    GLint ok = GL_FALSE;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048] = {};
        glGetShaderInfoLog(sh, sizeof log - 1, nullptr, log);
        if (error) *error = std::string(label) + " compile failed: " + log;
        glDeleteShader(sh);
        return 0;
    }
    return sh;
}

/// Compile + link a vertex/fragment pair loaded from the shader directory.
GLuint buildProgram(const std::filesystem::path& dir, const char* vertName,
                    const char* fragName, std::string* error) {
    std::string vsSrc, fsSrc;
    if (!readTextFile(dir / vertName, &vsSrc)) {
        if (error)
            *error = "cannot read shader "
                   + platform::pathToUtf8(dir / vertName);
        return 0;
    }
    if (!readTextFile(dir / fragName, &fsSrc)) {
        if (error)
            *error = "cannot read shader "
                   + platform::pathToUtf8(dir / fragName);
        return 0;
    }
    const GLuint vs = compileStage(GL_VERTEX_SHADER, vsSrc, vertName, error);
    if (!vs) return 0;
    const GLuint fs = compileStage(GL_FRAGMENT_SHADER, fsSrc, fragName, error);
    if (!fs) {
        glDeleteShader(vs);
        return 0;
    }
    const GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    // Stages are owned by the program after link; flag for deletion now.
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = GL_FALSE;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048] = {};
        glGetProgramInfoLog(prog, sizeof log - 1, nullptr, log);
        if (error) *error = std::string(vertName) + "+" + fragName
                          + " link failed: " + log;
        glDeleteProgram(prog);
        return 0;
    }
    return prog;
}

// ---------------------------------------------------------------------------
// Mesh construction helpers. Vertices carry position + face normal + tint;
// flat shading comes from duplicating vertices per face (the mesh shader
// flips the normal toward the viewer, so triangle winding never matters).
// ---------------------------------------------------------------------------

/// Interleaved vertex layout for the foil/VG mesh VBO (9 floats, 36 bytes).
struct MeshVertex {
    float px, py, pz; ///< Position, lattice cell space.
    float nx, ny, nz; ///< Face normal (flat shading).
    float r, g, b;    ///< Per-part tint.
};

void appendTriangle(std::vector<MeshVertex>& v, const Vec3f& a, const Vec3f& b,
                    const Vec3f& c, const Vec3f& n, const Vec3f& color) {
    for (const Vec3f& p : {a, b, c}) {
        v.push_back({p.x, p.y, p.z, n.x, n.y, n.z, color.x, color.y, color.z});
    }
}

void appendQuad(std::vector<MeshVertex>& v, const Vec3f& a, const Vec3f& b,
                const Vec3f& c, const Vec3f& d, const Vec3f& n,
                const Vec3f& color) {
    appendTriangle(v, a, b, c, n, color);
    appendTriangle(v, a, c, d, n, color);
}

/// Generic box from a corner and three edge vectors — used for VG vanes
/// (oriented thin plates). Face normals are the normalized edge cross
/// products; orientation is irrelevant (see flat-shading note above).
void appendParallelepiped(std::vector<MeshVertex>& v, const Vec3f& p0,
                          const Vec3f& e1, const Vec3f& e2, const Vec3f& e3,
                          const Vec3f& color) {
    const Vec3f n12 = normalized(cross(e1, e2));
    const Vec3f n23 = normalized(cross(e2, e3));
    const Vec3f n31 = normalized(cross(e3, e1));
    // Two faces per edge pair: at the base corner and offset by the third edge.
    appendQuad(v, p0, p0 + e1, p0 + e1 + e2, p0 + e2, n12, color);
    appendQuad(v, p0 + e3, p0 + e1 + e3, p0 + e1 + e2 + e3, p0 + e2 + e3, n12, color);
    appendQuad(v, p0, p0 + e2, p0 + e2 + e3, p0 + e3, n23, color);
    appendQuad(v, p0 + e1, p0 + e1 + e2, p0 + e1 + e2 + e3, p0 + e1 + e3, n23, color);
    appendQuad(v, p0, p0 + e3, p0 + e3 + e1, p0 + e1, n31, color);
    appendQuad(v, p0 + e2, p0 + e2 + e3, p0 + e2 + e3 + e1, p0 + e2 + e1, n31, color);
}

/// Triangular wedge prism (the Ramp VG type): right-triangle profile in the
/// (length, height) plane extruded along the width axis. Tall edge at the
/// downstream end, matching the voxelizer's wedge description (vg.h).
void appendWedge(std::vector<MeshVertex>& v, const Vec3f& root,
                 const Vec3f& length, const Vec3f& height, const Vec3f& width,
                 const Vec3f& color) {
    // Profile corners at the width-min face: upstream toe, downstream base,
    // downstream tip; the +width face repeats them.
    const Vec3f a0 = root;
    const Vec3f b0 = root + length;
    const Vec3f c0 = root + length + height;
    const Vec3f a1 = a0 + width, b1 = b0 + width, c1 = c0 + width;

    const Vec3f nSide = normalized(cross(length, height)); // triangle faces
    appendTriangle(v, a0, b0, c0, nSide, color);
    appendTriangle(v, a1, b1, c1, nSide, color);
    // Bottom (length x width), back (height x width), hypotenuse quads.
    appendQuad(v, a0, b0, b1, a1, normalized(cross(length, width)), color);
    appendQuad(v, b0, c0, c1, b1, normalized(cross(height, width)), color);
    appendQuad(v, c0, a0, a1, c1, normalized(cross(a0 - c0, width)), color);
}

/// Signed area of a 2D polygon (positive = CCW). Drives winding decisions for
/// the side-wall outward normals.
float polygonSignedArea(const std::vector<Vec2f>& pts) {
    float a = 0.0f;
    for (std::size_t i = 0; i < pts.size(); ++i) {
        const Vec2f& p = pts[i];
        const Vec2f& q = pts[(i + 1) % pts.size()];
        a += p.x * q.y - q.x * p.y;
    }
    return 0.5f * a;
}

/// Ear-clipping triangulation of a simple polygon (the airfoil outline is one
/// after duplicate stripping). O(n^2) on ~400 points — microseconds, and only
/// runs on geometry edits. Input must wind CCW. Falls back to a fan if the
/// numeric ear test stalls on degenerate slivers so it can never loop forever.
std::vector<std::array<int, 3>> earClipTriangulate(const std::vector<Vec2f>& pts) {
    std::vector<std::array<int, 3>> tris;
    const int n = static_cast<int>(pts.size());
    if (n < 3) return tris;
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;

    auto cross2 = [](const Vec2f& o, const Vec2f& a, const Vec2f& b) {
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    };
    auto pointInTri = [&](const Vec2f& p, const Vec2f& a, const Vec2f& b,
                          const Vec2f& c) {
        // Same-side test; boundary counts as inside (conservative for ears).
        const float d1 = cross2(a, b, p), d2 = cross2(b, c, p), d3 = cross2(c, a, p);
        const bool hasNeg = (d1 < 0) || (d2 < 0) || (d3 < 0);
        const bool hasPos = (d1 > 0) || (d2 > 0) || (d3 > 0);
        return !(hasNeg && hasPos);
    };

    int guard = 0;
    while (idx.size() > 3) {
        bool clipped = false;
        const int m = static_cast<int>(idx.size());
        for (int i = 0; i < m; ++i) {
            const int i0 = idx[(i + m - 1) % m], i1 = idx[i], i2 = idx[(i + 1) % m];
            const Vec2f &a = pts[i0], &b = pts[i1], &c = pts[i2];
            // Convex corner of a CCW polygon has positive cross product.
            if (cross2(a, b, c) <= 1e-12f) continue;
            // No other remaining vertex may sit inside the candidate ear.
            bool blocked = false;
            for (int j : idx) {
                if (j == i0 || j == i1 || j == i2) continue;
                if (pointInTri(pts[j], a, b, c)) { blocked = true; break; }
            }
            if (blocked) continue;
            tris.push_back({i0, i1, i2});
            idx.erase(idx.begin() + i);
            clipped = true;
            break;
        }
        // Degenerate remainder (collinear slivers): fan the rest and stop.
        if (!clipped || ++guard > 4 * n) {
            for (std::size_t i = 1; i + 1 < idx.size(); ++i)
                tris.push_back({idx[0], idx[i], idx[i + 1]});
            return tris;
        }
    }
    tris.push_back({idx[0], idx[1], idx[2]});
    return tris;
}

// ---------------------------------------------------------------------------
// Section -> lattice transform. This MUST mirror the voxelizer contract
// (geom/voxelizer.h): rotate the chord-unit polygon about quarter-chord
// (0.25, 0) by -aoa_deg, scale by chordCells, anchor quarter-chord at
// (layout.anchorX, layout.anchorY). Any drift here and the rendered foil
// detaches from the simulated voxels.
// ---------------------------------------------------------------------------

struct SectionTransform {
    float cosT = 1.0f, sinT = 0.0f;
    float scale = 256.0f;
    float tx = 0.0f, ty = 0.0f;

    static SectionTransform make(float aoa_deg, const DomainLayout& layout) {
        SectionTransform xf;
        const float theta = -aoa_deg * (kPiF / 180.0f); // voxelizer convention
        xf.cosT = std::cos(theta);
        xf.sinT = std::sin(theta);
        xf.scale = static_cast<float>(layout.chordCells);
        xf.tx = layout.anchorX();
        xf.ty = layout.anchorY();
        return xf;
    }

    /// Chord-unit point -> lattice cell coordinates (full transform).
    Vec2f point(const Vec2f& p) const {
        const float x = p.x - 0.25f, y = p.y; // about quarter-chord
        return {(x * cosT - y * sinT) * scale + tx,
                (x * sinT + y * cosT) * scale + ty};
    }

    /// Chord-unit DIRECTION -> lattice direction (rotation only, length kept).
    Vec2f direction(const Vec2f& v) const {
        return {v.x * cosT - v.y * sinT, v.x * sinT + v.y * cosT};
    }
};

/// Default palette-end value per slice field: lattice speeds run u_lat~0.08
/// with wake peaks near 2x; shear-layer vorticity at the default resolution
/// is a few 1e-2 per step; pressure deviation cs^2*(rho-1) is O(u^2).
float defaultSliceScale(SliceField field) {
    switch (field) {
        case SliceField::SpeedMag:   return 0.15f;
        case SliceField::VorticityZ: return 0.04f;
        case SliceField::Pressure:   return 0.01f;
    }
    return 0.15f;
}

// ---------------------------------------------------------------------------
// CUDA-registered GL image + cached surface object. The graphics resource is
// registered exactly once; the surface object is (re)created only when the
// mapped cudaArray actually changes (in practice: once), with a stream sync
// guarding against destroying a descriptor a queued kernel still uses.
// ---------------------------------------------------------------------------

struct InteropImage {
    cudaGraphicsResource* res = nullptr;
    cudaSurfaceObject_t surf = 0;
    cudaArray_t cachedArray = nullptr;

    /// Call after the resource is mapped: refresh the surface object if the
    /// backing array moved. Returns the first CUDA error.
    cudaError_t ensureSurface(cudaStream_t stream) {
        cudaArray_t arr = nullptr;
        cudaError_t err = cudaGraphicsSubResourceGetMappedArray(&arr, res, 0, 0);
        if (err != cudaSuccess) return err;
        if (arr == cachedArray && surf != 0) return cudaSuccess;
        if (surf) {
            // A queued kernel may still hold the old descriptor — drain first.
            cudaStreamSynchronize(stream);
            cudaDestroySurfaceObject(surf);
            surf = 0;
        }
        cudaResourceDesc rd{};
        rd.resType = cudaResourceTypeArray;
        rd.res.array.array = arr;
        err = cudaCreateSurfaceObject(&surf, &rd);
        if (err == cudaSuccess) cachedArray = arr;
        return err;
    }

    /// Full release: surface object (after a drain) + unregistration.
    void release(cudaStream_t stream) {
        if (surf) {
            cudaStreamSynchronize(stream);
            cudaDestroySurfaceObject(surf);
            surf = 0;
        }
        if (res) {
            cudaGraphicsUnregisterResource(res);
            res = nullptr;
        }
        cachedArray = nullptr;
    }
};

/// Full 4x4 inverse (cofactor method) — used once per frame to hand the
/// raymarch shaders a clip->world matrix so they can un-project the scene
/// depth buffer into world space. Column-major to match Mat4f (element
/// (row r, col c) at m[c*4 + r]). Returns false (identity out) if singular.
bool invertMat4(const float m[16], float out[16]) {
    float inv[16];
    inv[0] =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15]
            + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
    inv[4] = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15]
            - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
    inv[8] =  m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15]
            + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14]
            - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
    inv[1] = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15]
            - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
    inv[5] =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15]
            + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
    inv[9] = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15]
            - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
    inv[13] =  m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14]
            + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
    inv[2] =  m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15]
            + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];
    inv[6] = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15]
            - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];
    inv[10] =  m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15]
            + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14]
            - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];
    inv[3] = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11]
            - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6];
    inv[7] =  m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11]
            + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11]
            - m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5];
    inv[15] =  m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10]
            + m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5];

    float det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
    if (det == 0.0f) {
        for (int i = 0; i < 16; ++i) out[i] = (i % 5 == 0) ? 1.0f : 0.0f;
        return false;
    }
    det = 1.0f / det;
    for (int i = 0; i < 16; ++i) out[i] = inv[i] * det;
    return true;
}

/// Unit cube as 12 triangles (36 vertices, positions only) for the raycast
/// proxy geometry; the vertex shader scales it by the grid dimensions.
constexpr float kUnitCube[36 * 3] = {
    // -X face
    0,0,0, 0,0,1, 0,1,1,  0,0,0, 0,1,1, 0,1,0,
    // +X face
    1,0,0, 1,1,0, 1,1,1,  1,0,0, 1,1,1, 1,0,1,
    // -Y face
    0,0,0, 1,0,0, 1,0,1,  0,0,0, 1,0,1, 0,0,1,
    // +Y face
    0,1,0, 0,1,1, 1,1,1,  0,1,0, 1,1,1, 1,1,0,
    // -Z face
    0,0,0, 0,1,0, 1,1,0,  0,0,0, 1,1,0, 1,0,0,
    // +Z face
    0,0,1, 1,0,1, 1,1,1,  0,0,1, 1,1,1, 0,1,1,
};

} // namespace

// ===========================================================================
// Impl: every GL handle and CUDA interop registration lives here, off the
// header (viz.h promises no GL/interop types leak to includers).
// ===========================================================================

struct Visualizer::Impl {
    GridDims dims{};
    int particleCount = 0;
    cudaStream_t stream = nullptr;
    bool initialized = false;
    unsigned long long frame = 0; ///< updateFields call counter (Q cadence, RNG seeds).

    std::filesystem::path shaderDir;

    // -- programs + cached uniform locations --
    GLuint progParticles = 0, progSlice = 0, progMesh = 0, progQ = 0, progVol = 0;
    GLint uPViewProj = -1, uPPointSize = -1, uPColormap = -1, uPAlpha = -1;
    GLint uSViewProj = -1, uSTex = -1;
    GLint uMViewProj = -1, uMEye = -1;
    // Q isosurface: opaque first-hit raycast — writes gl_FragDepth, so the
    // hardware z-buffer handles occlusion (no scene-depth texture needed).
    GLint uQViewProj = -1, uQDims = -1, uQEye = -1, uQThresh = -1, uQVol = -1;
    GLint uQSpeedVol = -1, uQColorByVel = -1, uQColormap = -1;
    // Velocity volume (the hero wind-tunnel smoke mode).
    GLint uVViewProj = -1, uVDims = -1, uVEye = -1, uVVol = -1, uVDepth = -1;
    GLint uVViewport = -1, uVInvVP = -1, uVColormap = -1, uVSlowOpacity = -1,
          uVDensity = -1, uVFreestream = -1;

    // -- particle pool (hero mode) --
    GLuint particleVAO = 0, posVBO = 0, keyVBO = 0;
    cudaGraphicsResource* posRes = nullptr; ///< Registered ONCE per pool lifetime.
    cudaGraphicsResource* keyRes = nullptr;

    // -- foil + VG mesh --
    GLuint meshVAO = 0, meshVBO = 0;
    GLsizei meshVertexCount = 0;

    // -- slice planes: one texture per axis, registered once at init --
    GLuint sliceVAO = 0, sliceVBO = 0;
    GLuint sliceTex[3] = {0, 0, 0};
    int sliceW[3] = {0, 0, 0};
    int sliceH[3] = {0, 0, 0};
    InteropImage slice[3];

    // -- Q-criterion volume (lazy: ~cellCount bytes of VRAM only if used) --
    GLuint qTex = 0;
    InteropImage qVol;
    GLuint cubeVAO = 0, cubeVBO = 0;
    bool qAvailable = false;   ///< Shader compiled; volume may still be lazy.
    bool qCreateFailed = false;///< Latched so a failing create doesn't retry per frame.

    // -- velocity volume (lazy R16F 3D texture: 2 bytes/cell when used) --
    GLuint velTex = 0;
    InteropImage velVol;
    bool volAvailable = false;    ///< volume.* shaders compiled OK.
    bool velCreateFailed = false; ///< Latched create failure (no per-frame retry).

    // -- scene-depth copy: the opaque pass's depth, copied into a texture so
    // the volume/Q raymarchers can occlude against the foil silhouette. Sized
    // to the framebuffer and reallocated only when the viewport size changes. --
    GLuint depthTex = 0;
    int depthW = 0, depthH = 0;

    // ---- helpers ----------------------------------------------------------

    /// Create the particle VBO pair + VAO, register both with CUDA (the one
    /// registration for this pool's lifetime), and seed the pool.
    bool createParticlePool(int count, std::string* error) {
        glGenVertexArrays(1, &particleVAO);
        glBindVertexArray(particleVAO);

        glGenBuffers(1, &posVBO);
        glBindBuffer(GL_ARRAY_BUFFER, posVBO);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(count) * 4 * sizeof(float), nullptr,
                     GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);

        glGenBuffers(1, &keyVBO);
        glBindBuffer(GL_ARRAY_BUFFER, keyVBO);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(count) * sizeof(float), nullptr,
                     GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), nullptr);
        glBindVertexArray(0);

        // Register ONCE (plan section 13). Positions are read-modify-write by
        // the advection kernel; color keys are write-only, hence the discard
        // hint that lets the driver skip a sync copy.
        if (auto err = cudaGraphicsGLRegisterBuffer(
                &posRes, posVBO, cudaGraphicsRegisterFlagsNone);
            err != cudaSuccess) {
            if (error) *error = std::string("position VBO registration failed: ")
                              + cudaGetErrorString(err);
            return false;
        }
        if (auto err = cudaGraphicsGLRegisterBuffer(
                &keyRes, keyVBO, cudaGraphicsRegisterFlagsWriteDiscard);
            err != cudaSuccess) {
            if (error) *error = std::string("color VBO registration failed: ")
                              + cudaGetErrorString(err);
            return false;
        }
        particleCount = count;

        // Seed: randomized positions/ages so the pool starts in steady state.
        cudaGraphicsResource* both[2] = {posRes, keyRes};
        if (auto err = cudaGraphicsMapResources(2, both, stream);
            err != cudaSuccess) {
            if (error) *error = std::string("seed map failed: ")
                              + cudaGetErrorString(err);
            return false;
        }
        float4* positions = nullptr;
        float* keys = nullptr;
        std::size_t bytes = 0;
        cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&positions),
                                             &bytes, posRes);
        cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&keys),
                                             &bytes, keyRes);
        const cudaError_t seedErr =
            launchParticleSeed(positions, keys, count, dims, 0xF01DCAFEu, stream);
        cudaGraphicsUnmapResources(2, both, stream);
        if (seedErr != cudaSuccess) {
            if (error) *error = std::string("particle seed failed: ")
                              + cudaGetErrorString(seedErr);
            return false;
        }
        return true;
    }

    /// Tear down the particle pool: unregister both interop handles BEFORE
    /// deleting the GL buffers (deleting a still-registered buffer is UB).
    void destroyParticlePool() {
        if (posRes) { cudaGraphicsUnregisterResource(posRes); posRes = nullptr; }
        if (keyRes) { cudaGraphicsUnregisterResource(keyRes); keyRes = nullptr; }
        if (posVBO) { glDeleteBuffers(1, &posVBO); posVBO = 0; }
        if (keyVBO) { glDeleteBuffers(1, &keyVBO); keyVBO = 0; }
        if (particleVAO) { glDeleteVertexArrays(1, &particleVAO); particleVAO = 0; }
        particleCount = 0;
    }

    /// Allocate the per-axis slice textures and register each exactly once.
    /// Sizes follow the texel->cell mapping documented in particles.cuh.
    bool createSliceTextures(std::string* error) {
        const int w[3] = {dims.nz, dims.nx, dims.nx};
        const int h[3] = {dims.ny, dims.nz, dims.ny};
        for (int axis = 0; axis < 3; ++axis) {
            sliceW[axis] = w[axis];
            sliceH[axis] = h[axis];
            glGenTextures(1, &sliceTex[axis]);
            glBindTexture(GL_TEXTURE_2D, sliceTex[axis]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w[axis], h[axis], 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            if (auto err = cudaGraphicsGLRegisterImage(
                    &slice[axis].res, sliceTex[axis], GL_TEXTURE_2D,
                    cudaGraphicsRegisterFlagsSurfaceLoadStore);
                err != cudaSuccess) {
                if (error) *error = std::string("slice texture registration failed: ")
                                  + cudaGetErrorString(err);
                return false;
            }
        }
        glBindTexture(GL_TEXTURE_2D, 0);
        return true;
    }

    /// Lazy Q-volume creation: a full-grid R8 3D texture (~1 byte/cell) plus
    /// its one-time registration. Only runs when the user first enables the
    /// raycast mode, keeping the default VRAM footprint lean.
    bool createQVolume() {
        glGenTextures(1, &qTex);
        glBindTexture(GL_TEXTURE_3D, qTex);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, dims.nx, dims.ny, dims.nz, 0,
                     GL_RED, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_3D, 0);
        if (auto err = cudaGraphicsGLRegisterImage(
                &qVol.res, qTex, GL_TEXTURE_3D,
                cudaGraphicsRegisterFlagsSurfaceLoadStore);
            err != cudaSuccess) {
            std::fprintf(stderr, "[viz] Q volume registration failed: %s\n",
                         cudaGetErrorString(err));
            glDeleteTextures(1, &qTex);
            qTex = 0;
            return false;
        }
        return true;
    }

    /// Lazy velocity-volume creation: a full-grid single-channel float 3D
    /// texture (R32F: 4 bytes/cell) plus its one-time CUDA registration. Runs
    /// when the velocity-volume mode (or Q velocity coloring) first needs it.
    /// R32F (not R16F) so the CUDA-side surf3Dwrite of a 4-byte float matches
    /// the texel stride exactly — a 16-bit texel would make every write a
    /// 2-byte overflow and corrupt neighbouring memory. The smooth float ramp
    /// also avoids the banding an R8 would show in the faint slow-air haze.
    bool createVelVolume() {
        glGenTextures(1, &velTex);
        glBindTexture(GL_TEXTURE_3D, velTex);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, dims.nx, dims.ny, dims.nz, 0,
                     GL_RED, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_3D, 0);
        if (auto err = cudaGraphicsGLRegisterImage(
                &velVol.res, velTex, GL_TEXTURE_3D,
                cudaGraphicsRegisterFlagsSurfaceLoadStore);
            err != cudaSuccess) {
            std::fprintf(stderr, "[viz] velocity volume registration failed: %s\n",
                         cudaGetErrorString(err));
            glDeleteTextures(1, &velTex);
            velTex = 0;
            return false;
        }
        return true;
    }

    /// Ensure the scene-depth copy texture matches the framebuffer size, then
    /// copy the current (post-opaque-pass) depth buffer into it. The raymarch
    /// shaders sample this to terminate rays at the foil surface — without it
    /// the volume would composite over solid geometry from every angle.
    void captureSceneDepth(int w, int h) {
        if (w <= 0 || h <= 0) return;
        if (depthTex == 0 || w != depthW || h != depthH) {
            if (depthTex == 0) glGenTextures(1, &depthTex);
            glBindTexture(GL_TEXTURE_2D, depthTex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, w, h, 0,
                         GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            depthW = w;
            depthH = h;
        } else {
            glBindTexture(GL_TEXTURE_2D, depthTex);
        }
        // Copy straight from the default framebuffer's depth attachment — no
        // FBO needed, and the opaque mesh + slices have already written depth.
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, w, h);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
};

// ===========================================================================
// Visualizer
// ===========================================================================

Visualizer::Visualizer() : impl_(std::make_unique<Impl>()) {}
Visualizer::~Visualizer() { shutdown(); }

bool Visualizer::init(const GridDims& dims, int particleCount,
                      cudaStream_t stream, std::string* error) {
    shutdown(); // re-init support (resolution preset changes)

    impl_->dims = dims;
    impl_->stream = stream;
    impl_->frame = 0;
    if (dims.cellCount() <= 0 || particleCount <= 0) {
        if (error) *error = "visualizer init: invalid grid dims or particle count";
        return false;
    }

    // ---- shaders ----------------------------------------------------------
    impl_->shaderDir = findShaderDirectory();
    if (impl_->shaderDir.empty()) {
        if (error) {
            *error = "assets/shaders not found (searched up from "
                   + platform::pathToUtf8(platform::executableDirectory())
                   + " and the working directory)";
        }
        return false;
    }
    impl_->progParticles =
        buildProgram(impl_->shaderDir, "particles.vert", "particles.frag", error);
    if (!impl_->progParticles) return false;
    impl_->progSlice =
        buildProgram(impl_->shaderDir, "slice.vert", "slice.frag", error);
    if (!impl_->progSlice) return false;
    impl_->progMesh =
        buildProgram(impl_->shaderDir, "mesh.vert", "mesh.frag", error);
    if (!impl_->progMesh) return false;
    // Q raycast is the stretch mode (plan 9.1): a shader failure here must
    // not take down the core renderer, so it degrades to "mode unavailable".
    std::string qErr;
    impl_->progQ =
        buildProgram(impl_->shaderDir, "qraycast.vert", "qraycast.frag", &qErr);
    impl_->qAvailable = impl_->progQ != 0;
    if (!impl_->qAvailable) {
        std::fprintf(stderr, "[viz] Q raycast disabled: %s\n", qErr.c_str());
    }
    // Velocity volume (the hero wind-tunnel smoke mode) — also a soft-fail
    // optional stage: if it can't compile, the renderer simply falls back to
    // particles/slices instead of taking the whole app down.
    std::string volErr;
    impl_->progVol =
        buildProgram(impl_->shaderDir, "volume.vert", "volume.frag", &volErr);
    impl_->volAvailable = impl_->progVol != 0;
    if (!impl_->volAvailable) {
        std::fprintf(stderr, "[viz] velocity volume disabled: %s\n",
                     volErr.c_str());
    }

    // Uniform locations resolved once; -1 silently no-ops in glUniform*.
    impl_->uPViewProj  = glGetUniformLocation(impl_->progParticles, "uViewProj");
    impl_->uPPointSize = glGetUniformLocation(impl_->progParticles, "uPointSize");
    impl_->uPColormap  = glGetUniformLocation(impl_->progParticles, "uColormap");
    impl_->uPAlpha     = glGetUniformLocation(impl_->progParticles, "uAlphaScale");
    impl_->uSViewProj  = glGetUniformLocation(impl_->progSlice, "uViewProj");
    impl_->uSTex       = glGetUniformLocation(impl_->progSlice, "uField");
    impl_->uMViewProj  = glGetUniformLocation(impl_->progMesh, "uViewProj");
    impl_->uMEye       = glGetUniformLocation(impl_->progMesh, "uEye");
    if (impl_->qAvailable) {
        impl_->uQViewProj = glGetUniformLocation(impl_->progQ, "uViewProj");
        impl_->uQDims     = glGetUniformLocation(impl_->progQ, "uDims");
        impl_->uQEye      = glGetUniformLocation(impl_->progQ, "uEye");
        impl_->uQThresh   = glGetUniformLocation(impl_->progQ, "uThreshold");
        impl_->uQVol      = glGetUniformLocation(impl_->progQ, "uVolume");
        impl_->uQSpeedVol  = glGetUniformLocation(impl_->progQ, "uSpeedVolume");
        impl_->uQColorByVel = glGetUniformLocation(impl_->progQ, "uColorByVel");
        impl_->uQColormap  = glGetUniformLocation(impl_->progQ, "uColormap");
    }
    if (impl_->volAvailable) {
        impl_->uVViewProj   = glGetUniformLocation(impl_->progVol, "uViewProj");
        impl_->uVDims       = glGetUniformLocation(impl_->progVol, "uDims");
        impl_->uVEye        = glGetUniformLocation(impl_->progVol, "uEye");
        impl_->uVVol        = glGetUniformLocation(impl_->progVol, "uVolume");
        impl_->uVDepth      = glGetUniformLocation(impl_->progVol, "uSceneDepth");
        impl_->uVViewport   = glGetUniformLocation(impl_->progVol, "uViewport");
        impl_->uVInvVP      = glGetUniformLocation(impl_->progVol, "uInvViewProj");
        impl_->uVColormap   = glGetUniformLocation(impl_->progVol, "uColormap");
        impl_->uVSlowOpacity = glGetUniformLocation(impl_->progVol, "uSlowOpacity");
        impl_->uVDensity    = glGetUniformLocation(impl_->progVol, "uDensity");
        impl_->uVFreestream = glGetUniformLocation(impl_->progVol, "uFreestream");
    }

    // ---- particle pool (the two interop buffer registrations) -------------
    if (!impl_->createParticlePool(particleCount, error)) return false;

    // ---- slice textures (three interop image registrations) ---------------
    if (!impl_->createSliceTextures(error)) return false;

    // Slice quad: 4 verts x (pos3 + uv2), rewritten per draw (slices move).
    glGenVertexArrays(1, &impl_->sliceVAO);
    glBindVertexArray(impl_->sliceVAO);
    glGenBuffers(1, &impl_->sliceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, impl_->sliceVBO);
    glBufferData(GL_ARRAY_BUFFER, 4 * 5 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                          reinterpret_cast<const void*>(3 * sizeof(float)));

    // ---- foil/VG mesh buffers (filled by uploadGeometry) -------------------
    glGenVertexArrays(1, &impl_->meshVAO);
    glBindVertexArray(impl_->meshVAO);
    glGenBuffers(1, &impl_->meshVBO);
    glBindBuffer(GL_ARRAY_BUFFER, impl_->meshVBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex),
                          reinterpret_cast<const void*>(3 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex),
                          reinterpret_cast<const void*>(6 * sizeof(float)));

    // ---- raycast proxy cube -------------------------------------------------
    glGenVertexArrays(1, &impl_->cubeVAO);
    glBindVertexArray(impl_->cubeVAO);
    glGenBuffers(1, &impl_->cubeVBO);
    glBindBuffer(GL_ARRAY_BUFFER, impl_->cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof kUnitCube, kUnitCube, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    impl_->initialized = true;
    return true;
}

void Visualizer::shutdown() {
    if (!impl_) return;
    Impl& im = *impl_;
    if (!im.initialized && !im.posRes && !im.progParticles) return;

    // CUDA side first: surface objects + unregistrations must precede GL
    // object deletion (a registered buffer/texture must outlive its handle).
    im.destroyParticlePool();
    for (auto& s : im.slice) s.release(im.stream);
    im.qVol.release(im.stream);
    im.velVol.release(im.stream);

    // GL side. The context is still current here — main.cpp tears the
    // visualizer down before destroying the window.
    auto delTex = [](GLuint& t) { if (t) { glDeleteTextures(1, &t); t = 0; } };
    for (GLuint& t : im.sliceTex) delTex(t);
    delTex(im.qTex);
    delTex(im.velTex);
    delTex(im.depthTex);
    im.depthW = im.depthH = 0;
    auto delBuf = [](GLuint& b) { if (b) { glDeleteBuffers(1, &b); b = 0; } };
    delBuf(im.sliceVBO);
    delBuf(im.meshVBO);
    delBuf(im.cubeVBO);
    auto delVao = [](GLuint& v) { if (v) { glDeleteVertexArrays(1, &v); v = 0; } };
    delVao(im.sliceVAO);
    delVao(im.meshVAO);
    delVao(im.cubeVAO);
    auto delProg = [](GLuint& p) { if (p) { glDeleteProgram(p); p = 0; } };
    delProg(im.progParticles);
    delProg(im.progSlice);
    delProg(im.progMesh);
    delProg(im.progQ);
    delProg(im.progVol);

    im.meshVertexCount = 0;
    im.qAvailable = false;
    im.qCreateFailed = false;
    im.volAvailable = false;
    im.velCreateFailed = false;
    im.initialized = false;
}

// ---------------------------------------------------------------------------
// Geometry upload: extruded outline prism + VG vane boxes (plan 9.1 — no
// marching cubes; the render mesh is built from the same 2D polygon and the
// same quarter-chord transform the voxelizer stamps, so mesh and voxels agree).
// ---------------------------------------------------------------------------

void Visualizer::uploadGeometry(const AirfoilGeometry& airfoil,
                                const std::vector<VGParams>& vgs, float aoa_deg,
                                const DomainLayout& layout) {
    if (!impl_->initialized || !airfoil.isValid()) return;

    const SectionTransform xf = SectionTransform::make(aoa_deg, layout);
    const float spanZ = static_cast<float>(layout.dims.nz);
    const Vec3f foilColor(0.62f, 0.65f, 0.70f); // neutral aluminum gray
    const Vec3f vgColor(0.95f, 0.55f, 0.15f);   // high-vis orange vanes

    // ---- outline cleanup: strip consecutive duplicates + the closing repeat
    std::vector<Vec2f> outline;
    outline.reserve(airfoil.points.size());
    for (const Vec2f& p : airfoil.points) {
        if (!outline.empty()) {
            const Vec2f d = p - outline.back();
            if (std::fabs(d.x) < 1e-7f && std::fabs(d.y) < 1e-7f) continue;
        }
        outline.push_back(p);
    }
    if (outline.size() > 2) {
        const Vec2f d = outline.front() - outline.back();
        if (std::fabs(d.x) < 1e-7f && std::fabs(d.y) < 1e-7f) outline.pop_back();
    }
    if (outline.size() < 3) return;

    // Transform to lattice space and force CCW winding so the side-wall
    // outward normal rule below holds regardless of the source ordering.
    std::vector<Vec2f> poly(outline.size());
    for (std::size_t i = 0; i < outline.size(); ++i) poly[i] = xf.point(outline[i]);
    if (polygonSignedArea(poly) < 0.0f) std::reverse(poly.begin(), poly.end());

    std::vector<MeshVertex> verts;
    verts.reserve(poly.size() * 12 + vgs.size() * 72 + 256);

    // ---- side walls: one quad per outline edge, extruded z = 0..nz ---------
    const int n = static_cast<int>(poly.size());
    for (int i = 0; i < n; ++i) {
        const Vec2f& a = poly[i];
        const Vec2f& b = poly[(i + 1) % n];
        const Vec2f e = b - a;
        // CCW polygon: interior lies left of the edge direction, so the
        // outward normal is the right-hand perpendicular (e.y, -e.x).
        const Vec2f nrm = normalized(Vec2f{e.y, -e.x});
        appendQuad(verts, Vec3f(a.x, a.y, 0.0f), Vec3f(b.x, b.y, 0.0f),
                   Vec3f(b.x, b.y, spanZ), Vec3f(a.x, a.y, spanZ),
                   Vec3f(nrm.x, nrm.y, 0.0f), foilColor);
    }

    // ---- end caps: ear-clip the section once, stamp at both span faces -----
    const auto tris = earClipTriangulate(poly);
    for (const auto& t : tris) {
        appendTriangle(verts, Vec3f(poly[t[0]].x, poly[t[0]].y, 0.0f),
                       Vec3f(poly[t[1]].x, poly[t[1]].y, 0.0f),
                       Vec3f(poly[t[2]].x, poly[t[2]].y, 0.0f),
                       Vec3f(0, 0, -1), foilColor);
        appendTriangle(verts, Vec3f(poly[t[0]].x, poly[t[0]].y, spanZ),
                       Vec3f(poly[t[1]].x, poly[t[1]].y, spanZ),
                       Vec3f(poly[t[2]].x, poly[t[2]].y, spanZ),
                       Vec3f(0, 0, 1), foilColor);
    }

    // ---- VG vanes: actual placed boxes (plan spec), mirroring the vg.h
    // voxelization geometry: thin plates h tall along the local normal,
    // l = length_h*h long, yawed +/-beta about the normal, root on the
    // suction surface, repeated across the span.
    for (const VGParams& vg : vgs) {
        const SurfaceFrame frame = surfaceFrameAt(airfoil, vg.x_c, /*upper=*/true);
        if (!frame.valid) continue; // out-of-range station: nothing to draw

        // Surface frame -> lattice space (same AoA rotation as the foil).
        const Vec2f p2 = xf.point(frame.point);
        const Vec2f t2 = xf.direction(normalized(frame.tangent));
        const Vec2f n2 = xf.direction(normalized(frame.normal));
        const Vec3f T(t2.x, t2.y, 0.0f); // downstream
        const Vec3f N(n2.x, n2.y, 0.0f); // outward (off the suction surface)

        const float h = vg.height_c * static_cast<float>(layout.chordCells);
        if (h <= 0.0f) continue;
        const float len = std::max(vg.length_h, 0.1f) * h;
        const float thick = 1.5f; // 1-2 cell plate thickness (plan 6.1)
        const float pitch = vg.pitch_c * static_cast<float>(layout.chordCells);
        const float gap = vg.gap_h * h;
        const int unitCount = std::max(vg.count, 1);

        // One yawed vane plate (or wedge) rooted at span position zc.
        auto emitVane = [&](float zc, float betaDeg) {
            const float beta = betaDeg * (kPiF / 180.0f);
            // Vane axis must reproduce the voxelizer's FROZEN convention
            // (vg.cpp stampVane: d3 = (cosB*t.x, cosB*t.y, +sinB)): positive
            // beta sweeps the downstream end toward +z. On the upper surface
            // cross(N, T) = (0, 0, -1), so a Rodrigues rotation about N would
            // yaw the drawn vane OPPOSITE to the voxelized solid (z-mirror) —
            // blend the +z axis in directly instead. The AoA rotation is
            // purely in-plane, so the z component is unaffected.
            const Vec3f L = normalized(T * std::cos(beta)
                                       + Vec3f(0.0f, 0.0f, 1.0f) * std::sin(beta));
            const Vec3f W = normalized(cross(N, L)); // in-surface, across the vane
            const Vec3f root(p2.x, p2.y, zc);
            if (vg.type == VGType::Ramp) {
                // Wedge: toe upstream, full height at the downstream end,
                // one device-height wide (vg.h convention).
                appendWedge(verts, root - L * (len * 0.5f) - W * (h * 0.5f),
                            L * len, N * h, W * h, vgColor);
            } else {
                // Thin plate centered on the station footprint.
                appendParallelepiped(verts,
                                     root - L * (len * 0.5f) - W * (thick * 0.5f),
                                     L * len, N * h, W * thick, vgColor);
            }
        };

        for (int u = 0; u < unitCount; ++u) {
            // Units centered on mid-span, every pitch_c chords (plan 6.1).
            const float zc = 0.5f * spanZ
                           + (static_cast<float>(u)
                              - 0.5f * static_cast<float>(unitCount - 1)) * pitch;
            switch (vg.type) {
                case VGType::CounterRotatingPair: {
                    // Mirrored incidence; commonFlowDown swaps which side
                    // toes in, flipping the shared-vortex direction.
                    const float s = vg.commonFlowDown ? 1.0f : -1.0f;
                    emitVane(zc - 0.5f * gap, +s * vg.beta_deg);
                    emitVane(zc + 0.5f * gap, -s * vg.beta_deg);
                    break;
                }
                case VGType::SingleVane:
                case VGType::CoRotatingArray:
                case VGType::Ramp:
                    emitVane(zc, vg.beta_deg);
                    break;
            }
        }
    }

    // ---- upload -------------------------------------------------------------
    glBindBuffer(GL_ARRAY_BUFFER, impl_->meshVBO);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(verts.size() * sizeof(MeshVertex)),
                 verts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    impl_->meshVertexCount = static_cast<GLsizei>(verts.size());
}

void Visualizer::uploadStlMesh(const StlMesh& mesh) {
    if (!impl_->initialized) return;

    // One flat-shaded vertex triple per facet, same interleaved layout and
    // shader as the foil prism — the STL simply replaces the VBO contents.
    // Vertices are already in lattice-cell space (post applyNormalization),
    // matching what voxelizeStl stamped, so the drawn solid and the voxel
    // solid coincide. An empty mesh clears the buffer (vertex count 0); the
    // app restores the prism via uploadGeometry() when leaving STL mode.
    std::vector<MeshVertex> verts;
    verts.reserve(mesh.triangles.size() * 3);
    const Vec3f stlColor(0.62f, 0.65f, 0.70f); // match the foil's aluminum tint

    for (const StlTriangle& t : mesh.triangles) {
        // Geometric normal from the winding — many exporters write garbage
        // facet normals (stl.h documents the file normal as untrusted). The
        // mesh shader flips normals toward the viewer, so the sign is free;
        // only zero-area slivers fall back to whatever the file claimed.
        Vec3f n = cross(t.v1 - t.v0, t.v2 - t.v0);
        const float len2 = dot(n, n);
        n = (len2 > 1e-20f) ? n * (1.0f / std::sqrt(len2)) : t.normal;
        appendTriangle(verts, t.v0, t.v1, t.v2, n, stlColor);
    }

    glBindBuffer(GL_ARRAY_BUFFER, impl_->meshVBO);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(verts.size() * sizeof(MeshVertex)),
                 verts.empty() ? nullptr : verts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    impl_->meshVertexCount = static_cast<GLsizei>(verts.size());
}

// ---------------------------------------------------------------------------
// Per-frame field update: ONE map/unmap bracket around all GL-touching CUDA
// work (plan 9.3), single stream throughout.
// ---------------------------------------------------------------------------

cudaError_t Visualizer::updateFields(DeviceVelocityField vel, const float* rho,
                                     const std::uint8_t* flags, float dtSteps,
                                     const VizSettings& settings) {
    Impl& im = *impl_;
    if (!im.initialized) return cudaSuccess;
    // The solver may not have valid macroscopic arrays yet (pre-init UI
    // frames); skipping is correct — particles simply hold still.
    if (!vel.u || !vel.v || !vel.w || vel.dims.cellCount() <= 0) return cudaSuccess;

    ++im.frame;

    // ---- lazy Q-volume creation (one-time registration on first enable) ----
    if (settings.showQRaycast && im.qAvailable && !im.qTex && !im.qCreateFailed) {
        if (!im.createQVolume()) im.qCreateFailed = true;
    }

    // ---- lazy velocity-volume creation: needed by the volume mode itself OR
    // when the Q isosurface is colored by velocity (it samples the same tex). --
    const bool wantVel = (settings.showVelocityVolume && im.volAvailable)
                      || (settings.showQRaycast && im.qAvailable
                          && settings.qColorByVelocity);
    if (wantVel && !im.velTex && !im.velCreateFailed) {
        if (!im.createVelVolume()) im.velCreateFailed = true;
    }

    // ---- decide which slice slot drives each axis (one texture per axis) ---
    const SliceConfig* axisSlot[3] = {nullptr, nullptr, nullptr};
    if (settings.showSlices) {
        for (const SliceConfig& cfg : settings.slices) {
            const int axis = static_cast<int>(cfg.axis);
            if (cfg.enabled && axis >= 0 && axis < 3 && !axisSlot[axis]) {
                axisSlot[axis] = &cfg; // first enabled slot per axis wins
            }
        }
    }

    const bool doParticles = settings.showParticles && im.posRes && im.particleCount > 0;
    // Cadences are anchored at frame 1 (frame counter pre-increments above)
    // so a freshly enabled mode fills its texture THIS frame instead of
    // showing an empty volume for up to N frames.
    const int qEvery = std::max(settings.qUpdateEveryNFrames, 1);
    const bool doQ = settings.showQRaycast && im.qTex
                  && ((im.frame - 1) % static_cast<unsigned long long>(qEvery) == 0);
    // Velocity volume fill: on its update cadence, whenever the mode OR the Q
    // velocity coloring needs fresh speed data in the texture.
    const int volEvery = std::max(settings.volumeUpdateEveryNFrames, 1);
    const bool doVel = wantVel && im.velTex
                    && ((im.frame - 1) % static_cast<unsigned long long>(volEvery) == 0);

    // ---- gather the resources this frame actually touches -------------------
    cudaGraphicsResource* resources[7];
    int nRes = 0;
    if (doParticles) {
        resources[nRes++] = im.posRes;
        resources[nRes++] = im.keyRes;
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (axisSlot[axis]) resources[nRes++] = im.slice[axis].res;
    }
    if (doQ) resources[nRes++] = im.qVol.res;
    if (doVel) resources[nRes++] = im.velVol.res;
    if (nRes == 0) return cudaSuccess;

    cudaError_t firstErr = cudaGraphicsMapResources(nRes, resources, im.stream);
    if (firstErr != cudaSuccess) return firstErr;
    auto note = [&firstErr](cudaError_t e) {
        if (firstErr == cudaSuccess && e != cudaSuccess) firstErr = e;
    };

    // ---- particles -----------------------------------------------------------
    if (doParticles) {
        float4* positions = nullptr;
        float* keys = nullptr;
        std::size_t bytes = 0;
        note(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&positions), &bytes, im.posRes));
        note(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&keys), &bytes, im.keyRes));
        if (positions && keys) {
            ParticleAdvectParams prm;
            // Cap the advection time: after a long stepN burst (selftest runs
            // 200) a literal dt would launch tracers across the whole domain
            // in one frame. Visual continuity beats temporal exactness here.
            prm.dtSteps = std::min(dtSteps, 8.0f);
            prm.ageRate = 0.002f;
            prm.seed = static_cast<unsigned int>(im.frame * 2654435761ull);
            prm.colorBy = settings.particleColorBy == ParticleColorBy::VorticityMag ? 1 : 0;
            prm.colorScale = settings.particleColorBy == ParticleColorBy::VorticityMag
                                 ? settings.particleVorticityColorScale
                                 : settings.particleSpeedColorScale;
            note(launchParticleAdvectRK2(positions, keys, im.particleCount, vel,
                                         flags, prm, im.stream));
        }
    }

    // ---- slices ----------------------------------------------------------------
    for (int axis = 0; axis < 3; ++axis) {
        const SliceConfig* cfg = axisSlot[axis];
        if (!cfg) continue;
        note(im.slice[axis].ensureSurface(im.stream));
        if (!im.slice[axis].surf) continue;

        const int extent[3] = {im.dims.nx, im.dims.ny, im.dims.nz};
        SliceFillParams sp;
        sp.axis = axis;
        // Negative cell means "domain center" (the SliceConfig default).
        sp.cell = cfg->cell < 0 ? extent[axis] / 2
                                : std::clamp(cfg->cell, 0, extent[axis] - 1);
        sp.field = static_cast<int>(cfg->field);
        sp.colormap = cfg->colormap == Colormap::Coolwarm ? 1 : 0;
        sp.scale = defaultSliceScale(cfg->field) * std::max(cfg->rangeScale, 1e-3f);
        sp.width = im.sliceW[axis];
        sp.height = im.sliceH[axis];
        note(launchSliceFill(im.slice[axis].surf, vel, rho, flags, sp, im.stream));
    }

    // ---- Q volume ---------------------------------------------------------------
    if (doQ) {
        note(im.qVol.ensureSurface(im.stream));
        if (im.qVol.surf) {
            note(launchQCriterionVolume(im.qVol.surf, vel, flags,
                                        std::max(settings.qScale, 1e-12f),
                                        im.stream));
        }
    }

    // ---- velocity volume (wind-tunnel smoke speed field) ------------------------
    if (doVel) {
        note(im.velVol.ensureSurface(im.stream));
        if (im.velVol.surf) {
            VelocityVolumeParams vp;
            vp.speedScale = std::max(settings.velocitySpeedScale, 1e-6f);
            vp.width  = im.dims.nx;
            vp.height = im.dims.ny;
            vp.depth  = im.dims.nz;
            note(launchVelocityVolume(im.velVol.surf, vel, flags, vp, im.stream));
        }
    }

    note(cudaGraphicsUnmapResources(nRes, resources, im.stream));
    return firstErr;
}

// ---------------------------------------------------------------------------
// Drawing. Order matters (plan 9.1): opaque foil mesh writes depth first,
// slices next (opaque, depth-tested), then additive depth-tested particles
// with depth writes off, and the translucent Q raycast last.
// ---------------------------------------------------------------------------

void Visualizer::drawFrame(const OrbitCamera& camera, const VizSettings& settings,
                           int viewportWidth, int viewportHeight) {
    Impl& im = *impl_;
    glViewport(0, 0, viewportWidth, viewportHeight);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // pure black: the hero isosurface
                                          // and additive particles both read
                                          // best against zero background
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (!im.initialized || viewportWidth <= 0 || viewportHeight <= 0) return;

    const float aspect = static_cast<float>(viewportWidth)
                       / static_cast<float>(viewportHeight);
    const Mat4f viewProj = camera.projMatrix(aspect) * camera.viewMatrix();
    const Vec3f eye = camera.eye();

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE); // prisms are viewed from both sides; shader
                             // flips normals toward the camera anyway

    // Clip->world inverse, computed once per frame: the raymarch passes
    // un-project the sampled scene depth into world space to occlude the
    // volume against the foil silhouette. (The camera keeps near/far private,
    // so reconstructing world position is both cleaner and angle-exact.)
    float invVP[16];
    invertMat4(viewProj.m, invVP);

    // ---- 1. foil + VG mesh (opaque, depth-writing) --------------------------
    if (settings.showFoilMesh && im.meshVertexCount > 0) {
        glUseProgram(im.progMesh);
        glUniformMatrix4fv(im.uMViewProj, 1, GL_FALSE, viewProj.m);
        glUniform3f(im.uMEye, eye.x, eye.y, eye.z);
        glBindVertexArray(im.meshVAO);
        // Wireframe option: draw edges only. Wireframe still writes depth, so
        // it occludes the volume behind its near edges (a thin cage over the
        // smoke) rather than vanishing.
        if (settings.foilWireframe) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDrawArrays(GL_TRIANGLES, 0, im.meshVertexCount);
        if (settings.foilWireframe) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    // ---- 2. slice planes (opaque, depth-writing) -----------------------------
    if (settings.showSlices) {
        glUseProgram(im.progSlice);
        glUniformMatrix4fv(im.uSViewProj, 1, GL_FALSE, viewProj.m);
        glUniform1i(im.uSTex, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindVertexArray(im.sliceVAO);

        bool axisDone[3] = {false, false, false};
        for (const SliceConfig& cfg : settings.slices) {
            const int axis = static_cast<int>(cfg.axis);
            if (!cfg.enabled || axis < 0 || axis > 2 || axisDone[axis]) continue;
            axisDone[axis] = true; // one texture per axis: first slot wins

            const float nx = static_cast<float>(im.dims.nx);
            const float ny = static_cast<float>(im.dims.ny);
            const float nz = static_cast<float>(im.dims.nz);
            const int extent[3] = {im.dims.nx, im.dims.ny, im.dims.nz};
            const float c = (cfg.cell < 0 ? extent[axis] / 2
                                          : std::clamp(cfg.cell, 0, extent[axis] - 1))
                          + 0.5f; // plane through the cell CENTER

            // Quad corners follow the kernel's texel->cell mapping
            // (particles.cuh) so the texture lands undistorted on the plane.
            float quad[20];
            if (axis == 0) { // X plane: u <-> z, v <-> y
                const float v[20] = {c, 0,  0,  0, 0,  c, 0,  nz, 1, 0,
                                     c, ny, nz, 1, 1,  c, ny, 0,  0, 1};
                std::memcpy(quad, v, sizeof v);
            } else if (axis == 1) { // Y plane: u <-> x, v <-> z
                const float v[20] = {0,  c, 0,  0, 0,  nx, c, 0,  1, 0,
                                     nx, c, nz, 1, 1,  0,  c, nz, 0, 1};
                std::memcpy(quad, v, sizeof v);
            } else { // Z plane: u <-> x, v <-> y
                const float v[20] = {0,  0,  c, 0, 0,  nx, 0,  c, 1, 0,
                                     nx, ny, c, 1, 1,  0,  ny, c, 0, 1};
                std::memcpy(quad, v, sizeof v);
            }
            glBindBuffer(GL_ARRAY_BUFFER, im.sliceVBO);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof quad, quad);
            glBindTexture(GL_TEXTURE_2D, im.sliceTex[axis]);
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
        }
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    // ---- 3. Q-criterion isosurface (opaque, depth-writing) ------------------
    // First-hit raycast drawn like solid geometry: the fragment stage writes
    // the hit's true depth, so the z-buffer resolves foil-vs-filament
    // occlusion both ways and later passes are occluded by the vortex skins.
    // Back faces rasterize (front culled) so the march still covers the
    // volume with the camera inside the domain box.
    const bool willVol = settings.showVelocityVolume && im.volAvailable
                      && im.velTex;
    const bool willQ = settings.showQRaycast && im.qAvailable && im.qTex;
    if (willQ) {
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);
        glUseProgram(im.progQ);
        glUniformMatrix4fv(im.uQViewProj, 1, GL_FALSE, viewProj.m);
        glUniform3f(im.uQDims, static_cast<float>(im.dims.nx),
                    static_cast<float>(im.dims.ny), static_cast<float>(im.dims.nz));
        glUniform3f(im.uQEye, eye.x, eye.y, eye.z);
        glUniform1f(im.uQThresh, std::clamp(settings.qThreshold, 0.005f, 0.99f));
        // Velocity coloring only when the speed volume actually exists.
        const bool qColorVel = settings.qColorByVelocity && im.velTex != 0;
        glUniform1i(im.uQColorByVel, qColorVel ? 1 : 0);
        glUniform1i(im.uQColormap, static_cast<int>(settings.qColormap));
        // Unit 0: Q volume, unit 1: speed volume.
        glUniform1i(im.uQVol, 0);
        glUniform1i(im.uQSpeedVol, 1);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, im.qTex);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_3D, qColorVel ? im.velTex : im.qTex);
        glBindVertexArray(im.cubeVAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_3D, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, 0);
        glDisable(GL_CULL_FACE);
    }

    // ---- 3.5 capture the opaque scene depth (mesh + slices + Q skins have
    // all written it) into a texture so the translucent fog below terminates
    // at solid surfaces. Particles draw later but don't write depth (additive
    // glow). Only needed when the fog pass actually runs this frame. --
    if (willVol) {
        im.captureSceneDepth(viewportWidth, viewportHeight);
    }

    // ---- 4. particles: additive, depth-TESTED against the mesh but not
    // depth-written (a million points writing depth would occlude each other
    // and flicker) — plan 9.1.
    if (settings.showParticles && im.particleCount > 0) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE); // additive accumulation
        glDepthMask(GL_FALSE);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glUseProgram(im.progParticles);
        glUniformMatrix4fv(im.uPViewProj, 1, GL_FALSE, viewProj.m);
        glUniform1f(im.uPPointSize, std::max(settings.particlePointSize, 1.0f));
        glUniform1i(im.uPColormap,
                    settings.particleColormap == Colormap::Coolwarm ? 1 : 0);
        // Brightness compensates pool density: more particles, dimmer points,
        // same integrated glow.
        const float alpha = std::clamp(
            0.28f * std::sqrt(1.0e6f / static_cast<float>(im.particleCount)),
            0.02f, 1.0f);
        glUniform1f(im.uPAlpha, alpha);
        glBindVertexArray(im.particleVAO);
        glDrawArrays(GL_POINTS, 0, im.particleCount);
        glDisable(GL_PROGRAM_POINT_SIZE);
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
    }

    // ---- 4. velocity volume (the hero wind-tunnel smoke mode) -----------------
    // Translucent over-compositing; rasterize BACK faces so the march still
    // covers the volume with the camera inside the box. Reads the depth copy
    // to occlude against the foil (fixes the foil floating over the field).
    if (willVol) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_FALSE);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);
        glUseProgram(im.progVol);
        glUniformMatrix4fv(im.uVViewProj, 1, GL_FALSE, viewProj.m);
        glUniform3f(im.uVDims, static_cast<float>(im.dims.nx),
                    static_cast<float>(im.dims.ny), static_cast<float>(im.dims.nz));
        glUniform3f(im.uVEye, eye.x, eye.y, eye.z);
        glUniform2f(im.uVViewport, static_cast<float>(viewportWidth),
                    static_cast<float>(viewportHeight));
        glUniformMatrix4fv(im.uVInvVP, 1, GL_FALSE, invVP);
        glUniform1i(im.uVColormap, static_cast<int>(settings.volumeColormap));
        glUniform1f(im.uVSlowOpacity, std::clamp(settings.slowAirOpacity, 0.0f, 1.0f));
        glUniform1f(im.uVDensity, std::clamp(settings.volumeDensity, 0.0f, 1.0f));
        // Freestream baseline in the SAME normalized units the volume stores
        // (speed / speedScale), so the shader's disturbance is measured from
        // the actual inflow rather than a hardcoded level.
        glUniform1f(im.uVFreestream,
                    std::max(settings.freestreamLatticeSpeed, 1e-4f)
                        / std::max(settings.velocitySpeedScale, 1e-6f));
        // Unit 0: speed volume, unit 1: scene depth.
        glUniform1i(im.uVVol, 0);
        glUniform1i(im.uVDepth, 1);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, im.velTex);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, im.depthTex);
        glBindVertexArray(im.cubeVAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, 0);
        glDisable(GL_CULL_FACE);
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
    }

    glBindVertexArray(0);
    glUseProgram(0);
}

// ---------------------------------------------------------------------------
// Pool resize: the ONLY sanctioned re-registration point — the GL buffers
// themselves are recreated, so each new buffer gets its own single
// registration. Old pool stays live until the new one fully succeeds.
// ---------------------------------------------------------------------------

bool Visualizer::resizeParticlePool(int newCount, std::string* error) {
    Impl& im = *impl_;
    if (!im.initialized) {
        if (error) *error = "resizeParticlePool before init";
        return false;
    }
    if (newCount <= 0) {
        if (error) *error = "particle count must be positive";
        return false;
    }
    if (newCount == im.particleCount) return true;

    // Stash the old pool handles so failure can roll back untouched.
    const GLuint oldVAO = im.particleVAO, oldPos = im.posVBO, oldKey = im.keyVBO;
    cudaGraphicsResource* oldPosRes = im.posRes;
    cudaGraphicsResource* oldKeyRes = im.keyRes;
    const int oldCount = im.particleCount;
    im.particleVAO = im.posVBO = im.keyVBO = 0;
    im.posRes = im.keyRes = nullptr;

    if (!im.createParticlePool(newCount, error)) {
        // Roll back: drop whatever half-built state exists, restore the old pool.
        im.destroyParticlePool();
        im.particleVAO = oldVAO;
        im.posVBO = oldPos;
        im.keyVBO = oldKey;
        im.posRes = oldPosRes;
        im.keyRes = oldKeyRes;
        im.particleCount = oldCount;
        return false;
    }

    // New pool is live; release the old one (unregister, then delete buffers).
    if (oldPosRes) cudaGraphicsUnregisterResource(oldPosRes);
    if (oldKeyRes) cudaGraphicsUnregisterResource(oldKeyRes);
    GLuint bufs[2] = {oldPos, oldKey};
    glDeleteBuffers(2, bufs);
    GLuint vao = oldVAO;
    if (vao) glDeleteVertexArrays(1, &vao);
    return true;
}

// ---------------------------------------------------------------------------
// Screenshot: back-buffer readback -> minimal valid PNG (plan 9.1 demands a
// real PNG writer, no BMP rename tricks).
// ---------------------------------------------------------------------------

bool Visualizer::screenshotPNG(const std::filesystem::path& path,
                               std::string* error) {
    // Read the current back buffer; GL returns rows bottom-up, PNG wants
    // top-down, so flip while packing.
    GLint vp[4] = {};
    glGetIntegerv(GL_VIEWPORT, vp);
    const int w = vp[2], h = vp[3];
    if (w <= 0 || h <= 0) {
        if (error) *error = "viewport has zero size";
        return false;
    }
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(w) * h * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    std::vector<std::uint8_t> flipped(pixels.size());
    const std::size_t stride = static_cast<std::size_t>(w) * 3;
    for (int y = 0; y < h; ++y) {
        std::copy_n(pixels.data() + static_cast<std::size_t>(y) * stride, stride,
                    flipped.data() + static_cast<std::size_t>(h - 1 - y) * stride);
    }

    const std::vector<std::uint8_t> png = encodePNG(flipped.data(), w, h);

    std::error_code ec;
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path(), ec);
    }
    std::FILE* fp = nullptr;
#ifdef _WIN32
    _wfopen_s(&fp, path.wstring().c_str(), L"wb");
#else
    fp = std::fopen(path.string().c_str(), "wb");
#endif
    if (!fp) {
        // pathToUtf8, not string(): the error text must not itself throw on
        // a non-ASCII path (the very case most likely to hit this branch).
        if (error)
            *error = "cannot open " + platform::pathToUtf8(path) + " for writing";
        return false;
    }
    const bool ok = std::fwrite(png.data(), 1, png.size(), fp) == png.size();
    std::fclose(fp);
    if (!ok && error) *error = "short write to " + platform::pathToUtf8(path);
    return ok;
}

} // namespace foilcfd
