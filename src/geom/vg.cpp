// Vortex generator placement and voxelization (plan section 6): vanes as thin
// boxes 1-2 cells thick yawed +/-beta about the local surface normal with the
// root following the suction surface, counter-rotating pairs, co-rotating
// arrays, ramp wedges, and the Lin-2002 placement guidance helpers (Mission
// statement; sources in docs/CITATIONS.md — Lin 2002, Strausak 2021).
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include "vg.h"

#include <algorithm>
#include <cmath>

namespace foilcfd {
namespace {

constexpr float kPi = 3.14159265358979f;

// Raw flag bytes for the stamping loops (field stores uint8, not the enum).
constexpr std::uint8_t kFluid = static_cast<std::uint8_t>(CellFlag::Fluid);
constexpr std::uint8_t kSolid = static_cast<std::uint8_t>(CellFlag::Solid);

/// @brief Unpadded linear index x + nx*(y + ny*z) (z already in range).
inline long long cellIndex(int x, int y, int z, const GridDims& dims) {
    return static_cast<long long>(x)
         + static_cast<long long>(dims.nx)
               * (static_cast<long long>(y)
                  + static_cast<long long>(dims.ny) * static_cast<long long>(z));
}

/// @brief Mark one lattice cell Solid if it is currently Fluid. z wraps
/// periodically (the spanwise BC is periodic, plan 4.2, so a VG array sliding
/// off one z face must reappear on the other); x/y outside the interior are
/// skipped so domain-face flags are never overwritten.
inline void stampCell(std::vector<std::uint8_t>& flags, const GridDims& dims,
                      int x, int y, int z) {
    if (x < 1 || x > dims.nx - 2 || y < 1 || y > dims.ny - 2) return;
    const int zw = ((z % dims.nz) + dims.nz) % dims.nz; // periodic wrap
    const std::size_t idx = static_cast<std::size_t>(x)
                          + static_cast<std::size_t>(dims.nx)
                          * (static_cast<std::size_t>(y)
                             + static_cast<std::size_t>(dims.ny)
                             * static_cast<std::size_t>(zw));
    if (flags[idx] == kFluid) flags[idx] = kSolid;
}

// ===========================================================================
// Blade-profile geometry. A profile is defined by two per-slice functions of
// the fractional chordwise position t in [0,1] along the blade (t=0 at the
// upstream end, t=1 downstream): the local TOP height (fraction of hCells) and
// the local HALF-THICKNESS (fraction of the nominal halfW). The rasterizer
// (stampVane) walks t and fills [-embed, top*hCells] x [-thick*halfW, +thick]
// per slice — so a profile is just a pair of shape curves, nothing else moves.
// ===========================================================================

/// @brief Local top-height as a FRACTION of hCells at fractional length t.
/// Rectangle is flat; Delta/Wedge ramp 0->1; Trapezoid ramps then holds a flat
/// top over the central `taper` fraction; Parabolic is an elliptical cap.
inline float profileTopFrac(VGProfile profile, float t, float taper) {
    switch (profile) {
    case VGProfile::Rectangle:
    case VGProfile::AirfoilSection:
    case VGProfile::CustomStl: // unused (stl path), keep full height for slabs
        return 1.0f;
    case VGProfile::Delta:
    case VGProfile::Wedge:
        // Right triangle: height rises linearly from the upstream point to full
        // height at the downstream end (the classic delta / Wheeler ramp).
        return t;
    case VGProfile::Trapezoid: {
        // Clipped delta: ramp up over the front ramp-fraction, then a flat top.
        // taper in [0,1] is the FLAT-TOP fraction; (1-taper) is the front ramp.
        const float flat = std::clamp(taper, 0.0f, 1.0f);
        const float ramp = std::max(1e-3f, 1.0f - flat);
        return std::clamp(t / ramp, 0.0f, 1.0f);
    }
    case VGProfile::Parabolic: {
        // Elliptical cap: tallest at mid-length, tapering smoothly to the ends.
        // top = sqrt(1 - (2t-1)^2) maps t=0.5 -> 1, t=0 or 1 -> 0.
        const float u = 2.0f * t - 1.0f;
        return std::sqrt(std::max(0.0f, 1.0f - u * u));
    }
    }
    return 1.0f;
}

/// @brief Local half-thickness as a FRACTION of halfW at fractional length t.
/// Only AirfoilSection tapers thickness — a NACA 4-digit symmetric distribution
/// so the blade has a rounded leading edge and a sharp trailing edge (a thin
/// streamlined cross-section instead of a flat plate). All other profiles keep
/// the constant plate thickness (1.0).
inline float profileThickFrac(VGProfile profile, float t) {
    if (profile != VGProfile::AirfoilSection) return 1.0f;
    // NACA symmetric half-thickness, normalized so its MAX (at t~0.3) is 1.0.
    // yt(x) = 5*tmax*(0.2969 sqrt(x) - 0.1260 x - 0.3516 x^2 + 0.2843 x^3
    //                 - 0.1015 x^4); the tmax factor cancels in the ratio.
    const float x = std::clamp(t, 0.0f, 1.0f);
    const float yt = 0.2969f * std::sqrt(x) - 0.1260f * x - 0.3516f * x * x
                   + 0.2843f * x * x * x - 0.1015f * x * x * x * x;
    // The bracket peaks at ~0.10003 (at x~0.30); divide so the ratio reaches 1
    // at the blade's thickest point (NOT 0.1015, which is just the x^4 term).
    constexpr float kNacaPeak = 0.10003f;
    return std::clamp(yt / kNacaPeak, 0.0f, 1.0f);
}

/// @brief Stamp one vane (or ramp wedge) into the flag field.
///
/// The device is sampled as a parametric slab and rasterized by dense
/// sub-cell sampling (0.45-cell steps in every direction) — analytic
/// stamping rather than cell-membership testing, because thin plates tested
/// per-cell can vanish entirely when thinner than a cell (see the FluidX3D
/// finding recorded in docs/CITATIONS.md "house findings").
///
/// Frame construction per chordwise sample s along the vane axis:
///   - The ROOT follows the airfoil surface (plan 6.1): each s maps back to a
///     chordwise station, the surface frame is re-queried there, and the
///     column rises along the LOCAL outward normal. The root is additionally
///     embedded 1.5 cells INTO the foil so curvature can never open a gap
///     under the vane.
///   - Yaw beta rotates the vane axis about the local surface normal: the
///     axis is cos(beta)*tangent + sin(beta)*spanwise, so positive beta
///     sweeps the downstream end of the vane toward +z.
///   - AoA is applied exactly like the airfoil voxelizer (rotation about the
///     quarter-chord by -aoa), so vanes stay glued to the surface at any AoA.
///
/// @param airfoil  Section supplying the surface frames.
/// @param aoaRad   Angle of attack in radians.
/// @param layout   Grid placement/scale.
/// @param flags    Flag field, modified in place (Fluid -> Solid only).
/// @param x_c      Chordwise station of the vane CENTER.
/// @param zCenter  Spanwise center of the vane in lattice cells.
/// @param betaRad  Yaw about the surface normal (signed).
/// @param hCells   Device height in cells (>= 1).
/// @param lenCells Device length along its own axis in cells.
/// @param halfW    Nominal half-thickness (vane) or half-width (wedge) in cells.
/// @param profile  Blade outline/section (Rectangle/Delta/Trapezoid/Parabolic/
///                 AirfoilSection/Wedge). The per-slice top-height and thickness
///                 follow this; placement/yaw/AoA are profile-independent.
/// @param taper    Trapezoid flat-top fraction (ignored by other profiles).
void stampVane(const AirfoilGeometry& airfoil, float aoaRad,
               const DomainLayout& layout, std::vector<std::uint8_t>& flags,
               float x_c, float zCenter, float betaRad, float hCells,
               float lenCells, float halfW, VGProfile profile, float taper,
               std::vector<VaneSlab>* slabsOut = nullptr) {
    // Center frame gives the reference tangent used to convert axis distance
    // into a chordwise-station advance (vanes are ~3h long, a few % chord —
    // treating the tangent's x-projection as constant over that span is fine).
    const SurfaceFrame center = surfaceFrameAt(airfoil, x_c, /*upper=*/true);
    if (!center.valid) return;

    // Build the analytic OBB descriptor from the CENTER slice (one slab per
    // vane is enough — the side faces are near-flat over a few-% chord). The
    // axes match the per-slice frame below; the box spans the embedded root
    // through the crest in height, the full length, and the half-thickness.
    if (slabsOut) {
        const float chordC = static_cast<float>(layout.chordCells);
        const Vec2f cpr = rotated(center.point - Vec2f(0.25f, 0.0f), -aoaRad);
        const Vec2f cnr = rotated(center.normal, -aoaRad);
        const Vec2f ctr = rotated(center.tangent, -aoaRad);
        const Vec3f cRoot(layout.anchorX() + cpr.x * chordC,
                          layout.anchorY() + cpr.y * chordC, zCenter);
        const Vec3f cn3(cnr.x, cnr.y, 0.0f);
        const Vec3f cd3(std::cos(betaRad) * ctr.x, std::cos(betaRad) * ctr.y,
                        std::sin(betaRad));
        const Vec3f cb3 = normalized(cross(cn3, cd3));
        // Height spans [-kRootEmbed, hCells]; its mid-point lifts the center.
        constexpr float kRootEmbedSlab = 1.5f;
        const float halfHt = 0.5f * (hCells + kRootEmbedSlab);
        const float midHt  = 0.5f * (hCells - kRootEmbedSlab);
        VaneSlab slab;
        slab.center     = cRoot + cn3 * midHt;
        slab.nAxis      = normalized(cn3);
        slab.dAxis      = normalized(cd3);
        slab.bAxis      = cb3;
        slab.halfHeight = halfHt;
        slab.halfLength = 0.5f * lenCells;
        slab.halfThick  = halfW;
        slabsOut->push_back(slab);
    }

    const float cosB = std::cos(betaRad);
    const float sinB = std::sin(betaRad);
    const float chord = static_cast<float>(layout.chordCells);
    const float ax = layout.anchorX();
    const float ay = layout.anchorY();
    const Vec2f quarterChord(0.25f, 0.0f);

    // 0.45-cell sampling: strictly below half a cell in every direction, so
    // no cell the slab passes through can be skipped over.
    constexpr float kStep = 0.45f;
    constexpr float kRootEmbed = 1.5f; // cells buried below the surface

    for (float s = -0.5f * lenCells; s <= 0.5f * lenCells + 1e-4f; s += kStep) {
        // Chordwise station of this slice: the in-surface-plane advance is
        // s*cos(beta) cells along the tangent, whose x-projection converts
        // cells to delta(x/c) via the chord length.
        const float station = std::clamp(
            x_c + (s * cosB * center.tangent.x) / chord, 0.0f, 1.0f);
        const SurfaceFrame f = surfaceFrameAt(airfoil, station, /*upper=*/true);
        if (!f.valid) continue;

        // Apply AoA exactly as voxelizeAirfoil does: the surface POINT
        // rotates about the quarter-chord; direction vectors rotate only.
        const Vec2f pr = rotated(f.point - quarterChord, -aoaRad);
        const Vec2f nr = rotated(f.normal, -aoaRad);
        const Vec2f tr = rotated(f.tangent, -aoaRad);

        // Lift the 2D frame into lattice 3-space; z advances with the yaw.
        const Vec3f root(ax + pr.x * chord, ay + pr.y * chord,
                         zCenter + s * sinB);
        const Vec3f n3(nr.x, nr.y, 0.0f);                       // wall-normal
        const Vec3f d3(cosB * tr.x, cosB * tr.y, sinB);          // vane axis
        const Vec3f b3 = normalized(cross(n3, d3));              // thickness

        // Per-slice shape from the blade profile: fractional position t along
        // the length (0 upstream -> 1 downstream) drives the local top-height
        // and (AirfoilSection only) the local thickness.
        const float t   = (s + 0.5f * lenCells) / std::max(lenCells, 1e-3f);
        const float top = hCells * profileTopFrac(profile, t, taper);
        // Thickness floors at half a cell so a tapered nose never aliases away
        // entirely (analytic-stamping rule); the embedded root always stamps
        // full thickness so the seat stays watertight.
        const float wHalf = std::max(0.5f, halfW * profileThickFrac(profile, t));

        // Rasterize the slice column: height u (embedded root included) by
        // thickness w. Below the surface the column keeps full thickness so the
        // seat seals; above it follows the (possibly tapered) profile thickness.
        for (float u = -kRootEmbed; u <= top + 1e-4f; u += kStep) {
            const float colHalf = (u < 0.0f) ? std::max(0.5f, halfW) : wHalf;
            for (float w = -colHalf; w <= colHalf + 1e-4f; w += kStep) {
                const Vec3f p = root + n3 * u + b3 * w;
                stampCell(flags, layout.dims,
                          static_cast<int>(std::floor(p.x)),
                          static_cast<int>(std::floor(p.y)),
                          static_cast<int>(std::floor(p.z)));
            }
        }
    }
}

/// @brief Surface-sample one triangle into the flag field at sub-cell density.
///
/// Walks the triangle in barycentric space at ~0.45-cell steps (the same
/// strictly-below-half-a-cell guarantee stampVane uses, so a slab thinner than
/// a cell can never be skipped over) and marks the cell each sample lands in
/// Solid. Step counts are derived from the triangle's longest edge in cells, so
/// big facets get more samples and tiny ones stay cheap. This SHELL-stamps the
/// mesh surface — which is exactly what an LBM bounce-back wall needs; the
/// interior staying Fluid is correct and matches stampVane's plate stamping.
inline void rasterizeTriangleSurface(std::vector<std::uint8_t>& flags,
                                     const GridDims& dims, const Vec3f& a,
                                     const Vec3f& b, const Vec3f& c) {
    constexpr float kStep = 0.45f;
    // Sample density: enough that adjacent samples are < kStep cells apart along
    // each edge. Longest edge in cells / kStep, with a floor of 1 subdivision.
    const float e0 = length(b - a);
    const float e1 = length(c - a);
    const int nU = std::max(1, static_cast<int>(std::ceil(std::max(e0, e1) / kStep)));
    // Barycentric sweep: u along (b-a), v along (c-a), u+v <= 1 stays inside.
    for (int iu = 0; iu <= nU; ++iu) {
        const float u = static_cast<float>(iu) / static_cast<float>(nU);
        const int nV = std::max(1, static_cast<int>(std::ceil((1.0f - u)
                                   * std::max(e0, length(c - b)) / kStep)));
        for (int iv = 0; iv <= nV; ++iv) {
            const float v = (static_cast<float>(iv) / static_cast<float>(nV))
                            * (1.0f - u);
            const Vec3f p = a + (b - a) * u + (c - a) * v;
            stampCell(flags, dims, static_cast<int>(std::floor(p.x)),
                      static_cast<int>(std::floor(p.y)),
                      static_cast<int>(std::floor(p.z)));
        }
    }
}

/// @brief Stamp one CustomStl VG unit: seat the unit-normalized mesh on the
/// suction surface at x_c, fix its orientation (flip + 90-deg steps), scale it
/// to the device height, map it into the seated surface frame, and shell-stamp
/// every transformed triangle.
///
/// The seating frame is built IDENTICALLY to stampVane (vg.cpp center-slice
/// block) so an STL vane sits and rotates exactly like a parametric one at any
/// AoA, with beta yaw already baked into the d3/b3 axes. The stored mesh is
/// unit longest-extent centered at origin (normalizeVgMeshUnit), so scaling by
/// hCells re-seats it correctly at ANY level's chord resolution with no
/// per-level state — the same property that lets parametric hCells grow with
/// the patch.
///
/// Canonical mesh axis convention (after import normalization): x = thickness
/// (-> b3), y = up/height (-> n3 wall-normal), z = length/chord (-> d3 vane
/// axis). The import axis-preset dropdown lets the user pick which file axes map
/// to these; flip/rotSteps fix the leftover wrong-facing cases here.
///
/// @param unitMesh  Unit-normalized VG mesh (App::vgMeshes[stlMeshId]).
/// @param airfoil   Section supplying the surface frame.
/// @param aoaRad    Angle of attack in radians.
/// @param layout    Grid placement/scale.
/// @param flags     Flag field, modified in place (Fluid -> Solid only).
/// @param x_c       Chordwise station of the unit center.
/// @param zCenter   Spanwise center of the unit in lattice cells.
/// @param betaRad   Yaw about the surface normal (signed).
/// @param hCells    Device height in cells (>= 1) — the mesh's overall scale.
/// @param flip      Negate the canonical up-axis (fixes upside-down meshes).
/// @param rotSteps  Extra 90-deg yaw steps about the canonical up-axis (0..3).
/// @param mirror    Reflect the blade across its chordwise-vertical plane (negate
///                  the thickness axis) so the +z blade of a counter-rotating
///                  pair is a true mirror image of the -z blade, even for an
///                  asymmetric mesh. Combine with a flipped betaRad for the pair.
void stampStlVane(const StlMesh& unitMesh, const AirfoilGeometry& airfoil,
                  float aoaRad, const DomainLayout& layout,
                  std::vector<std::uint8_t>& flags, float x_c, float zCenter,
                  float betaRad, float hCells, bool flip, int rotSteps,
                  bool mirror) {
    const SurfaceFrame center = surfaceFrameAt(airfoil, x_c, /*upper=*/true);
    if (!center.valid || unitMesh.triangles.empty()) return;

    // Seated surface frame at the unit center — same construction as stampVane.
    const float cosB = std::cos(betaRad);
    const float sinB = std::sin(betaRad);
    const float chord = static_cast<float>(layout.chordCells);
    const float ax = layout.anchorX();
    const float ay = layout.anchorY();
    const Vec2f quarterChord(0.25f, 0.0f);
    constexpr float kRootEmbed = 1.5f; // bury the root like the parametric vane

    const Vec2f pr = rotated(center.point - quarterChord, -aoaRad);
    const Vec2f nr = rotated(center.normal, -aoaRad);
    const Vec2f tr = rotated(center.tangent, -aoaRad);
    // Surface point (the vane BASE seats here); the embed is applied per-vertex
    // in the up direction below, not baked into the root, so the crest still
    // reaches surface + hCells exactly like the parametric column.
    const Vec3f n3 = normalized(Vec3f(nr.x, nr.y, 0.0f));        // wall-normal
    const Vec3f root = Vec3f(ax + pr.x * chord, ay + pr.y * chord, zCenter);
    const Vec3f d3 = normalized(Vec3f(cosB * tr.x, cosB * tr.y, sinB)); // length
    const Vec3f b3 = normalized(cross(n3, d3));                  // thickness

    // Local orientation fix applied in canonical mesh space BEFORE the frame
    // map: rotSteps*90 deg about the up-axis (y), then optional upside-down
    // flip. Precompute the in-plane (x,z) rotation for the chosen step.
    const int steps = ((rotSteps % 4) + 4) % 4;
    const float rotRad = static_cast<float>(steps) * (kPi * 0.5f);
    const float rc = std::cos(rotRad), rs = std::sin(rotRad);

    // Transform + shell-stamp every triangle. The canonical mesh has its BASE at
    // y = 0 and UP extent = 1 (normalizeVgMeshUnit), so vy in [0,1] is the
    // fractional height up the device.
    auto toWorld = [&](const Vec3f& vert) {
        float vx = vert.x, vy = vert.y, vz = vert.z;
        // Mirror across the chordwise-vertical plane (negate thickness) BEFORE
        // yaw so an asymmetric mesh reflects into a true mirror image for the
        // counter-rotating pair's far blade.
        if (mirror) { vx = -vx; }
        // Upside-down fix mirrors about mid-height so the mesh stays in [0,1]
        // (a naive negation would push it below the surface).
        if (flip) { vy = 1.0f - vy; }
        // Yaw about up-axis y: rotate the (x,z) plane.
        const float rx = vx * rc + vz * rs;
        const float rz = -vx * rs + vz * rc;
        vx = rx; vz = rz;
        // Map canonical axes -> seated frame. Thickness (x->b3) and length
        // (z->d3) scale by hCells (the device's overall size). Height maps the
        // base to -kRootEmbed (buried) and the crest to +hCells, reproducing the
        // parametric vane's embedded skirt + full protruding height.
        const float up = -kRootEmbed + vy * (hCells + kRootEmbed);
        return root + b3 * (vx * hCells) + n3 * up + d3 * (vz * hCells);
    };
    for (const StlTriangle& t : unitMesh.triangles) {
        rasterizeTriangleSurface(flags, layout.dims, toWorld(t.v0),
                                 toWorld(t.v1), toWorld(t.v2));
    }
}

} // namespace

SurfaceFrame vgPlacementFrame(const AirfoilGeometry& airfoil,
                              const VGParams& params) {
    // VGs mount on the suction surface only in v1 (plan 6.1).
    return surfaceFrameAt(airfoil, params.x_c, /*upper=*/true);
}

void voxelizeVG(const VGParams& vg, const AirfoilGeometry& airfoil,
                float aoa_deg, const DomainLayout& layout,
                std::vector<std::uint8_t>& flags,
                std::vector<VaneSlab>* slabsOut,
                const std::vector<StlMesh>* unitVgMeshes) {
    // Disabled VGs are skipped entirely — they contribute no solid cells.
    if (!vg.enabled) return;
    if (!airfoil.isValid() || layout.dims.nz < 1) return;

    // Device dimensions in lattice cells. The 1-cell floor mirrors the
    // analytic-stamping rule (solids must never be thinner than one cell or
    // they alias away); the under-resolution UI warning (vgUnderResolved)
    // still fires below 8 cells — we stamp SOMETHING honest, the guard tells
    // the user it is too coarse to trust (plan 6.1).
    const float h   = std::max(1.0f, vg.height_c * static_cast<float>(layout.chordCells));
    const float len = std::max(1.0f, vg.length_h * h);
    // Vanes are "thin boxes 1-2 cells thick" (plan 6.1): 1 cell for small
    // devices, growing to 2 cells once the vane is tall enough to afford it.
    const float thick = std::clamp(h / 6.0f, 1.0f, 2.0f);
    const float beta = vg.beta_deg * kPi / 180.0f;
    const float aoaRad = aoa_deg * kPi / 180.0f;
    const float pitch = vg.pitch_c * static_cast<float>(layout.chordCells);
    const int count = std::max(1, vg.count);
    const float x_c = std::clamp(vg.x_c, 0.0f, 1.0f);
    // Units are centered on the mid-span; the periodic z wrap in stampCell
    // handles arrays wider than the domain gracefully.
    const float zMid = 0.5f * static_cast<float>(layout.dims.nz);

    // Arrangement and blade shape are independent: resolve the real ones from
    // the (possibly legacy) type/profile fields.
    const VGType arrangement  = effectiveArrangement(vg);
    const VGProfile profile   = effectiveProfile(vg);

    // Resolve the CustomStl mesh up-front when the blade is a mesh: an
    // unresolved id (no mesh list, or an out-of-range / empty handle)
    // contributes nothing — the caller detects this via warnIfVgMeshMissing and
    // logs it once, so we just bail quietly.
    const StlMesh* stlMesh = nullptr;
    if (profile == VGProfile::CustomStl) {
        if (!unitVgMeshes || vg.stlMeshId < 0
            || vg.stlMeshId >= static_cast<int>(unitVgMeshes->size())
            || (*unitVgMeshes)[vg.stlMeshId].triangles.empty()) {
            return;
        }
        stlMesh = &(*unitVgMeshes)[vg.stlMeshId];
    }

    // Per-profile nominal half-thickness:
    //   Wedge          -> half the device height (a chunky ramp prism),
    //   AirfoilSection -> thicknessRatio * blade length (a real streamlined
    //                     section whose NACA distribution peaks at this value),
    //   everything else-> the thin 1-2 cell flat plate.
    float halfW;
    if (profile == VGProfile::Wedge) {
        halfW = std::max(0.5f, 0.5f * h);
    } else if (profile == VGProfile::AirfoilSection) {
        halfW = std::max(0.5f, 0.5f * std::clamp(vg.thicknessRatio, 0.02f, 0.4f)
                                    * len);
    } else {
        halfW = 0.5f * thick;
    }

    // Stamp ONE blade of the configured shape at spanwise center zc, yaw
    // betaRad, optionally mirror-imaged (the far blade of a counter-rotating
    // pair). The shape choice (parametric profile vs. user mesh) is orthogonal
    // to the arrangement, so any arrangement composes with any blade here.
    auto stampBlade = [&](float zc, float betaRad, bool mirror) {
        if (profile == VGProfile::CustomStl) {
            // Mesh blade: no VaneSlab (a single OBB can't model a triangle
            // soup, so q-LIBB falls back to plain bounce-back — documented).
            stampStlVane(*stlMesh, airfoil, aoaRad, layout, flags, x_c, zc,
                         betaRad, h, vg.stlFlip, vg.stlRotSteps, mirror);
        } else {
            // Parametric plate/wedge: the profile drives per-slice top-height
            // and thickness. (Mirror is implicit in the flipped betaRad for the
            // parametric pair — its plate is symmetric about its own axis.)
            stampVane(airfoil, aoaRad, layout, flags, x_c, zc, betaRad, h, len,
                      halfW, profile, vg.taper, slabsOut);
        }
    };

    for (int i = 0; i < count; ++i) {
        const float zc = zMid
                       + (static_cast<float>(i) - 0.5f * static_cast<float>(count - 1))
                       * pitch;
        switch (arrangement) {
        case VGType::SingleVane:
        case VGType::CoRotatingArray:
            // One blade per unit, all yawed the same way — a single vane is
            // just a co-rotating array of count 1.
            stampBlade(zc, beta, /*mirror=*/false);
            break;
        case VGType::CounterRotatingPair: {
            // Two mirror-image blades per unit, centers gap_h device heights
            // apart. commonFlowDown=true toes the trailing edges IN (the
            // flight-proven Strausak arrangement); false toes them out. The far
            // (+z) blade is yawed the opposite way AND geometry-mirrored so an
            // asymmetric STL blade forms a true mirror pair.
            const float halfGap = 0.5f * vg.gap_h * h;
            const float betaNear = vg.commonFlowDown ? beta : -beta;
            stampBlade(zc - halfGap, betaNear, /*mirror=*/false);
            stampBlade(zc + halfGap, -betaNear, /*mirror=*/true);
            break;
        }
        default: // Ramp / CustomStl legacy values already mapped to Single above
            stampBlade(zc, beta, /*mirror=*/false);
            break;
        }
    }
}

std::vector<std::uint8_t> buildFlagsWithVGs(
    const std::vector<VGParams>& vgs, const AirfoilGeometry& airfoil,
    float aoa_deg, const DomainLayout& layout,
    const std::vector<std::uint8_t>& cleanFoilFlags,
    std::vector<VaneSlab>* slabsOut,
    const std::vector<StlMesh>* unitVgMeshes) {
    // Plan 6.2 flow: copy the cached clean mask, OR every VG in, run the TE
    // closure once at the end so vane roots get the same single-cell-gap
    // sealing the foil TE does. The clean flags are never modified — they are
    // the warm-start cache key's geometry half.
    std::vector<std::uint8_t> flags = cleanFoilFlags;
    for (const VGParams& vg : vgs) {
        voxelizeVG(vg, airfoil, aoa_deg, layout, flags, slabsOut, unitVgMeshes);
    }
    closeTrailingEdgeGaps(layout.dims, flags);
    return flags;
}

// ===========================================================================
// Interpolated bounce-back (q-LIBB) link list construction.
// ===========================================================================

namespace {

/// @brief Entry parameter t in [0,1] where the ray origin + t*dir first enters
/// the oriented box @p slab, or -1 if it misses within the segment. Standard
/// slab method projected onto the OBB's three orthonormal axes.
float rayOBBEntry(const Vec3f& origin, const Vec3f& dir, const VaneSlab& slab) {
    // Vector from the box center to the ray origin, in world cells.
    const Vec3f p = origin - slab.center;
    // Project origin and direction onto each box axis -> three 1D slab tests.
    const Vec3f axes[3]   = {slab.nAxis, slab.dAxis, slab.bAxis};
    const float halfs[3]  = {slab.halfHeight, slab.halfLength, slab.halfThick};
    float tMin = 0.0f, tMax = 1.0f; // clamp to the link segment [0,1]
    for (int a = 0; a < 3; ++a) {
        const float e = dot(p, axes[a]);    // origin offset along this axis
        const float f = dot(dir, axes[a]);  // direction component
        const float h = halfs[a];
        if (std::fabs(f) < 1e-7f) {
            // Ray parallel to this slab: miss unless the origin is between.
            if (-e - h > 0.0f || -e + h < 0.0f) return -1.0f;
        } else {
            float t1 = (-e - h) / f;
            float t2 = (-e + h) / f;
            if (t1 > t2) std::swap(t1, t2);
            tMin = std::max(tMin, t1);
            tMax = std::min(tMax, t2);
            if (tMin > tMax) return -1.0f;
        }
    }
    return tMin; // first entry into the box along the segment
}

} // namespace

QLinkList buildVaneQLinks(const GridDims& dims,
                          const std::vector<std::uint8_t>& activeFlags,
                          const std::vector<std::uint8_t>& cleanFlags,
                          const std::vector<VaneSlab>& slabs) {
    QLinkList out;
    if (slabs.empty() || activeFlags.size() != cleanFlags.size()) return out;

    const std::uint8_t kInterface =
        static_cast<std::uint8_t>(CellFlag::Interface);

    // True when (x,y,z) (z wrapped) is Solid in the given field.
    auto solidAt = [&](const std::vector<std::uint8_t>& f, int x, int y, int z) {
        if (x < 0 || x >= dims.nx || y < 0 || y >= dims.ny) return false;
        const int zw = ((z % dims.nz) + dims.nz) % dims.nz;
        return f[cellIndex(x, y, zw, dims)] == kSolid;
    };
    auto flagAt = [&](int x, int y, int z) -> std::uint8_t {
        if (x < 0 || x >= dims.nx || y < 0 || y >= dims.ny) return kSolid;
        const int zw = ((z % dims.nz) + dims.nz) % dims.nz;
        return activeFlags[cellIndex(x, y, zw, dims)];
    };

    for (int z = 0; z < dims.nz; ++z) {
        for (int y = 0; y < dims.ny; ++y) {
            for (int x = 0; x < dims.nx; ++x) {
                const long long cell = cellIndex(x, y, z, dims);
                if (activeFlags[cell] != kFluid) continue;

                for (int q = 1; q < kQ; ++q) {
                    // Pull neighbor (x - c_q): the device kernel bounces here.
                    const int sx = x - kCx[q];
                    const int sy = y - kCy[q];
                    const int sz = z - kCz[q];
                    // Must be VANE solid: Solid live, Fluid in the clean foil.
                    if (!solidAt(activeFlags, sx, sy, sz)) continue;
                    if (solidAt(cleanFlags, sx, sy, sz)) continue; // foil, skip

                    // Ray from THIS fluid node toward the vane neighbor. dir is
                    // the OUTGOING lattice direction (-c_q in the pull picture),
                    // pointing fluid -> solid, so q is measured from the fluid.
                    const Vec3f origin(static_cast<float>(x) + 0.5f,
                                       static_cast<float>(y) + 0.5f,
                                       static_cast<float>(z) + 0.5f);
                    const Vec3f dir(static_cast<float>(-kCx[q]),
                                    static_cast<float>(-kCy[q]),
                                    static_cast<float>(-kCz[q]));
                    // Intersect against every slab; take the nearest entry.
                    float qFrac = -1.0f;
                    for (const VaneSlab& slab : slabs) {
                        const float t = rayOBBEntry(origin, dir, slab);
                        if (t >= 0.0f && (qFrac < 0.0f || t < qFrac)) qFrac = t;
                    }
                    if (qFrac < 0.0f) continue; // no analytic hit -> plain BB

                    // Clamp guard: outside [kQLibbMin, 1] keep plain BB.
                    if (qFrac < kQLibbMin || qFrac > 1.0f) continue;

                    // The wall sits on the link between this fluid node and the
                    // solid neighbor at (x - c_q). The SECOND fluid node, one
                    // step further from the wall (the q<1/2 Bouzidi branch needs
                    // it), is therefore on the OPPOSITE side: x_ff = x + c_q.
                    const int ffx = x + kCx[q];
                    const int ffy = y + kCy[q];
                    const int ffz = z + kCz[q];
                    const std::uint8_t ffFlag = flagAt(ffx, ffy, ffz);
                    if (ffFlag == kInterface) continue; // stale source, skip
                    const std::uint8_t ffIsFluid =
                        (ffFlag == kFluid) ? 1u : 0u;

                    out.cellIdx.push_back(cell);
                    out.dir.push_back(static_cast<std::uint8_t>(q));
                    out.ffFluid.push_back(ffIsFluid);
                    out.q.push_back(qFrac);
                }
            }
        }
    }
    return out;
}

void densifyQLinks(const QLinkList& links, long long ncells,
                   std::vector<std::uint8_t>& qFrac,
                   std::vector<std::uint32_t>& ffMask) {
    qFrac.assign(static_cast<std::size_t>(ncells) * kQ, 0);
    ffMask.assign(static_cast<std::size_t>(ncells), 0u);
    for (std::size_t i = 0; i < links.size(); ++i) {
        const long long cell = links.cellIdx[i];
        const int q = links.dir[i];
        // Quantize q in (0,1] to a non-zero byte (1..255); 0 stays "no q-link".
        const float qc = std::clamp(links.q[i], kQLibbMin, 1.0f);
        std::uint8_t b = static_cast<std::uint8_t>(
            std::lround(qc * 255.0f));
        if (b == 0) b = 1; // never collide with the "no link" sentinel
        qFrac[static_cast<std::size_t>(cell) * kQ + q] = b;
        if (links.ffFluid[i])
            ffMask[static_cast<std::size_t>(cell)] |= (1u << q);
    }
}

// ===========================================================================
// Lin-2002 guidance (docs/CITATIONS.md: Lin, Prog. Aerospace Sci. 38 (2002)
// 389-420 — place VGs 5-10 h upstream of separation onset, h = 0.1-1.0 d99
// with the low-profile sweet spot at 0.2-0.5 d99; Strausak 2021 flight-proven
// defaults x/c ~ 0.07, +/-15 deg counter-rotating pairs, l = 3h).
// ===========================================================================

GuidanceBand recommendedHeightBand(float delta99_c) {
    GuidanceBand band;
    if (delta99_c <= 0.0f) return band; // no measured BL -> nothing to anchor
    // Lin 2002: device height h = 0.1 .. 1.0 * delta99. Returned directly in
    // h/c units so the UI can compare against VGParams::height_c verbatim.
    band.minVal = kLinHeightMinDelta99 * delta99_c;
    band.maxVal = kLinHeightMaxDelta99 * delta99_c;
    band.valid = true;
    return band;
}

GuidanceBand recommendedSweetSpotHeightBand(float delta99_c) {
    GuidanceBand band;
    if (delta99_c <= 0.0f) return band;
    // Lin 2002 low-profile sweet spot: h = 0.2 .. 0.5 * delta99 captured most
    // of the separation control at a fraction of the device drag.
    band.minVal = kLinSweetSpotMinDelta99 * delta99_c;
    band.maxVal = kLinSweetSpotMaxDelta99 * delta99_c;
    band.valid = true;
    return band;
}

GuidanceBand recommendedStationBand(float separationXc, float height_c) {
    GuidanceBand band;
    // separationXc < 0 is the LBMSolver::separationOnsetXc() "attached flow"
    // sentinel — with no separation onset there is no station to anchor.
    if (separationXc < 0.0f || height_c <= 0.0f) return band;
    // Lin 2002: 5..10 device heights UPSTREAM of separation onset. The larger
    // multiple lies farther upstream, hence it forms the MIN end of the
    // station band; both ends clamp at the LE.
    band.minVal = std::max(0.0f, separationXc - kLinUpstreamMaxHeights * height_c);
    band.maxVal = std::max(0.0f, separationXc - kLinUpstreamMinHeights * height_c);
    band.valid = band.maxVal > 0.0f; // separation at the very LE -> nowhere to go
    return band;
}

} // namespace foilcfd
