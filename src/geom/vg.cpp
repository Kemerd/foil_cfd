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
/// @param halfW    Half-thickness (vane) or half-width (ramp) in cells.
/// @param ramp     True for the right-triangular wedge: local height rises
///                 linearly from 0 at the upstream end to hCells downstream.
void stampVane(const AirfoilGeometry& airfoil, float aoaRad,
               const DomainLayout& layout, std::vector<std::uint8_t>& flags,
               float x_c, float zCenter, float betaRad, float hCells,
               float lenCells, float halfW, bool ramp) {
    // Center frame gives the reference tangent used to convert axis distance
    // into a chordwise-station advance (vanes are ~3h long, a few % chord —
    // treating the tangent's x-projection as constant over that span is fine).
    const SurfaceFrame center = surfaceFrameAt(airfoil, x_c, /*upper=*/true);
    if (!center.valid) return;

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

        // Ramp wedge: height ramps 0 -> h from upstream end to downstream
        // end (right-triangular profile); vanes are full height everywhere.
        const float top = ramp
            ? hCells * (s + 0.5f * lenCells) / std::max(lenCells, 1e-3f)
            : hCells;

        // Rasterize the slice column: height u (embedded root included) by
        // thickness w. Every sample marks the cell containing it.
        for (float u = -kRootEmbed; u <= top + 1e-4f; u += kStep) {
            for (float w = -halfW; w <= halfW + 1e-4f; w += kStep) {
                const Vec3f p = root + n3 * u + b3 * w;
                stampCell(flags, layout.dims,
                          static_cast<int>(std::floor(p.x)),
                          static_cast<int>(std::floor(p.y)),
                          static_cast<int>(std::floor(p.z)));
            }
        }
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
                std::vector<std::uint8_t>& flags) {
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

    for (int i = 0; i < count; ++i) {
        const float zc = zMid
                       + (static_cast<float>(i) - 0.5f * static_cast<float>(count - 1))
                       * pitch;
        switch (vg.type) {
        case VGType::SingleVane:
        case VGType::CoRotatingArray:
            // One vane per unit, all yawed the same way — a single vane is
            // just a co-rotating array of count 1.
            stampVane(airfoil, aoaRad, layout, flags, x_c, zc, beta, h, len,
                      0.5f * thick, /*ramp=*/false);
            break;
        case VGType::CounterRotatingPair: {
            // Two mirrored vanes per unit, centers gap_h device heights
            // apart. Orientation convention: commonFlowDown=true toes the
            // trailing edges IN (the -z vane sweeps toward +z and vice
            // versa), the flight-proven arrangement of the Strausak recipe;
            // false toes them out (common flow up between the pair).
            const float halfGap = 0.5f * vg.gap_h * h;
            const float betaNear = vg.commonFlowDown ? beta : -beta;
            stampVane(airfoil, aoaRad, layout, flags, x_c, zc - halfGap,
                      betaNear, h, len, 0.5f * thick, /*ramp=*/false);
            stampVane(airfoil, aoaRad, layout, flags, x_c, zc + halfGap,
                      -betaNear, h, len, 0.5f * thick, /*ramp=*/false);
            break;
        }
        case VGType::Ramp:
            // Right-triangular wedge prism in the same placement frame
            // (plan 6.1). Wheeler-ramp-like proportions: spanwise width of
            // one device height (never under one cell), height rising
            // linearly to h at the downstream end.
            stampVane(airfoil, aoaRad, layout, flags, x_c, zc, beta, h, len,
                      std::max(0.5f, 0.5f * h), /*ramp=*/true);
            break;
        }
    }
}

std::vector<std::uint8_t> buildFlagsWithVGs(
    const std::vector<VGParams>& vgs, const AirfoilGeometry& airfoil,
    float aoa_deg, const DomainLayout& layout,
    const std::vector<std::uint8_t>& cleanFoilFlags) {
    // Plan 6.2 flow: copy the cached clean mask, OR every VG in, run the TE
    // closure once at the end so vane roots get the same single-cell-gap
    // sealing the foil TE does. The clean flags are never modified — they are
    // the warm-start cache key's geometry half.
    std::vector<std::uint8_t> flags = cleanFoilFlags;
    for (const VGParams& vg : vgs) {
        voxelizeVG(vg, airfoil, aoa_deg, layout, flags);
    }
    closeTrailingEdgeGaps(layout.dims, flags);
    return flags;
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
