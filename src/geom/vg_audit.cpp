// Vortex-strength audit implementation (see vg_audit.h for the contract):
// Wendt-correlation prediction, crossflow-plane circulation integration,
// and the orchestration that turns both into the UI's honesty meter.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#include "vg_audit.h"

#include <algorithm>
#include <cmath>

namespace foilcfd {

float wendtCirculationLattice(float betaRad, float ueLat, float hCells,
                              float vaneChordCells, float delta99Cells) {
    if (ueLat <= 0.0f || hCells <= 0.0f || vaneChordCells <= 0.0f
        || delta99Cells <= 0.0f)
        return 0.0f;
    // Effective aspect ratio of a wall-mounted half-wing (Dudek 2005):
    // the wall mirror doubles the span, the 4/pi the elliptic equivalence.
    const float ar = 8.0f * hCells / (3.14159265f * vaneChordCells);
    // Prandtl lifting line with regression constants and the boundary
    // layer's retarding tanh (Wendt 2001 Eq. 6 / Dudek 2005 Eq. 3).
    return kWendtK1 * std::fabs(betaRad) * ueLat * vaneChordCells
         * std::tanh(kWendtK3 * hCells / delta99Cells)
         / (1.0f + kWendtK2 / ar);
}

PlaneCirculation integrateStreamwiseCirculation(const std::vector<float>& v,
                                                const std::vector<float>& w,
                                                int ny, int nz,
                                                int yMin, int yMax) {
    PlaneCirculation out;
    const std::size_t need = static_cast<std::size_t>(ny)
                           * static_cast<std::size_t>(nz);
    if (v.size() < need || w.size() < need || nz < 3) return out;
    yMin = std::max(yMin, 0);
    yMax = std::min(yMax, ny - 1);
    if (yMax <= yMin) return out;

    auto at = [ny](const std::vector<float>& a, int y, int z) {
        return a[static_cast<std::size_t>(y)
                 + static_cast<std::size_t>(ny) * static_cast<std::size_t>(z)];
    };

    // omega_x = dw/dy - dv/dz, cell area 1: the band sum IS the circulation
    // content split by sign. y derivatives stay inside the band (one-sided
    // at its edges) so solid rows below never contaminate the integral; z
    // wraps periodically like the lattice does.
    for (int z = 0; z < nz; ++z) {
        const int zm = (z == 0) ? nz - 1 : z - 1;
        const int zp = (z == nz - 1) ? 0 : z + 1;
        for (int y = yMin; y <= yMax; ++y) {
            float dwdy;
            if (y == yMin)      dwdy = at(w, y + 1, z) - at(w, y, z);
            else if (y == yMax) dwdy = at(w, y, z) - at(w, y - 1, z);
            else                dwdy = 0.5f * (at(w, y + 1, z) - at(w, y - 1, z));
            const float dvdz = 0.5f * (at(v, y, zp) - at(v, y, zm));
            const float om = dwdy - dvdz;
            if (om > 0.0f) out.pos += om;
            else           out.neg += om;
        }
    }
    return out;
}

BandCirculation integrateBandCirculation(const std::vector<float>& v,
                                         const std::vector<float>& w,
                                         int ny, int nz, int yMin, int yMax,
                                         int z0, int z1) {
    BandCirculation out;
    const std::size_t need = static_cast<std::size_t>(ny)
                           * static_cast<std::size_t>(nz);
    if (v.size() < need || w.size() < need || nz < 3) return out;
    yMin = std::max(yMin, 0);
    yMax = std::min(yMax, ny - 1);
    z0 = std::max(z0, 0);
    z1 = std::min(z1, nz - 1);
    if (yMax <= yMin || z1 < z0) return out;

    auto at = [ny](const std::vector<float>& a, int y, int z) {
        return a[static_cast<std::size_t>(y)
                 + static_cast<std::size_t>(ny) * static_cast<std::size_t>(z)];
    };

    // Same stencils as the full-span integral; each spanwise column lands
    // in either the vane-row zone or the ambient-floor bucket.
    for (int z = 0; z < nz; ++z) {
        const bool inZone = (z >= z0 && z <= z1);
        PlaneCirculation& dst = inZone ? out.zone : out.ambient;
        if (inZone) ++out.zoneCols; else ++out.ambientCols;
        const int zm = (z == 0) ? nz - 1 : z - 1;
        const int zp = (z == nz - 1) ? 0 : z + 1;
        for (int y = yMin; y <= yMax; ++y) {
            float dwdy;
            if (y == yMin)      dwdy = at(w, y + 1, z) - at(w, y, z);
            else if (y == yMax) dwdy = at(w, y, z) - at(w, y - 1, z);
            else                dwdy = 0.5f * (at(w, y + 1, z) - at(w, y - 1, z));
            const float dvdz = 0.5f * (at(v, y, zp) - at(v, y, zm));
            const float om = dwdy - dvdz;
            if (om > 0.0f) dst.pos += om;
            else           dst.neg += om;
        }
    }
    return out;
}

VGAuditReadout auditVGVortexStrength(const LBMSolver& solver,
                                     const VGParams& vg, int chordCells,
                                     float ueLat, float delta99Cells) {
    VGAuditReadout r;
    if (chordCells <= 0 || ueLat <= 0.0f || delta99Cells <= 0.0f
        || vg.count <= 0)
        return r;

    const float h = vgHeightCells(vg, chordCells);
    if (h <= 0.0f) return r;
    const float vaneChord = vg.length_h * h;
    const float betaRad = vg.beta_deg * 3.14159265f / 180.0f;
    r.gammaPredicted =
        wendtCirculationLattice(betaRad, ueLat, h, vaneChord, delta99Cells);
    if (r.gammaPredicted <= 0.0f) return r;

    // Audit plane: vane trailing edge + the rollup distance, in chord
    // fractions (vane chord projects ~1:1 onto x at these incidences).
    r.planeXc = std::clamp(
        vg.x_c + vg.height_c * (vg.length_h + kAuditPlaneHeightsDownstream),
        0.02f, 0.97f);
    const int ix = solver.latticeXForChordStation(r.planeXc);
    if (ix < 0) return r;
    const int ys = solver.suctionSurfaceY(ix);
    if (ys < 0) return r;

    const GridDims dims = solver.dims();
    const int band = std::max(2, static_cast<int>(
                                     std::ceil(kAuditBandHeights * h)));

    // Spanwise vane-row zone: stampVane centers the array on the span; the
    // zone covers every unit plus 3h of meander margin per side. Whatever
    // remains is the ambient-floor strip — same plane, same resolution,
    // same boundary layer, so its per-column vorticity content is the
    // statistically matched estimate of "what this band reads with no
    // vortex in it".
    const float pitchCells = vg.pitch_c * static_cast<float>(chordCells);
    const float rowHalf = 0.5f * (static_cast<float>(vg.count - 1) * pitchCells
                                  + vg.gap_h * h)
                        + 3.0f * h;
    const int mid = dims.nz / 2;
    const int z0 = mid - static_cast<int>(std::ceil(rowHalf));
    const int z1 = mid + static_cast<int>(std::ceil(rowHalf));

    std::vector<float> pv, pw;
    if (!solver.downloadCrossflowPlane(ix, pv, pw)) return r;
    const BandCirculation b = integrateBandCirculation(
        pv, pw, dims.ny, dims.nz, ys + 1, std::min(dims.ny - 2, ys + band),
        z0, z1);
    if (b.zoneCols <= 0) return r;

    // Net vane contribution per sign: zone minus the area-scaled ambient
    // floor, clamped at zero (an honest "indistinguishable from noise"
    // reads 0, never negative). Too-narrow ambient strips estimate nothing
    // — skip the subtraction rather than scale up garbage.
    float floorPos = 0.0f, floorNeg = 0.0f;
    if (b.ambientCols >= 4) {
        const float scale = static_cast<float>(b.zoneCols)
                          / static_cast<float>(b.ambientCols);
        floorPos = b.ambient.pos * scale;
        floorNeg = -b.ambient.neg * scale;
    }
    const float posNet = std::max(0.0f, b.zone.pos - floorPos);
    const float negNet = std::max(0.0f, -b.zone.neg - floorNeg);

    // Per-vortex strength by configuration type: counter-rotating layouts
    // shed one vortex of each sign per unit; co-rotating layouts shed
    // same-signed vortices and the minority sign is treated as noise.
    float gamma = 0.0f;
    const float n = static_cast<float>(vg.count);
    switch (vg.type) {
        case VGType::CounterRotatingPair:
        case VGType::Ramp:
            gamma = (posNet + negNet) / (2.0f * n);
            break;
        case VGType::SingleVane:
        case VGType::CoRotatingArray:
            gamma = std::max(posNet, negNet) / n;
            break;
    }

    r.gammaMeasured = gamma;
    r.ratio = gamma / r.gammaPredicted;
    r.valid = true;
    return r;
}

} // namespace foilcfd
