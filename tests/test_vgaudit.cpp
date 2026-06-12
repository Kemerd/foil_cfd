// VG vortex-audit math checks (geom/vg_audit.h): the plane-circulation
// integrator must recover the known circulation of a synthetic Lamb-Oseen
// counter-rotating pair to within 3%, and the Wendt-correlation prediction
// must reproduce a hand-computed spot value and behave physically
// (monotonic in incidence and height, saturating with h/delta).
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#include <cmath>
#include <cstdio>
#include <vector>

#include "geom/vg_audit.h"
#include "test_util.h"

using namespace foilcfd;
using namespace foilcfd::testutil;

namespace {

/// @brief Superpose one Lamb-Oseen vortex of circulation gamma and core
/// radius rc at (y0, z0) onto a crossflow plane (index y + ny*z). Velocity:
/// v_theta(r) = gamma/(2 pi r) (1 - exp(-r^2/rc^2)), giving the compact
/// vorticity omega(r) = gamma/(pi rc^2) exp(-r^2/rc^2) — over 3 rc of
/// window the integral captures > 99.9% of gamma.
void addLambOseen(std::vector<float>& v, std::vector<float>& w, int ny, int nz,
                  float y0, float z0, float gamma, float rc) {
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            const float dy = static_cast<float>(y) - y0;
            const float dz = static_cast<float>(z) - z0;
            const float r2 = std::max(dy * dy + dz * dz, 1e-6f);
            const float swirl = gamma / (2.0f * 3.14159265f * r2)
                              * (1.0f - std::exp(-r2 / (rc * rc)));
            const std::size_t i = static_cast<std::size_t>(y)
                                + static_cast<std::size_t>(ny)
                                      * static_cast<std::size_t>(z);
            v[i] += -swirl * dz; // tangential direction (-dz, +dy)/r * v_t*r
            w[i] += swirl * dy;
        }
    }
}

void testLambOseenRecovery() {
    const int ny = 64, nz = 64;
    const float gamma = 0.8f, rc = 4.0f;
    std::vector<float> v(static_cast<std::size_t>(ny) * nz, 0.0f);
    std::vector<float> w(v.size(), 0.0f);
    // Counter-rotating pair, both cores well inside the band and 3 rc clear
    // of every edge and of each other.
    addLambOseen(v, w, ny, nz, 24.0f, 16.0f, +gamma, rc);
    addLambOseen(v, w, ny, nz, 24.0f, 48.0f, -gamma, rc);

    const PlaneCirculation c =
        integrateStreamwiseCirculation(v, w, ny, nz, 2, ny - 3);
    std::printf("lamb-oseen: pos %.4f neg %.4f (planted +/-%.4f)\n",
                c.pos, c.neg, gamma);
    TCHECK_MSG(approxRel(c.pos, gamma, 0.03), "pos %.4f vs %.4f", c.pos, gamma);
    TCHECK_MSG(approxRel(-c.neg, gamma, 0.03), "neg %.4f vs %.4f", c.neg, -gamma);

    // A band that excludes both cores must read near zero on both signs.
    const PlaneCirculation far_ =
        integrateStreamwiseCirculation(v, w, ny, nz, 44, ny - 3);
    TCHECK_MSG(far_.pos < 0.05f * gamma && -far_.neg < 0.05f * gamma,
               "empty band reads %.4f / %.4f", far_.pos, far_.neg);

    // Zone split (the audit's noise-floor machinery): a zone holding both
    // cores leaves a near-empty ambient strip; a zone holding only the
    // positive core pushes the negative one into the ambient bucket.
    const BandCirculation both =
        integrateBandCirculation(v, w, ny, nz, 2, ny - 3, 8, 56);
    TCHECK_MSG(approxRel(both.zone.pos, gamma, 0.03)
                   && approxRel(-both.zone.neg, gamma, 0.03),
               "zone missed a core: %.4f / %.4f", both.zone.pos, both.zone.neg);
    TCHECK_MSG(both.ambient.pos < 0.03f * gamma
                   && -both.ambient.neg < 0.03f * gamma,
               "ambient strip not clean: %.4f / %.4f",
               both.ambient.pos, both.ambient.neg);
    TCHECK(both.zoneCols == 49 && both.ambientCols == nz - 49);

    const BandCirculation half =
        integrateBandCirculation(v, w, ny, nz, 2, ny - 3, 8, 24);
    TCHECK_MSG(approxRel(half.zone.pos, gamma, 0.03),
               "positive core missing from narrow zone: %.4f", half.zone.pos);
    TCHECK_MSG(approxRel(-half.ambient.neg, gamma, 0.03),
               "negative core not in ambient bucket: %.4f", half.ambient.neg);
}

void testWendtPrediction() {
    // Hand-computed spot value: beta = 16 deg, ue = 0.05, h = 8, c = 24,
    // delta = 10 -> AR = 64/(24 pi) = 0.8488, tanh(1.41*0.8) = 0.8104,
    // denominator 1.5655 -> Gamma = 0.2793 (worked by hand from the
    // constants; locks the formula against accidental edits).
    const float beta = 16.0f * 3.14159265f / 180.0f;
    const float g = wendtCirculationLattice(beta, 0.05f, 8.0f, 24.0f, 10.0f);
    TCHECK_MSG(approxEq(g, 0.2793, 2e-3), "spot value %.4f", g);

    // Monotonic in incidence and in height (fixed everything else).
    TCHECK(wendtCirculationLattice(1.2f * beta, 0.05f, 8, 24, 10) > g);
    TCHECK(wendtCirculationLattice(beta, 0.05f, 10, 24, 10) > g);
    // Saturating in h/delta: far outside the BL, thickness stops mattering.
    const float deep = wendtCirculationLattice(beta, 0.05f, 8, 24, 8.0f / 5.0f);
    const float deeper = wendtCirculationLattice(beta, 0.05f, 8, 24, 8.0f / 50.0f);
    TCHECK_MSG(approxRel(deep, deeper, 0.01), "tanh not saturating");
    // Guards: any non-positive ingredient yields zero, not garbage.
    TCHECK(wendtCirculationLattice(beta, 0.0f, 8, 24, 10) == 0.0f);
    TCHECK(wendtCirculationLattice(beta, 0.05f, 0, 24, 10) == 0.0f);
}

} // namespace

int main() {
    testLambOseenRecovery();
    testWendtPrediction();
    return finish("test_vgaudit");
}
