// M6 (ISLBM stretched mesh, 2026-06-16): correctness gate for the interpolated-
// gather streaming. Two checks, both through the public LBMSolver API:
//
//   (1) UNIFORM-COLLAPSE. Feed ISLBM a FLAT wall-distance field (every cell
//       equidistant from "the wall"), so the spacing profile is uniform: dx is
//       constant, the per-link foot shift is exactly 1 index cell, and the
//       quadratic Lagrange weights collapse to {0,1,0}. The interpolated gather
//       must then reproduce the exact integer pull to round-off. We assert the
//       ISLBM Taylor-Green decay tracks the uniform-kernel control (same grid,
//       same scaling) within a tight band at every sample — if the gather had a
//       bug it would diverge from the control immediately.
//
//   (2) GRADED STABILITY. Re-init with a genuinely STRETCHED wall-distance field
//       (a linear ramp, so dx grows toward kMaxStretchGrowth away from one face)
//       and assert the run stays finite and still decays — the gather must be
//       stable, not just identical-when-uniform.
//
// Both reuse the M0 Taylor-Green machinery; the decay horizon is shortened (the
// point here is the gather, not a 500x decay) to stay well inside the budget.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "lbm_test_util.h"
#include "sim/lbm_solver.h"
#include "sim/units.h"
#include "test_util.h"

using namespace foilcfd;
using namespace foilcfd::testutil;

namespace {

constexpr int   kN          = 48;     // smaller than M0 — gather check, not decay
constexpr int   kTotalSteps = 8000;   // enough to expose a gather divergence
constexpr int   kSampleEvery = 500;
constexpr float kAmplitude  = 0.05f;

LatticeScaling decayScaling() {
    constexpr float uLat      = 1.0e-3f;
    constexpr float nuLatGoal  = 0.01f;
    PhysicalParams phys;
    phys.chord_m = 1.0f;
    phys.nu_m2s  = kNuAir;
    phys.airspeed_ms = kNuAir * uLat * static_cast<float>(kN)
                     / (nuLatGoal * phys.chord_m);
    return computeScaling(phys, kN, uLat);
}

void buildTaylorGreenF(const GridDims& dims, std::vector<float>& f) {
    const std::size_t n = static_cast<std::size_t>(dims.cellCount());
    f.resize(static_cast<std::size_t>(kQ) * n);
    const float k = 2.0f * 3.14159265358979f / static_cast<float>(kN);
    for (int z = 0; z < dims.nz; ++z)
        for (int y = 0; y < dims.ny; ++y)
            for (int x = 0; x < dims.nx; ++x) {
                const std::size_t c =
                    static_cast<std::size_t>(cellIndex(dims, x, y, z));
                const float u =  kAmplitude * std::sin(k * x) * std::cos(k * y)
                                            * std::cos(k * z);
                const float v = -kAmplitude * std::cos(k * x) * std::sin(k * y)
                                            * std::cos(k * z);
                for (int q = 0; q < kQ; ++q)
                    f[static_cast<std::size_t>(q) * n + c] =
                        equilibrium(q, 1.0f, u, v, 0.0f);
            }
}

// Run a TG decay and record the KE sample series. When @p wallDist is non-empty
// the solver is put in ISLBM mode with that field; otherwise it runs uniform.
std::vector<double> runDecay(const GridDims& dims, const LatticeScaling& scaling,
                             const std::vector<std::uint8_t>& flags,
                             const std::vector<float>& wallDist, bool& ok) {
    std::vector<double> ke;
    LBMSolver solver;
    std::string err;
    if (!solver.init(dims, scaling, flags, nullptr, &err)) {
        std::printf("  init failed: %s\n", err.c_str());
        ok = false;
        return ke;
    }
    if (!wallDist.empty()) {
        if (!solver.initStretchMode(wallDist, &err)) {
            std::printf("  initStretchMode failed: %s\n", err.c_str());
            ok = false;
            return ke;
        }
        TCHECK(solver.stretchActive());
    }
    std::vector<float> hostF;
    buildTaylorGreenF(dims, hostF);
    if (uploadF(solver, hostF) != cudaSuccess) { ok = false; return ke; }

    HostMoments m = momentsFromF(hostF, dims);
    ke.push_back(kineticEnergy(m, flags));
    for (int step = kSampleEvery; step <= kTotalSteps; step += kSampleEvery) {
        if (solver.stepN(kSampleEvery) != cudaSuccess) { ok = false; break; }
        if (solver.nanDetected()) {
            std::printf("  NaN at step %d: %s\n", step,
                        solver.nanDiagnosis().c_str());
            ok = false;
            break;
        }
        if (downloadMoments(solver, m) != cudaSuccess) { ok = false; break; }
        const double e = kineticEnergy(m, flags);
        if (!std::isfinite(e)) { ok = false; break; }
        ke.push_back(e);
    }
    return ke;
}

} // namespace

int main() {
    const GridDims dims{kN, kN, kN};
    const LatticeScaling scaling = decayScaling();
    TCHECK(!scaling.tauClamped);
    const std::vector<std::uint8_t> flags = openDomainFlags(dims);
    const long long ncells = dims.cellCount();

    // --- control: uniform kernel (no ISLBM) ---
    bool ctrlOk = true;
    const std::vector<double> keUniform =
        runDecay(dims, scaling, flags, {}, ctrlOk);
    TCHECK_MSG(ctrlOk, "uniform control run failed");
    TCHECK(keUniform.size() > 2 && keUniform.front() > 0.0);

    // --- (1) uniform-collapse: ISLBM with a FLAT wall-distance field ---
    // Every cell the same distance from the wall -> uniform dx -> the gather
    // must reproduce the integer pull. Use a constant 10 cells so the spacing
    // sits on the dxMin plateau everywhere.
    bool flatOk = true;
    const std::vector<float> flatDist(static_cast<std::size_t>(ncells), 10.0f);
    const std::vector<double> keFlat =
        runDecay(dims, scaling, flags, flatDist, flatOk);
    TCHECK_MSG(flatOk, "ISLBM flat-field run diverged");
    TREQUIRE(keFlat.size() == keUniform.size());
    // The two series must track tightly: a uniform mesh makes the gather an
    // identity, so only round-off (and the 1-cell collar, which is also exact
    // here since there are no solids) separates them. Allow 1% per sample.
    double maxRel = 0.0;
    for (std::size_t i = 1; i < keFlat.size(); ++i) {
        const double rel = std::fabs(keFlat[i] - keUniform[i])
                         / std::max(1e-30, keUniform[i]);
        maxRel = std::max(maxRel, rel);
    }
    std::printf("uniform-collapse: max relative KE deviation = %.3e\n", maxRel);
    TCHECK_MSG(maxRel < 1e-2, "ISLBM flat field deviates %.2e from uniform "
               "(>1%% — gather is not collapsing to the integer pull)", maxRel);

    // --- (2) graded stability: a genuinely stretched field ---
    // Distance grows linearly from one y face (0) to the other (large), so dx
    // ramps toward the growth cap. The run must stay finite and still decay.
    bool gradOk = true;
    std::vector<float> rampDist(static_cast<std::size_t>(ncells));
    for (int z = 0; z < dims.nz; ++z)
        for (int y = 0; y < dims.ny; ++y)
            for (int x = 0; x < dims.nx; ++x)
                rampDist[static_cast<std::size_t>(cellIndex(dims, x, y, z))] =
                    static_cast<float>(y); // 0..N-1 cells from the y=0 "wall"
    const std::vector<double> keGraded =
        runDecay(dims, scaling, flags, rampDist, gradOk);
    TCHECK_MSG(gradOk, "ISLBM graded-field run diverged");
    TCHECK(keGraded.size() > 2);
    TCHECK_MSG(keGraded.back() < keGraded.front(),
               "graded ISLBM did not dissipate: %.3e -> %.3e",
               keGraded.front(), keGraded.back());

    return finish("m6_stretch");
}
