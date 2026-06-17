// M6b (ISLBM A/B gate, 2026-06-16): the force-accuracy proof that the stretched-
// mesh gradient mode produces the SAME aerodynamics as the proven uniform/
// cascade path on a real shedding case, BEFORE the cascade is ever torn out.
//
// A Re-150 circular cylinder (D = 32 cells) sheds a von Karman street. We run it
// twice on the same grid + scaling:
//   (A) uniform kernel (control), and
//   (B) ISLBM stretched mesh (cells finest at the cylinder wall, coarsening
//       outward via the wall-distance field; the cylinder's bounce-back wall is
//       wrapped by the exact-pull collar, the wake is bulk-gathered).
// Both measure mean drag Cd (streamwise momentum exchange) and the shedding
// Strouhal (Goertzel scan of the lift signal). PASS iff ISLBM's Strouhal stays
// in the physical band AND its mean Cd matches the control within tolerance —
// i.e. the gradient mode is engineering-equivalent to the uniform solver.
//
// Smaller/shorter than m2_cylinder (this is an A/B comparison, not an absolute
// Strouhal certification) to stay inside the test budget.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "geom/voxelizer.h"   // buildWallDistanceField
#include "lbm_test_util.h"
#include "sim/lbm_core.cuh"
#include "sim/lbm_solver.h"
#include "sim/units.h"
#include "test_util.h"

using namespace foilcfd;
using namespace foilcfd::testutil;

namespace {

constexpr int   kNx = 420, kNy = 300, kNz = 4;
constexpr float kD       = 32.0f;
constexpr float kCenterX = 120.0f;
constexpr float kCenterY = 150.0f;
constexpr float kULat    = 0.08f;
constexpr float kRe      = 150.0f;
constexpr int   kTransientSteps = 18000;
constexpr int   kSampleEvery    = 16;
constexpr int   kSampleCount    = 2200;

double goertzelPower(const std::vector<double>& x, double cyclesPerSample) {
    const double w = 2.0 * 3.14159265358979323846 * cyclesPerSample;
    const double coeff = 2.0 * std::cos(w);
    double s1 = 0.0, s2 = 0.0;
    for (double v : x) {
        const double s0 = v + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    return s1 * s1 + s2 * s2 - coeff * s1 * s2;
}

struct Result {
    bool   ok      = false;
    double meanCd  = 0.0;  ///< Mean drag coefficient.
    double strouhal = 0.0; ///< Peak shedding Strouhal.
    double liftStd = 0.0;  ///< Lift oscillation amplitude (shedding present?).
};

// Run the cylinder case; when @p islbm is true the solver is put in ISLBM mode
// with a wall-distance field built from the cylinder solids.
Result runCylinder(const GridDims& dims, const LatticeScaling& scaling,
                   const std::vector<std::uint8_t>& flags, double dMeasured,
                   bool islbm) {
    Result r;
    LBMSolver solver;
    std::string err;
    if (!solver.init(dims, scaling, flags, nullptr, &err)) {
        std::printf("  init failed: %s\n", err.c_str());
        return r;
    }
    if (islbm) {
        const std::vector<float> wallDist =
            buildWallDistanceField(dims, flags);
        if (!solver.initStretchMode(wallDist, &err)) {
            std::printf("  initStretchMode failed: %s\n", err.c_str());
            return r;
        }
        const StretchInfo si = solver.stretchInfo();
        std::printf("  ISLBM mesh: dx x%.1f range, growth x%.4f/y%.4f, "
                    "tau %.3f->%.3f\n",
                    si.dxMax / si.dxMin, si.growthX, si.growthY, si.tauWall,
                    si.tauFar);
    }
    solver.setStartupRampEnabled(false);
    solver.reset();

    // Impulsive start + antisymmetric v seed (as m2) so the street locks in fast.
    {
        const std::size_t n = static_cast<std::size_t>(dims.cellCount());
        std::vector<float> f0(static_cast<std::size_t>(kQ) * n);
        const float twoPi = 2.0f * 3.14159265358979f;
        const auto solidFlag = static_cast<std::uint8_t>(CellFlag::Solid);
        for (int z = 0; z < dims.nz; ++z)
            for (int j = 0; j < dims.ny; ++j)
                for (int i = 0; i < dims.nx; ++i) {
                    const std::size_t c =
                        static_cast<std::size_t>(cellIndex(dims, i, j, z));
                    const bool solid = flags[c] == solidFlag;
                    const float u = solid ? 0.0f : kULat;
                    const float v = solid ? 0.0f
                        : 0.1f * kULat
                          * std::sin(twoPi * static_cast<float>(i) / kNx)
                          * std::sin(3.14159265f * static_cast<float>(j) / kNy);
                    for (int q = 0; q < kQ; ++q)
                        f0[static_cast<std::size_t>(q) * n + c] =
                            equilibrium(q, 1.0f, u, v, 0.0f);
                }
        if (uploadF(solver, f0) != cudaSuccess) return r;
    }

    for (int done = 0; done < kTransientSteps; done += 1000) {
        if (solver.stepN(1000) != cudaSuccess) return r;
        if (solver.nanDetected()) {
            std::printf("  NaN during transient: %s\n",
                        solver.nanDiagnosis().c_str());
            return r;
        }
    }

    float* dForce = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dForce), 3 * sizeof(float))
        != cudaSuccess)
        return r;
    std::vector<double> lift, drag;
    lift.reserve(kSampleCount);
    drag.reserve(kSampleCount);
    for (int s = 0; s < kSampleCount; ++s) {
        if (solver.stepN(kSampleEvery) != cudaSuccess) { cudaFree(dForce); return r; }
        const DeviceLatticeView view = solver.latticeView();
        if (launchForceReduction(view, DeviceForceAccumulator{dForce}, nullptr)
            != cudaSuccess) { cudaFree(dForce); return r; }
        float force[3] = {0, 0, 0};
        if (cudaMemcpy(force, dForce, sizeof(force), cudaMemcpyDeviceToHost)
            != cudaSuccess) { cudaFree(dForce); return r; }
        lift.push_back(static_cast<double>(force[1]));
        drag.push_back(static_cast<double>(force[0]));
    }
    cudaFree(dForce);
    if (solver.nanDetected()) return r;

    // Mean drag -> Cd = Fx / (1/2 rho u^2 D), rho=1, per-unit-span (divide by nz
    // fluid planes). Same normalization for both runs, so the ratio is what we
    // compare; absolute value just sanity-checks the regime.
    double meanFx = 0.0;
    for (double d : drag) meanFx += d;
    meanFx /= static_cast<double>(drag.size());
    const double q = 0.5 * kULat * kULat * dMeasured * static_cast<double>(kNz);
    r.meanCd = meanFx / q;

    // Lift spectrum -> Strouhal.
    double mean = 0.0;
    for (double v : lift) mean += v;
    mean /= static_cast<double>(lift.size());
    double var = 0.0;
    std::vector<double> sig(lift.size());
    const double n1 = static_cast<double>(lift.size() - 1);
    for (std::size_t i = 0; i < lift.size(); ++i) {
        const double cen = lift[i] - mean;
        var += cen * cen;
        const double hann =
            0.5 * (1.0 - std::cos(2.0 * 3.14159265358979 * i / n1));
        sig[i] = cen * hann;
    }
    r.liftStd = std::sqrt(var / static_cast<double>(lift.size()));
    double bestSt = 0.0, bestPower = -1.0;
    for (double st = 0.10; st <= 0.30; st += 0.0005) {
        const double cps = st * kULat / dMeasured * static_cast<double>(kSampleEvery);
        const double p = goertzelPower(sig, cps);
        if (p > bestPower) { bestPower = p; bestSt = st; }
    }
    r.strouhal = bestSt;
    r.ok = true;
    return r;
}

} // namespace

int main() {
    const GridDims dims{kNx, kNy, kNz};
    PhysicalParams phys;
    phys.chord_m     = 1.0f;
    phys.nu_m2s      = kNuAir;
    phys.airspeed_ms = kRe * kNuAir / phys.chord_m;
    const LatticeScaling scaling =
        computeScaling(phys, static_cast<int>(kD), kULat);
    TCHECK(!scaling.tauClamped);

    // Voxelize the cylinder; measure the stamped diameter for normalization.
    std::vector<std::uint8_t> flags = openDomainFlags(dims);
    int iMin = kNx, iMax = -1;
    for (int z = 0; z < dims.nz; ++z)
        for (int j = 0; j < dims.ny; ++j)
            for (int i = 0; i < dims.nx; ++i) {
                const float dx = static_cast<float>(i) - kCenterX;
                const float dy = static_cast<float>(j) - kCenterY;
                if (dx * dx + dy * dy <= 0.25f * kD * kD) {
                    flags[static_cast<std::size_t>(cellIndex(dims, i, j, z))] =
                        static_cast<std::uint8_t>(CellFlag::Solid);
                    if (z == 0 && j == static_cast<int>(kCenterY)) {
                        iMin = std::min(iMin, i);
                        iMax = std::max(iMax, i);
                    }
                }
            }
    const double dMeasured = static_cast<double>(iMax - iMin + 1);

    std::printf("--- control: uniform kernel ---\n");
    const Result ctrl = runCylinder(dims, scaling, flags, dMeasured, false);
    TCHECK_MSG(ctrl.ok, "uniform control run failed");
    std::printf("  uniform: Cd = %.4f, St = %.4f, liftStd = %.3e\n",
                ctrl.meanCd, ctrl.strouhal, ctrl.liftStd);

    std::printf("--- ISLBM stretched mesh ---\n");
    const Result isl = runCylinder(dims, scaling, flags, dMeasured, true);
    TCHECK_MSG(isl.ok, "ISLBM run failed/diverged");
    std::printf("  ISLBM: Cd = %.4f, St = %.4f, liftStd = %.3e\n",
                isl.meanCd, isl.strouhal, isl.liftStd);

    if (ctrl.ok && isl.ok) {
        // Both must actually shed.
        TCHECK_MSG(ctrl.liftStd > 1e-3 && isl.liftStd > 1e-3,
                   "no shedding (ctrl %.2e, ISLBM %.2e)", ctrl.liftStd,
                   isl.liftStd);
        // ISLBM Strouhal in the physical band (literature ~0.18 at Re 150,
        // blockage-raised here; the same band m2_cylinder certifies).
        TCHECK_MSG(isl.strouhal >= 0.16 && isl.strouhal <= 0.22,
                   "ISLBM Strouhal %.4f out of band", isl.strouhal);
        // Strouhal A/B: the two must agree closely (wake physics unchanged).
        TCHECK_MSG(std::fabs(isl.strouhal - ctrl.strouhal) <= 0.02,
                   "Strouhal mismatch: ISLBM %.4f vs uniform %.4f",
                   isl.strouhal, ctrl.strouhal);
        // Mean-Cd A/B: the force the project lives on. The cylinder wall (all
        // collar cells, exact pull) plus the bulk-gathered wake should land Cd
        // within ~10% of the uniform run — the coarsened far field changes the
        // wake resolution slightly, but the near-body force must track.
        const double cdRel = std::fabs(isl.meanCd - ctrl.meanCd)
                           / std::max(1e-9, std::fabs(ctrl.meanCd));
        std::printf("Cd A/B: ISLBM %.4f vs uniform %.4f (%.1f%%)\n",
                    isl.meanCd, ctrl.meanCd, 100.0 * cdRel);
        TCHECK_MSG(cdRel <= 0.10, "ISLBM mean Cd deviates %.1f%% from uniform",
                   100.0 * cdRel);
    }

    return finish("m6b_stretch_cylinder");
}
