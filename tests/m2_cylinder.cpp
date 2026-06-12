// M2 milestone (plan section 10): voxelized circular cylinder, D = 40 cells,
// at Re 150 in the standard open domain (inlet/outlet, free-slip y, periodic
// z). The von Karman street's lift oscillation is sampled via the public
// momentum-exchange reduction (launchForceReduction on the solver's active
// buffer), and the dominant shedding frequency is located with a Goertzel
// scan over candidate Strouhal numbers. PASS iff the spectral peak lands in
// St = 0.17 .. 0.21 (literature: St ~ 0.183 at Re 150). Budget < 5 minutes.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "lbm_test_util.h"
#include "sim/lbm_core.cuh"
#include "sim/lbm_solver.h"
#include "sim/units.h"
#include "test_util.h"

using namespace foilcfd;
using namespace foilcfd::testutil;

namespace {

// Domain: 14 D long, 10 D tall (10% blockage), 4 cells deep (quasi-2D).
// Blockage raises the measured Strouhal a few percent above the unbounded
// value (~0.183 at Re 150); 10% keeps the spectral peak mid-band instead of
// brushing the 0.21 acceptance ceiling.
// (kCenterX/Y naming avoids the D3Q19 kCx/kCy lattice arrays in lbm_core.cuh.)
constexpr int   kNx = 560, kNy = 400, kNz = 4;
constexpr float kD       = 40.0f;  // nominal cylinder diameter in cells
constexpr float kCenterX = 160.0f; // center 4 D from the inlet
constexpr float kCenterY = 200.0f; // mid-height
constexpr float kULat    = 0.08f;
constexpr float kRe      = 150.0f;

// Transient long enough for the street to lock in (~9 shedding periods,
// ~3.5 flow-throughs), then ~17 periods of lift sampled for the spectrum.
// Bumped to absorb the startup INLET-VELOCITY ramp (the field now accelerates
// from rest, so shedding locks in later than under the old impulsive start).
constexpr int kTransientSteps = 40000;
constexpr int kSampleEvery    = 16;    // steps between lift samples
constexpr int kSampleCount    = 3000;  // -> 48000 sampled steps

/// @brief Goertzel single-bin DFT power at normalized frequency `cyclesPerSample`
/// (classic 2nd-order resonator recurrence — O(n) per candidate frequency,
/// exact DFT magnitude at the probed bin, no power-of-two length needed).
double goertzelPower(const std::vector<double>& x, double cyclesPerSample) {
    const double w = 2.0 * 3.14159265358979323846 * cyclesPerSample;
    const double coeff = 2.0 * std::cos(w);
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (double v : x) {
        s0 = v + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    // |X(w)|^2 from the final recurrence state.
    return s1 * s1 + s2 * s2 - coeff * s1 * s2;
}

} // namespace

int main() {
    const GridDims dims{kNx, kNy, kNz};

    // Re = u_lat * D / nu_lat: reverse-engineer physical inputs (the "chord"
    // here is the cylinder diameter) so computeScaling delivers it exactly.
    PhysicalParams phys;
    phys.chord_m     = 1.0f;
    phys.nu_m2s      = kNuAir;
    phys.airspeed_ms = kRe * kNuAir / phys.chord_m;
    const LatticeScaling scaling =
        computeScaling(phys, static_cast<int>(kD), kULat);
    TCHECK(!scaling.tauClamped);
    TCHECK(approxRel(scaling.nu_lat, kULat * kD / kRe, 1e-3)); // ~0.02133

    // Open-domain boundary flags + the voxelized cylinder disk, extruded
    // across z. The stamped diameter is measured back from the mask so the
    // Strouhal normalization uses the geometry that actually exists.
    std::vector<std::uint8_t> flags = openDomainFlags(dims);
    int iMin = kNx, iMax = -1;
    for (int z = 0; z < dims.nz; ++z) {
        for (int j = 0; j < dims.ny; ++j) {
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
        }
    }
    const double dMeasured = static_cast<double>(iMax - iMin + 1);
    TCHECK_MSG(dMeasured >= kD - 2.0 && dMeasured <= kD + 2.0,
               "stamped diameter = %.0f cells", dMeasured);

    LBMSolver solver;
    std::string err;
    const bool initOk = solver.init(dims, scaling, flags, nullptr, &err);
    TCHECK_MSG(initOk, "solver init failed: %s", err.c_str());
    if (!initOk) return finish("m2_cylinder");

    // Impulsive uniform start plus a deliberate mirror-asymmetric v ripple:
    // at Re 150 the street self-starts from noise eventually, but seeding the
    // antisymmetric mode deterministically keeps the transient inside budget
    // and makes the run reproducible. Injected as host-built equilibrium via
    // the snapshot-restore path.
    {
        const std::size_t n = static_cast<std::size_t>(dims.cellCount());
        std::vector<float> f0(static_cast<std::size_t>(kQ) * n);
        const float twoPi = 2.0f * 3.14159265358979f;
        for (int z = 0; z < dims.nz; ++z) {
            for (int j = 0; j < dims.ny; ++j) {
                for (int i = 0; i < dims.nx; ++i) {
                    const std::size_t c =
                        static_cast<std::size_t>(cellIndex(dims, i, j, z));
                    const bool solid = flags[c]
                        == static_cast<std::uint8_t>(CellFlag::Solid);
                    // Nonzero v at mid-height breaks the y-mirror symmetry
                    // (the mirror maps v -> -v); zero at the slip walls.
                    const float u = solid ? 0.0f : kULat;
                    const float v = solid ? 0.0f
                        : 0.1f * kULat
                          * std::sin(twoPi * static_cast<float>(i) / kNx)
                          * std::sin(3.14159265f * static_cast<float>(j) / kNy);
                    for (int q = 0; q < kQ; ++q)
                        f0[static_cast<std::size_t>(q) * n + c] =
                            equilibrium(q, 1.0f, u, v, 0.0f);
                }
            }
        }
        TREQUIRE(uploadF(solver, f0) == cudaSuccess);
    }

    // ---- transient: let the wake develop and the street lock in ----
    for (int done = 0; done < kTransientSteps; done += 1000) {
        const cudaError_t serr = solver.stepN(1000);
        TCHECK_MSG(serr == cudaSuccess, "stepN failed in transient: %s",
                   cudaGetErrorString(serr));
        if (serr != cudaSuccess) return finish("m2_cylinder");
        TCHECK_MSG(!solver.nanDetected(), "NaN during transient: %s",
                   solver.nanDiagnosis().c_str());
        if (solver.nanDetected()) return finish("m2_cylinder");
    }

    // ---- lift sampling via the momentum-exchange reduction ----
    // The device accumulator is owned by the test; launchForceReduction
    // zeroes it before each launch per the lbm_core.cuh contract.
    float* dForce = nullptr;
    TREQUIRE(cudaMalloc(reinterpret_cast<void**>(&dForce), 3 * sizeof(float))
             == cudaSuccess);
    std::vector<double> lift;
    lift.reserve(kSampleCount);
    for (int s = 0; s < kSampleCount; ++s) {
        const cudaError_t serr = solver.stepN(kSampleEvery);
        if (serr != cudaSuccess) {
            TCHECK_MSG(false, "stepN failed at sample %d: %s", s,
                       cudaGetErrorString(serr));
            break;
        }
        // latticeView() hands back the PADDED base pointers the launch
        // wrappers index with — deviceFlags() is ghost-offset for unpadded
        // consumers, and a view built from it reads flags one z-plane high.
        const DeviceLatticeView view = solver.latticeView();
        const cudaError_t ferr =
            launchForceReduction(view, DeviceForceAccumulator{dForce}, nullptr);
        if (ferr != cudaSuccess) {
            TCHECK_MSG(false, "force reduction failed: %s",
                       cudaGetErrorString(ferr));
            break;
        }
        float force[3] = {0.0f, 0.0f, 0.0f};
        if (cudaMemcpy(force, dForce, sizeof(force), cudaMemcpyDeviceToHost)
            != cudaSuccess) {
            TCHECK_MSG(false, "force readback failed at sample %d", s);
            break;
        }
        lift.push_back(static_cast<double>(force[1])); // Fy = lift direction
        if ((s + 1) % 500 == 0)
            std::printf("sample %4d / %d  Fy = %+.5e\n", s + 1, kSampleCount,
                        lift.back());
    }
    cudaFree(dForce);
    TREQUIRE(static_cast<int>(lift.size()) == kSampleCount);
    TCHECK(!solver.nanDetected());

    // ---- spectral analysis: detrend, Hann window, Goertzel St scan ----
    double mean = 0.0;
    for (double v : lift) mean += v;
    mean /= static_cast<double>(lift.size());
    double var = 0.0;
    std::vector<double> sig(lift.size());
    const double n1 = static_cast<double>(lift.size() - 1);
    for (std::size_t i = 0; i < lift.size(); ++i) {
        const double centered = lift[i] - mean;
        var += centered * centered;
        // Hann window suppresses leakage from the finite, non-integer number
        // of shedding periods in the record.
        const double hann =
            0.5 * (1.0 - std::cos(2.0 * 3.14159265358979 * i / n1));
        sig[i] = centered * hann;
    }
    var /= static_cast<double>(lift.size());
    // A real vortex street oscillates hard (|Cl'| ~ 0.5 -> Fy amplitude
    // ~0.25 lattice units here); a flat signal means no shedding at all.
    TCHECK_MSG(std::sqrt(var) > 1e-3, "lift std %.3e — no oscillation detected",
               std::sqrt(var));

    // Scan candidate Strouhal numbers. St = f * D / u with f in cycles/step;
    // cycles/sample = f * kSampleEvery (Nyquist-safe: 0.30 -> 0.0096 << 0.5).
    double bestSt = 0.0, bestPower = -1.0, totalPower = 0.0;
    int nCandidates = 0;
    for (double st = 0.10; st <= 0.30; st += 0.0005) {
        const double cyclesPerSample =
            st * kULat / dMeasured * static_cast<double>(kSampleEvery);
        const double p = goertzelPower(sig, cyclesPerSample);
        totalPower += p;
        ++nCandidates;
        if (p > bestPower) {
            bestPower = p;
            bestSt = st;
        }
    }
    const double meanPower = totalPower / nCandidates;
    std::printf("peak St = %.4f (power %.3e, %.1fx scan mean)\n", bestSt,
                bestPower, bestPower / (meanPower > 0.0 ? meanPower : 1.0));

    // The peak must be a genuine spectral line, not broadband noise...
    TCHECK_MSG(bestPower > 5.0 * meanPower,
               "peak only %.1fx the scan mean — no clear shedding line",
               bestPower / (meanPower > 0.0 ? meanPower : 1.0));
    // ...and it must land in the published Re-150 Strouhal band.
    TCHECK_MSG(bestSt >= 0.17 && bestSt <= 0.21,
               "Strouhal %.4f outside 0.17..0.21", bestSt);

    return finish("m2_cylinder");
}
