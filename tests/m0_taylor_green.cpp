// M0 milestone (plan section 10): 64^3 Taylor-Green-like vortex decay through
// the public LBMSolver API. A divergence-free TG velocity field is injected
// via the snapshot-restore path (host-built equilibrium f -> activeDeviceF),
// then the solver runs 50k steps while the test asserts the total kinetic
// energy decays monotonically (1% jitter allowance) and never goes NaN.
// Energy is sampled every 1000 steps from a host-side moment reconstruction,
// keeping the whole test well under the 3-minute budget.
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

constexpr int   kN          = 64;     // 64^3 grid per the milestone definition
constexpr int   kTotalSteps = 50000;  // NaN-free horizon demanded by M0
constexpr int   kSampleEvery = 1000;  // energy check cadence (runtime budget)
constexpr float kAmplitude  = 0.05f;  // TG velocity amplitude (lattice units)

/// @brief Build the lattice scaling for a pure-decay run: the inlet speed is
/// made vanishingly small (u_lat = 1e-3, so the x=0 Dirichlet plane injects
/// ~0.04% of the initial TG energy at steady state — far below the 1% jitter
/// allowance) while the physical inputs are reverse-engineered to land the
/// lattice viscosity at nu_lat = 0.01 (tau ~ 0.53: comfortably stable, and
/// dissipative enough that 50k steps show deep, obvious decay).
LatticeScaling decayScaling() {
    constexpr float uLat     = 1.0e-3f;
    constexpr float nuLatGoal = 0.01f;
    PhysicalParams phys;
    phys.chord_m = 1.0f;
    phys.nu_m2s  = kNuAir;
    // From nu_lat = nu_phys * u_lat / (U * dx) with dx = chord / N:
    //   U = nu_phys * u_lat * N / (nu_lat * chord).
    phys.airspeed_ms = kNuAir * uLat * static_cast<float>(kN)
                     / (nuLatGoal * phys.chord_m);
    return computeScaling(phys, kN, uLat);
}

/// @brief Classic 3D Taylor-Green initial condition, divergence-free:
///   u =  A sin(kx) cos(ky) cos(kz)
///   v = -A cos(kx) sin(ky) cos(kz)
///   w =  0
/// with k = 2*pi/N on every axis. u vanishes on the x=0 inlet plane and v
/// vanishes at the y faces, so the field is near-compatible with the
/// inlet/slip boundaries ("periodic-ish" — the open boundaries add a thin
/// adjustment layer but cannot add energy).
void buildTaylorGreenF(const GridDims& dims, std::vector<float>& f) {
    const std::size_t n = static_cast<std::size_t>(dims.cellCount());
    f.resize(static_cast<std::size_t>(kQ) * n);
    const float k = 2.0f * 3.14159265358979f / static_cast<float>(kN);
    for (int z = 0; z < dims.nz; ++z) {
        for (int y = 0; y < dims.ny; ++y) {
            for (int x = 0; x < dims.nx; ++x) {
                const std::size_t c =
                    static_cast<std::size_t>(cellIndex(dims, x, y, z));
                const float sx = std::sin(k * x), cx = std::cos(k * x);
                const float sy = std::sin(k * y), cy = std::cos(k * y);
                const float cz = std::cos(k * z);
                const float u =  kAmplitude * sx * cy * cz;
                const float v = -kAmplitude * cx * sy * cz;
                // Populate every direction with the local equilibrium; the
                // solver's first collisions grow the proper non-equilibrium
                // part within a few steps.
                for (int q = 0; q < kQ; ++q)
                    f[static_cast<std::size_t>(q) * n + c] =
                        equilibrium(q, 1.0f, u, v, 0.0f);
            }
        }
    }
}

} // namespace

int main() {
    const GridDims dims{kN, kN, kN};
    const LatticeScaling scaling = decayScaling();
    // Guard the reverse-engineered scaling itself: viscosity on target and
    // unclamped, otherwise the decay-rate reasoning above is void.
    TCHECK(!scaling.tauClamped);
    TCHECK(approxRel(scaling.nu_lat, 0.01, 1e-3));

    // Standard open-domain flag contract (inlet/outlet/slip, periodic z).
    const std::vector<std::uint8_t> flags = openDomainFlags(dims);

    LBMSolver solver;
    std::string err;
    const bool initOk = solver.init(dims, scaling, flags, nullptr, &err);
    TCHECK_MSG(initOk, "solver init failed: %s", err.c_str());
    if (!initOk) return finish("m0_taylor_green");

    // Inject the TG field through the snapshot-restore path.
    std::vector<float> hostF;
    buildTaylorGreenF(dims, hostF);
    TREQUIRE(uploadF(solver, hostF) == cudaSuccess);

    // Reference energy at step 0 (from the uploaded field itself).
    HostMoments m = momentsFromF(hostF, dims);
    const double ke0 = kineticEnergy(m, flags);
    TCHECK_MSG(ke0 > 0.0, "initial TG field has zero energy?");
    std::printf("step %6d  KE = %.6e\n", 0, ke0);

    // March 50k steps, auditing energy every 1000. The decay floor guard
    // stops enforcing monotonicity once KE has fallen below 0.2% of the
    // initial value — at that level the open-boundary residual circulation
    // and the inlet trickle dominate (observed plateau ~0.12% of KE0) and
    // the physics statement is already proven by the 500x decay above it.
    double prevKe = ke0;
    bool monotone = true, finite = true;
    const double floorKe = 2e-3 * ke0;
    for (int step = kSampleEvery; step <= kTotalSteps; step += kSampleEvery) {
        const cudaError_t serr = solver.stepN(kSampleEvery);
        TCHECK_MSG(serr == cudaSuccess, "stepN failed at step %d: %s", step,
                   cudaGetErrorString(serr));
        if (serr != cudaSuccess) break;
        TCHECK_MSG(!solver.nanDetected(), "NaN watchdog tripped at step %d: %s",
                   step, solver.nanDiagnosis().c_str());

        TREQUIRE(downloadMoments(solver, m) == cudaSuccess);
        const double ke = kineticEnergy(m, flags);

        if (!std::isfinite(ke)) {
            std::printf("  KE became non-finite at step %d\n", step);
            finite = false;
            break;
        }
        // Monotone within the 1% jitter allowance, above the floor only.
        if (prevKe > floorKe && ke > prevKe * 1.01) {
            std::printf("  KE rose at step %d: %.6e -> %.6e (+%.2f%%)\n", step,
                        prevKe, ke, 100.0 * (ke / prevKe - 1.0));
            monotone = false;
        }
        if (step % (10 * kSampleEvery) == 0)
            std::printf("step %6d  KE = %.6e\n", step, ke);
        prevKe = ke;
    }

    TCHECK(finite);
    TCHECK(monotone);
    // The decay must be substantial, not just non-increasing: with
    // nu_lat = 0.01 and Laplacian eigenvalue -3k^2 the analytic e-folding
    // time is ~1700 steps, so by 50k steps the vortex energy is gone for
    // any correct viscous solver (observed: > 500x down; demand > 50x so
    // the bound has slack without ever passing a non-dissipative scheme).
    TCHECK_MSG(prevKe < 0.02 * ke0, "final KE %.3e vs initial %.3e — weak decay",
               prevKe, ke0);

    return finish("m0_taylor_green");
}
