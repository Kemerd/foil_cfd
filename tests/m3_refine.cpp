// M3 milestone (plan M-refine): two-level coupling sanity. A uniform
// freestream is the exact steady state of the open domain (equilibrium init
// at u_lat matches the inlet Dirichlet; slip walls are tangential; the outlet
// is zero-gradient), so with a solid-free refinement patch two-way coupled in
// the middle of the domain the field must STAY uniform: any interpolation,
// rescaling, restriction, or ghost-plane bug at the interface shows up as a
// velocity disturbance radiating from the patch box. The test marches 2000
// coarse steps (4000 fine) and asserts the velocity field never deviates from
// the freestream by more than 1% anywhere (observed: round-off level), the
// watchdog stays quiet, and the refinement bookkeeping reports active.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "lbm_test_util.h"
#include "sim/lbm_refine.cuh"
#include "sim/lbm_solver.h"
#include "sim/units.h"
#include "test_util.h"

using namespace foilcfd;
using namespace foilcfd::testutil;

namespace {

constexpr int kNx = 192, kNy = 96, kNz = 16;
constexpr int kTotalSteps  = 2000; // coarse steps (= 4000 fine sub-steps)
constexpr int kSampleEvery = 500;

/// @brief Comfortable-viscosity scaling (mirrors m0's reverse-engineering):
/// u_lat = 0.05 with nu_lat = 0.02 -> tau = 0.56, far from both the clamp and
/// the equilibrium validity bound. chordCells is nominal (no foil here).
LatticeScaling testScaling() {
    constexpr float uLat      = 0.05f;
    constexpr float nuLatGoal = 0.02f;
    constexpr int   nc        = 64;
    PhysicalParams phys;
    phys.chord_m = 1.0f;
    phys.nu_m2s  = kNuAir;
    phys.airspeed_ms = kNuAir * uLat * static_cast<float>(nc)
                     / (nuLatGoal * phys.chord_m);
    return computeScaling(phys, nc, uLat);
}

/// @brief Fine flag field for a solid-free patch: all Fluid plus the 2-cell
/// Interface shell on the x/y faces (z stays periodic). Local re-implementation
/// of the geom-layer stamper so this test links against sim/ sources only.
std::vector<std::uint8_t> solidFreeFineFlags(const GridDims& fineDims) {
    std::vector<std::uint8_t> flags(
        static_cast<std::size_t>(fineDims.cellCount()),
        static_cast<std::uint8_t>(CellFlag::Fluid));
    const auto iface = static_cast<std::uint8_t>(CellFlag::Interface);
    for (int z = 0; z < fineDims.nz; ++z) {
        for (int y = 0; y < fineDims.ny; ++y) {
            for (int x = 0; x < fineDims.nx; ++x) {
                const bool shell = x < kInterfaceShellFine
                                || x >= fineDims.nx - kInterfaceShellFine
                                || y < kInterfaceShellFine
                                || y >= fineDims.ny - kInterfaceShellFine;
                if (shell)
                    flags[static_cast<std::size_t>(
                        cellIndex(fineDims, x, y, z))] = iface;
            }
        }
    }
    return flags;
}

} // namespace

int main() {
    const GridDims dims{kNx, kNy, kNz};
    const LatticeScaling scaling = testScaling();
    TCHECK(!scaling.tauClamped);

    const std::vector<std::uint8_t> flags = openDomainFlags(dims);

    LBMSolver solver;
    std::string err;
    const bool initOk = solver.init(dims, scaling, flags, nullptr, &err);
    TCHECK_MSG(initOk, "solver init failed: %s", err.c_str());
    if (!initOk) return finish("m3_refine");

    // Patch in the middle of the domain, clear of every boundary face.
    PatchBox box;
    box.x0 = 60; box.x1 = 130;
    box.y0 = 30; box.y1 = 66;
    const GridDims fineDims = fineDimsFor(box, dims);
    std::printf("patch [%d,%d)x[%d,%d) -> fine %d x %d x %d\n",
                box.x0, box.x1, box.y0, box.y1,
                fineDims.nx, fineDims.ny, fineDims.nz);

    const bool refOk =
        solver.initRefinement(box, 2, solidFreeFineFlags(fineDims), &err);
    TCHECK_MSG(refOk, "initRefinement failed: %s", err.c_str());
    if (!refOk) return finish("m3_refine");
    TCHECK(solver.refinementInfo().active);
    // No solids anywhere: the force-source heuristic must stay on the fine
    // grid (vacuously, all zero solids are "covered" by the patch).
    TCHECK(solver.refinementInfo().forcesFromFine);

    // The freestream reference. The field now initializes AT REST and the inlet
    // ramps up (no impulsive shock), so u = u_lat is the SETTLED state, not the
    // t=0 state. March one flow-through past the velocity ramp so the domain is
    // uniformly at u_lat before we assert the interface holds it. nx/u_lat ~=
    // 3840 steps for one flow-through; 4500 leaves margin.
    const float u0 = scaling.u_lat;
    const float tol = 0.01f * u0; // 1% of freestream
    {
        const cudaError_t werr = solver.stepN(4500);
        TCHECK_MSG(werr == cudaSuccess, "warm-up stepN failed: %s",
                   cudaGetErrorString(werr));
        TCHECK_MSG(!solver.nanDetected(), "watchdog tripped during warm-up: %s",
                   solver.nanDiagnosis().c_str());
    }

    bool clean = true;
    for (int step = kSampleEvery; step <= kTotalSteps; step += kSampleEvery) {
        const cudaError_t serr = solver.stepN(kSampleEvery);
        TCHECK_MSG(serr == cudaSuccess, "stepN failed at step %d: %s", step,
                   cudaGetErrorString(serr));
        if (serr != cudaSuccess) break;
        TCHECK_MSG(!solver.nanDetected(), "watchdog tripped at step %d: %s",
                   step, solver.nanDiagnosis().c_str());

        HostMoments m;
        TREQUIRE(downloadMoments(solver, m) == cudaSuccess);

        // Max deviation from the freestream over every fluid cell — the
        // interface ring around the patch is exactly where a bug would leak.
        float maxDev = 0.0f;
        long long maxCell = -1;
        for (std::size_t i = 0; i < m.u.size(); ++i) {
            if (flags[i] != static_cast<std::uint8_t>(CellFlag::Fluid)) continue;
            const float dev = std::sqrt(
                (m.u[i] - u0) * (m.u[i] - u0)
                + m.v[i] * m.v[i] + m.w[i] * m.w[i]);
            if (dev > maxDev) {
                maxDev = dev;
                maxCell = static_cast<long long>(i);
            }
        }
        std::printf("step %5d  max |u - u0| = %.3e (%.4f%% of u0)\n", step,
                    maxDev, 100.0f * maxDev / u0);
        if (maxDev > tol) {
            const int x = static_cast<int>(maxCell % kNx);
            const int y = static_cast<int>((maxCell / kNx) % kNy);
            const int z = static_cast<int>(maxCell / (static_cast<long long>(kNx) * kNy));
            std::printf("  worst cell (%d, %d, %d) — patch is [%d,%d)x[%d,%d)\n",
                        x, y, z, box.x0, box.x1, box.y0, box.y1);
            clean = false;
        }
    }
    TCHECK_MSG(clean, "freestream disturbed by the refinement interface");

    return finish("m3_refine");
}
