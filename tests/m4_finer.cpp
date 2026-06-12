// M4-finer milestone (nested VG patch): THREE-level coupling sanity. Same logic
// as m3_refine one rung deeper — a uniform freestream is the exact steady state
// of the open domain, so with BOTH a solid-free fine patch AND a solid-free
// nested ("finer") patch two-way coupled inside it, the field must STAY uniform.
// Any bug in the fine->finer interpolation/rescale, the finer sub-stepping, the
// finer->fine restriction, the buffer time-level bookkeeping, or the finer
// ghost planes shows up as a velocity disturbance radiating from the nested box.
// The test marches 2000 coarse steps (4000 fine, 8000 finer sub-steps) and
// asserts the velocity field never deviates from the freestream by more than 1%
// anywhere, the watchdog stays quiet, and the refinement bookkeeping reports
// both levels active.
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
constexpr int kTotalSteps  = 2000; // coarse steps (= 4000 fine, 8000 finer)
constexpr int kSampleEvery = 500;

/// @brief Comfortable-viscosity scaling (identical to m3_refine): u_lat = 0.05
/// with nu_lat = 0.02 -> tau = 0.56, far from both the clamp and the
/// equilibrium validity bound.
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

/// @brief Solid-free flag field at a level's dims: all Fluid plus the 2-cell
/// Interface shell on the x/y faces (z stays periodic). Works for any level —
/// the fine patch and the nested finer patch share the same shell convention.
/// Local re-implementation of the geom stamper so this test links sim/ only.
std::vector<std::uint8_t> solidFreeFlags(const GridDims& d) {
    std::vector<std::uint8_t> flags(
        static_cast<std::size_t>(d.cellCount()),
        static_cast<std::uint8_t>(CellFlag::Fluid));
    const auto iface = static_cast<std::uint8_t>(CellFlag::Interface);
    for (int z = 0; z < d.nz; ++z) {
        for (int y = 0; y < d.ny; ++y) {
            for (int x = 0; x < d.nx; ++x) {
                const bool shell = x < kInterfaceShellFine
                                || x >= d.nx - kInterfaceShellFine
                                || y < kInterfaceShellFine
                                || y >= d.ny - kInterfaceShellFine;
                if (shell)
                    flags[static_cast<std::size_t>(cellIndex(d, x, y, z))] =
                        iface;
            }
        }
    }
    return flags;
}

/// @brief Solid-free flags PLUS a solid slab, so the fill/restriction must
/// handle solid cells inside the patch. Models a VG vane sitting near the box
/// edge: the interface-fill stencil straddles the slab, which is exactly the
/// configuration that injected garbage populations and diverged before the fill
/// was made solid-aware. Slab is given in this level's own cells.
std::vector<std::uint8_t> flagsWithSlab(const GridDims& d, int sx0, int sx1,
                                        int sy0, int sy1) {
    std::vector<std::uint8_t> flags = solidFreeFlags(d);
    const auto solid = static_cast<std::uint8_t>(CellFlag::Solid);
    for (int z = 0; z < d.nz; ++z)
        for (int y = sy0; y < sy1; ++y)
            for (int x = sx0; x < sx1; ++x) {
                const std::size_t i =
                    static_cast<std::size_t>(cellIndex(d, x, y, z));
                if (flags[i] != static_cast<std::uint8_t>(CellFlag::Interface))
                    flags[i] = solid; // never overwrite the shell
            }
    return flags;
}

/// @brief Run a solid-in-box configuration purely for STABILITY: a slab near
/// the nested box edge used to inject garbage through the fill stencil and blow
/// up. Here we only assert the watchdog never trips over a long march — the
/// wake makes the field non-uniform, so the freestream check does not apply.
bool runSlabStability() {
    const GridDims dims{kNx, kNy, kNz};
    const LatticeScaling scaling = testScaling();

    LBMSolver solver;
    std::string err;
    if (!solver.init(dims, scaling, openDomainFlags(dims), nullptr, &err)) {
        std::printf("slab: solver init failed: %s\n", err.c_str());
        return false;
    }

    PatchBox box;
    box.x0 = 60; box.x1 = 130;
    box.y0 = 30; box.y1 = 66;
    const GridDims fineDims = fineDimsFor(box, dims);
    if (!solver.initRefinement(box, 2, solidFreeFlags(fineDims), &err)) {
        std::printf("slab: initRefinement failed: %s\n", err.c_str());
        return false;
    }

    PatchBox finerBox;
    finerBox.x0 = 40; finerBox.x1 = 100;
    finerBox.y0 = 24; finerBox.y1 = 56;
    const GridDims finerDims = fineDimsFor(finerBox, fineDims, 2);
    // A slab hugging the finer box's lower-x interior, a few cells off the
    // shell — close enough that the fill stencil for the shell ring reaches
    // into it. This is the VG-vane-near-the-edge case.
    const std::vector<std::uint8_t> finerSlab =
        flagsWithSlab(finerDims, /*sx0=*/8, /*sx1=*/16,
                      /*sy0=*/finerDims.ny / 2 - 6, /*sy1=*/finerDims.ny / 2 + 6);
    if (!solver.initFinerRefinement(finerBox, 2, finerSlab, &err)) {
        std::printf("slab: initFinerRefinement failed: %s\n", err.c_str());
        return false;
    }

    // March well past where the divergence used to hit (step ~200).
    for (int step = 0; step < 4; ++step) {
        const cudaError_t serr = solver.stepN(500);
        if (serr != cudaSuccess) {
            std::printf("slab: stepN failed: %s\n", cudaGetErrorString(serr));
            return false;
        }
        if (solver.nanDetected()) {
            std::printf("slab: DIVERGED at ~%d steps: %s\n", (step + 1) * 500,
                        solver.nanDiagnosis().c_str());
            return false;
        }
    }
    std::printf("slab: stable over 2000 steps with a solid in the nested box\n");
    return true;
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
    if (!initOk) return finish("m4_finer");

    // Fine patch in the middle of the domain, clear of every boundary face.
    PatchBox box;
    box.x0 = 60; box.x1 = 130;
    box.y0 = 30; box.y1 = 66;
    const GridDims fineDims = fineDimsFor(box, dims);
    std::printf("fine patch [%d,%d)x[%d,%d) -> fine %d x %d x %d\n",
                box.x0, box.x1, box.y0, box.y1,
                fineDims.nx, fineDims.ny, fineDims.nz);

    const bool refOk =
        solver.initRefinement(box, 2, solidFreeFlags(fineDims), &err);
    TCHECK_MSG(refOk, "initRefinement failed: %s", err.c_str());
    if (!refOk) return finish("m4_finer");

    // Nested ("finer") box, in FINE cells: a small region well inside the fine
    // patch and clear of its Interface shell (the fine patch is 140 x 72 fine
    // cells; this box sits comfortably in the interior).
    PatchBox finerBox;
    finerBox.x0 = 40; finerBox.x1 = 100;
    finerBox.y0 = 24; finerBox.y1 = 56;
    const GridDims finerDims = fineDimsFor(finerBox, fineDims, 2);
    std::printf("nested box [%d,%d)x[%d,%d) (fine cells) -> finer %d x %d x %d\n",
                finerBox.x0, finerBox.x1, finerBox.y0, finerBox.y1,
                finerDims.nx, finerDims.ny, finerDims.nz);

    const bool finerOk =
        solver.initFinerRefinement(finerBox, 2, solidFreeFlags(finerDims), &err);
    TCHECK_MSG(finerOk, "initFinerRefinement failed: %s", err.c_str());
    if (!finerOk) return finish("m4_finer");

    const RefinementInfo ri = solver.refinementInfo();
    TCHECK(ri.active);        // fine level up
    TCHECK(ri.finerActive);   // nested level up
    TCHECK(ri.finerFactor == 2);
    // Effective resolution vs coarse is 2x (fine) * 2x (finer) = 4x.
    TCHECK(ri.factor * ri.finerFactor == 4);

    // The freestream reference. The field now initializes AT REST and the inlet
    // ramps up (no impulsive shock), so u = u_lat is the SETTLED state. March one
    // flow-through past the velocity ramp so the domain is uniformly at u_lat
    // before asserting the nested interface holds it. ~3840 steps/flow-through.
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

        // Max deviation from the freestream over every fluid cell — the nested
        // box's restriction writes the coarse macro arrays through the fine
        // overlap, so a finer-level bug surfaces in the coarse field here.
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
            std::printf("  worst cell (%d, %d, %d)\n", x, y, z);
            clean = false;
        }
    }
    TCHECK_MSG(clean, "freestream disturbed by the nested refinement interface");

    // Regression guard for the solid-aware fill: a vane-like slab near the
    // nested box edge used to poison the fill stencil and diverge. It must now
    // stay stable (the fill skips solid corners, the vane is resolved by the
    // child grid's bounce-back).
    TCHECK_MSG(runSlabStability(),
               "nested patch diverged with a solid in the box (fill not "
               "skipping solid parent corners?)");

    return finish("m4_finer");
}
