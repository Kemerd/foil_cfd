// M1 milestone (plan section 10): lid-driven cavity at Re 1000 on a 2D-thin
// 128 x 128 x 4 grid, driven through the public LBMSolver API. The moving lid
// is realized with the solver's own boundary vocabulary: the top row carries
// the Inlet flag (equilibrium Dirichlet at u = (u_lat, 0, 0)) while the other
// three walls are Solid (half-way bounce-back). After running to steady state
// the u-velocity profile along the vertical centerline is compared against
// the Ghia, Ghia & Shin (1982) reference points, hard-coded below, within 7%
// of the lid speed at every station.
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

constexpr int   kNx = 128, kNy = 128, kNz = 4;
constexpr float kULat       = 0.08f;   // lid speed in lattice units
constexpr float kRe         = 1000.0f; // cavity Reynolds number
constexpr int   kChunkSteps = 4000;    // steps between convergence probes
constexpr int   kMinSteps   = 40000;   // never declare convergence before this
constexpr int   kMaxSteps   = 240000;  // hard cap (~95 lid transits)
constexpr double kTol       = 0.07;    // |u_sim - u_ghia| <= 7% of lid speed

// Cavity interior: fluid rows/columns 1 .. n-2; the bounce-back wall planes
// sit half a cell outside the last fluid cell, so the physical cavity side is
// L = n - 2 cells, spanning lattice coordinate 0.5 .. n - 1.5 on each axis.
constexpr int   kL = kNx - 2;

// -----------------------------------------------------------------------
// Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982), Table I: u-velocity along
// the vertical line through the cavity center, Re = 1000 (129x129 grid).
// y is measured from the stationary bottom wall (0) to the lid (1).
// The y = 0 and y = 1 entries are the trivial wall/lid values and are
// excluded from the comparison (they test the BC stencil, not the flow).
// -----------------------------------------------------------------------
struct GhiaPoint { double y, u; };
constexpr GhiaPoint kGhiaU[] = {
    {0.0547, -0.18109}, {0.0625, -0.20196}, {0.0703, -0.22220},
    {0.1016, -0.29730}, {0.1719, -0.38289}, {0.2813, -0.27805},
    {0.4531, -0.10648}, {0.5000, -0.06080}, {0.6172,  0.05702},
    {0.7344,  0.18719}, {0.8516,  0.33304}, {0.9531,  0.46604},
    {0.9609,  0.51117}, {0.9688,  0.57492}, {0.9766,  0.65928},
};

/// @brief Cavity flag field: Fluid bulk, Solid left/right/bottom walls, the
/// lid as an Inlet row (equilibrium Dirichlet at the lattice inflow speed —
/// exactly a lid moving in +x). Solid wins at the two lid corners so the
/// velocity discontinuity lives inside bounce-back cells, not the Dirichlet
/// row. z gets no flags: the solver's periodic z fits the quasi-2D setup.
std::vector<std::uint8_t> cavityFlags(const GridDims& dims) {
    std::vector<std::uint8_t> flags(
        static_cast<std::size_t>(dims.cellCount()),
        static_cast<std::uint8_t>(CellFlag::Fluid));
    for (int z = 0; z < dims.nz; ++z) {
        for (int x = 0; x < dims.nx; ++x) // lid first, walls overwrite corners
            flags[static_cast<std::size_t>(cellIndex(dims, x, dims.ny - 1, z))] =
                static_cast<std::uint8_t>(CellFlag::Inlet);
        for (int x = 0; x < dims.nx; ++x)
            flags[static_cast<std::size_t>(cellIndex(dims, x, 0, z))] =
                static_cast<std::uint8_t>(CellFlag::Solid);
        for (int y = 0; y < dims.ny; ++y) {
            flags[static_cast<std::size_t>(cellIndex(dims, 0, y, z))] =
                static_cast<std::uint8_t>(CellFlag::Solid);
            flags[static_cast<std::size_t>(cellIndex(dims, dims.nx - 1, y, z))] =
                static_cast<std::uint8_t>(CellFlag::Solid);
        }
    }
    return flags;
}

/// @brief Extract the normalized u(y) profile on the vertical centerline:
/// average over the spanwise direction and the two columns straddling the
/// geometric center (the cavity center x = (nx-1)/2 falls between columns).
/// @return u / u_lid for fluid rows j = 1 .. ny-2 (index 0 <-> row 1).
std::vector<double> centerlineU(const HostMoments& m, const GridDims& dims) {
    const int cxLo = dims.nx / 2 - 1, cxHi = dims.nx / 2; // 63 and 64
    std::vector<double> profile(static_cast<std::size_t>(dims.ny - 2), 0.0);
    for (int j = 1; j <= dims.ny - 2; ++j) {
        double sum = 0.0;
        for (int z = 0; z < dims.nz; ++z) {
            sum += m.u[static_cast<std::size_t>(cellIndex(dims, cxLo, j, z))];
            sum += m.u[static_cast<std::size_t>(cellIndex(dims, cxHi, j, z))];
        }
        profile[static_cast<std::size_t>(j - 1)] =
            sum / (2.0 * dims.nz) / static_cast<double>(kULat);
    }
    return profile;
}

/// @brief Linear interpolation of the centerline profile at normalized cavity
/// height yNorm in [0,1]. Fluid row j sits at yNorm = (j - 0.5) / L (wall
/// plane at lattice y = 0.5 from half-way bounce-back).
double profileAt(const std::vector<double>& profile, double yNorm) {
    const double yCell = yNorm * kL + 0.5;       // back to lattice row space
    const double s = yCell - 1.0;                // profile[0] is row j = 1
    const int lo = static_cast<int>(std::floor(s));
    const int hi = lo + 1;
    if (lo < 0) return profile.front();
    if (hi >= static_cast<int>(profile.size())) return profile.back();
    const double t = s - lo;
    return profile[static_cast<std::size_t>(lo)] * (1.0 - t)
         + profile[static_cast<std::size_t>(hi)] * t;
}

} // namespace

int main() {
    const GridDims dims{kNx, kNy, kNz};

    // Re = u_lat * L / nu_lat with L = nx - 2: reverse-engineer the physical
    // inputs (chord 1 m of air) so computeScaling lands exactly there.
    PhysicalParams phys;
    phys.chord_m     = 1.0f;
    phys.nu_m2s      = kNuAir;
    phys.airspeed_ms = kRe * kNuAir / phys.chord_m; // Re_target = 1000
    const LatticeScaling scaling = computeScaling(phys, kL, kULat);
    TCHECK(!scaling.tauClamped);
    TCHECK(approxRel(scaling.nu_lat, kULat * kL / kRe, 1e-3)); // ~0.0101

    const std::vector<std::uint8_t> flags = cavityFlags(dims);
    LBMSolver solver;
    std::string err;
    const bool initOk = solver.init(dims, scaling, flags, nullptr, &err);
    TCHECK_MSG(initOk, "solver init failed: %s", err.c_str());
    if (!initOk) return finish("m1_cavity");

    // The cavity must start from rest, not from the uniform-inflow field the
    // solver's cold start assumes for the wind-tunnel domain: overwrite with
    // a quiescent equilibrium via the snapshot-restore path.
    {
        const std::size_t n = static_cast<std::size_t>(dims.cellCount());
        std::vector<float> rest(static_cast<std::size_t>(kQ) * n);
        for (int q = 0; q < kQ; ++q) {
            const float feq = equilibrium(q, 1.0f, 0.0f, 0.0f, 0.0f);
            for (std::size_t c = 0; c < n; ++c)
                rest[static_cast<std::size_t>(q) * n + c] = feq;
        }
        TREQUIRE(uploadF(solver, rest) == cudaSuccess);
    }

    // March to steady state: probe the centerline profile every chunk and
    // stop when it freezes (max change < 5e-4 of lid speed) after the
    // minimum step count, or at the hard cap.
    HostMoments moments;
    std::vector<double> profile, prevProfile;
    int steps = 0;
    while (steps < kMaxSteps) {
        const cudaError_t serr = solver.stepN(kChunkSteps);
        steps += kChunkSteps;
        TCHECK_MSG(serr == cudaSuccess, "stepN failed at step %d: %s", steps,
                   cudaGetErrorString(serr));
        if (serr != cudaSuccess) break;
        TCHECK_MSG(!solver.nanDetected(), "NaN watchdog tripped at step %d: %s",
                   steps, solver.nanDiagnosis().c_str());
        if (solver.nanDetected()) break;

        TREQUIRE(downloadMoments(solver, moments) == cudaSuccess);
        profile = centerlineU(moments, dims);

        // Profile must stay finite even before convergence.
        bool finite = true;
        for (double v : profile)
            if (!std::isfinite(v)) finite = false;
        TCHECK_MSG(finite, "non-finite centerline value at step %d", steps);
        if (!finite) break;

        if (!prevProfile.empty()) {
            double maxDelta = 0.0;
            for (std::size_t i = 0; i < profile.size(); ++i)
                maxDelta = std::max(maxDelta,
                                    std::fabs(profile[i] - prevProfile[i]));
            if (steps % (5 * kChunkSteps) == 0)
                std::printf("step %6d  profile max-delta = %.3e\n", steps, maxDelta);
            if (steps >= kMinSteps && maxDelta < 5e-4) {
                std::printf("converged at step %d (max-delta %.3e)\n", steps, maxDelta);
                break;
            }
        }
        prevProfile = profile;
    }
    TREQUIRE(!profile.empty());

    // Compare against Ghia et al. at every reference station, in units of
    // the lid speed (relative-to-local-u comparison would blow up where the
    // profile crosses zero — normalizing by U_lid is the standard practice).
    std::printf("%8s %10s %10s %8s\n", "y", "u_ghia", "u_sim", "delta");
    for (const GhiaPoint& g : kGhiaU) {
        const double uSim = profileAt(profile, g.y);
        const double delta = std::fabs(uSim - g.u);
        std::printf("%8.4f %10.5f %10.5f %8.4f%s\n", g.y, g.u, uSim, delta,
                    delta <= kTol ? "" : "  <-- FAIL");
        TCHECK_MSG(delta <= kTol,
                   "y=%.4f: u_sim=%.5f vs Ghia %.5f (|delta| %.4f > %.2f)",
                   g.y, uSim, g.u, delta, kTol);
    }

    return finish("m1_cavity");
}
