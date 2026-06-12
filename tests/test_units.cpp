// Pure-host checks of sim/units.h: dx/dt derivation, conversion round-trips,
// effective-vs-target Reynolds computation, the tau stability clamp at its
// boundary, the u_lat cap, the startup viscosity ramp, and the force
// normalization (plan section 4.3 — units.h is the single scaling authority,
// so this test pins its contract before any solver physics depends on it).
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include <cmath>

#include "sim/units.h"
#include "test_util.h"

using namespace foilcfd;
using namespace foilcfd::testutil;

namespace {

// -----------------------------------------------------------------------
// dx/dt derivation and the lattice<->physical conversion round trips.
// -----------------------------------------------------------------------
void checkScalingDerivation() {
    PhysicalParams phys;
    phys.chord_m     = 1.2f;
    phys.airspeed_ms = 30.0f;
    phys.nu_m2s      = kNuAir;
    const LatticeScaling s = computeScaling(phys, 256, 0.08f);

    // dx = c / N_c and dt = u_lat * dx / U, straight from plan 4.3.
    TCHECK(approxRel(s.dx, 1.2 / 256.0, 1e-6));
    TCHECK(approxRel(s.dt, 0.08 * (1.2 / 256.0) / 30.0, 1e-6));

    // The defining property of the velocity scale: the physical airspeed must
    // map exactly onto u_lat, and the chord onto N_c cells.
    TCHECK(approxRel(s.velocityToLattice(30.0f), 0.08, 1e-6));
    TCHECK(approxRel(s.lengthToLattice(1.2f), 256.0, 1e-6));

    // Round trips through every converter pair must be identity (to FP32 eps).
    TCHECK(approxRel(s.velocityToPhysical(s.velocityToLattice(13.7f)), 13.7, 1e-5));
    TCHECK(approxRel(s.lengthToPhysical(s.lengthToLattice(0.31f)), 0.31, 1e-5));
    TCHECK(approxRel(s.stepsToTime(s.timeToSteps(2.5f)), 2.5, 1e-5));

    // One flow-through of a 768-long domain at u_lat = 0.08 is 9600 steps.
    TCHECK(approxRel(s.flowThroughSteps(768), 9600.0, 1e-6));

    // Echoed inputs must survive so a scaling object is self-describing.
    TCHECK(s.chordCells == 256);
    TCHECK(approxRel(s.u_lat, 0.08, 1e-7));
}

// -----------------------------------------------------------------------
// Effective-Re computation, unclamped regime: a deliberately low target Re
// must be reproduced exactly (reEffective == reTarget, tau above the clamp).
// -----------------------------------------------------------------------
void checkEffectiveReUnclamped() {
    // Build Re = 1000 from physical inputs: U = Re * nu / c with c = 1 m.
    PhysicalParams phys;
    phys.chord_m     = 1.0f;
    phys.nu_m2s      = kNuAir;
    phys.airspeed_ms = 1000.0f * kNuAir / phys.chord_m;
    const LatticeScaling s = computeScaling(phys, 128, 0.08f);

    // nu_lat follows the closed form u_lat * N_c / Re = 0.08*128/1000.
    TCHECK(approxRel(s.nu_lat_target, 0.08 * 128.0 / 1000.0, 1e-4));
    TCHECK(!s.tauClamped);
    TCHECK(approxRel(s.tau, 3.0 * 0.08 * 128.0 / 1000.0 + 0.5, 1e-4));

    // Unclamped means the simulation honestly delivers the requested Re.
    TCHECK(approxRel(s.reTarget, 1000.0, 1e-3));
    TCHECK(approxRel(s.reEffective, s.reTarget, 1e-3));
}

// -----------------------------------------------------------------------
// Effective-Re computation, clamped regime: flight-scale Re (~2.4e6) is
// unreachable at N_c = 256; tau must pin at kMinTau and the effective Re
// must report the truth — u_lat * N_c / nu_min — instead of the target.
// -----------------------------------------------------------------------
void checkEffectiveReClamped() {
    PhysicalParams phys;          // defaults: 1.2 m chord, 30 m/s, air
    const LatticeScaling s = computeScaling(phys, 256, 0.08f);

    TCHECK(approxRel(s.reTarget, 30.0 * 1.2 / kNuAir, 1e-4)); // ~2.43e6
    TCHECK(s.tauClamped);
    TCHECK(approxRel(s.tau, kMinTau, 1e-7));

    // The achievable viscosity floor and the resulting effective Re.
    const double nuMin = (kMinTau - 0.5) / 3.0;
    TCHECK(approxRel(s.nu_lat, nuMin, 1e-4));
    TCHECK(approxRel(s.reEffective, 0.08 * 256.0 / nuMin, 1e-3));
    TCHECK(s.reEffective < s.reTarget);

    // maxEffectiveRe must agree with the clamped scaling's effective Re.
    TCHECK(approxRel(maxEffectiveRe(256, 0.08f), s.reEffective, 1e-3));
}

// -----------------------------------------------------------------------
// Tau clamp boundary: straddle maxEffectiveRe by +/-2% and verify the
// clamp engages exactly on the unreachable side.
// -----------------------------------------------------------------------
void checkTauClampBoundary() {
    const int   nc   = 192;
    const float ulat = 0.08f;
    const float reBoundary = maxEffectiveRe(nc, ulat);

    auto scaleFor = [&](float re) {
        PhysicalParams phys;
        phys.chord_m     = 1.0f;
        phys.nu_m2s      = kNuAir;
        phys.airspeed_ms = re * kNuAir; // U = Re * nu / c, c = 1
        return computeScaling(phys, nc, ulat);
    };

    // 2% inside the achievable range: no clamp, Re delivered as asked.
    const LatticeScaling below = scaleFor(reBoundary * 0.98f);
    TCHECK(!below.tauClamped);
    TCHECK(below.tau > kMinTau);
    TCHECK(approxRel(below.reEffective, below.reTarget, 2e-3));

    // 2% beyond: clamp engages, tau pins, effective Re saturates at the max.
    const LatticeScaling above = scaleFor(reBoundary * 1.02f);
    TCHECK(above.tauClamped);
    TCHECK(approxRel(above.tau, kMinTau, 1e-7));
    TCHECK(approxRel(above.reEffective, reBoundary, 2e-3));
    TCHECK(above.reEffective < above.reTarget);
}

// -----------------------------------------------------------------------
// u_lat hard cap: requests above kMaxULat must be clamped, never honored
// (above ~0.12 the D3Q19 equilibrium expansion breaks down).
// -----------------------------------------------------------------------
void checkULatCap() {
    PhysicalParams phys;
    const LatticeScaling s = computeScaling(phys, 256, 0.5f);
    TCHECK(approxRel(s.u_lat, kMaxULat, 1e-7));
    // dt must be derived from the CLAMPED u_lat, not the request.
    TCHECK(approxRel(s.dt, kMaxULat * s.dx / phys.airspeed_ms, 1e-5));

    // At/below the cap the request passes through untouched.
    const LatticeScaling ok = computeScaling(phys, 256, 0.05f);
    TCHECK(approxRel(ok.u_lat, 0.05, 1e-7));
}

// -----------------------------------------------------------------------
// Startup viscosity ramp (plan 4.3): 4x viscosity at step 0, LINEAR IN NU
// down to target over 2*nx steps, constant afterwards.
// -----------------------------------------------------------------------
void checkStartupRamp() {
    // Unclamped scaling so nu_lat is a clean nonzero number.
    PhysicalParams phys;
    phys.chord_m     = 1.0f;
    phys.nu_m2s      = kNuAir;
    phys.airspeed_ms = 500.0f * kNuAir;
    const LatticeScaling s = computeScaling(phys, 128, 0.08f);
    const int nx = 768;
    const long long rampSteps = 2LL * nx; // kStartupRampNxFactor * nx

    // Step 0: tau for 4x the target viscosity.
    TCHECK(approxRel(rampedTau(s, 0, nx), 3.0 * 4.0 * s.nu_lat + 0.5, 1e-5));

    // Midpoint: nu = nu * (4 + 0.5*(1-4)) = 2.5 * nu (linear in viscosity).
    TCHECK(approxRel(rampedTau(s, rampSteps / 2, nx),
                     3.0 * 2.5 * s.nu_lat + 0.5, 1e-5));

    // At and beyond the ramp end: exactly the run tau, forever.
    TCHECK(approxRel(rampedTau(s, rampSteps, nx), s.tau, 1e-6));
    TCHECK(approxRel(rampedTau(s, rampSteps * 10, nx), s.tau, 1e-6));

    // The ramp must never undershoot the target viscosity (monotone decrease).
    float prev = rampedTau(s, 0, nx);
    bool monotone = true;
    for (long long step = 1; step <= rampSteps; step += 64) {
        const float now = rampedTau(s, step, nx);
        if (now > prev + 1e-6f || now < s.tau - 1e-6f) monotone = false;
        prev = now;
    }
    TCHECK(monotone);
}

// -----------------------------------------------------------------------
// Force normalization: C = F / (0.5 * u_lat^2 * N_c * nz) with rho_lat = 1.
// -----------------------------------------------------------------------
void checkForceCoefficient() {
    PhysicalParams phys;
    const LatticeScaling s = computeScaling(phys, 256, 0.08f);
    const int nz = 96;
    const float qref = 0.5f * 0.08f * 0.08f * 256.0f * 96.0f;

    // A force of exactly 1.5 * qref must read back as coefficient 1.5.
    TCHECK(approxRel(s.forceToCoefficient(1.5f * qref, nz), 1.5, 1e-5));
    // Zero force is coefficient zero; the guard path returns 0 (never NaN).
    TCHECK(s.forceToCoefficient(0.0f, nz) == 0.0f);
    TCHECK(std::isfinite(s.forceToCoefficient(1.0f, 0)));
}

} // namespace

int main() {
    checkScalingDerivation();
    checkEffectiveReUnclamped();
    checkEffectiveReClamped();
    checkTauClampBoundary();
    checkULatCap();
    checkStartupRamp();
    checkForceCoefficient();
    return finish("test_units");
}
