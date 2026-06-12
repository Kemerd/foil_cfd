// Physical <-> lattice unit conversion. This header is the ONLY place in the
// codebase where dx/dt/viscosity scaling, the tau stability clamp, and the
// effective-vs-target Reynolds number computation live (plan section 4.3).
// Every other module asks this header; nobody re-derives scaling locally.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <algorithm>
#include <cmath>

namespace foilcfd {

// ---------------------------------------------------------------------------
// Hard solver-stability constants. These are physics-of-the-method limits,
// not tunables; changing them changes what "stable" means everywhere.
// ---------------------------------------------------------------------------

/// Default lattice Mach proxy: inlet speed in lattice units. 0.08 keeps the
/// compressibility error of the LBM equilibrium (O(Ma^2)) under ~1%.
inline constexpr float kDefaultULat = 0.08f;

/// Absolute ceiling on u_lat. Above ~0.12 the D3Q19 equilibrium expansion
/// breaks down and TRT goes unstable regardless of viscosity.
inline constexpr float kMaxULat = 0.12f;

/// Minimum permitted relaxation time. tau -> 0.5 is the zero-viscosity limit;
/// 0.5005 leaves just enough physical viscosity that the Smagorinsky eddy
/// term can carry the rest of the stabilization (plan section 4.3).
inline constexpr float kMinTau = 0.5005f;

/// Kinematic viscosity of air at ~15 C [m^2/s]; the UI default.
inline constexpr float kNuAir = 1.48e-5f;

/// TRT "magic parameter" Lambda = 3/16: best wall location accuracy with
/// half-way bounce-back (plan section 4.1). Lives here because it is part of
/// the unit/relaxation contract shared by solver and tests.
inline constexpr float kTRTMagicLambda = 3.0f / 16.0f;

/// Smagorinsky constant for the LES subgrid model (plan section 4.1).
/// 0.1733 is the Smagorinsky-Lilly derivation C = (1/pi)*(2/(3*C_K))^(3/4)
/// with Kolmogorov constant C_K = 3/2 — the value proven out by GPU-LBM aero
/// runs at effectively unbounded Reynolds number. The previous 0.12 gave
/// less than HALF the eddy damping (the prefactor goes with Cs^2) and let
/// separated shear layers at high AoA run away at the tau stability clamp
/// (deep-stall divergence around 2 flow-throughs).
inline constexpr float kSmagorinskyCs = 0.1733f;

// ---------------------------------------------------------------------------
// User-facing physical inputs.
// ---------------------------------------------------------------------------

/// @brief What the user specifies: a real wing section in real air.
struct PhysicalParams {
    float chord_m     = 1.2f;    ///< Chord length [m].
    float airspeed_ms = 30.0f;   ///< Freestream airspeed [m/s].
    float nu_m2s      = kNuAir;  ///< Kinematic viscosity [m^2/s].
};

// ---------------------------------------------------------------------------
// Resolution presets and the High Fidelity mode bundle.
// ---------------------------------------------------------------------------

/// @brief Chord-resolution presets exposed in the UI (plan section 9.2).
/// Grid dimensions scale proportionally with chord cells so the domain keeps
/// the same physical extent (default layout: 3*N_c x 1.25*N_c x 0.375*N_c).
enum class ResolutionPreset {
    Fast,     ///< 192 cells/chord — interactive exploration.
    Default,  ///< 256 cells/chord — the standard grid (768 x 320 x 96).
    Fine,     ///< 320 cells/chord — High Fidelity entry level.
    Ultra,    ///< 384 cells/chord — High Fidelity, large VRAM only.
};

/// @brief Cells per chord for a given preset.
constexpr int chordCellsFor(ResolutionPreset p) {
    switch (p) {
        case ResolutionPreset::Fast:    return 192;
        case ResolutionPreset::Default: return 256;
        case ResolutionPreset::Fine:    return 320;
        case ResolutionPreset::Ultra:   return 384;
    }
    return 256;
}

/// @brief Settings bundle applied when the user toggles High Fidelity mode
/// (Mission statement: engineering-useful deltas for VG placement decisions).
///
/// High Fidelity trades interactivity for accuracy: finer grid, lower lattice
/// Mach (less compressibility error), force averaging over many flow-throughs,
/// and the dual-resolution Richardson trend readout with an error bar on
/// Cl/Cd deltas.
struct HighFidelityPreset {
    bool             enabled            = false;
    ResolutionPreset resolution         = ResolutionPreset::Fine;
    float            u_lat              = 0.05f; ///< Lowered from 0.08 — halves Ma^2 error.
    float            forceEmaFlowThroughs = 8.0f;  ///< EMA window in flow-through times.
    bool             richardsonReadout  = true;  ///< Run Fast-grid companion, extrapolate
                                                 ///< Cl/Cd trend, display an error bar.
};

/// @brief The standard (non-HiFi) defaults, for symmetry with the above.
struct StandardPreset {
    ResolutionPreset resolution           = ResolutionPreset::Default;
    float            u_lat                = kDefaultULat;
    float            forceEmaFlowThroughs = 1.0f;  ///< ~one flow-through (plan 4.4).
};

// ---------------------------------------------------------------------------
// The derived lattice scaling — the single conversion authority.
// ---------------------------------------------------------------------------

/// @brief Everything derived from (PhysicalParams, N_c, u_lat). Construct via
/// computeScaling(); never fill by hand, so the clamp logic cannot be bypassed.
struct LatticeScaling {
    // --- inputs echoed back (so a scaling object is self-describing) ---
    PhysicalParams phys;          ///< The physical inputs this scaling was built from.
    int   chordCells = 256;      ///< N_c: chord length in lattice cells.
    float u_lat      = kDefaultULat; ///< Inlet speed in lattice units (clamped to kMaxULat).

    // --- derived conversion factors ---
    float dx = 0.0f;             ///< Cell size [m] = chord_m / N_c.
    float dt = 0.0f;             ///< Time step [s] = u_lat * dx / airspeed_ms.

    // --- derived lattice-side quantities ---
    float nu_lat_target = 0.0f;  ///< Viscosity in lattice units the physics asks for.
    float nu_lat        = 0.0f;  ///< Viscosity actually used (after the tau clamp).
    float tau           = 0.0f;  ///< Relaxation time = 3*nu_lat + 0.5, clamped >= kMinTau.
    bool  tauClamped    = false; ///< True when the target Re was unreachable.

    // --- the honesty numbers (plan: DISPLAY BOTH in the UI) ---
    float reTarget    = 0.0f;    ///< Re the user asked for: U*c/nu_phys (often ~2.5e6).
    float reEffective = 0.0f;    ///< Re actually simulated: u_lat*N_c/nu_lat.

    // ------ conversion helpers (lattice <-> physical) ------

    /// @brief Convert a physical velocity [m/s] to lattice units.
    float velocityToLattice(float v_ms) const { return v_ms * dt / dx; }

    /// @brief Convert a lattice velocity to physical units [m/s].
    float velocityToPhysical(float v_lat) const { return v_lat * dx / dt; }

    /// @brief Convert a physical time span [s] to a (fractional) step count.
    float timeToSteps(float seconds) const { return seconds / dt; }

    /// @brief Convert a step count to physical seconds.
    float stepsToTime(float steps) const { return steps * dt; }

    /// @brief Convert a physical length [m] to lattice cells.
    float lengthToLattice(float meters) const { return meters / dx; }

    /// @brief Convert lattice cells to physical meters.
    float lengthToPhysical(float cells) const { return cells * dx; }

    /// @brief One flow-through time expressed in lattice steps: the time for
    /// the freestream to cross the full domain length nx (used to size the
    /// force EMA window and the convergence gate).
    float flowThroughSteps(int nx) const {
        return static_cast<float>(nx) / std::max(u_lat, 1e-6f);
    }

    /// @brief Lift/drag force coefficient normalization. Momentum-exchange
    /// forces come out in lattice units; the reference dynamic pressure is
    /// 0.5 * rho_lat * u_lat^2 over the projected area (chord x span), with
    /// rho_lat = 1 by LBM convention. C = F_lat / (0.5 * u_lat^2 * N_c * nz).
    /// @param forceLat Lattice-unit force component (x for drag, y for lift).
    /// @param spanCells Spanwise extent nz of the domain in cells.
    float forceToCoefficient(float forceLat, int spanCells) const {
        const float qref = 0.5f * u_lat * u_lat
                         * static_cast<float>(chordCells)
                         * static_cast<float>(spanCells);
        return (qref > 0.0f) ? forceLat / qref : 0.0f;
    }
};

/// @brief Build the full lattice scaling from physical inputs (the ONLY
/// sanctioned constructor for LatticeScaling).
///
/// Derivation (plan section 4.3):
///   dx = c_phys / N_c
///   dt = u_lat * dx / U_phys              (fixes the velocity scale)
///   nu_lat = nu_phys * dt / dx^2          (diffusive scaling of viscosity)
///   tau    = 3 * nu_lat + 0.5             (BGK/TRT relation, cs^2 = 1/3)
/// If tau falls below kMinTau the target Reynolds number is unreachable at
/// this resolution: we clamp tau, recompute the achievable nu_lat, and report
/// the *effective* Re alongside the target so the UI can show both numbers.
///
/// @param phys       Physical chord/airspeed/viscosity.
/// @param chordCells Chord resolution N_c (cells per chord).
/// @param u_lat      Desired lattice inlet speed; clamped to kMaxULat.
/// @return Fully-populated, internally-consistent LatticeScaling.
inline LatticeScaling computeScaling(const PhysicalParams& phys,
                                     int chordCells,
                                     float u_lat = kDefaultULat) {
    LatticeScaling s;
    s.phys       = phys;
    s.chordCells = chordCells;
    s.u_lat      = std::min(u_lat, kMaxULat);

    // Length and time scales follow directly from chord resolution + velocity scale.
    s.dx = phys.chord_m / static_cast<float>(chordCells);
    s.dt = s.u_lat * s.dx / phys.airspeed_ms;

    // Viscosity converts with the diffusive factor dt/dx^2.
    s.nu_lat_target = phys.nu_m2s * s.dt / (s.dx * s.dx);

    // tau = 3*nu + 1/2 in lattice units (cs^2 = 1/3). Clamp for stability:
    // below kMinTau the scheme is unusable, so we accept a higher viscosity
    // (= lower effective Re) and tell the user the truth about it.
    const float tauTarget = 3.0f * s.nu_lat_target + 0.5f;
    s.tau        = std::max(tauTarget, kMinTau);
    s.tauClamped = (tauTarget < kMinTau);
    s.nu_lat     = (s.tau - 0.5f) / 3.0f;

    // Both Reynolds numbers, for the "simulating at Re X (target Y)" readout.
    s.reTarget    = phys.airspeed_ms * phys.chord_m / phys.nu_m2s;
    s.reEffective = s.u_lat * static_cast<float>(chordCells) / s.nu_lat;
    return s;
}

/// @brief The maximum Reynolds number achievable at a given resolution and
/// lattice speed, i.e. with tau pinned at the clamp. Useful for the UI to
/// show "what resolution would I need" guidance without building a scaling.
inline float maxEffectiveRe(int chordCells, float u_lat = kDefaultULat) {
    const float nuMin = (kMinTau - 0.5f) / 3.0f;
    return std::min(u_lat, kMaxULat) * static_cast<float>(chordCells) / nuMin;
}

/// @brief Lattice scaling of a refined level derived from its parent
/// (plan M-refine, acoustic scaling): dx and dt shrink by the factor, u_lat
/// is unchanged, and the lattice viscosity GROWS by the factor so the
/// physical viscosity matches across levels:
///   nu_phys = nu_lat * dx^2/dt  ->  nu_lat_f = m * nu_lat_c.
/// Consequently tau_f = 3*m*nu_lat_c + 1/2 = m*(tau_c - 1/2) + 1/2, which is
/// FARTHER from the stability floor than the parent — a coupled fine level
/// can never be the stability bottleneck. Both Reynolds numbers are shared
/// across levels (Re_eff = u_lat * m*N_c / (m*nu_lat) is invariant): the
/// refined level buys spatial resolution at the same Re, not a higher Re.
/// @param parent The coarse level's scaling (from computeScaling()).
/// @param factor Refinement factor m (2 for the v1 patch).
/// @return Fully-populated fine-level scaling.
inline LatticeScaling refinedScaling(const LatticeScaling& parent, int factor) {
    LatticeScaling f = parent;
    const float m = static_cast<float>(factor);
    f.chordCells    = parent.chordCells * factor;
    f.dx            = parent.dx / m;
    f.dt            = parent.dt / m;
    f.nu_lat_target = parent.nu_lat_target * m;
    f.nu_lat        = parent.nu_lat * m;
    f.tau           = 3.0f * f.nu_lat + 0.5f;
    // reTarget/reEffective are level-invariant; tauClamped describes the
    // PARENT's clamp state (the fine tau is derived, never clamped itself).
    return f;
}

// ---------------------------------------------------------------------------
// Startup viscosity ramp (plan section 4.3): fresh runs start at 4x the target
// viscosity and ramp linearly to target over the first 2*nx steps, killing the
// impulsive-start pressure shock that otherwise rings through the domain.
// ---------------------------------------------------------------------------

/// @brief Viscosity multiplier at the start of the ramp.
inline constexpr float kStartupNuMultiplier = 4.0f;

/// @brief Ramp length in units of nx (domain lengths of steps).
inline constexpr int kStartupRampNxFactor = 2;

/// @brief Effective relaxation time during the startup ramp.
/// @param scaling   The run's lattice scaling (provides the target tau).
/// @param step      Current step index since the fresh start (0-based).
/// @param nx        Domain length in cells; ramp lasts kStartupRampNxFactor * nx steps.
/// @return tau to use this step: starts at the 4x-viscosity tau, linearly
///         relaxes to scaling.tau, then stays there.
inline float rampedTau(const LatticeScaling& scaling, long long step, int nx) {
    const long long rampSteps =
        static_cast<long long>(kStartupRampNxFactor) * static_cast<long long>(nx);
    if (step >= rampSteps) return scaling.tau;
    // Interpolate in viscosity space (linear nu ramp), then convert to tau —
    // ramping tau directly would make the nu ramp slightly non-linear.
    const float t      = static_cast<float>(step) / static_cast<float>(rampSteps);
    const float nuNow  = scaling.nu_lat * (kStartupNuMultiplier + t * (1.0f - kStartupNuMultiplier));
    return 3.0f * nuNow + 0.5f;
}

} // namespace foilcfd
