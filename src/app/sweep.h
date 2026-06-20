// Testing Suite (2026-06-19): a general, fully-configurable parameter sweep
// runner. It drives the SAME solver the interactive UI uses, one case at a
// time, to convergence — so it works on a CLEAN airfoil (a Cl/Cd/L-D-vs-AoA
// polar) just as well as on a VG configuration (parametric OR custom-STL). VGs
// are one OPTIONAL sweep dimension, never required.
//
// This header is the pure-host model: the case expansion, the ranking, and the
// CSV serialization. It deliberately knows nothing about the solver or the
// frame loop — main.cpp owns the in-frame state machine (App::SweepRun) that
// actually steps each case, so the sweep never blocks the UI and the user can
// watch every case run live and scrub any result back into the interactive view.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <string>
#include <vector>

#include "../geom/vg.h"          // VGParams
#include "../geom/vg_audit.h"    // VGAuditReadout
#include "../sim/lbm_solver.h"   // ForceReadout

namespace foilcfd {

// ===========================================================================
// Sweep configuration (the editable mirror lives on UIParams::sweepParams).
// ===========================================================================

/// @brief Which VG parameter the sweep varies across its steps. None = the VGs
/// (if any) stay fixed and only AoA varies — i.e. a plain polar of whatever
/// geometry is currently configured.
enum class SweepVgAxis { None, Pitch, Beta, Station };

/// @brief A full sweep definition. The case list is the Cartesian product of
/// the AoA set (outer) and the VG-axis steps (inner); with vgAxis == None there
/// is exactly one VG configuration per AoA (the base one, or none).
struct SweepParams {
    /// AoA values to sweep [deg]. Editable list. Default = a cruise point plus
    /// the high-alpha band where VG separation control matters most.
    std::vector<float> aoaDegs{4.0f, 15.0f, 16.0f, 17.0f, 18.0f, 20.0f};
    bool        includeVgs = false;    ///< Apply the base VG config to every
                                       ///< case (off = clean-airfoil polar).
    SweepVgAxis vgAxis = SweepVgAxis::None; ///< VG parameter to vary (needs
                                       ///< includeVgs and a non-empty base VG).
    float vgMin = 0.04f;               ///< VG-axis sweep range, low end.
    float vgMax = 0.10f;               ///< VG-axis sweep range, high end.
    int   vgSteps = 3;                 ///< Number of VG-axis samples (>= 1).

    /// Extra flow-throughs to AVERAGE past the force gate before recording a
    /// case — longer than the interactive default so a sweep number is trusted.
    float averagingFlowThroughs = 3.0f;
    /// Hard safety cap on solver steps per case: a case that never opens the
    /// force gate (stalled / pathological) is recorded non-converged and the
    /// sweep moves on rather than hanging forever.
    long long maxStepsPerCase = 4'000'000;
};

// ===========================================================================
// One case + its result.
// ===========================================================================

/// @brief A single resolved sweep case: an AoA and the exact VG configuration
/// to run. vgs is empty for a clean-airfoil case.
struct SweepCase {
    float aoaDeg = 0.0f;
    bool  hasVgs = false;
    std::vector<VGParams> vgs;      ///< The VG config for this case (may be empty).
    float vgAxisValue = 0.0f;       ///< The swept VG-axis value (for the table /
                                    ///< CSV); 0 when vgAxis == None.
};

/// @brief The measured outcome of one case after it converged (or failed).
struct SweepResult {
    SweepCase      config;
    ForceReadout   forces;          ///< Averaged Cl/Cd/L-D (.valid gates trust).
    VGAuditReadout audit;           ///< Wendt honesty ratio (valid only w/ VGs).
    bool           converged = false; ///< Force gate opened within the cap.
    bool           diverged  = false; ///< NaN / CUDA failure during the case.
    long long      steps     = 0;     ///< Solver steps this case ran.
    double         wallSeconds = 0.0; ///< Wall-clock time the case took.
};

// ===========================================================================
// Pure-host operations (unit-testable; no solver, no globals).
// ===========================================================================

/// @brief Expand a SweepParams + the user's base VG list into the full ordered
/// case list (AoA outer, VG-axis inner). With includeVgs == false (or an empty
/// base list) every case is clean. With vgAxis == None there is one VG case per
/// AoA; otherwise vgSteps cases per AoA, each with the swept parameter applied
/// to EVERY base VG entry.
/// @param params   The sweep definition.
/// @param baseVgs  The user's currently-configured VG entries (the template).
/// @return Ordered case list (empty when aoaDegs is empty).
std::vector<SweepCase> buildSweepCases(const SweepParams& params,
                                       const std::vector<VGParams>& baseVgs);

/// @brief Rank results best-first: primary key L/D median (descending), falling
/// back to the live L/D when the median is unavailable; ties broken by the VG
/// audit ratio (descending) when VGs are present. Diverged and non-converged
/// results always sink to the bottom (they are untrustworthy regardless of the
/// numbers they happened to print).
/// @param results The recorded results.
/// @return Indices into @p results, best first.
std::vector<int> rankSweepResults(const std::vector<SweepResult>& results);

/// @brief Serialize results to a CSV string (RFC-4180-ish: a header row + one
/// row per result, numbers unquoted). Column order matches the on-screen table:
/// case index, AoA, VG axis value, Cl, Cd, L/D, audit ratio, status, steps,
/// seconds. Pure string builder so it can be unit-tested without disk I/O.
/// @param results The recorded results (any order; rows follow input order).
/// @return The CSV text, newline-terminated per row.
std::string sweepResultsToCsv(const std::vector<SweepResult>& results);

} // namespace foilcfd
