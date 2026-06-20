// Testing Suite host model (2026-06-19): sweep-case expansion, result ranking,
// and CSV serialization. No solver / frame-loop coupling — see App::SweepRun in
// main.cpp for the in-frame state machine that consumes these.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include "sweep.h"

#include <algorithm>
#include <cstdio>

namespace foilcfd {
namespace {

/// @brief Apply one swept VG-axis value to a copy of the base VG list. The same
/// value is written to EVERY base entry (the sweep varies one parameter across
/// the whole array, which is what a builder comparing spacings/angles wants).
std::vector<VGParams> applyVgAxis(const std::vector<VGParams>& base,
                                  SweepVgAxis axis, float value) {
    std::vector<VGParams> out = base;
    for (VGParams& vg : out) {
        switch (axis) {
        case SweepVgAxis::Pitch:   vg.pitch_c  = value; break;
        case SweepVgAxis::Beta:    vg.beta_deg = value; break;
        case SweepVgAxis::Station: vg.x_c      = value; break;
        case SweepVgAxis::None:    break; // no change — caller shouldn't get here
        }
    }
    return out;
}

/// @brief The L/D used for ranking: the trailing-median when the gate is open,
/// else the live ratio. Diverged/non-converged callers handle separately.
float rankLD(const SweepResult& r) {
    return (r.forces.ldMedian != 0.0f) ? r.forces.ldMedian : r.forces.liftToDrag;
}

} // namespace

std::vector<SweepCase> buildSweepCases(const SweepParams& params,
                                       const std::vector<VGParams>& baseVgs) {
    std::vector<SweepCase> cases;
    const bool useVgs = params.includeVgs && !baseVgs.empty();

    for (const float aoa : params.aoaDegs) {
        if (!useVgs) {
            // Clean-airfoil case (a plain polar point).
            SweepCase c;
            c.aoaDeg = aoa;
            c.hasVgs = false;
            cases.push_back(std::move(c));
            continue;
        }
        if (params.vgAxis == SweepVgAxis::None) {
            // Fixed VG config, only AoA varies.
            SweepCase c;
            c.aoaDeg = aoa;
            c.hasVgs = true;
            c.vgs    = baseVgs;
            cases.push_back(std::move(c));
            continue;
        }
        // Sweep the chosen VG axis across [vgMin, vgMax] in vgSteps samples.
        const int steps = std::max(1, params.vgSteps);
        for (int s = 0; s < steps; ++s) {
            const float t = (steps == 1)
                                ? 0.0f
                                : static_cast<float>(s)
                                      / static_cast<float>(steps - 1);
            const float value = params.vgMin + (params.vgMax - params.vgMin) * t;
            SweepCase c;
            c.aoaDeg      = aoa;
            c.hasVgs      = true;
            c.vgs         = applyVgAxis(baseVgs, params.vgAxis, value);
            c.vgAxisValue = value;
            cases.push_back(std::move(c));
        }
    }
    return cases;
}

std::vector<int> rankSweepResults(const std::vector<SweepResult>& results) {
    std::vector<int> order(results.size());
    for (int i = 0; i < static_cast<int>(results.size()); ++i) order[i] = i;

    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const SweepResult& ra = results[a];
        const SweepResult& rb = results[b];
        // Trustworthy results (converged, not diverged) always rank above
        // untrustworthy ones, regardless of the numbers they printed.
        const bool ga = ra.converged && !ra.diverged;
        const bool gb = rb.converged && !rb.diverged;
        if (ga != gb) return ga; // a good, b bad -> a first

        // Primary: higher L/D is better.
        const float la = rankLD(ra), lb = rankLD(rb);
        if (std::abs(la - lb) > 1e-4f) return la > lb;

        // Tiebreak: a stronger (closer-to-Wendt) measured vortex when VGs exist.
        if (ra.audit.valid && rb.audit.valid)
            return ra.audit.ratio > rb.audit.ratio;
        return false; // stable_sort keeps the original order for true ties
    });
    return order;
}

std::string sweepResultsToCsv(const std::vector<SweepResult>& results) {
    std::string csv =
        "case,aoa_deg,vg_axis_value,cl,cd,l_over_d,audit_ratio,status,"
        "steps,seconds\n";
    char row[256];
    for (std::size_t i = 0; i < results.size(); ++i) {
        const SweepResult& r = results[i];
        const char* status = r.diverged    ? "diverged"
                             : r.converged ? "converged"
                                           : "timeout";
        // Averaged coefficients when the gate opened, else the live values so a
        // timeout row still carries whatever the run reached.
        const float cl = r.converged ? r.forces.clAvg : r.forces.cl;
        const float cd = r.converged ? r.forces.cdAvg : r.forces.cd;
        const float ld = r.converged ? r.forces.ldMedian : r.forces.liftToDrag;
        const float ratio = r.audit.valid ? r.audit.ratio : 0.0f;
        std::snprintf(row, sizeof row,
                      "%zu,%.2f,%.5f,%.5f,%.6f,%.4f,%.4f,%s,%lld,%.2f\n",
                      i, r.config.aoaDeg, r.config.vgAxisValue, cl, cd, ld,
                      ratio, status, r.steps, r.wallSeconds);
        csv += row;
    }
    return csv;
}

} // namespace foilcfd
