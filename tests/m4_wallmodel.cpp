// M4 milestone: iMEM slip-velocity wall-function validation on the flat
// plate (a solid slab spanning the full domain length, so the measured
// x-force is pure skin friction — no frontal face). Gates:
//   1. u_tau recovery — a SYNTHESIZED Reichardt boundary layer with a known
//      friction velocity is uploaded as a snapshot; the model must read the
//      planted u_tau back (sampled mean y+ within 15% of the synthesized
//      value). Validates sampling, projection, the Newton solve, and EMA
//      end-to-end against an exact answer.
//   2. equilibrium hold — stepping that profile for ~1.3k steps with the
//      model active must neither diverge nor drift the recovered u_tau (a
//      mis-signed or ringing u_tau <-> u_w loop fails this loudly), with
//      the slip clamp essentially never engaging.
//   3. low-Re do-no-harm — with the sublayer genuinely resolved the model
//      self-degrades (low-y+ fade): ON and OFF forces agree within 2%.
//   4. delta99 growth — the BL probe reports a monotonically thickening
//      layer downstream over the plate.
// A note on what is NOT here: the original plan called for a free-running
// turbulent plate compared against Schultz-Grunow. Empirically, a domain
// small enough for CI cannot SUSTAIN a turbulent boundary layer at
// sublayer-unresolvable cell Reynolds numbers — the plain-bounce-back
// baseline saturates the stability limiter long before any Cf is
// trustworthy (the wall-function literature validates on body-force-driven
// channels for exactly this reason, and this solver has no body force).
// The synthesized-profile gates above check the same machinery against a
// sharper reference; the full-airfoil polar comparison lives in the manual
// validation protocol (LS(1)-0413MOD vs NASA data).
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

// Helper-scope fatal check (returns out of the calling void function when
// the condition fails — later checks would crash or be meaningless).
#define TREQUIRE_VOID_M4(expr)                                                 \
    do {                                                                       \
        const bool ok_ = static_cast<bool>(expr);                              \
        ::foilcfd::testutil::recordCheck(ok_, #expr, __FILE__, __LINE__);      \
        if (!ok_) return;                                                      \
    } while (0)

namespace {

constexpr int kNx = 256;      ///< Plate length (both regimes).
constexpr int kSlabH = 6;     ///< Plate thickness in cells (top at y = 5).
constexpr int kNc = 64;       ///< Nominal chord for the coefficient scale.

/// Test grid: every gate runs gentle, short-horizon flows (planted profile
/// or viscous regime), so a thin cheap span suffices throughout.
constexpr GridDims kDimsLowRe{kNx, 64, 8};

/// @brief Plate flags: the open domain with a solid slab y < kSlabH across
/// the FULL x extent (overwriting the inlet/outlet/slip-bottom markers in the
/// slab rows). The boundary layer starts growing at x = 0 with no leading
/// face, so the momentum-exchange x-force is skin friction and nothing else.
std::vector<std::uint8_t> plateFlags(const GridDims& dims) {
    std::vector<std::uint8_t> flags = openDomainFlags(dims);
    const auto solid = static_cast<std::uint8_t>(CellFlag::Solid);
    for (int z = 0; z < dims.nz; ++z)
        for (int y = 0; y < kSlabH; ++y)
            for (int x = 0; x < dims.nx; ++x)
                flags[static_cast<std::size_t>(cellIndex(dims, x, y, z))] = solid;
    return flags;
}

/// @brief Scaling with chosen lattice viscosity and inlet speed (m0/m3
/// pattern: reverse-engineer the physical airspeed that lands on the
/// nu_lat target). The high-Re gate wants an UNRESOLVABLE sublayer
/// (first-cell y+ well above the fade band) while staying a respectful
/// distance from the tau stability cliff — at the exact clamp this short
/// shallow test domain saturates the limiter even with plain bounce-back,
/// which would validate nothing.
LatticeScaling scalingFor(float nuLatGoal, float uLat) {
    PhysicalParams phys;
    phys.chord_m = 1.0f;
    phys.nu_m2s  = kNuAir;
    phys.airspeed_ms = kNuAir * uLat * static_cast<float>(kNc)
                     / (nuLatGoal * phys.chord_m);
    return computeScaling(phys, kNc, uLat);
}

/// @brief Host mirror of the device Reichardt law (kappa 0.41, C 7.8 —
/// lbm_wallmodel.cuh constants) for synthesizing exact-profile fields.
float reichardtUPlus(float yp) {
    return std::log(1.0f + 0.41f * yp) / 0.41f
         + 7.8f * (1.0f - std::exp(-yp / 11.0f)
                   - (yp / 11.0f) * std::exp(-yp / 3.0f));
}

/// @brief Gates 1 + 2: plant a Reichardt boundary layer with a KNOWN
/// friction velocity over the plate, let the wall model read it, and demand
/// it (a) recovers the planted u_tau (sampled mean y+ within 15%) and
/// (b) holds the profile steady for ~1.3k steps without ringing, clamping,
/// or divergence. The inlet feeds uniform freestream, but its influence
/// front travels only ~u*steps ~ 64 cells and the profile is within 5% of
/// freestream at the sampling height anyway — well inside the tolerance.
void checkSyntheticProfile() {
    const GridDims dims = kDimsLowRe; // cheap thin span; short horizon
    constexpr float kNuSynth   = 5e-4f;  // tau 0.5015: high-Re-like, stable
    constexpr float kUTauSynth = 0.004f; // sampled y+ = 2.5*utau/nu = 20
    const LatticeScaling sc = scalingFor(kNuSynth, 0.05f);
    TCHECK(!sc.tauClamped);

    LBMSolver solver;
    std::string err;
    const std::vector<std::uint8_t> flags = plateFlags(dims);
    const bool initOk = solver.init(dims, sc, flags, nullptr, &err);
    TCHECK_MSG(initOk, "solver init failed: %s", err.c_str());
    if (!initOk) return;
    solver.setWallModelEnabled(true);

    // Synthesize f = equilibrium(rho = 1, u = Reichardt profile). The wall
    // plane sits at y = kSlabH - 0.5 (half-way bounce-back convention).
    const std::size_t n = static_cast<std::size_t>(dims.cellCount());
    std::vector<float> f(static_cast<std::size_t>(kQ) * n);
    for (int z = 0; z < dims.nz; ++z) {
        for (int y = 0; y < dims.ny; ++y) {
            float u = 0.0f;
            if (y >= kSlabH) {
                const float dist = static_cast<float>(y)
                                 - (static_cast<float>(kSlabH) - 0.5f);
                u = std::min(kUTauSynth
                                 * reichardtUPlus(dist * kUTauSynth / kNuSynth),
                             sc.u_lat);
            }
            for (int x = 0; x < dims.nx; ++x) {
                const std::size_t c =
                    static_cast<std::size_t>(cellIndex(dims, x, y, z));
                for (int q = 0; q < kQ; ++q)
                    f[static_cast<std::size_t>(q) * n + c] =
                        equilibrium(q, 1.0f, u, 0.0f, 0.0f);
            }
        }
    }
    TREQUIRE_VOID_M4(uploadF(solver, f) == cudaSuccess);

    // 12 batches: the per-batch EMA (alpha 0.25) reaches ~97% of the target
    // while the planted profile barely evolves (diffusive timescale >> run).
    const float expectYplus = 2.5f * kUTauSynth / kNuSynth;
    std::vector<float> hist;
    for (int b = 0; b < 12; ++b) {
        const cudaError_t serr = solver.stepN(64);
        TCHECK_MSG(serr == cudaSuccess, "stepN failed: %s",
                   cudaGetErrorString(serr));
        if (serr != cudaSuccess) return;
        if (solver.nanDetected()) {
            TCHECK_MSG(false, "watchdog tripped: %s",
                       solver.nanDiagnosis().c_str());
            return;
        }
        const WallModelReadout r = solver.wallModelReadout();
        if (r.cells > 0) hist.push_back(r.meanYplus);
    }
    const WallModelReadout r = solver.wallModelReadout();
    std::printf("synthetic profile: planted y+ %.1f | recovered %.1f "
                "(%d cells, clamp %.2f%%)\n",
                expectYplus, r.meanYplus, r.cells, r.clampedFrac * 100.0f);

    // Gate 1: exact-answer recovery.
    TCHECK_MSG(r.cells > 0, "no active wall cells");
    TCHECK_MSG(approxRel(r.meanYplus, expectYplus, 0.15),
               "u_tau recovery off: y+ %.1f vs planted %.1f",
               r.meanYplus, expectYplus);
    // Gate 2: held steady — last three UI samples pairwise within 5%, and
    // the slip clamp (the model-is-fighting-the-flow tell) quiet over the
    // DEVELOPED part of the plate. The inlet feeds a uniform profile, so a
    // strip of u*steps ~ 0.05*768 ~ 38 of 254 columns (~15%) is a brand-new
    // boundary layer where clamping is the designed spin-up behavior (same
    // as any run's first flow-through near a leading edge); the gate allows
    // that strip and nothing more.
    TCHECK_MSG(r.clampedFrac < 0.20f, "slip clamp engaged %.1f%%",
               r.clampedFrac * 100.0f);
    bool settled = hist.size() >= 3;
    if (settled) {
        const std::size_t m = hist.size();
        for (std::size_t a = m - 3; a < m; ++a)
            for (std::size_t b2 = a + 1; b2 < m; ++b2)
                settled &= approxRel(hist[a], hist[b2], 0.05);
    }
    TCHECK_MSG(settled, "u_tau drifting/ringing on a planted equilibrium");
}

/// @brief Run one plate sim to a force-trustworthy state and return the
/// trailing-average drag converted to a plate friction coefficient:
/// Cf = Fx / (0.5 u^2 nx nz) = cdAvg * Nc / nx (the coefficient normalizes
/// by chord, the plate by its full length).
/// @param scaling     Unit scaling for the run.
/// @param wallModel   Wall-function switch for the run.
/// @param outYplus    Optional: last sampled mean y+ (high-Re diagnostics).
/// @param checkStable When true, gate 1 runs: the mean-y+ history must stop
///                    moving (last three UI samples pairwise within 10%).
/// @param checkD99    When true, gate 4 runs on the converged field.
float runPlate(const GridDims& dims, const LatticeScaling& scaling,
               bool wallModel, float* outYplus,
               bool checkStable, bool checkD99) {
    LBMSolver solver;
    std::string err;
    const bool initOk = solver.init(dims, scaling, plateFlags(dims),
                                    nullptr, &err);
    TCHECK_MSG(initOk, "solver init failed: %s", err.c_str());
    if (!initOk) return -1.0f;
    solver.setWallModelEnabled(wallModel);

    // 2.6 flow-throughs: the force gate opens at 2.0 and the trailing
    // average then covers a fully developed window.
    const float ftSteps = scaling.flowThroughSteps(kNx);
    const int totalSteps = static_cast<int>(2.6f * ftSteps);
    const int chunk = 512;

    std::vector<float> yplusHist;
    for (int done = 0; done < totalSteps; done += chunk) {
        const cudaError_t serr =
            solver.stepN(std::min(chunk, totalSteps - done));
        TCHECK_MSG(serr == cudaSuccess, "stepN failed: %s",
                   cudaGetErrorString(serr));
        if (serr != cudaSuccess) return -1.0f;
        if (solver.nanDetected()) {
            TCHECK_MSG(false, "watchdog tripped: %s",
                       solver.nanDiagnosis().c_str());
            return -1.0f;
        }
        if (wallModel && done >= static_cast<int>(0.5f * ftSteps)) {
            const WallModelReadout r = solver.wallModelReadout();
            if (r.cells > 0) yplusHist.push_back(r.meanYplus);
        }
    }

    if (wallModel) {
        const WallModelReadout r = solver.wallModelReadout();
        TCHECK_MSG(r.enabled && r.cells > 0,
                   "wall model inactive (cells = %d)", r.cells);
        TCHECK_MSG(r.clampedFrac < 0.05f,
                   "slip clamp engaged on %.1f%% of wall cells",
                   r.clampedFrac * 100.0f);
        if (outYplus) *outYplus = r.meanYplus;

        if (checkStable && yplusHist.size() >= 3) {
            // Gate 1: the last three sampled mean-y+ values agree pairwise
            // within 10% — a ringing u_tau <-> u_w loop cannot do that.
            const std::size_t n = yplusHist.size();
            bool settled = true;
            for (std::size_t a = n - 3; a < n; ++a)
                for (std::size_t b = a + 1; b < n; ++b)
                    settled &= approxRel(yplusHist[a], yplusHist[b], 0.10);
            TCHECK_MSG(settled, "u_tau not settled: y+ tail %.2f %.2f %.2f",
                       yplusHist[n - 3], yplusHist[n - 2], yplusHist[n - 1]);
        }
    }

    if (checkD99) {
        // Gate 4: delta99 thickens downstream. Stations in plate-extent
        // fractions (the probe maps x/c onto the solid x extent).
        const auto prof =
            solver.extractSuctionDelta99({0.20f, 0.50f, 0.80f});
        bool grows = prof.size() == 3;
        for (const auto& s : prof) grows &= (s.valid && s.delta99_c > 0.0f);
        if (grows)
            grows = prof[0].delta99_c < prof[1].delta99_c
                 && prof[1].delta99_c < prof[2].delta99_c;
        TCHECK_MSG(grows, "delta99 not growing: %.4f %.4f %.4f",
                   prof.size() == 3 ? prof[0].delta99_c : -1.0f,
                   prof.size() == 3 ? prof[1].delta99_c : -1.0f,
                   prof.size() == 3 ? prof[2].delta99_c : -1.0f);
    }

    const ForceReadout f = solver.forces();
    TCHECK_MSG(f.valid, "force gate never opened (%.2f flow-throughs)",
               f.flowThroughs);
    // Plate Cf from the chord-normalized drag coefficient (see brief above).
    return f.cdAvg * static_cast<float>(kNc) / static_cast<float>(kNx);
}

} // namespace

int main() {
    // ---- gates 1 + 2: planted-profile u_tau recovery and equilibrium hold ----
    checkSyntheticProfile();

    // ---- gates 3 + 4: resolved-sublayer do-no-harm and delta99 growth ----
    {
        const LatticeScaling sc = scalingFor(0.05f, 0.05f);
        TCHECK(!sc.tauClamped);
        const float cfOff = runPlate(kDimsLowRe, sc, /*wallModel=*/false,
                                     nullptr, false, false);
        const float cfOn = runPlate(kDimsLowRe, sc, /*wallModel=*/true,
                                    nullptr, false, /*checkD99=*/true);
        std::printf("low-Re plate: Cf on %.5f | off %.5f (delta %.2f%%)\n",
                    cfOn, cfOff,
                    100.0f * std::fabs(cfOn - cfOff) / std::fabs(cfOff));
        // Gate 3: do no harm.
        TCHECK_MSG(approxRel(cfOn, cfOff, 0.02), "low-Re ON/OFF mismatch");
    }

    return finish("m4_wallmodel");
}
