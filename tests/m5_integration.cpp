// M5 integration protocol: the wall model, the resolved-vane machinery, and
// the vortex audit working together on a real (NACA 4412) airfoil at the
// app's honest operating point (tau at the stability clamp, effective Re
// ~2e4 at chord 64). Five runs, three gates:
//   1. VG efficacy — at a post-separation AoA (plain bounce-back: the
//      equilibrium wall function keeps this section attached through 16
//      degrees at this effective Re, a known property of equilibrium wall
//      stress), adding a VG row placed 5-10 heights upstream of the
//      measured onset (Lin 2002) must move separation aft, not forward;
//   2. resolution monotonicity — the MEASURED shed circulation of a
//      marginally-resolved vane must rise when the patch factor goes
//      2x -> 4x (the auto-factor guard's premise), with the audit ratio
//      clearing its own marginal band at both factors;
//   3. wall-model sanity — toggling the wall model shifts the measured
//      circulation by real boundary-layer physics (~20-25% here), never
//      by the order-of-magnitude disagreement that signals a broken
//      measurement (the upstream-floor bug read 480%); gate at 35%.
// Wall-clock dominated by the 4x-patch run (~m^4 cost); budget minutes.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "geom/airfoil.h"
#include "geom/vg.h"
#include "geom/vg_audit.h"
#include "geom/voxelizer.h"
#include "lbm_test_util.h"
#include "sim/lbm_solver.h"
#include "sim/units.h"
#include "test_util.h"

using namespace foilcfd;
using namespace foilcfd::testutil;

namespace {

constexpr int kNc = 64;                       ///< Chord cells (CI budget).
const GridDims kDims{3 * kNc, 96, 32};        ///< Plan-4.6 proportions-ish.
constexpr float kSettleFT = 2.2f;             ///< Flow-throughs before reads.

/// @brief The app's real scaling path: physical chord/airspeed at this
/// resolution clamps tau and reports the honest effective Re — exactly the
/// regime the product runs in.
LatticeScaling appScaling() {
    PhysicalParams phys;
    phys.chord_m     = 1.2f;
    phys.airspeed_ms = 35.0f;
    return computeScaling(phys, kNc, 0.05f);
}

/// @brief What one protocol run reports back.
struct CaseResult {
    bool  ok = false;        ///< Ran to the readout point without diverging.
    float sepXc = -1.0f;     ///< Separation onset (clean-surface probe).
    int   attachedAft = 0;   ///< Attached stations in x/c 0.30..0.95 (the
                             ///< region DOWNSTREAM of any vane row: the
                             ///< onset probe reads the row's own wake as
                             ///< "separation", this count does not).
    VGAuditReadout audit;    ///< Averaged vortex audit (VG cases only).
};

/// Stations for the attachment count — clear of the gate-1 vane row
/// (which ends near x/c 0.19 vane-chord included) and of the TE itself.
const std::vector<float> kAftStations = {0.30f, 0.35f, 0.40f, 0.45f, 0.50f,
                                         0.55f, 0.60f, 0.65f, 0.70f, 0.75f,
                                         0.80f, 0.85f, 0.90f, 0.95f};

/// @brief Build flags, optional refinement patch, run to kSettleFT, read
/// out. Mirrors main.cpp's geometry pipeline (clean OR VG-merged flags,
/// fine level with VG stamping between voxelize and shell, clean surface
/// references on both levels).
CaseResult runCase(const AirfoilGeometry& foil, float aoaDeg,
                   const std::vector<VGParams>& vgs, int factor,
                   bool wallModel) {
    CaseResult res;
    DomainLayout layout;
    layout.dims = kDims;
    layout.chordCells = kNc;
    const LatticeScaling scaling = appScaling();

    const std::vector<std::uint8_t> clean =
        buildCleanFoilFlags(foil, aoaDeg, layout);
    std::vector<std::uint8_t> active = clean;
    if (!vgs.empty())
        active = buildFlagsWithVGs(vgs, foil, aoaDeg, layout, clean);

    LBMSolver solver;
    std::string err;
    if (!solver.init(layout.dims, scaling, active, nullptr, &err)) {
        std::printf("  init failed: %s\n", err.c_str());
        return res;
    }
    solver.setSurfaceReference(clean);
    solver.setWallModelEnabled(wallModel);

    if (factor >= 2) {
        // Patch around the VG-merged solids, margins as the app defaults.
        auto cellsOf = [](float chords) {
            return std::max(2, static_cast<int>(std::lround(
                                   chords * static_cast<float>(kNc))));
        };
        const PatchBox box = derivePatchBox(layout.dims, active,
                                            cellsOf(0.20f), cellsOf(0.50f),
                                            cellsOf(0.10f), cellsOf(0.20f));
        if (box.valid()) {
            const DomainLayout fineLayout = makeFineLayout(layout, box, factor);
            std::vector<std::uint8_t> fine(
                static_cast<std::size_t>(fineLayout.dims.cellCount()),
                static_cast<std::uint8_t>(CellFlag::Fluid));
            voxelizeAirfoil(foil, aoaDeg, fineLayout, fine);
            std::vector<std::uint8_t> fineClean = fine; // foil-only snapshot
            closeTrailingEdgeGaps(fineLayout.dims, fineClean);
            for (const VGParams& vg : vgs)
                voxelizeVG(vg, foil, aoaDeg, fineLayout, fine);
            closeTrailingEdgeGaps(fineLayout.dims, fine);
            stampInterfaceShell(fineLayout.dims, fine);
            if (!solver.initRefinement(box, factor, fine, &err)) {
                std::printf("  refinement unavailable: %s\n", err.c_str());
            } else {
                solver.setRefinedSurfaceReference(fineClean);
            }
        }
    }

    // March to the readout point, watching the watchdog.
    const int totalSteps = static_cast<int>(
        kSettleFT * scaling.flowThroughSteps(layout.dims.nx));
    for (int done = 0; done < totalSteps; done += 256) {
        if (solver.stepN(std::min(256, totalSteps - done)) != cudaSuccess)
            return res;
        if (solver.nanDetected()) {
            std::printf("  DIVERGED: %s\n", solver.nanDiagnosis().c_str());
            return res;
        }
    }

    res.sepXc = solver.separationOnsetXc();

    // Attachment census over the aft half: a station is "attached" when the
    // delta99 probe found forward near-wall flow there (Delta99Sample::valid
    // is exactly that test). Vane wakes can fool the ONSET scan; they cannot
    // fool a count taken wholly downstream of the row.
    for (const Delta99Sample& s : solver.extractSuctionDelta99(kAftStations))
        if (s.valid) ++res.attachedAft;

    // Vortex audit, averaged over three snapshots ~0.1 flow-throughs apart
    // (the wake is unsteady; a single plane sample is noisier than the
    // verdict bands deserve). BL inputs come from the live delta99 probe at
    // the vane station, exactly as the app feeds the audit.
    if (!vgs.empty()) {
        const int gapSteps = static_cast<int>(
            0.1f * scaling.flowThroughSteps(layout.dims.nx));
        float sumMeas = 0.0f, sumPred = 0.0f;
        int samples = 0;
        for (int snap = 0; snap < 3; ++snap) {
            if (snap > 0 && solver.stepN(gapSteps) != cudaSuccess) break;
            const auto prof = solver.extractSuctionDelta99({vgs[0].x_c});
            if (prof.empty() || !prof[0].valid) continue;
            const VGAuditReadout a = auditVGVortexStrength(
                solver, vgs[0], kNc, prof[0].ueEdge,
                prof[0].delta99_c * static_cast<float>(kNc));
            if (!a.valid) continue;
            sumMeas += a.gammaMeasured;
            sumPred += a.gammaPredicted;
            ++samples;
            res.audit = a;
        }
        if (samples > 0) {
            res.audit.gammaMeasured = sumMeas / static_cast<float>(samples);
            res.audit.gammaPredicted = sumPred / static_cast<float>(samples);
            res.audit.ratio = res.audit.gammaMeasured
                            / std::max(res.audit.gammaPredicted, 1e-9f);
            res.audit.valid = true;
        } else {
            res.audit = VGAuditReadout{};
        }
    }
    res.ok = true;
    return res;
}

/// @brief Counter-rotating VG row description used across the gates.
std::vector<VGParams> vgRow(float xc, float heightC) {
    VGParams vg = defaultVGParams();
    vg.x_c = xc;
    vg.height_c = heightC;
    vg.pitch_c = 0.10f;
    vg.count = 3;
    return {vg};
}

} // namespace

int main() {
    const AirfoilLoadResult gen = generateNACA4("4412");
    TREQUIRE(gen.ok);
    const AirfoilGeometry& foil = gen.airfoil;
    const LatticeScaling sc = appScaling();
    std::printf("operating point: effective Re %.2e (target %.2e, tau %s)\n",
                sc.reEffective, sc.reTarget,
                sc.tauClamped ? "clamped" : "free");

    // ---- gate 1: VG efficacy at a separated AoA --------------------------
    // Runs with the wall model OFF: the equilibrium wall function keeps
    // this section attached through AoA 16 at the test's effective Re
    // (verified empirically — equilibrium wall stress is well known to
    // resist separation; the product's "trust deltas" guidance covers it).
    // VG efficacy — vanes moving a real separation aft — is a geometry +
    // solver gate, and the plain-bounce-back path separates readily here.
    {
        const float aoa = 16.0f;
        std::printf("[gate 1] VG-off baseline at AoA %.0f (plain BB)...\n",
                    aoa);
        const CaseResult off = runCase(foil, aoa, {}, 2, false);
        TCHECK_MSG(off.ok, "baseline diverged");
        TCHECK_MSG(off.sepXc >= 0.0f, "no separation at AoA %.0f — gate "
                   "needs a separated baseline", aoa);
        if (off.ok && off.sepXc >= 0.0f) {
            // Lin 2002: 5-10 heights upstream of the onset; clamp into the
            // sensible station range.
            const float h = 0.05f;
            const float xc = std::clamp(off.sepXc - 7.0f * h, 0.04f, 0.50f);
            std::printf("[gate 1] onset x/c %.3f -> VG row at x/c %.3f\n",
                        off.sepXc, xc);
            const CaseResult on = runCase(foil, aoa, vgRow(xc, h), 2, false);
            TCHECK_MSG(on.ok, "VG-on run diverged");
            std::printf("[gate 1] onset (info only; reads vane wakes): "
                        "off %.3f -> on %.3f\n", off.sepXc, on.sepXc);
            std::printf("[gate 1] attached aft stations (of %d): "
                        "off %d -> on %d\n",
                        static_cast<int>(kAftStations.size()),
                        off.attachedAft, on.attachedAft);
            // The mission gate: the aft half of the suction surface must be
            // measurably LESS separated with the vane row working.
            TCHECK_MSG(on.ok && on.attachedAft > off.attachedAft,
                       "VGs did not reduce aft separation (%d -> %d)",
                       off.attachedAft, on.attachedAft);
        }
    }

    // ---- gates 2 + 3: audit monotonicity in patch factor, wall-model
    // neutrality of the measurement (attached AoA, small vane) -------------
    {
        // height_c 0.06 -> 3.8 coarse / 7.7 fine(2x) / 15.4 fine(4x) cells:
        // marginal at 2x, resolved at 4x — exactly the contrast gate 2
        // exists to demonstrate. (A sub-voxel vane sheds nothing at ANY
        // tested factor and the audit honestly reads ~0 across the board,
        // which demonstrates only the honesty.)
        const float aoa = 8.0f;
        const std::vector<VGParams> row = vgRow(0.10f, 0.06f);
        std::printf("[gate 2] 2x patch...\n");
        const CaseResult f2 = runCase(foil, aoa, row, 2, true);
        std::printf("[gate 2] 4x patch (the slow one)...\n");
        const CaseResult f4 = runCase(foil, aoa, row, 4, true);
        TCHECK_MSG(f2.ok && f2.audit.valid, "2x audit unavailable");
        TCHECK_MSG(f4.ok && f4.audit.valid, "4x audit unavailable");
        if (f2.audit.valid && f4.audit.valid) {
            std::printf("[gate 2] audit ratio: 2x %.3f -> 4x %.3f "
                        "(measured %.4f -> %.4f, predicted %.4f -> %.4f)\n",
                        f2.audit.ratio, f4.audit.ratio,
                        f2.audit.gammaMeasured, f4.audit.gammaMeasured,
                        f2.audit.gammaPredicted, f4.audit.gammaPredicted);
            // The physics claim is on the MEASURED circulation: a finer
            // vane sheds more. The ratio's denominator moves too (the
            // finer patch thins the measured delta99, raising the Wendt
            // prediction), so ratio monotonicity would conflate the two.
            TCHECK_MSG(f4.audit.gammaMeasured > f2.audit.gammaMeasured,
                       "shed circulation did not improve with resolution");
            // And both configurations must clear the audit's own
            // trust floor — a reading near zero would mean the honesty
            // meter calls its own integration noise.
            TCHECK_MSG(f2.audit.ratio > kAuditRatioMarginal
                           && f4.audit.ratio > kAuditRatioMarginal,
                       "audit below the marginal band (%.2f / %.2f)",
                       f2.audit.ratio, f4.audit.ratio);
        }

        std::printf("[gate 3] 2x patch, wall model OFF...\n");
        const CaseResult f2off = runCase(foil, aoa, row, 2, false);
        TCHECK_MSG(f2off.ok && f2off.audit.valid, "WM-off audit unavailable");
        if (f2.audit.valid && f2off.audit.valid) {
            const float rel = std::fabs(f2.audit.gammaMeasured
                                        - f2off.audit.gammaMeasured)
                            / std::max(f2.audit.gammaMeasured, 1e-9f);
            std::printf("[gate 3] measured gamma: WM on %.4f | off %.4f "
                        "(%.1f%%)\n", f2.audit.gammaMeasured,
                        f2off.audit.gammaMeasured, rel * 100.0f);
            // The wall model legitimately changes the boundary layer the
            // vane sits in (that is its purpose), and ~20-25% of measured
            // circulation shift at this regime is real BL physics. The
            // gate guards against measurement-infrastructure breakage —
            // the broken-floor failure mode read a 480% disagreement.
            TCHECK_MSG(rel < 0.35f,
                       "wall model perturbs the measured vortex by %.0f%% — "
                       "audit infrastructure suspect", rel * 100.0f);
        }
    }

    return finish("m5_integration");
}
