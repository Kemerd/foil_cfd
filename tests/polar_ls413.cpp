// Manual validation tool (NOT a CTest): sweep the LS(1)-0413 lift polar at
// the app's default operating point (1.2 m chord, 35.76 m/s -> target
// Re 2.84e6, Fast-preset 192 cells/chord, 2x patch) with the wall model ON,
// plus spot points with it OFF, and write the results to
// validation/polar_ls413_wm.csv for comparison against the digitized NASA
// TM X-72843 curves (Re 2.2e6 and 4.3e6 bracket the target).
// Run time: roughly a minute per point on an RTX 5090 — budget ~15 minutes.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "geom/airfoil.h"
#include "geom/voxelizer.h"
#include "sim/lbm_solver.h"
#include "sim/units.h"

using namespace foilcfd;

namespace {

constexpr int kNc = 192; ///< Fast preset.

/// @brief One polar point: run the clean foil at one AoA to a trustworthy
/// force readout (3 flow-throughs: gate at 2, trailing window fully inside
/// developed flow) and report the windowed coefficients.
struct PolarPoint {
    float aoa = 0.0f;
    bool  wallModel = true;
    bool  ok = false;     ///< False = diverged or init failure.
    float cl = 0.0f, cd = 0.0f;
    float yplusMean = 0.0f;
};

PolarPoint runPoint(const AirfoilGeometry& foil, float aoa, bool wallModel,
                    bool usePatch = true) {
    PolarPoint pt;
    pt.aoa = aoa;
    pt.wallModel = wallModel;

    DomainLayout layout;
    layout.dims = GridDims{3 * kNc, kNc + kNc / 4, 3 * kNc / 8};
    layout.chordCells = kNc;

    PhysicalParams phys;
    phys.chord_m     = 1.2f;
    phys.airspeed_ms = 35.7632f; // the app default (80 mph)
    // HiFi-style lattice speed: at u_lat 0.08 the first sweep diverged on
    // the fine level beyond AoA 12 (cold impulsive starts at the tau clamp
    // — the app survives these via mesh-sequencing preconvergence, which
    // this standalone tool does not replicate). 0.05 lowers the lattice
    // Mach enough to ride out the start-up transient at every AoA.
    const LatticeScaling scaling = computeScaling(phys, kNc, 0.05f);

    const std::vector<std::uint8_t> clean =
        buildCleanFoilFlags(foil, aoa, layout);

    LBMSolver solver;
    std::string err;
    if (!solver.init(layout.dims, scaling, clean, nullptr, &err)) {
        std::printf("  init failed: %s\n", err.c_str());
        return pt;
    }
    solver.setSurfaceReference(clean);
    solver.setWallModelEnabled(wallModel);

    // 2x patch, app-default margins — the configuration a user actually
    // runs, so the validation validates the product.
    auto cellsOf = [](float chords) {
        return std::max(2, static_cast<int>(std::lround(
                               chords * static_cast<float>(kNc))));
    };
    const PatchBox box = derivePatchBox(layout.dims, clean,
                                        cellsOf(0.20f), cellsOf(0.50f),
                                        cellsOf(0.10f), cellsOf(0.20f));
    if (usePatch && box.valid()) {
        const DomainLayout fineLayout = makeFineLayout(layout, box, 2);
        std::vector<std::uint8_t> fine(
            static_cast<std::size_t>(fineLayout.dims.cellCount()),
            static_cast<std::uint8_t>(CellFlag::Fluid));
        voxelizeAirfoil(foil, aoa, fineLayout, fine);
        std::vector<std::uint8_t> fineClean = fine;
        closeTrailingEdgeGaps(fineLayout.dims, fineClean);
        closeTrailingEdgeGaps(fineLayout.dims, fine);
        stampInterfaceShell(fineLayout.dims, fine);
        if (solver.initRefinement(box, 2, fine, &err))
            solver.setRefinedSurfaceReference(fineClean);
        else
            std::printf("  (patch unavailable: %s)\n", err.c_str());
    }

    // 5 flow-throughs: the trailing force window covers [3, 5] FT, fully
    // clear of the impulsive-start transient (the first sweep's 3 FT left
    // the window sampling start-up drag spikes — Cd read 5-10x high).
    const int totalSteps =
        static_cast<int>(5.0f * scaling.flowThroughSteps(layout.dims.nx));
    for (int done = 0; done < totalSteps; done += 512) {
        if (solver.stepN(std::min(512, totalSteps - done)) != cudaSuccess)
            return pt;
        if (solver.nanDetected()) {
            std::printf("  DIVERGED at AoA %.1f: %s\n", aoa,
                        solver.nanDiagnosis().c_str());
            return pt;
        }
    }

    const ForceReadout f = solver.forces();
    if (!f.valid) {
        std::printf("  force gate never opened at AoA %.1f\n", aoa);
        return pt;
    }
    pt.cl = f.clAvg;
    pt.cd = f.cdAvg;
    pt.yplusMean = solver.wallModelReadout().meanYplus;
    pt.ok = true;
    return pt;
}

} // namespace

int main(int argc, char** argv) {
    const AirfoilLoadResult load =
        loadAirfoilDat(FOILCFD_AIRFOIL_DIR "/ls413.dat");
    if (!load.ok) {
        std::printf("ls413.dat load failed: %s\n",
                    load.rejectionReason.c_str());
        return 1;
    }

    // --probe: the drag-anomaly isolation experiment. The full sweeps read
    // absolute Cd 5-10x above any plausible value while Cl stayed sane.
    // Three AoA-0 cases isolate where the excess lives: coarse-only plain
    // walls, the 2x patch (fine-grid force path + restriction), and the
    // wall model on top of the patch.
    if (argc > 1 && std::string(argv[1]) == "--probe") {
        struct { const char* label; bool wm; bool patch; } cases[] = {
            {"coarse, plain BB ", false, false},
            {"2x patch, plain BB", false, true},
            {"2x patch, WM on  ", true, true},
        };
        std::printf("Cd anomaly probe, AoA 0:\n");
        for (const auto& c : cases) {
            const PolarPoint pt = runPoint(load.airfoil, 0.0f, c.wm, c.patch);
            std::printf("  %s  Cl %7.3f  Cd %7.4f%s\n", c.label, pt.cl,
                        pt.cd, pt.ok ? "" : "  FAILED");
            std::fflush(stdout);
        }
        return 0;
    }

    const PhysicalParams phys{}; // defaults only used for the banner below
    std::printf("LS(1)-0413 polar, chord %d cells, 2x patch, target Re "
                "2.84e6 — NASA TM X-72843 brackets: Re 2.2e6 / 4.3e6\n\n",
                kNc);
    std::printf("%6s %5s %8s %8s %8s\n", "aoa", "WM", "Cl", "Cd", "y+mean");

    std::vector<PolarPoint> results;
    const float sweepOn[]  = {0.0f, 4.0f, 8.0f, 10.0f, 12.0f,
                              14.0f, 16.0f, 18.0f, 20.0f};
    const float sweepOff[] = {8.0f, 14.0f, 18.0f};
    for (const float aoa : sweepOn) {
        const PolarPoint pt = runPoint(load.airfoil, aoa, true);
        results.push_back(pt);
        std::printf("%6.1f %5s %8.3f %8.4f %8.0f%s\n", pt.aoa, "on",
                    pt.cl, pt.cd, pt.yplusMean, pt.ok ? "" : "  FAILED");
        std::fflush(stdout);
    }
    for (const float aoa : sweepOff) {
        const PolarPoint pt = runPoint(load.airfoil, aoa, false);
        results.push_back(pt);
        std::printf("%6.1f %5s %8.3f %8.4f %8s%s\n", pt.aoa, "off",
                    pt.cl, pt.cd, "-", pt.ok ? "" : "  FAILED");
        std::fflush(stdout);
    }

    // CSV for the validation records (same contract as the XFOIL polars:
    // one header line, then plain rows).
    if (FILE* csv = std::fopen("validation/polar_ls413_wm.csv", "w")) {
        std::fprintf(csv, "aoa,wall_model,cl,cd,yplus_mean,ok\n");
        for (const PolarPoint& pt : results) {
            std::fprintf(csv, "%.1f,%d,%.4f,%.5f,%.1f,%d\n", pt.aoa,
                         pt.wallModel ? 1 : 0, pt.cl, pt.cd, pt.yplusMean,
                         pt.ok ? 1 : 0);
        }
        std::fclose(csv);
        std::printf("\nwrote validation/polar_ls413_wm.csv\n");
    } else {
        std::printf("\ncould not write validation CSV (run from repo root)\n");
    }
    (void)phys;
    return 0;
}
