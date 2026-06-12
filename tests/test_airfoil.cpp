// geom/airfoil checks (plan 5.1/5.2): parse the in-tree LS(1)-0413 Lednicer
// file (45+45 points, blunt TE) and a real Selig file from airfoils/uiuc/,
// asserting point counts, chord normalization, and LE-at-origin; verify a
// malformed file is rejected WITH a stated reason; and sanity-check the NACA
// 2412 generator (max thickness ~12% near x/c 0.30, max camber ~2% near 0.40).
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "geom/airfoil.h"
#include "test_util.h"

using namespace foilcfd;
using namespace foilcfd::testutil;

// The build system injects the absolute airfoils/ directory so the test runs
// from any working directory (ctest launches from the build tree).
#ifndef FOILCFD_AIRFOIL_DIR
#error "FOILCFD_AIRFOIL_DIR must be defined by tests/CMakeLists.txt"
#endif

// TREQUIRE (test_util.h) returns the int summary from main; helper functions
// need a void flavor that bails out of the helper instead.
#define TREQUIRE_VOID(expr)                                                    \
    do {                                                                       \
        const bool ok_ = static_cast<bool>(expr);                              \
        ::foilcfd::testutil::recordCheck(ok_, #expr, __FILE__, __LINE__);      \
        if (!ok_) return;                                                      \
    } while (0)

namespace {

// -----------------------------------------------------------------------
// Shared geometry assertions: every successfully loaded foil must come out
// normalized (chord exactly [0,1], LE at the origin) per the airfoil.h
// contract, regardless of source format.
// -----------------------------------------------------------------------
void checkNormalized(const AirfoilGeometry& foil) {
    TREQUIRE_VOID(!foil.points.empty());
    float minX = foil.points[0].x, maxX = foil.points[0].x;
    std::size_t minXIdx = 0;
    for (std::size_t i = 0; i < foil.points.size(); ++i) {
        const float x = foil.points[i].x;
        if (x < minX) { minX = x; minXIdx = i; }
        maxX = std::max(maxX, x);
    }
    // Chord spans exactly [0,1] after normalization.
    TCHECK_MSG(approxEq(minX, 0.0, 1e-3), "min x = %.6f", minX);
    TCHECK_MSG(approxEq(maxX, 1.0, 1e-3), "max x = %.6f", maxX);
    // The leading edge (the min-x point) sits at the origin.
    TCHECK_MSG(approxEq(foil.points[minXIdx].y, 0.0, 2e-3),
               "LE y = %.6f", foil.points[minXIdx].y);
    // leIndex must point at (or right next to) that min-x point: the loop
    // ordering contract says [0..leIndex] is the upper surface.
    TCHECK(foil.leIndex > 0
           && foil.leIndex < static_cast<int>(foil.points.size()) - 1);
    TCHECK_MSG(approxEq(foil.points[static_cast<std::size_t>(foil.leIndex)].x,
                        0.0, 5e-3),
               "points[leIndex].x = %.6f",
               foil.points[static_cast<std::size_t>(foil.leIndex)].x);
    // All y values stay section-plausible (a parse bug shows up as wild y).
    bool yBounded = true;
    for (const Vec2f& p : foil.points)
        if (p.y < -0.5f || p.y > 0.5f) yBounded = false;
    TCHECK(yBounded);
}

// -----------------------------------------------------------------------
// Lednicer format: airfoils/ls413.dat — LS(1)-0413, header comment lines,
// "45. 45." count line, upper LE->TE, blank, lower LE->TE, blunt TE.
// Converted to the canonical Selig loop this is 89-90 points (45 + 45 with
// the shared LE point merged or kept).
// -----------------------------------------------------------------------
void checkLednicerLS413() {
    const std::filesystem::path path =
        std::filesystem::path(FOILCFD_AIRFOIL_DIR) / "ls413.dat";
    const AirfoilLoadResult r = loadAirfoilDat(path);
    TCHECK_MSG(r.ok, "rejected: %s", r.rejectionReason.c_str());
    if (!r.ok) return;

    TCHECK(r.format == DatFormat::Lednicer);
    // 45 upper + 45 lower; the loop may merge the shared LE point and/or a
    // coincident TE point, so allow the small dedup window around 90.
    const int n = static_cast<int>(r.airfoil.points.size());
    TCHECK_MSG(n >= 86 && n <= 91, "loop point count = %d (expected 86..91)", n);
    checkNormalized(r.airfoil);

    // Both loop ends are trailing-edge points (x ~ 1): Lednicer surfaces were
    // re-stitched into TE -> upper -> LE -> lower -> TE order.
    TCHECK(approxEq(r.airfoil.points.front().x, 1.0, 1e-3));
    TCHECK(approxEq(r.airfoil.points.back().x, 1.0, 1e-3));

    // Upper surface really is on top: mean y of [0..leIndex] must exceed mean
    // y of [leIndex..end] (LS(1)-0413 is strongly cambered, so the gap is big).
    const std::size_t le = static_cast<std::size_t>(r.airfoil.leIndex);
    double upper = 0.0, lower = 0.0;
    for (std::size_t i = 0; i <= le; ++i) upper += r.airfoil.points[i].y;
    for (std::size_t i = le; i < r.airfoil.points.size(); ++i)
        lower += r.airfoil.points[i].y;
    upper /= static_cast<double>(le + 1);
    lower /= static_cast<double>(r.airfoil.points.size() - le);
    TCHECK_MSG(upper > lower + 0.01, "upper mean y %.4f vs lower %.4f", upper, lower);
}

// -----------------------------------------------------------------------
// Selig format: airfoils/uiuc/a18.dat — name line then 41 coordinate pairs
// in one TE -> upper -> LE -> lower -> TE loop.
// -----------------------------------------------------------------------
void checkSeligA18() {
    const std::filesystem::path path =
        std::filesystem::path(FOILCFD_AIRFOIL_DIR) / "uiuc" / "a18.dat";
    const AirfoilLoadResult r = loadAirfoilDat(path);
    TCHECK_MSG(r.ok, "rejected: %s", r.rejectionReason.c_str());
    if (!r.ok) return;

    TCHECK(r.format == DatFormat::Selig);
    // The file carries exactly 41 pairs; tolerate a +/-1 dedup/closure delta.
    const int n = static_cast<int>(r.airfoil.points.size());
    TCHECK_MSG(n >= 40 && n <= 42, "point count = %d (expected 40..42)", n);
    checkNormalized(r.airfoil);
    // The display name comes from the file's first line.
    TCHECK_MSG(r.airfoil.name.find("A18") != std::string::npos,
               "name = '%s'", r.airfoil.name.c_str());
}

// -----------------------------------------------------------------------
// Rejection paths: the .dat zoo is messy and the loader must say WHY a file
// failed (plan section 13), never silently return an empty/garbage section.
// -----------------------------------------------------------------------
void checkMalformedRejected() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path();

    // Case 1: non-numeric junk where coordinates should be.
    const fs::path junkPath = dir / "foilcfd_test_junk.dat";
    {
        std::ofstream out(junkPath);
        out << "Definitely An Airfoil\n"
               "0.5 hello\n"
               "this line is prose, not coordinates\n"
               "1.0 0.0 0.5 extra tokens everywhere\n";
    }
    const AirfoilLoadResult junk = loadAirfoilDat(junkPath);
    TCHECK(!junk.ok);
    TCHECK_MSG(!junk.rejectionReason.empty(),
               "rejection must carry a human-readable reason");
    std::printf("  junk file reason: %s\n", junk.rejectionReason.c_str());

    // Case 2: syntactically fine but far too few points (< 20 minimum).
    const fs::path tinyPath = dir / "foilcfd_test_tiny.dat";
    {
        std::ofstream out(tinyPath);
        out << "Tiny Triangle\n"
               "1.0 0.0\n0.5 0.1\n0.0 0.0\n0.5 -0.1\n1.0 0.0\n";
    }
    const AirfoilLoadResult tiny = loadAirfoilDat(tinyPath);
    TCHECK(!tiny.ok);
    TCHECK(!tiny.rejectionReason.empty());
    std::printf("  tiny file reason: %s\n", tiny.rejectionReason.c_str());

    // Case 3: the file does not exist at all.
    const AirfoilLoadResult missing =
        loadAirfoilDat(dir / "foilcfd_test_does_not_exist.dat");
    TCHECK(!missing.ok);
    TCHECK(!missing.rejectionReason.empty());

    std::error_code ec;
    fs::remove(junkPath, ec);
    fs::remove(tinyPath, ec);
}

// -----------------------------------------------------------------------
// NACA 2412 generator sanity. Reconstruct thickness/camber distributions by
// interpolating upper and lower surface y at common x stations:
//   thickness t(x) ~ y_u(x) - y_l(x), camber m(x) ~ (y_u(x) + y_l(x)) / 2.
// (The vertical-cut approximation differs from the perpendicular-to-camber
// definition by < 0.5% at 2% camber — well inside the tolerances below.)
// Expectations: max t ~ 0.12 near x = 0.30, max camber ~ 0.02 at x = 0.40.
// -----------------------------------------------------------------------
float surfaceYAt(const AirfoilGeometry& foil, float x, bool upper) {
    // Walk the requested half of the loop and linearly interpolate the
    // bracketing segment. Upper runs points[0..leIndex], lower the rest.
    const int begin = upper ? 0 : foil.leIndex;
    const int end   = upper ? foil.leIndex
                            : static_cast<int>(foil.points.size()) - 1;
    for (int i = begin; i < end; ++i) {
        const Vec2f a = foil.points[static_cast<std::size_t>(i)];
        const Vec2f b = foil.points[static_cast<std::size_t>(i + 1)];
        const float lo = std::min(a.x, b.x), hi = std::max(a.x, b.x);
        if (x < lo || x > hi || hi - lo < 1e-9f) continue;
        const float t = (x - a.x) / (b.x - a.x);
        return a.y + (b.y - a.y) * t;
    }
    return 0.0f; // outside the surface span; callers only probe interior x
}

void checkNACA2412() {
    const AirfoilLoadResult r = generateNACA4("2412");
    TCHECK_MSG(r.ok, "rejected: %s", r.rejectionReason.c_str());
    if (!r.ok) return;

    // ~200 points per surface requested by default; demand a dense outline.
    TCHECK(static_cast<int>(r.airfoil.points.size()) >= 150);
    checkNormalized(r.airfoil);

    // Scan the interior for the thickness and camber extrema.
    float maxT = 0.0f, maxTAt = 0.0f, maxM = 0.0f, maxMAt = 0.0f;
    for (float x = 0.02f; x <= 0.98f; x += 0.005f) {
        const float yu = surfaceYAt(r.airfoil, x, /*upper=*/true);
        const float yl = surfaceYAt(r.airfoil, x, /*upper=*/false);
        const float t  = yu - yl;
        const float m  = 0.5f * (yu + yl);
        if (t > maxT) { maxT = t; maxTAt = x; }
        if (m > maxM) { maxM = m; maxMAt = x; }
    }
    // 12% thickness: allow the closed-TE polynomial + vertical-cut slack.
    TCHECK_MSG(maxT > 0.113f && maxT < 0.127f, "max thickness = %.4f", maxT);
    TCHECK_MSG(maxTAt > 0.24f && maxTAt < 0.36f, "max thickness at x = %.3f", maxTAt);
    // 2% camber at 40% chord (the "2" and "4" digits).
    TCHECK_MSG(maxM > 0.017f && maxM < 0.023f, "max camber = %.4f", maxM);
    TCHECK_MSG(maxMAt > 0.34f && maxMAt < 0.46f, "max camber at x = %.3f", maxMAt);

    // Generator input validation must reject non-4-digit codes with reasons.
    TCHECK(!generateNACA4("241").ok);
    TCHECK(!generateNACA4("24x2").ok);
    TCHECK(!generateNACA4("24122").ok);
    TCHECK(!generateNACA4("241").rejectionReason.empty());
}

} // namespace

int main() {
    checkLednicerLS413();
    checkSeligA18();
    checkMalformedRejected();
    checkNACA2412();
    return finish("test_airfoil");
}
