// app/aircraft_manifest checks (plan 15.5): the REAL in-tree manifest parses
// completely (75 rows) with the Glasair III resolving its factory-modified
// GA(W)-2 file from airfoils/uiuc/ case-insensitively; quoted fields keep
// their embedded commas through the CSV reader; and malformed rows (junk
// field counts, unterminated quotes, missing identity columns) are skipped
// WITH a stated reason instead of crashing — the CSV is user-editable, so the
// parser's failure mode is the feature's whole robustness story.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "app/aircraft_manifest.h"
#include "test_util.h"

using namespace foilcfd;
using namespace foilcfd::testutil;

// The build system injects the absolute airfoils/ directory so the test runs
// from any working directory (ctest launches from the build tree).
#ifndef FOILCFD_AIRFOIL_DIR
#error "FOILCFD_AIRFOIL_DIR must be defined by tests/CMakeLists.txt"
#endif

// TREQUIRE (test_util.h) returns the int summary from main; helper functions
// need a void flavor that bails out of the helper instead (as test_airfoil).
#define TREQUIRE_VOID(expr)                                                    \
    do {                                                                       \
        const bool ok_ = static_cast<bool>(expr);                              \
        ::foilcfd::testutil::recordCheck(ok_, #expr, __FILE__, __LINE__);      \
        if (!ok_) return;                                                      \
    } while (0)

namespace {

// -----------------------------------------------------------------------
// The shipped manifest: every one of the 75 data rows must survive the
// parser (the file is the feature's ground truth — a row silently dropped
// here would vanish from the Aircraft section), and dat resolution against
// airfoils/ + airfoils/uiuc/ must find the files the manifest claims.
// -----------------------------------------------------------------------
void checkRealManifest() {
    const std::filesystem::path dir(FOILCFD_AIRFOIL_DIR);
    const AircraftManifest m = loadAircraftManifest(dir);

    TCHECK_MSG(m.loadError.empty(), "load error: %s", m.loadError.c_str());
    TCHECK_MSG(m.entries.size() == 75, "parsed %d rows (expected 75)",
               static_cast<int>(m.entries.size()));
    TCHECK_MSG(m.skippedRows.empty(), "%d rows unexpectedly skipped",
               static_cast<int>(m.skippedRows.size()));
    for (const std::string& reason : m.skippedRows) {
        std::printf("  skipped: %s\n", reason.c_str());
    }

    // Glasair III: factory-modified GA(W)-2, dat_root = ls413mod.dat, which
    // lives in airfoils/uiuc/ — resolution must find it there and flag data.
    const AircraftEntry* glasair3 = nullptr;
    for (const AircraftEntry& e : m.entries) {
        if (e.manufacturer == "Stoddard-Hamilton" && e.model == "Glasair III") {
            glasair3 = &e;
        }
    }
    TREQUIRE_VOID(glasair3 != nullptr);
    TCHECK_MSG(glasair3->hasRootData, "Glasair III root '%s' did not resolve",
               glasair3->datRoot.c_str());
    TCHECK_MSG(glasair3->resolvedRoot.filename() == "ls413mod.dat",
               "resolvedRoot = %s",
               glasair3->resolvedRoot.filename().string().c_str());
    TCHECK(glasair3->resolvedRoot.is_absolute());
    TCHECK(std::filesystem::exists(glasair3->resolvedRoot));
    TCHECK(glasair3->hasData());

    // A few structural spot checks on known rows: rows that claim no dat
    // files must resolve to none (the Van's RV fleet), and a row whose root
    // and tip name DIFFERENT files must resolve both (the UI's root/tip
    // selector keys off exactly this).
    int rvRows = 0, glastarOk = 0;
    for (const AircraftEntry& e : m.entries) {
        if (e.manufacturer == "Van's" && e.airfoilRoot.find("23013") != std::string::npos) {
            ++rvRows;
            TCHECK(!e.hasRootData && !e.hasTipData && !e.hasData());
        }
        if (e.model == "GlaStar / Sportsman 2+2") {
            ++glastarOk;
            TCHECK(e.hasRootData && e.hasTipData);
            TCHECK(e.resolvedRoot != e.resolvedTip);
        }
    }
    TCHECK_MSG(rvRows == 4, "found %d NACA-23013.5 Van's rows (expected 4)", rvRows);
    TCHECK_MSG(glastarOk == 1, "found %d GlaStar rows (expected 1)", glastarOk);
}

// -----------------------------------------------------------------------
// Quoted-field handling: commas inside double quotes are DATA, "" is a
// literal quote, and CRLF terminators never leak into field values.
// -----------------------------------------------------------------------
void checkQuotedCommaFields() {
    const std::string csv =
        "manufacturer,model,category,airfoil_root,airfoil_tip,dat_root,dat_tip,notes\r\n"
        "Cessna,\"172, 175, 182\",GA-single,NACA 2412,NACA 2412,naca2412.dat,"
        "naca2412.dat,\"early models, pre-1973, used a 0012 tip\"\r\n"
        "Pitts,\"S-1 \"\"Special\"\"\",aerobatic,NACA M-6,NACA M-6,,,\r\n";
    std::vector<std::string> skipped;
    const std::vector<AircraftEntry> rows = parseAircraftManifestCsv(csv, &skipped);

    TCHECK_MSG(rows.size() == 2, "parsed %d rows (expected 2)",
               static_cast<int>(rows.size()));
    TCHECK_MSG(skipped.empty(), "unexpected skips: %d",
               static_cast<int>(skipped.size()));
    if (rows.size() != 2) return;
    // The embedded commas survived as ONE model / notes field each.
    TCHECK_MSG(rows[0].model == "172, 175, 182", "model = '%s'",
               rows[0].model.c_str());
    TCHECK_MSG(rows[0].notes == "early models, pre-1973, used a 0012 tip",
               "notes = '%s'", rows[0].notes.c_str());
    TCHECK(rows[0].datRoot == "naca2412.dat");
    // The RFC-4180 "" escape decodes to a literal double quote.
    TCHECK_MSG(rows[1].model == "S-1 \"Special\"", "model = '%s'",
               rows[1].model.c_str());
}

// -----------------------------------------------------------------------
// Malformed input: every bad shape must be SKIPPED with a reason while the
// good rows around it still parse — and nothing may crash (plan 13 spirit:
// be defensive and report why).
// -----------------------------------------------------------------------
void checkMalformedRowsSkipped() {
    const std::string csv =
        "manufacturer,model,category,airfoil_root,airfoil_tip,dat_root,dat_tip,notes\n"
        "Good,Plane One,homebuilt,NACA 2412,NACA 2412,,,fine row\n"
        "Bad,Too Many,fields,a,b,c,d,unquoted, comma, overflow\n"   // > 8 fields
        ",No Manufacturer,homebuilt,X,X,,,identity missing\n"        // empty maker
        "Short,Row Padded\n"                                         // < 8: padded, OK
        "\n"                                                         // blank: ignored
        "Good,Plane Two,trainer,NACA 0012,NACA 0012,,,also fine\n"
        "Bad,Unterminated,\"quote never closes...";                  // EOF in quotes
    std::vector<std::string> skipped;
    const std::vector<AircraftEntry> rows = parseAircraftManifestCsv(csv, &skipped);

    // 3 good rows survive (two full + the padded short one); 3 bad rows skip.
    TCHECK_MSG(rows.size() == 3, "parsed %d rows (expected 3)",
               static_cast<int>(rows.size()));
    TCHECK_MSG(skipped.size() == 3, "skipped %d rows (expected 3)",
               static_cast<int>(skipped.size()));
    for (const std::string& reason : skipped) {
        TCHECK_MSG(!reason.empty(), "skip reason must be human-readable");
        std::printf("  skip reason: %s\n", reason.c_str());
    }
    if (rows.size() != 3) return;
    // The short row was padded with empty trailing fields, not rejected.
    TCHECK(rows[1].manufacturer == "Short" && rows[1].model == "Row Padded");
    TCHECK(rows[1].category.empty() && rows[1].notes.empty());
    // Good rows on either side of the bad ones were untouched.
    TCHECK(rows[0].model == "Plane One");
    TCHECK(rows[2].model == "Plane Two");

    // Resolution over a synthetic list never crashes on a missing directory
    // either — entries simply stay unresolved.
    std::vector<AircraftEntry> copy = rows;
    resolveManifestDatFiles(copy,
                            std::filesystem::path(FOILCFD_AIRFOIL_DIR)
                                / "definitely_not_a_directory");
    for (const AircraftEntry& e : copy) TCHECK(!e.hasData());
}

} // namespace

int main() {
    checkRealManifest();
    checkQuotedCommaFields();
    checkMalformedRowsSkipped();
    return finish("test_manifest");
}
