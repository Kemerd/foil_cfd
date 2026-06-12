// Aircraft -> airfoil manifest (plan section 15.5): parse the user-editable
// airfoils/aircraft_manifest.csv (robust RFC-4180-ish CSV: quoted fields,
// CRLF, UTF-8 BOM, missing trailing fields), resolve each row's dat_root /
// dat_tip against airfoils/ and airfoils/uiuc/ case-insensitively, and link
// resolved files back to the scanned airfoil catalog so the UI loads them
// through the exact same path as the .dat dropdown.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "../geom/airfoil.h" // AirfoilCatalogEntry (catalog linking)

namespace foilcfd {

/// @brief One aircraft row from aircraft_manifest.csv, plus the on-disk
/// resolution computed at load time. Raw CSV columns are kept verbatim (the
/// manifest README promises designations stay as-documented); the resolved_*
/// fields are what the UI actually acts on.
struct AircraftEntry {
    // -- the eight CSV columns, verbatim (MANIFEST_README.md contract) --
    std::string manufacturer; ///< Airframe maker, e.g. "Stoddard-Hamilton".
    std::string model;        ///< Model / variant range, e.g. "Glasair III".
    std::string category;     ///< homebuilt / GA-single / warbird / ... (badge text).
    std::string airfoilRoot;  ///< Root airfoil designation as documented.
    std::string airfoilTip;   ///< Tip airfoil designation as documented.
    std::string datRoot;      ///< Coordinate filename claimed for the root ("" = none).
    std::string datTip;       ///< Coordinate filename claimed for the tip ("" = none).
    std::string notes;        ///< Caveats line (row tooltip in the UI).

    // -- resolution against airfoils/ + airfoils/uiuc/ (load-time) --
    std::filesystem::path resolvedRoot; ///< Absolute path of datRoot, empty if missing.
    std::filesystem::path resolvedTip;  ///< Absolute path of datTip, empty if missing.
    bool hasRootData = false; ///< datRoot named a file that exists on disk.
    bool hasTipData  = false; ///< datTip named a file that exists on disk.

    // -- catalog linkage (linkManifestToCatalog) --
    int catalogIndexRoot = -1; ///< Index of resolvedRoot in the scanned catalog (-1 = none).
    int catalogIndexTip  = -1; ///< Index of resolvedTip in the scanned catalog (-1 = none).

    /// @brief True when at least one section's coordinates exist on disk —
    /// the row is selectable in the UI iff this holds.
    bool hasData() const { return hasRootData || hasTipData; }
};

/// @brief Full manifest load result: the parsed rows plus diagnostics. The
/// skipped-row reasons are also logged to stderr by loadAircraftManifest so
/// a user editing the CSV sees WHY their row vanished (plan 13 spirit: be
/// defensive and report rejections, never crash).
struct AircraftManifest {
    std::vector<AircraftEntry> entries; ///< Successfully parsed rows, file order.
    std::vector<std::string> skippedRows; ///< One human-readable reason per skipped row.
    std::string loadError; ///< Non-empty when the CSV itself could not be read.
};

/// @brief Parse manifest CSV TEXT into rows (no disk access — unit-testable).
///
/// Tolerates the full messy-CSV zoo: double-quoted fields containing commas,
/// newlines, and "" escapes; CRLF and bare-LF line endings; a UTF-8 BOM; and
/// rows with missing trailing fields (padded empty). A leading header row
/// (first field == "manufacturer", case-insensitive) is skipped. Malformed
/// rows — unterminated quote, more than 8 fields (an unescaped comma), or an
/// empty manufacturer/model — are skipped with a reason, never fatal.
/// @param csvText        Entire CSV contents as UTF-8 text.
/// @param skippedReasons Optional sink for one reason string per skipped row.
/// @return Parsed entries in file order, dat fields UNRESOLVED (paths empty).
std::vector<AircraftEntry> parseAircraftManifestCsv(
    const std::string& csvText, std::vector<std::string>* skippedReasons = nullptr);

/// @brief Resolve every entry's datRoot/datTip against @p airfoilsDir and
/// @p airfoilsDir/uiuc (in that precedence order), matching filenames
/// CASE-INSENSITIVELY (the UIUC zoo mixes .dat/.DAT and the manifest is
/// hand-edited). Fills resolvedRoot/resolvedTip with weakly-canonical
/// absolute paths and the hasRootData/hasTipData flags.
/// @param entries     Rows from parseAircraftManifestCsv (mutated in place).
/// @param airfoilsDir The airfoils/ data directory (same one the catalog scans).
void resolveManifestDatFiles(std::vector<AircraftEntry>& entries,
                             const std::filesystem::path& airfoilsDir);

/// @brief Convenience load: read @p airfoilsDir/aircraft_manifest.csv, parse,
/// resolve dat files, and log every skipped row to stderr. Called at startup
/// and again from the airfoil "Rescan" button (plan 15.5: the manifest is
/// user-editable and re-read on refresh). A missing CSV is NOT an error —
/// the feature simply renders empty with loadError explaining why.
/// @param airfoilsDir The airfoils/ data directory.
AircraftManifest loadAircraftManifest(const std::filesystem::path& airfoilsDir);

/// @brief Map each entry's resolved paths onto indices into the scanned
/// airfoil catalog (case-insensitive canonical-path match). The UI then loads
/// an aircraft by setting selectedDatIndex to catalogIndexRoot/Tip and raising
/// reloadAirfoil — byte-for-byte the same code path as the .dat dropdown, so
/// loading logic exists in exactly one place (plan 15.5 UI contract).
/// @param entries Resolved manifest rows (mutated in place).
/// @param catalog The catalog from scanAirfoilDirectory (recursive, so every
///                file resolveManifestDatFiles can find is present in it).
void linkManifestToCatalog(std::vector<AircraftEntry>& entries,
                           const std::vector<AirfoilCatalogEntry>& catalog);

} // namespace foilcfd
