// aircraft_manifest.h implementation: a small, defensive CSV reader for the
// user-editable aircraft_manifest.csv (quoted fields with embedded commas /
// newlines / "" escapes, CRLF, UTF-8 BOM, padded short rows, skip-with-reason
// for malformed ones), plus the case-insensitive dat-file resolution against
// airfoils/ + airfoils/uiuc/ and the catalog-index linking the UI loads from.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include "aircraft_manifest.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <system_error>
#include <unordered_map>

#include "../platform/platform.h" // pathToUtf8 (inline, no link dependency)

namespace foilcfd {

namespace {

/// The manifest's fixed column count (MANIFEST_README.md contract).
constexpr std::size_t kManifestColumns = 8;

/// @brief ASCII-lowercase a copy of @p s. Filenames and CSV header keywords
/// are ASCII in practice; multibyte UTF-8 sequences pass through untouched
/// (tolower is only applied to single bytes < 0x80 via the unsigned-char cast).
std::string toLowerAscii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

/// @brief Strip leading/trailing ASCII whitespace (spaces, tabs, stray CR
/// from hand-edited CRLF files) from a field.
std::string trimmedField(const std::string& s) {
    std::size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

/// @brief One raw CSV record: its fields, the 1-based line it started on
/// (for error messages), and whether a quote was left unterminated at EOF.
struct CsvRecord {
    std::vector<std::string> fields;
    int  startLine = 0;
    bool unterminatedQuote = false;
};

/// @brief Split the whole CSV text into records, RFC-4180 style but lenient:
/// a field starting with '"' runs until the closing quote ("" = literal '"',
/// commas/newlines inside are data); outside quotes ',' splits fields, '\n'
/// ends the record, and '\r' is dropped (CRLF tolerance). A UTF-8 BOM on the
/// first byte triplet is stripped. Never throws — the manifest is user-edited
/// and a typo must degrade to a skipped row, not a crash (plan 15.5).
std::vector<CsvRecord> splitCsvRecords(const std::string& text) {
    std::vector<CsvRecord> records;
    std::size_t i = 0;
    // Strip the UTF-8 BOM some Windows editors prepend on save.
    if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF
        && static_cast<unsigned char>(text[1]) == 0xBB
        && static_cast<unsigned char>(text[2]) == 0xBF) {
        i = 3;
    }

    CsvRecord rec;
    rec.startLine = 1;
    std::string field;
    bool inQuotes = false;
    bool recordHasContent = false; // distinguishes "a," from a blank line
    int  line = 1;

    // Close out the current field / record. Fields are whitespace-trimmed so
    // a hand-edited "foo, bar" row still parses cleanly (the manifest never
    // needs significant leading/trailing spaces inside a field).
    auto endField = [&] {
        rec.fields.push_back(trimmedField(field));
        field.clear();
    };
    auto endRecord = [&](int nextLine) {
        if (recordHasContent) {
            endField();
            records.push_back(std::move(rec));
        }
        rec = CsvRecord{};
        rec.startLine = nextLine;
        field.clear();
        recordHasContent = false;
    };

    for (; i < text.size(); ++i) {
        const char c = text[i];
        if (inQuotes) {
            if (c == '"') {
                // "" inside quotes is an escaped literal quote (RFC 4180).
                if (i + 1 < text.size() && text[i + 1] == '"') {
                    field += '"';
                    ++i;
                } else {
                    inQuotes = false; // closing quote; trailing junk is kept
                }
            } else {
                if (c == '\n') ++line; // quoted newlines are data, keep counting
                field += c;
            }
            recordHasContent = true;
        } else if (c == '"') {
            // Opening quote (mid-field quotes are tolerated as literal text
            // by falling through here too — lenient by design).
            inQuotes = true;
            recordHasContent = true;
        } else if (c == ',') {
            endField();
            recordHasContent = true;
        } else if (c == '\n') {
            ++line;
            endRecord(line);
        } else if (c != '\r') { // CRLF: drop the CR, the '\n' branch closes
            field += c;
            recordHasContent = true;
        }
    }
    // EOF: flush a final record with no trailing newline; an unterminated
    // quote is flagged so the caller can skip the row WITH a reason.
    rec.unterminatedQuote = inQuotes;
    endRecord(line);
    return records;
}

} // namespace

// ===========================================================================
// Public API (aircraft_manifest.h contract).
// ===========================================================================

std::vector<AircraftEntry> parseAircraftManifestCsv(
    const std::string& csvText, std::vector<std::string>* skippedReasons) {
    std::vector<AircraftEntry> entries;
    auto skip = [&](const CsvRecord& rec, const std::string& why) {
        if (skippedReasons) {
            skippedReasons->push_back(
                "line " + std::to_string(rec.startLine) + ": " + why);
        }
    };

    bool first = true;
    for (const CsvRecord& rec : splitCsvRecords(csvText)) {
        // Header row: skipped by content (not position) so a manifest whose
        // header was deleted still parses every data row.
        if (first) {
            first = false;
            if (!rec.fields.empty()
                && toLowerAscii(rec.fields[0]) == "manufacturer") {
                continue;
            }
        }
        // -- malformed-row triage (skip with a reason, never crash) --
        if (rec.unterminatedQuote) {
            skip(rec, "unterminated quoted field");
            continue;
        }
        if (rec.fields.size() > kManifestColumns) {
            skip(rec, "too many fields (" + std::to_string(rec.fields.size())
                          + " > 8) — unquoted comma in a field?");
            continue;
        }
        // Short rows are padded (missing trailing fields are legal), but a
        // row without both identity columns cannot name an aircraft at all.
        std::vector<std::string> f = rec.fields;
        f.resize(kManifestColumns);
        if (f[0].empty() || f[1].empty()) {
            skip(rec, "missing manufacturer and/or model");
            continue;
        }

        // -- a well-formed row: map columns onto the entry, verbatim --
        AircraftEntry e;
        e.manufacturer = f[0];
        e.model        = f[1];
        e.category     = f[2];
        e.airfoilRoot  = f[3];
        e.airfoilTip   = f[4];
        e.datRoot      = f[5];
        e.datTip       = f[6];
        e.notes        = f[7];
        entries.push_back(std::move(e));
    }
    return entries;
}

void resolveManifestDatFiles(std::vector<AircraftEntry>& entries,
                             const std::filesystem::path& airfoilsDir) {
    namespace fs = std::filesystem;

    // Build a lowercase-filename -> absolute-path index over the two search
    // directories ONCE (the manifest has dozens of rows; per-row directory
    // scans would be O(rows x files)). airfoils/ is added first so it wins
    // over uiuc/ on a duplicate filename, matching the README's precedence.
    std::unordered_map<std::string, fs::path> index;
    auto addDirectory = [&](const fs::path& dir) {
        std::error_code ec;
        if (!fs::is_directory(dir, ec)) return; // optional dir; skip quietly
        for (fs::directory_iterator it(dir, ec), end; !ec && it != end;
             it.increment(ec)) {
            std::error_code fec;
            if (!it->is_regular_file(fec)) continue;
            // UTF-8 + lowercase key: pathToUtf8 never throws on non-ASCII
            // names (unlike path::string() on MSVC), and the lowering makes
            // the lookup case-insensitive on every platform.
            const std::string key =
                toLowerAscii(platform::pathToUtf8(it->path().filename()));
            std::error_code cec;
            const fs::path canon = fs::weakly_canonical(it->path(), cec);
            index.try_emplace(key, cec ? it->path() : canon); // first dir wins
        }
    };
    addDirectory(airfoilsDir);
    addDirectory(airfoilsDir / "uiuc");

    // Resolve each entry's claimed filenames through the index. Empty dat
    // fields (the manifest's "no coordinates on disk" marker) stay empty.
    auto resolveOne = [&](const std::string& datName, fs::path& outPath,
                          bool& outHas) {
        outPath.clear();
        outHas = false;
        if (datName.empty()) return;
        const auto it = index.find(toLowerAscii(datName));
        if (it != index.end()) {
            outPath = it->second;
            outHas  = true;
        }
    };
    for (AircraftEntry& e : entries) {
        resolveOne(e.datRoot, e.resolvedRoot, e.hasRootData);
        resolveOne(e.datTip, e.resolvedTip, e.hasTipData);
        // Resolution invalidates any previous catalog linkage; the caller
        // re-links against the freshly scanned catalog afterwards.
        e.catalogIndexRoot = -1;
        e.catalogIndexTip  = -1;
    }
}

AircraftManifest loadAircraftManifest(const std::filesystem::path& airfoilsDir) {
    AircraftManifest manifest;
    const std::filesystem::path csvPath = airfoilsDir / "aircraft_manifest.csv";

    // Binary read: the parser owns ALL line-ending handling (CRLF vs LF), so
    // the stream must not translate anything behind its back.
    std::ifstream in(csvPath, std::ios::binary);
    if (!in) {
        manifest.loadError = "aircraft_manifest.csv not found in "
                           + platform::pathToUtf8(airfoilsDir);
        return manifest; // missing manifest = feature renders empty, not fatal
    }
    std::ostringstream buf;
    buf << in.rdbuf();
    if (in.bad()) { // read error mid-file (I/O fault, not a parse problem)
        manifest.loadError = "failed reading "
                           + platform::pathToUtf8(csvPath);
        return manifest;
    }

    manifest.entries = parseAircraftManifestCsv(buf.str(), &manifest.skippedRows);
    resolveManifestDatFiles(manifest.entries, airfoilsDir);

    // Surface every skipped row on stderr: the CSV is user-editable, and a
    // silently vanishing aircraft is exactly the confusion plan 13 forbids.
    for (const std::string& reason : manifest.skippedRows) {
        std::fprintf(stderr, "[foilcfd] aircraft_manifest.csv row skipped: %s\n",
                     reason.c_str());
    }
    return manifest;
}

void linkManifestToCatalog(std::vector<AircraftEntry>& entries,
                           const std::vector<AirfoilCatalogEntry>& catalog) {
    namespace fs = std::filesystem;

    // Canonical lowercase path -> catalog index. Canonicalizing BOTH sides
    // makes the comparison robust to mixed separators / relative segments
    // between the catalog scan root and the manifest resolution root.
    std::unordered_map<std::string, int> byPath;
    byPath.reserve(catalog.size());
    for (int i = 0; i < static_cast<int>(catalog.size()); ++i) {
        std::error_code ec;
        const fs::path canon =
            fs::weakly_canonical(catalog[static_cast<std::size_t>(i)].path, ec);
        byPath.try_emplace(
            toLowerAscii(platform::pathToUtf8(
                ec ? catalog[static_cast<std::size_t>(i)].path : canon)),
            i);
    }

    // resolvedRoot/resolvedTip are already weakly canonical (resolution
    // contract), so the lookup is a straight map hit per section.
    auto lookup = [&](const fs::path& resolved) -> int {
        if (resolved.empty()) return -1;
        const auto it = byPath.find(toLowerAscii(platform::pathToUtf8(resolved)));
        return it != byPath.end() ? it->second : -1;
    };
    for (AircraftEntry& e : entries) {
        e.catalogIndexRoot = lookup(e.resolvedRoot);
        e.catalogIndexTip  = lookup(e.resolvedTip);
    }
}

} // namespace foilcfd
