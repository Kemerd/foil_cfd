// Airfoil geometry sources: the NACA 4-digit generator (closed trailing edge,
// cosine spacing), the defensive UIUC .dat loader (Selig + Lednicer with
// human-readable rejection reasons), the airfoils/ catalog scan, and the
// surface point/tangent/normal frame query that VG placement seats vanes with
// (plan sections 5, 6.1, and the section-13 ".dat zoo" pitfall list).
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include "airfoil.h"

// UTF-8 path conversions: path::string() decodes via the ANSI code page on
// MSVC and can throw for non-ASCII filenames the UIUC zoo / users provide.
#include "../platform/platform.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>

namespace foilcfd {
namespace {

constexpr float kPi = 3.14159265358979f;

// ===========================================================================
// Small parsing helpers. The UIUC database is three decades of hand-typed
// files: CRLF and LF line endings, tabs, comma separators, stray blank lines,
// duplicate points, and the occasional descriptive header. Everything below
// is written to either tolerate a quirk or name it in the rejection reason.
// ===========================================================================

/// @brief One surviving (non-blank) input line plus its 1-based line number
/// in the original file, so rejection messages can point at the exact line.
struct NumberedLine {
    int lineNo = 0;
    std::string text;
};

/// @brief Strip carriage returns (CRLF files read in text or binary mode),
/// tabs, and leading/trailing spaces from a raw input line.
std::string trimmed(const std::string& s) {
    const char* ws = " \t\r\n\f\v";
    const std::size_t first = s.find_first_not_of(ws);
    if (first == std::string::npos) return {};
    const std::size_t last = s.find_last_not_of(ws);
    return s.substr(first, last - first + 1);
}

/// @brief Parse the first two floats on a line (whitespace or comma
/// separated). Extra trailing columns are tolerated — a few zoo files carry
/// a third annotation column. Returns false if either number is missing or
/// non-finite, which is also how name/header lines are recognized.
bool parseTwoFloats(const std::string& line, float& a, float& b) {
    const char* s = line.c_str();
    char* end = nullptr;
    const float va = std::strtof(s, &end);
    if (end == s) return false; // first token is not numeric
    // Skip separator run between the two numbers (spaces, tabs, one comma).
    const char* s2 = end;
    while (*s2 == ' ' || *s2 == '\t' || *s2 == ',') ++s2;
    const float vb = std::strtof(s2, &end);
    if (end == s2) return false; // second number missing (e.g. "12% smoothed")
    if (!std::isfinite(va) || !std::isfinite(vb)) return false;
    a = va;
    b = vb;
    return true;
}

/// @brief Signed shoelace area of a point loop (implicitly closed). Positive
/// for counter-clockwise winding; the canonical Selig loop (TE -> upper ->
/// LE -> lower -> TE) winds clockwise, so a healthy foil yields negative area.
float polygonArea(const std::vector<Vec2f>& pts) {
    double acc = 0.0; // double accumulator: ~400 small terms, avoid cancellation
    const std::size_t n = pts.size();
    for (std::size_t i = 0; i < n; ++i) {
        const Vec2f& p = pts[i];
        const Vec2f& q = pts[(i + 1) % n];
        acc += static_cast<double>(p.x) * q.y - static_cast<double>(q.x) * p.y;
    }
    return static_cast<float>(0.5 * acc);
}

/// @brief Remove consecutive (near-)duplicate points — duplicated LE points
/// are the single most common .dat quirk and break tangent estimation.
/// @param eps Distance below which two neighbors count as the same point.
void dedupeConsecutive(std::vector<Vec2f>& pts, float eps) {
    if (pts.size() < 2) return;
    std::vector<Vec2f> out;
    out.reserve(pts.size());
    out.push_back(pts.front());
    for (std::size_t i = 1; i < pts.size(); ++i) {
        const Vec2f d = pts[i] - out.back();
        if (length(d) > eps) out.push_back(pts[i]);
    }
    pts.swap(out);
}

/// @brief Index of the minimum-x point of a loop — the leading edge after
/// normalization, and the upper/lower surface split index.
int argMinX(const std::vector<Vec2f>& pts) {
    int best = 0;
    for (int i = 1; i < static_cast<int>(pts.size()); ++i) {
        if (pts[i].x < pts[static_cast<std::size_t>(best)].x) best = i;
    }
    return best;
}

/// @brief Normalize a raw point loop in place: chord scaled to [0,1] with the
/// LE translated to the origin (plan 5.2), then canonicalize the traversal
/// direction to upper-surface-first (the Selig convention every consumer —
/// voxelizer, surface frames, mesh extrusion — assumes). Fills leIndex.
/// @return Empty string on success, otherwise the rejection reason.
std::string normalizeLoop(std::vector<Vec2f>& pts, int& leIndex) {
    // Chord from the raw x extent — files arrive in chord fractions, inches,
    // or millimeters; the extent is the only unit-free chord measure.
    float minX = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    for (const Vec2f& p : pts) {
        minX = std::min(minX, p.x);
        maxX = std::max(maxX, p.x);
    }
    const float chord = maxX - minX;
    if (!(chord > 1e-6f)) return "degenerate chord (max x equals min x)";

    // Translate the LE vertex to the origin, scale the chord to unity. Using
    // the LE point's own y keeps the chord line passing through the origin.
    const Vec2f le = pts[static_cast<std::size_t>(argMinX(pts))];
    for (Vec2f& p : pts) p = (p - le) / chord;

    // Duplicate-point cleanup AFTER normalization so the epsilon is in chord
    // units regardless of the file's original scale.
    dedupeConsecutive(pts, 1e-6f);
    if (pts.size() < 20) {
        return "only " + std::to_string(pts.size())
             + " distinct points (minimum 20)";
    }

    // The LE must be interior to the loop: a loop that STARTS at the LE is
    // some ordering this loader does not understand — refuse rather than
    // voxelize garbage (plan 13: report why).
    leIndex = argMinX(pts);
    if (leIndex == 0 || leIndex == static_cast<int>(pts.size()) - 1) {
        return "leading edge at loop end (point " + std::to_string(leIndex)
             + " of " + std::to_string(pts.size()) + ") - unrecognized ordering";
    }

    // Canonical direction check: points[0..leIndex] must be the UPPER surface.
    // Compare mean y of the two halves; a lower-surface-first file (it happens)
    // is simply reversed into the canonical loop.
    double upperSum = 0.0, lowerSum = 0.0;
    for (int i = 0; i <= leIndex; ++i) upperSum += pts[static_cast<std::size_t>(i)].y;
    for (int i = leIndex; i < static_cast<int>(pts.size()); ++i)
        lowerSum += pts[static_cast<std::size_t>(i)].y;
    const double upperMean = upperSum / (leIndex + 1);
    const double lowerMean = lowerSum / (static_cast<int>(pts.size()) - leIndex);
    if (upperMean < lowerMean) {
        std::reverse(pts.begin(), pts.end());
        leIndex = argMinX(pts);
    }

    // Final shape sanity: a (near-)zero enclosed area means collinear points
    // or a self-cancelling outline — the parity voxelizer would produce noise.
    // Thinnest legitimate sections (~1% t/c) enclose area ~7e-3 c^2.
    if (std::fabs(polygonArea(pts)) < 1e-3f) {
        return "outline encloses (near-)zero area - collinear or corrupt points";
    }
    return {};
}

} // namespace

// ===========================================================================
// NACA 4-digit generator (plan 5.1)
// ===========================================================================

AirfoilLoadResult generateNACA4(const std::string& digits, int pointsPerSurface) {
    AirfoilLoadResult result;
    if (digits.size() != 4) {
        result.rejectionReason = "expected exactly 4 digits, got '" + digits + "'";
        return result;
    }
    for (char c : digits) {
        if (c < '0' || c > '9') {
            result.rejectionReason = "non-digit character in '" + digits + "'";
            return result;
        }
    }

    // Decode the standard NACA 4-digit parameters:
    //   m = maximum camber (first digit, % chord)
    //   p = position of maximum camber (second digit, tenths of chord)
    //   t = maximum thickness (last two digits, % chord)
    const float m = static_cast<float>(digits[0] - '0') / 100.0f;
    const float p = static_cast<float>(digits[1] - '0') / 10.0f;
    const float t = static_cast<float>((digits[2] - '0') * 10 + (digits[3] - '0'))
                  / 100.0f;
    if (t <= 0.0f) {
        result.rejectionReason = "thickness digits are 00 (zero-thickness section)";
        return result;
    }
    if (m > 0.0f && p <= 0.0f) {
        // e.g. "1012": nonzero camber with the camber position at the LE is
        // outside the 4-digit definition (division by p below).
        result.rejectionReason =
            "camber position digit is 0 while camber digit is nonzero";
        return result;
    }

    const int n = std::max(pointsPerSurface, 20);
    AirfoilGeometry& foil = result.airfoil;
    foil.name = "NACA " + digits;
    foil.points.reserve(static_cast<std::size_t>(2 * n - 1));

    // Per-station evaluation of the standard equations. Cosine spacing
    // x = (1 - cos(pi*k/(n-1)))/2 clusters points at both the LE (high
    // curvature) and TE (thin wedge) where the voxelizer needs them most.
    auto evalStation = [&](int k, Vec2f& upper, Vec2f& lower) {
        const float x = 0.5f * (1.0f - std::cos(kPi * static_cast<float>(k)
                                                / static_cast<float>(n - 1)));
        // Thickness half-distribution, CLOSED-TE variant: the final
        // coefficient is -0.1036 (vs the open-TE -0.1015) so yt(1) == 0
        // exactly and the TE closure pass has a watertight outline (plan 5.1).
        const float sqx = std::sqrt(x);
        const float yt = 5.0f * t
                       * (0.2969f * sqx - 0.1260f * x - 0.3516f * x * x
                          + 0.2843f * x * x * x - 0.1036f * x * x * x * x);
        // Mean camber line and its slope: two parabolic arcs meeting at x = p.
        float yc = 0.0f, dyc = 0.0f;
        if (m > 0.0f) {
            if (x < p) {
                yc  = (m / (p * p)) * (2.0f * p * x - x * x);
                dyc = (2.0f * m / (p * p)) * (p - x);
            } else {
                yc  = (m / ((1.0f - p) * (1.0f - p)))
                    * ((1.0f - 2.0f * p) + 2.0f * p * x - x * x);
                dyc = (2.0f * m / ((1.0f - p) * (1.0f - p))) * (p - x);
            }
        }
        // Thickness is applied PERPENDICULAR to the camber line (the exact
        // construction, not the thin-airfoil shortcut of adding yt to yc).
        const float theta = std::atan(dyc);
        const float st = std::sin(theta);
        const float ct = std::cos(theta);
        upper = {x - yt * st, yc + yt * ct};
        lower = {x + yt * st, yc - yt * ct};
    };

    // Assemble the canonical Selig loop: TE -> upper -> LE -> lower -> TE.
    // k runs n-1 (TE) down to 0 (LE) on the upper surface...
    for (int k = n - 1; k >= 0; --k) {
        Vec2f u, l;
        evalStation(k, u, l);
        foil.points.push_back(u);
    }
    foil.leIndex = n - 1; // yt(0) == 0, so the upper LE point is exactly (0,0)
    // ...then 1 (just aft of LE) up to n-1 (TE) on the lower surface; k = 0
    // is skipped because upper and lower coincide at the LE point.
    for (int k = 1; k < n; ++k) {
        Vec2f u, l;
        evalStation(k, u, l);
        foil.points.push_back(l);
    }
    result.ok = true;
    return result;
}

// ===========================================================================
// UIUC .dat loader (plan 5.2 / pitfall 13: be defensive, say why you refused)
// ===========================================================================

AirfoilLoadResult loadAirfoilDat(const std::filesystem::path& path) {
    AirfoilLoadResult result;

    std::ifstream in(path, std::ios::binary); // binary: we trim CR ourselves
    if (!in) {
        result.rejectionReason = "cannot open file";
        return result;
    }

    // Collect every non-blank line with its original line number — blank
    // lines are structure in Lednicer files but we tolerate them anywhere.
    // '#'-prefixed comment lines (annotated copies, e.g. our own ls413.dat,
    // carry provenance headers) are skipped the same way.
    std::vector<NumberedLine> lines;
    {
        std::string raw;
        int lineNo = 0;
        while (std::getline(in, raw)) {
            ++lineNo;
            std::string t = trimmed(raw);
            if (t.empty() || t[0] == '#') continue;
            lines.push_back({lineNo, std::move(t)});
        }
    }
    if (lines.empty()) {
        result.rejectionReason = "file is empty";
        return result;
    }

    // ---- Header: name line (optional — some zoo files start with data). ----
    std::size_t idx = 0;
    float a = 0.0f, b = 0.0f;
    if (!parseTwoFloats(lines[0].text, a, b)) {
        result.airfoil.name = lines[0].text;
        idx = 1;
    } else {
        // Headerless coordinate file: fall back to the filename as the name
        // (UTF-8 — stem().string() would throw for non-ASCII filenames).
        result.airfoil.name = platform::pathToUtf8(path.stem());
    }

    // ---- Format detection (plan 5.2): Lednicer iff the first post-name ----
    // line parses as two values > 1 — those are the upper/lower point counts
    // ("45.  45."), which no Selig coordinate pair can be (x <= ~1.01 in
    // chord-fraction files). The near-integer requirement additionally
    // protects millimeter-scaled Selig files whose first pair could be large.
    bool lednicer = false;
    int nUpper = 0, nLower = 0;
    if (idx < lines.size() && parseTwoFloats(lines[idx].text, a, b)) {
        const bool integral = std::fabs(a - std::round(a)) < 1e-3f
                           && std::fabs(b - std::round(b)) < 1e-3f;
        if (a > 1.5f && b > 1.5f && integral) {
            lednicer = true;
            nUpper = static_cast<int>(std::lround(a));
            nLower = static_cast<int>(std::lround(b));
            ++idx;
        }
    }
    result.format = lednicer ? DatFormat::Lednicer : DatFormat::Selig;

    // ---- Coordinate lines: every remaining line must be an x,y pair. ----
    std::vector<Vec2f> raw;
    raw.reserve(lines.size() - idx);
    for (; idx < lines.size(); ++idx) {
        float x = 0.0f, y = 0.0f;
        if (!parseTwoFloats(lines[idx].text, x, y)) {
            const std::string snippet = lines[idx].text.substr(0, 40);
            result.rejectionReason = "line " + std::to_string(lines[idx].lineNo)
                                   + ": expected two numbers, got '" + snippet + "'";
            return result;
        }
        raw.emplace_back(x, y);
    }

    // ---- Assemble the canonical loop (TE -> upper -> LE -> lower -> TE). ----
    std::vector<Vec2f>& pts = result.airfoil.points;
    if (lednicer) {
        // Lednicer stores upper LE->TE, then lower LE->TE (counts declared).
        const std::size_t need = static_cast<std::size_t>(nUpper)
                               + static_cast<std::size_t>(nLower);
        if (nUpper < 2 || nLower < 2) {
            result.rejectionReason = "Lednicer point counts implausible ("
                                   + std::to_string(nUpper) + " upper, "
                                   + std::to_string(nLower) + " lower)";
            return result;
        }
        if (raw.size() < need) {
            result.rejectionReason = "Lednicer header declares "
                                   + std::to_string(need) + " points but only "
                                   + std::to_string(raw.size())
                                   + " coordinate lines follow";
            return result;
        }
        // Reverse the upper surface to TE->LE, then append the lower LE->TE.
        pts.reserve(need);
        for (int i = nUpper - 1; i >= 0; --i)
            pts.push_back(raw[static_cast<std::size_t>(i)]);
        // Skip the lower surface's LE point when it duplicates the upper LE
        // (it almost always does) — the epsilon is relative to the raw chord.
        float minX = std::numeric_limits<float>::max();
        float maxX = std::numeric_limits<float>::lowest();
        for (const Vec2f& q : raw) {
            minX = std::min(minX, q.x);
            maxX = std::max(maxX, q.x);
        }
        const float eps = 1e-4f * std::max(maxX - minX, 1e-6f);
        const std::size_t lowerBegin = static_cast<std::size_t>(nUpper);
        std::size_t start = lowerBegin;
        if (length(raw[lowerBegin] - raw[0]) < eps) ++start;
        for (std::size_t i = start; i < need; ++i) pts.push_back(raw[i]);
    } else {
        // Selig files ARE the loop already.
        pts = std::move(raw);
    }

    if (pts.size() < 20) {
        result.rejectionReason = "only " + std::to_string(pts.size())
                               + " points (minimum 20)";
        return result;
    }

    // ---- Normalize + canonicalize; any failure reason passes through. ----
    if (std::string err = normalizeLoop(pts, result.airfoil.leIndex); !err.empty()) {
        result.rejectionReason = std::move(err);
        return result;
    }
    result.ok = true;
    return result;
}

// ===========================================================================
// Catalog scan (plan 5.2: dropdown lists filename + the file's name line)
// ===========================================================================

std::vector<AirfoilCatalogEntry> scanAirfoilDirectory(
    const std::filesystem::path& directory) {
    namespace fs = std::filesystem;
    std::vector<AirfoilCatalogEntry> entries;

    std::error_code ec;
    fs::recursive_directory_iterator it(
        directory, fs::directory_options::skip_permission_denied, ec);
    if (ec) return entries; // missing/unreadable directory -> empty catalog

    for (const fs::recursive_directory_iterator end; it != end; it.increment(ec)) {
        if (ec) break; // iteration error: return what we gathered so far
        if (!it->is_regular_file(ec)) continue;
        // Case-insensitive .dat extension match (the zoo has .DAT files too).
        std::string ext = platform::pathToUtf8(it->path().extension());
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext != ".dat") continue;

        // Cheap peek at the first content line only — full parsing happens
        // on selection, and unparseable files stay listed so the user can
        // select one and read its rejection reason (header contract). Blank
        // and '#' comment lines are skipped just like the loader does.
        AirfoilCatalogEntry entry;
        entry.path = it->path();
        // UTF-8 display name: one non-ASCII filename anywhere under
        // airfoils/ must not abort the whole startup scan with a throwing
        // ANSI conversion.
        entry.displayName = platform::pathToUtf8(it->path().filename());
        std::ifstream f(it->path(), std::ios::binary);
        std::string raw;
        while (f && std::getline(f, raw)) {
            const std::string first = trimmed(raw);
            if (first.empty() || first[0] == '#') continue;
            float a = 0.0f, b = 0.0f;
            // A numeric first content line means a headerless file: the
            // filename alone is the display name.
            if (!parseTwoFloats(first, a, b)) {
                entry.displayName += " - " + first;
            }
            break;
        }
        entries.push_back(std::move(entry));
    }

    // Stable, case-insensitive alphabetical order for the UI dropdown.
    std::sort(entries.begin(), entries.end(),
              [](const AirfoilCatalogEntry& l, const AirfoilCatalogEntry& r) {
                  const std::string lf = platform::pathToUtf8(l.path.filename());
                  const std::string rf = platform::pathToUtf8(r.path.filename());
                  return std::lexicographical_compare(
                      lf.begin(), lf.end(), rf.begin(), rf.end(),
                      [](unsigned char c1, unsigned char c2) {
                          return std::tolower(c1) < std::tolower(c2);
                      });
              });
    return entries;
}

// ===========================================================================
// Surface frame query (plan 6.1: "the vane root follows the surface")
// ===========================================================================

SurfaceFrame surfaceFrameAt(const AirfoilGeometry& airfoil, float x_c, bool upper) {
    SurfaceFrame frame;
    if (!airfoil.isValid() || x_c < 0.0f || x_c > 1.0f) return frame;
    const int n = static_cast<int>(airfoil.points.size());
    if (airfoil.leIndex <= 0 || airfoil.leIndex >= n - 1) return frame;

    // Requested surface span within the canonical loop: the upper surface is
    // points[0..leIndex] (running TE -> LE), the lower is points[leIndex..n-1]
    // (running LE -> TE).
    const int begin = upper ? 0 : airfoil.leIndex;
    const int end   = upper ? airfoil.leIndex : n - 1;

    // Find the segment whose x-interval brackets x_c. Surfaces are monotone
    // in x away from the nose, but cambered noses can fold slightly — so we
    // track the NEAREST segment by x-interval distance and fall back to it
    // when no exact bracket exists (e.g. x_c == 1.0 on a file whose upper
    // surface tops out at 0.9997). VG placement must not fail at the edges.
    int bestSeg = -1;
    float bestDist = std::numeric_limits<float>::max();
    for (int i = begin; i < end; ++i) {
        const Vec2f& pa = airfoil.points[static_cast<std::size_t>(i)];
        const Vec2f& pb = airfoil.points[static_cast<std::size_t>(i + 1)];
        const float lo = std::min(pa.x, pb.x);
        const float hi = std::max(pa.x, pb.x);
        const float d = (x_c < lo) ? (lo - x_c) : (x_c > hi ? x_c - hi : 0.0f);
        if (d < bestDist) {
            bestDist = d;
            bestSeg = i;
            if (d == 0.0f && hi - lo > 1e-9f) break; // exact bracket found
        }
    }
    if (bestSeg < 0) return frame;

    const Vec2f& pa = airfoil.points[static_cast<std::size_t>(bestSeg)];
    const Vec2f& pb = airfoil.points[static_cast<std::size_t>(bestSeg + 1)];
    // Linear interpolation parameter along the segment, clamped so nearest-
    // segment fallback lands on the closer endpoint instead of extrapolating.
    const float dx = pb.x - pa.x;
    const float tParam = (std::fabs(dx) > 1e-9f)
                       ? std::clamp((x_c - pa.x) / dx, 0.0f, 1.0f)
                       : 0.5f;
    frame.point = pa + (pb - pa) * tParam;

    // Tangent must point LE -> TE (downstream) regardless of which surface:
    // the upper span is stored TE -> LE, so its segment direction is flipped.
    frame.tangent = normalized(upper ? (pa - pb) : (pb - pa));

    // Outward normal from the canonical winding (clockwise around the foil
    // interior): rotating the downstream tangent +90 deg CCW points away from
    // the interior on the upper surface, and toward it on the lower — hence
    // the sign flip. This stays correct around the nose where the tangent is
    // near-vertical and any y-sign heuristic would break.
    frame.normal = upper ? perpCCW(frame.tangent)
                         : perpCCW(frame.tangent) * -1.0f;
    frame.valid = true;
    return frame;
}

} // namespace foilcfd
