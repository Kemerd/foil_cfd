// Airfoil geometry sources (plan section 5): NACA 4-digit generator and the
// UIUC .dat loader (Selig + Lednicer, defensive, with rejection reasons),
// plus the surface point/tangent/normal frame query that VG placement uses.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "../core/vec.h"

namespace foilcfd {

/// @brief Which .dat layout the loader detected (plan 5.2).
enum class DatFormat {
    Selig,    ///< Name line, then one loop TE -> upper -> LE -> lower -> TE.
    Lednicer, ///< Name, point-count line ("NU. NL."), upper LE->TE, blank, lower LE->TE.
};

/// @brief A normalized airfoil section: chord on [0,1], LE at the origin.
///
/// Canonical storage is the Selig loop ordering — `points` runs
/// TE -> upper surface -> LE -> lower surface -> back to TE — because that is
/// simultaneously a valid simple polygon for the voxelizer's point-in-polygon
/// test and the natural order for outline-extrusion mesh rendering. Lednicer
/// input is converted to this loop on load.
struct AirfoilGeometry {
    std::string name;          ///< Display name (the .dat name line, or "NACA xxxx").
    std::vector<Vec2f> points; ///< Closed outline loop (see ordering note above).
    int leIndex = 0;           ///< Index into `points` of the leading-edge point —
                               ///< points [0..leIndex] are the upper surface,
                               ///< [leIndex..size-1] the lower (loop ordering).

    /// @brief True when the geometry holds a plausible section (>= 20 points,
    /// nonzero area, normalized chord). Loader/generator guarantee this on
    /// success; default-constructed instances are invalid.
    bool isValid() const { return points.size() >= 20; }
};

/// @brief Local surface coordinate frame at a chordwise station, used to seat
/// VG vanes on the suction surface (plan section 6.1). 2D in the section
/// plane; the spanwise axis (z) completes the right-handed triad implicitly.
struct SurfaceFrame {
    Vec2f point;   ///< Surface point at the requested x/c (chord units).
    Vec2f tangent; ///< Unit tangent, pointing LE -> TE (downstream).
    Vec2f normal;  ///< Unit outward normal (away from the foil interior).
    bool  valid = false; ///< False if x/c was outside [0,1] or geometry invalid.
};

/// @brief Result wrapper for anything that can reject its input. The .dat zoo
/// is messy (plan section 13) — the loader must report WHY a file failed, and
/// the UI surfaces that string verbatim.
struct AirfoilLoadResult {
    bool ok = false;            ///< True when `airfoil` is usable.
    AirfoilGeometry airfoil;    ///< The loaded/generated section (when ok).
    DatFormat format = DatFormat::Selig; ///< Detected layout (loader only).
    std::string rejectionReason;///< Human-readable failure cause (when !ok),
                                ///< e.g. "line 14: expected 2 floats, got 'NACA'"
                                ///< or "only 12 points (minimum 20)".
};

/// @brief Generate a NACA 4-digit section (plan 5.1).
///
/// Standard equations: max camber m (1st digit, % chord), camber position p
/// (2nd digit, tenths), thickness t (last two digits, % chord). Uses the
/// CLOSED trailing-edge thickness polynomial (last coefficient -0.1036) so
/// the voxelizer's TE-gap closure has a watertight outline to work with.
/// Cosine point spacing clusters points at LE/TE (~`pointsPerSurface` each side).
/// @param digits           The 4-digit code as text, e.g. "2412", "0012".
/// @param pointsPerSurface Points per surface; default 200 per plan 5.1.
/// @return ok=false with a reason if `digits` is not exactly 4 digits.
AirfoilLoadResult generateNACA4(const std::string& digits, int pointsPerSurface = 200);

/// @brief Load a UIUC-style .dat file, auto-detecting Selig vs Lednicer
/// (Lednicer iff line 2 parses as two values > 1 — plan 5.2).
///
/// Defensive by design: tolerates CRLF, stray blank lines, duplicate LE
/// points, and TE gaps; normalizes chord to [0,1] with LE at origin; rejects
/// (with a stated reason) files with < 20 points, non-numeric coordinate
/// lines, or degenerate geometry.
/// @param path Path to the .dat file.
/// @return Result with the parsed section or a rejection reason.
AirfoilLoadResult loadAirfoilDat(const std::filesystem::path& path);

/// @brief Scan a directory (recursively) for .dat files and return their
/// paths paired with the display name from each file's name line. Used by the
/// UI dropdown at startup and on "refresh" (plan 5.2). Unparseable files are
/// still listed (so the user can select one and see its rejection reason).
struct AirfoilCatalogEntry {
    std::filesystem::path path; ///< Full path to the .dat file.
    std::string displayName;    ///< "<filename> — <name line>".
};
std::vector<AirfoilCatalogEntry> scanAirfoilDirectory(
    const std::filesystem::path& directory);

/// @brief Query the surface frame at chordwise station x/c (plan 6.1: "the
/// vane root follows the surface"). Interpolates along the polygon between
/// the bracketing outline points.
/// @param airfoil Normalized section geometry.
/// @param x_c     Chordwise station in [0,1].
/// @param upper   True for the suction (upper) surface — the only surface VGs
///                mount on in v1 — false for the pressure side.
/// @return Frame with valid=false on out-of-range input.
SurfaceFrame surfaceFrameAt(const AirfoilGeometry& airfoil, float x_c,
                            bool upper = true);

} // namespace foilcfd
