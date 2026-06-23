// Vortex generators (plan section 6): the parametric VG model (VGParams kept
// verbatim from the plan), surface placement frames, voxelization into the
// flag field, the under-resolution guard, and the Lin-2002 placement guidance
// helpers that power the VG overlay (Mission statement).
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "../sim/lbm_core.cuh"
#include "airfoil.h"
#include "stl.h"        // StlMesh + StlAxisPreset for custom-STL VGs
#include "voxelizer.h"

namespace foilcfd {

// ===========================================================================
// Parametric model — struct kept VERBATIM from plan section 6.1. Field names
// are API: the UI panel, voxelizer, and cache flow all bind to them.
// ===========================================================================

// VG configuration is two INDEPENDENT axes, so any arrangement can use any
// blade shape (a counter-rotating pair of delta blades, a co-rotating array of
// your custom STL, etc.):
//
//   VGType    = the ARRANGEMENT (how units are placed across the span).
//   VGProfile = the BLADE SHAPE (the outline/section of one blade).
//
// VGType::Ramp and VGType::CustomStl are LEGACY values kept so older configs
// still load; effectiveArrangement()/effectiveProfile() below migrate them to
// the (arrangement, profile) pair (Ramp -> Single + Wedge, CustomStl -> Single +
// CustomStl). New code sets `type` to a real arrangement and `profile` to the
// shape.
enum class VGType { SingleVane, CounterRotatingPair, CoRotatingArray,
                    Ramp,      // legacy: migrates to Single + Wedge
                    CustomStl }; // legacy: migrates to Single + CustomStl

// Blade outline/section. The first five are parametric plates whose top-height
// (and, for AirfoilSection, thickness) follow a formula along the blade length;
// Wedge is the right-triangular ramp prism; CustomStl swaps the whole blade for
// a user mesh. Placement, arraying, yaw, AoA and q-LIBB are identical across
// profiles — only the per-slice top-height / thickness differ.
enum class VGProfile { Rectangle, Delta, Trapezoid, Parabolic, AirfoilSection,
                       Wedge, CustomStl };

struct VGParams {
  VGType type;
  float x_c;        // chordwise station, 0..1 (typical 0.05–0.30)
  float height_c;   // device height / chord (typical 0.005–0.02)
  float length_h;   // vane length in heights (default 3.0)
  float beta_deg;   // incidence to local flow (default 16, range ±30)
  float pitch_c;    // spanwise spacing between units (arrays/pairs)
  float gap_h;      // intra-pair gap in heights (pairs; default 2.5)
  int   count;      // number of units across the span
  bool  commonFlowDown; // pair orientation
  bool  enabled = true; // when false: voxelized and rendered as ghost only

  // ---- blade shape (orthogonal to `type`) --------------------------------
  VGProfile profile = VGProfile::Rectangle; // blade outline/section
  float taper       = 0.5f;  // Trapezoid only: flat-top fraction of the length
                             // (0 -> pure delta point, 1 -> rectangle).
  float thicknessRatio = 0.12f; // AirfoilSection only: NACA max thickness / chord
                                // of the blade's streamlined cross-section.

  // ---- CustomStl fields (used when profile == CustomStl) -----------------
  // Kept as a copyable POD: the heavy triangle soup lives in App::vgMeshes,
  // and stlMeshId indexes into it, so VGParams stays cheap to copy (the VG
  // list, the sweep cases, and the warm-restart cache all copy it by value).
  int   stlMeshId  = -1;   // index into App::vgMeshes (-1 = none configured yet)
  int   stlRotSteps = 0;   // extra 0/90/180/270 deg yaw about the mount normal
  bool  stlFlip     = false; // flip upside-down (negate the wall-normal axis)
  StlAxisPreset stlAxis = StlAxisPreset::XYZ; // import-time axis remap
};

/// @brief The real arrangement of a VG entry, migrating the legacy Ramp /
/// CustomStl type values to SingleVane (their shape lives in the profile).
inline VGType effectiveArrangement(const VGParams& vg) {
    return (vg.type == VGType::Ramp || vg.type == VGType::CustomStl)
               ? VGType::SingleVane
               : vg.type;
}

/// @brief The real blade profile of a VG entry: an explicit non-Rectangle
/// `profile` wins; otherwise the legacy type values pick the shape (Ramp ->
/// Wedge, CustomStl -> CustomStl) so old configs render correctly. A modern
/// entry leaves `type` an arrangement and sets `profile` directly.
inline VGProfile effectiveProfile(const VGParams& vg) {
    if (vg.profile != VGProfile::Rectangle) return vg.profile;
    if (vg.type == VGType::Ramp)      return VGProfile::Wedge;
    if (vg.type == VGType::CustomStl) return VGProfile::CustomStl;
    return VGProfile::Rectangle;
}

/// @brief Sensible starting values matching the plan's "typical" annotations
/// and the Strausak flight-proven recipe defaults (x/c 0.07, beta ~15-16 deg,
/// l = 3h, counter-rotating pairs).
inline VGParams defaultVGParams() {
    VGParams p{};
    p.type           = VGType::CounterRotatingPair;
    p.x_c            = 0.07f;
    p.height_c       = 0.01f;
    p.length_h       = 3.0f;
    p.beta_deg       = 16.0f;
    p.pitch_c        = 0.06f;
    p.gap_h          = 2.5f;
    p.count          = 4;
    p.commonFlowDown = true;
    return p;
}

/// @brief The CFD-tuned champion config from the FoilCFD sibling study
/// (L:\Dev\glasair_vg_sim): a 6 mm DELTA vane, counter-rotating pairs, 10 deg
/// TOE-OUT, on the NASA/Langley LS(1)-0413 at the Glasair III aileron mid-station
/// (chord 0.9022 m). It was the study's double winner — highest steady Clmax
/// (1.709 @ a=18 vs clean 1.443 @ a=15) AND the lowest cruise drag tax (+6.2%).
/// Physical->chord-fraction mapping (chord = 902.2 mm): height 6/902.2 = 0.00665;
/// pair pitch 70/902.2 = 0.0776; intra-pair vane spacing P/2 = 35 mm = 5.83 h.
/// commonFlowDown=false encodes the measured toe-OUT optimum (splayed leading
/// edges, beats toe-in here). NOTE: this is a SMALL vane (h/c 0.0067) — it needs
/// the VG-resolution cascade to resolve honestly, exactly the path that exists.
inline VGParams defaultGlasairVG() {
    VGParams p{};
    p.type           = VGType::CounterRotatingPair;
    p.profile        = VGProfile::Delta;   // 6 mm delta (study winner)
    p.x_c            = 0.07f;               // front tips at 7% chord
    p.height_c       = 0.00665f;            // 6 mm / 902.2 mm
    p.length_h       = 3.0f;                // l = 3h (~18 mm)
    p.beta_deg       = 10.0f;               // 10 deg toe-OUT (measured optimum)
    p.pitch_c        = 0.0776f;             // 70 mm pair-to-pair / chord
    p.gap_h          = 5.83f;               // 35 mm intra-pair (P/2) / 6 mm
    p.count          = 4;
    p.commonFlowDown = false;               // toe-out (splayed LEs)
    return p;
}

// ===========================================================================
// Placement and voxelization
// ===========================================================================

/// @brief Compute the placement frame for a VG entry: the airfoil surface
/// point/tangent/normal at x_c on the UPPER (suction) surface — VGs mount
/// there only in v1 — expressed in normalized chord coordinates. The frame is
/// pre-AoA: voxelizeVGs applies the same quarter-chord rotation the airfoil
/// voxelizer uses, so vanes stay glued to the surface at any AoA.
/// @param airfoil The section the VG sits on.
/// @param params  VG entry (only x_c is consumed here).
/// @return SurfaceFrame with valid=false for out-of-range stations.
SurfaceFrame vgPlacementFrame(const AirfoilGeometry& airfoil, const VGParams& params);

/// @brief Voxelize one VG entry into the flag field, OR'ing SOLID over the
/// existing clean-foil mask (plan 6.2: airfoil mask OR VG voxels).
///
/// Geometry: vanes are thin boxes 1-2 cells thick, height_c*N_c tall along
/// the local surface normal, length_h heights long, yawed +/-beta_deg about
/// the local normal; pairs place two mirrored vanes gap_h heights apart
/// (commonFlowDown chooses orientation), arrays repeat every pitch_c along z,
/// centered on the span. Ramp type stamps a right-triangular wedge prism in
/// the same frame.
/// @param vg      VG entry to stamp.
/// @param airfoil Section providing the surface frame.
/// @param aoa_deg Current angle of attack (vanes rotate with the foil).
/// @param layout  Foil placement/scale inside the grid.
/// @param flags   Host flag field, modified in place.
/// @param slabsOut Optional: appends one VaneSlab OBB per stamped vane (the
///                 analytic side-face descriptor the q-LIBB build ray-intersects
///                 — see voxelizer.h). nullptr (default) keeps the legacy
///                 stamp-only behavior, so existing call sites are untouched.
///                 CustomStl vanes emit NO slab (q-LIBB falls back to plain
///                 bounce-back for them — a single OBB can't model a mesh).
/// @param unitVgMeshes Optional: the unit-normalized VG meshes (App::vgMeshes)
///                 a CustomStl entry's stlMeshId indexes into. nullptr (default)
///                 means parametric-only; a CustomStl entry with an unresolved
///                 id is skipped (the caller logs it once).
void voxelizeVG(const VGParams& vg, const AirfoilGeometry& airfoil, float aoa_deg,
                const DomainLayout& layout, std::vector<std::uint8_t>& flags,
                std::vector<VaneSlab>* slabsOut = nullptr,
                const std::vector<StlMesh>* unitVgMeshes = nullptr);

/// @brief Stamp a whole VG configuration: copy of the cached clean-foil flags
/// + every entry voxelized + one TE-closure pass at the end. This is the
/// function the UI edit flow calls on slider release (plan 6.2/9.2).
/// @param vgs           All VG entries currently configured.
/// @param airfoil       Section geometry.
/// @param aoa_deg       Angle of attack.
/// @param layout        Domain layout.
/// @param cleanFoilFlags The cached clean-foil flag field (NOT modified).
/// @param slabsOut Optional: appends the analytic VaneSlab OBBs of every stamped
///                 vane (for q-LIBB). nullptr keeps legacy behavior.
/// @param unitVgMeshes Optional: unit-normalized meshes for CustomStl entries
///                 (App::vgMeshes). nullptr = parametric-only.
/// @return New flag field = clean foil mask OR all VG voxels.
std::vector<std::uint8_t> buildFlagsWithVGs(
    const std::vector<VGParams>& vgs, const AirfoilGeometry& airfoil,
    float aoa_deg, const DomainLayout& layout,
    const std::vector<std::uint8_t>& cleanFoilFlags,
    std::vector<VaneSlab>* slabsOut = nullptr,
    const std::vector<StlMesh>* unitVgMeshes = nullptr);

// ===========================================================================
// Interpolated bounce-back (q-LIBB) link list. For every fluid cell whose pull
// of direction q lands on VANE solid, we store the exact fractional cut q in
// [0,1] of the true vane side-face along that link (ray-OBB intersection
// against the persisted VaneSlabs), so the hot kernel can place the wall at the
// real sub-cell position (Bouzidi 2001) instead of snapping to q=0.5. SoA so
// the solver cudaMemcpy's each array straight into its device mirror.
// ===========================================================================

/// Clamp on the stored cut fraction: the Bouzidi q>=1/2 branch divides by 2q,
/// and q->0 extrapolation is noisy, so links resolving outside [kQMin, 1] snap
/// to plain half-way bounce-back (fraction 0.5) rather than risk instability.
inline constexpr float kQLibbMin = 0.05f;

/// @brief One interpolated-bounce-back link: a (cell, direction) pair whose
/// pull crosses a vane side face, with the analytic cut fraction.
struct QLink {
    long long cellIdx; ///< UNPADDED fluid-cell index x + nx*(y + ny*z).
    std::uint8_t dir;  ///< Lattice direction q (1..18) that hits the vane.
    std::uint8_t ffFluid; ///< 1 if the second fluid node x-2c_q is Fluid (the
                          ///< q<1/2 two-node branch is usable), else 0.
    float q;           ///< Cut fraction in [kQLibbMin, 1], from the fluid node.
};

/// @brief SoA q-link list, one entry per cut link (built parallel to the wall
/// list but for the COMPLEMENTARY set: vane links, which the wall list drops).
struct QLinkList {
    std::vector<long long>    cellIdx;
    std::vector<std::uint8_t> dir;
    std::vector<std::uint8_t> ffFluid;
    std::vector<float>        q;
    std::size_t size() const { return cellIdx.size(); }
};

/// @brief Build the q-LIBB link list from the live flags + the analytic vane
/// slabs. For each fluid cell, each pull direction q whose upstream cell is
/// VANE solid (Solid in @p activeFlags, Fluid in @p cleanFlags) is ray-cast
/// against the nearest VaneSlab to recover the true cut fraction. Links whose
/// fraction falls outside [kQLibbMin, 1], or that would read an Interface shell
/// cell, are dropped (the kernel keeps plain bounce-back there).
/// @param dims        Grid dimensions of the flag fields.
/// @param activeFlags Live flag field (foil + VGs).
/// @param cleanFlags  Clean-foil flag field (VG-free) — vane provenance test.
/// @param slabs       Persisted vane OBBs (voxelizeVG/buildFlagsWithVGs output).
/// @return The q-link list; empty when there are no resolved vane links.
QLinkList buildVaneQLinks(const GridDims& dims,
                          const std::vector<std::uint8_t>& activeFlags,
                          const std::vector<std::uint8_t>& cleanFlags,
                          const std::vector<VaneSlab>& slabs);

/// @brief Densify a QLinkList into the per-cell device-field layout the hot
/// kernel reads (QLinkView): @p qFrac sized ncells*kQ (byte b = round(q*255),
/// 0 = no q-link), @p ffMask sized ncells (bit q set when the q<1/2 two-node
/// branch is usable). Both are zero-initialized, so cells/links without a stored
/// q decode to plain half-way bounce-back. Keeps the sim layer geom-free: the
/// app builds + densifies, the solver just uploads the raw arrays.
/// @param links   The sparse q-link list (buildVaneQLinks output).
/// @param ncells  Cell count of the level (dims.cellCount()).
/// @param qFrac   Output, resized to ncells*kQ.
/// @param ffMask  Output, resized to ncells.
void densifyQLinks(const QLinkList& links, long long ncells,
                   std::vector<std::uint8_t>& qFrac,
                   std::vector<std::uint32_t>& ffMask);

// ===========================================================================
// Resolution guard (plan 6.1): a vane shorter than ~8 cells is voxel noise,
// not a vortex generator. The UI warns rather than silently rendering junk.
// ===========================================================================

/// Minimum resolved vane height in cells before the under-resolution warning.
/// This is the FLOOR the honesty meter still warns below; the user-facing
/// "VG height target" (RefinementUIParams::vgTargetCells) drives the auto
/// refinement separately and can ask for more.
inline constexpr int kMinVGHeightCells = 8;

/// Ceiling the VG-resolution target can drive the cascade to. The fine patch
/// itself caps at kMaxRefineFactor (4x); the nested VG box doubles that to an
/// effective 8x over the coarse grid, which is the most resolution a vane can
/// usefully be given before the patch cost (m^4 time / m^3 VRAM, even on the
/// tiny nested box) stops paying off. Targets that would need more than this
/// keep the honest under-resolution warning instead of refining further.
inline constexpr int kMaxVGTargetFactor = 8;

/// @brief Height of the vane in lattice cells at the given chord resolution.
inline float vgHeightCells(const VGParams& vg, int chordCells) {
    return vg.height_c * static_cast<float>(chordCells);
}

/// @brief True when the VG is under-resolved at this grid (height_c * N_c <
/// kMinVGHeightCells). UI shows: "VG under-resolved — increase chord
/// resolution or VG size".
inline bool vgUnderResolved(const VGParams& vg, int chordCells) {
    return vgHeightCells(vg, chordCells) < static_cast<float>(kMinVGHeightCells);
}

/// @brief Smallest refinement factor that lifts EVERY configured vane to at
/// least @p targetCells of resolved height (vortex circulation is what a vane
/// sheds, and an under-resolved vane sheds it weak — the patch is the cheap fix
/// because it multiplies resolution only around the geometry). Returns 1 when
/// no patch is needed (no VGs, or all tall enough at the base grid); never
/// exceeds @p cap — a vane that stays short even at the cap needs a
/// chord-resolution or height change instead, which the under-resolution
/// warning continues to say.
///
/// The two trailing parameters generalize the old fixed behaviour: by default
/// the target is kMinVGHeightCells and the cap is kMaxRefineFactor, so every
/// legacy call site is unchanged. The VG-resolution slider passes the user's
/// target and the larger kMaxVGTargetFactor cap so the high-res zone actually
/// grows to MEET the requested cell count instead of stalling at 4x.
/// @param vgs        Configured VG entries.
/// @param chordCells BASE-grid chord resolution N_c.
/// @param targetCells Desired resolved vane height in cells (>= 1).
/// @param cap        Maximum factor this may return.
inline int recommendedRefineFactorForVGs(const std::vector<VGParams>& vgs,
                                         int chordCells,
                                         int targetCells = kMinVGHeightCells,
                                         int cap = kMaxRefineFactor) {
    int rec = 1;
    const float target = static_cast<float>(std::max(1, targetCells));
    for (const VGParams& vg : vgs) {
        if (!vg.enabled) continue; // disabled VGs don't drive refinement
        const float h = vgHeightCells(vg, chordCells);
        if (h <= 0.0f) continue;
        rec = std::max(rec, static_cast<int>(std::ceil(target / h)));
    }
    return std::min(rec, std::max(1, cap));
}

// ===========================================================================
// Lin-2002 placement guidance (Mission statement). Source: Lin, J.C., "Review
// of research on low-profile vortex generators to control boundary-layer
// separation", Prog. Aerospace Sci. 38 (2002):
//   - device height h = 0.1 .. 1.0 * delta99 (local BL thickness)
//   - place devices 5 .. 10 device-heights UPSTREAM of separation onset
// These helpers turn the live delta99 profile (LBMSolver::extractSuctionDelta99)
// and detected separation point into the recommendation bands the overlay draws.
// ===========================================================================

/// Lin 2002 height band, as fractions of local delta99.
inline constexpr float kLinHeightMinDelta99 = 0.1f;
inline constexpr float kLinHeightMaxDelta99 = 1.0f;
/// Lin 2002 "low-profile" sweet spot within that band: devices of 0.2-0.5
/// delta99 achieved most of the separation control at a fraction of the
/// device drag — the overlay highlights this inner band (docs/CITATIONS.md).
inline constexpr float kLinSweetSpotMinDelta99 = 0.2f;
inline constexpr float kLinSweetSpotMaxDelta99 = 0.5f;
/// Lin 2002 upstream placement distance band, in device heights.
inline constexpr float kLinUpstreamMinHeights = 5.0f;
inline constexpr float kLinUpstreamMaxHeights = 10.0f;

/// @brief A recommended [min,max] band in chord units, plus validity.
struct GuidanceBand {
    float minVal = 0.0f;
    float maxVal = 0.0f;
    bool  valid  = false; ///< False when inputs were unusable (e.g. attached
                          ///< flow: no separation point to anchor a station band).
};

/// @brief Recommended VG height band (h/c) at a station, from the measured
/// boundary-layer thickness there: h in [0.1, 1.0] * delta99 (Lin 2002).
/// @param delta99_c Boundary-layer thickness at the station, chord units
///                  (Delta99Sample::delta99_c). Non-positive -> invalid band.
/// @return Height band in h/c units, directly comparable to VGParams::height_c.
GuidanceBand recommendedHeightBand(float delta99_c);

/// @brief The low-profile sweet spot inside the full Lin band: h in
/// [0.2, 0.5] * delta99 — most of the separation-control benefit at a
/// fraction of the parasitic drag (Lin 2002). Drawn as the highlighted inner
/// band of the height overlay.
/// @param delta99_c Boundary-layer thickness at the station, chord units.
/// @return Sweet-spot band in h/c units; invalid when delta99_c <= 0.
GuidanceBand recommendedSweetSpotHeightBand(float delta99_c);

/// @brief Recommended chordwise station band for VG placement: 5-10 device
/// heights upstream of the detected separation onset (Lin 2002).
/// @param separationXc Separation onset station x/c (from
///                     LBMSolver::separationOnsetXc()); < 0 means attached
///                     flow and yields an invalid band.
/// @param height_c     Device height h/c the user has configured (distances
///                     are measured in this h). Non-positive -> invalid.
/// @return Station band [x/c_min, x/c_max], clamped into [0, separationXc].
GuidanceBand recommendedStationBand(float separationXc, float height_c);

} // namespace foilcfd
