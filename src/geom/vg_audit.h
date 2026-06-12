// Vortex-strength audit (Mission statement: the honesty meter for VG
// placement work). Measures the streamwise circulation actually shed by the
// voxelized vanes — integrated from a crossflow velocity plane a few vane
// heights downstream — and compares it against the Wendt empirical
// correlation (NASA/CR-2001-211144, as implemented by Dudek, AIAA-2005-1003
// Eq. 3). A resolved vane lands near the correlation; an under-resolved one
// sheds measurably weak, and THAT is the number that quietly corrupts
// VG-on/VG-off comparisons if nobody is watching it.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <vector>

#include "../sim/lbm_solver.h"
#include "vg.h"

namespace foilcfd {

// ===========================================================================
// Wendt correlation constants (least-squares fit over 0.13 < h/c < 2.62,
// 0.12 < h/delta < 2.60, M 0.2-0.6 — Wendt 2001 via Dudek 2005):
//   Gamma = k1 * alpha * V_e * c * tanh(k3 * h/delta) / (1 + k2/AR)
//   AR = 8h / (pi c)
// k1 replaces Prandtl's pi and k2 the elliptic-loading 2; the tanh carries
// the boundary layer's retarding influence on the vane's effective loading.
// ===========================================================================

inline constexpr float kWendtK1 = 1.61f;
inline constexpr float kWendtK2 = 0.48f;
inline constexpr float kWendtK3 = 1.41f;

/// Audit-plane placement: this many vane heights downstream of the vane
/// trailing edge. Wendt measured one vane CHORD downstream (vortex fully
/// rolled up); with l = 3h vanes, 3h is the same station.
inline constexpr float kAuditPlaneHeightsDownstream = 3.0f;

/// Wall-normal extent of the integration band, in vane heights: the shed
/// vortex sits near one h off the wall and meanders little this close to
/// the vane; 4h of band catches it with margin while excluding the outer
/// flow's incidental vorticity.
inline constexpr float kAuditBandHeights = 4.0f;

/// Verdict bands on measured/predicted (UI traffic light): above the first
/// the vane is trustworthy, between the two it is marginal, below the
/// second the configuration needs more resolution before its results mean
/// anything.
inline constexpr float kAuditRatioGood     = 0.8f;
inline constexpr float kAuditRatioMarginal = 0.5f;

/// @brief Wendt-correlation circulation in LATTICE units (pass every input
/// in the units of the grid the measurement runs on).
/// @param betaRad        Vane incidence to the local flow [radians].
/// @param ueLat          Local edge velocity [lattice units].
/// @param hCells         Vane height [cells].
/// @param vaneChordCells Vane chord length [cells] (= length_h * h).
/// @param delta99Cells   Local boundary-layer thickness [cells].
/// @return Predicted shed circulation [cells^2 / step]; 0 for bad inputs.
float wendtCirculationLattice(float betaRad, float ueLat, float hCells,
                              float vaneChordCells, float delta99Cells);

/// @brief Signed split of the streamwise-vorticity integral over a band of
/// a crossflow plane: pos collects clockwise, neg counter-clockwise (neg is
/// <= 0). Their difference is total |circulation| content; their sum the
/// net. Units: lattice (cell area = 1).
struct PlaneCirculation {
    float pos = 0.0f;
    float neg = 0.0f;
};

/// @brief Integrate omega_x = dw/dy - dv/dz over the band y in [yMin, yMax]
/// (all z, periodic) of a crossflow plane. Central differences; the rows
/// adjacent to the band edges use the in-band one-sided stencil so solid
/// cells below yMin never enter. Pure host math — unit-tested against a
/// synthetic Lamb-Oseen pair.
/// @param v    Crossflow y-velocity plane, ny*nz floats, index y + ny*z.
/// @param w    Crossflow z-velocity plane, same layout.
/// @param ny   Plane height (lattice ny).
/// @param nz   Plane depth (lattice nz, periodic).
/// @param yMin First row of the band (>= 1).
/// @param yMax Last row of the band (inclusive, <= ny-2).
PlaneCirculation integrateStreamwiseCirculation(const std::vector<float>& v,
                                                const std::vector<float>& w,
                                                int ny, int nz,
                                                int yMin, int yMax);

/// @brief Zone-split variant for the audit's noise floor: one pass over the
/// band that accumulates the VANE-ROW spanwise zone [z0, z1] and the
/// ambient remainder separately. The ambient strips sit at the SAME plane
/// and resolution as the zone, so their per-cell vorticity content is the
/// statistically matched floor estimate — an upstream plane is not (the
/// refinement patch changes what the boundary layer carries, and the
/// leading-edge region carries curvature content of its own).
struct BandCirculation {
    PlaneCirculation zone;    ///< Inside [z0, z1].
    PlaneCirculation ambient; ///< Everything else in the band.
    int zoneCols = 0;         ///< Spanwise columns inside the zone.
    int ambientCols = 0;      ///< Spanwise columns outside it.
};
BandCirculation integrateBandCirculation(const std::vector<float>& v,
                                         const std::vector<float>& w,
                                         int ny, int nz, int yMin, int yMax,
                                         int z0, int z1);

/// @brief One audit verdict for the UI VG panel.
struct VGAuditReadout {
    bool  valid = false;        ///< Measurement and prediction both usable.
    float gammaMeasured = 0.0f; ///< Per-vortex |circulation|, lattice units.
    float gammaPredicted = 0.0f;///< Wendt prediction, lattice units.
    float ratio = 0.0f;         ///< measured / predicted.
    float planeXc = 0.0f;       ///< Chordwise station of the audit plane.
};

/// @brief Run the full audit for one VG entry against the live solver
/// state: download the crossflow plane at the vane row's audit station,
/// integrate the shed circulation per vortex (counter-rotating types split
/// the signed halves; co-rotating types take the dominant sign), subtract
/// the ambient-vorticity floor measured on a plane upstream of the vanes,
/// and compare against the Wendt prediction built from the LIVE local edge
/// velocity and boundary-layer thickness.
/// Costs two small plane downloads — call at UI rate alongside the
/// delta99 guidance refresh, never per frame.
/// @param solver       Stepped solver (coarse macro fields; in the patch
///                     overlap these carry fine-derived moments already).
/// @param vg           The VG entry to audit.
/// @param chordCells   Coarse-grid chord resolution N_c.
/// @param ueLat        Edge velocity at the vane station [lattice], from
///                     the delta99 guidance profile.
/// @param delta99Cells BL thickness at the vane station [cells], same source.
/// @return Readout; .valid false when any ingredient was unusable.
VGAuditReadout auditVGVortexStrength(const LBMSolver& solver,
                                     const VGParams& vg, int chordCells,
                                     float ueLat, float delta99Cells);

} // namespace foilcfd
