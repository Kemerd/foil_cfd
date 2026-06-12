// Two-level grid refinement (ML-LBM, plan M-refine): the patch box type, the
// level-scaling relations, and the host-callable launch wrappers for the
// coarse<->fine coupling kernels — coarse-to-fine interface fill (trilinear +
// temporal interpolation with non-equilibrium rescaling), fine-to-coarse
// restriction (child average with the inverse rescale and an edge blend that
// suppresses seam vorticity), and the macroscopic trilinear upsample used by
// the mesh-sequencing presolve.
//
// Physics contract (runtime factor m in {2, 3, 4}, acoustic scaling):
//   dx_f = dx_c / m          dt_f = dt_c / m          u_lat unchanged
//   nu_lat_f = m * nu_lat_c  ->  tau_f = m*(tau_c - 1/2) + 1/2
// The fine level advances m sub-steps per coarse step. The equilibrium part
// of f is invariant under the level change (same rho, u); the non-equilibrium
// part carries the strain rate and rescales as
//   coarse->fine:  fneq_f = fneq_c * tau_f / (m * tau_c)
//   fine->coarse:  fneq_c = fneq_f * (m * tau_c) / tau_f
// (Dupuis & Chopard 2003 / Filippova & Haenel family; the rescale uses the
// BASE relaxation times — the local Smagorinsky adjustment is intentionally
// excluded, the standard practice for LES-coupled multi-level lattices.)
//
// Both Reynolds numbers are SHARED across levels: the fine tau is derived
// from the coarse tau, so the patch buys m-times spatial resolution (boundary
// layer cells, VG geometry) at the same effective Re — it does not raise Re.
//
// The fine level spans the FULL z extent (m*nz fine cells) and reuses the
// padded-ghost-z layout + refresh kernels of lbm_core unchanged. Its x/y
// faces carry a 2-fine-cell CellFlag::Interface shell written by the fill
// kernel before every fine sub-step; no inlet/outlet/slip flags exist at the
// fine level.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

#include "lbm_core.cuh"

namespace foilcfd {

// ===========================================================================
// Patch geometry.
// ===========================================================================

/// @brief Axis-aligned refinement patch in COARSE cell coordinates. The fine
/// grid covers [x0, x1) x [y0, y1) x [0, nz) at the chosen factor (full span).
struct PatchBox {
    int x0 = 0, y0 = 0;  ///< Inclusive lower corner (coarse cells).
    int x1 = 0, y1 = 0;  ///< Exclusive upper corner (coarse cells).

    int  width()  const { return x1 - x0; }
    int  height() const { return y1 - y0; }
    bool valid()  const { return width() > 8 && height() > 8; }
};

/// Default refinement factor; runtime factors 2..kMaxRefineFactor supported.
inline constexpr int kRefineFactor    = 2;
inline constexpr int kMaxRefineFactor = 4;

/// Interface-shell thickness on the fine level's x/y faces, in FINE cells.
/// Two cells: the outer ring feeds the pull of the inner ring, which feeds
/// the first true fluid ring — a fine fluid cell can never pull from a slot
/// the fill kernel did not just write.
inline constexpr int kInterfaceShellFine = 2;

/// Coarse-cell band adjacent to the patch boundary that the restriction
/// SKIPS: the coarse solution there must stay purely coarse-evolved, because
/// it is the source data for the next interface fill (restricting it would
/// short-circuit fine data back into the fine boundary condition).
inline constexpr int kRestrictBandCoarse = 2;

/// Width (in coarse cells) of the restriction's edge BLEND ramp: the first
/// cells inside the restricted region mix coarse-evolved and fine-restricted
/// populations with a weight ramping 0 -> 1 inward. A hard cutoff leaves a
/// velocity kink at the band edge that the Q-criterion lights up as a
/// spurious vortex sheet standing on the patch faces (observed in the fog
/// view); the ramp spreads the level hand-off over a few cells instead.
inline constexpr int kRestrictBlendCoarse = 3;

/// @brief Fine grid dimensions for a patch over the given coarse grid.
inline GridDims fineDimsFor(const PatchBox& box, const GridDims& coarse,
                            int factor = kRefineFactor) {
    GridDims d;
    d.nx = factor * box.width();
    d.ny = factor * box.height();
    d.nz = factor * coarse.nz;
    return d;
}

/// @brief Relaxation time at the fine level for a given coarse tau (acoustic
/// scaling): same physical viscosity on both levels.
inline float fineTauFor(float tauCoarse, int factor = kRefineFactor) {
    return static_cast<float>(factor) * (tauCoarse - 0.5f) + 0.5f;
}

// ===========================================================================
// Launch wrappers (implemented in lbm_refine.cu). Same conventions as
// lbm_core.cuh: explicit stream, cudaGetLastError() returned.
// ===========================================================================

/// @brief Coarse-to-fine fill: write interpolated, rescaled populations into
/// the fine buffer's Interface shell (or every non-solid cell in full-volume
/// mode). For each target fine cell, the 19 populations are trilinearly
/// interpolated from the 8 surrounding coarse cells — blended between the two
/// coarse time levels by @p timeWeight — then split into feq + fneq via the
/// interpolated moments and the fneq part is rescaled by tau_f/(m*tau_c).
///
/// The coarse ping-pong pair provides both time levels for free: pass the
/// pre-step buffer as @p coarseT0 and the post-step buffer as @p coarseT1F.
/// z interpolation rides the coarse ghost planes (periodic span).
/// @param coarseT0   Coarse lattice view at time t (padded base + dims/flags).
/// @param coarseT1F  Coarse f buffer at time t+1 (padded base; same dims).
/// @param fine       Fine lattice view; the buffer about to be PULLED from.
/// @param box        Patch box in coarse cells.
/// @param factor     Refinement factor m (2..kMaxRefineFactor).
/// @param timeWeight k/m before fine sub-step k (0, 1/m, ..., (m-1)/m).
/// @param tauCoarse  Coarse base relaxation time this step (ramped).
/// @param tauFine    Fine base relaxation time (fineTauFor(tauCoarse, m)).
/// @param fullVolume False: Interface shell only (the per-step path).
///                   True: every Fluid + Interface fine cell (seeding after
///                   init/reset/presolve; call once per fine buffer).
cudaError_t launchCoarseToFineFill(DeviceLatticeView coarseT0,
                                   const FPop* coarseT1F,
                                   DeviceLatticeView fine,
                                   PatchBox box, int factor, float timeWeight,
                                   float tauCoarse, float tauFine,
                                   bool fullVolume, cudaStream_t stream);

/// @brief Fine-to-coarse restriction: for every coarse Fluid cell inside the
/// patch (excluding the kRestrictBandCoarse band), average the populations of
/// its m^3 fine children (skipping Solid children near stair-step walls),
/// rescale fneq by (m*tau_c)/tau_f, BLEND with the existing coarse value over
/// the kRestrictBlendCoarse edge ramp, and overwrite the coarse post-collision
/// buffer. Optionally writes the coarse macroscopic arrays in the overlap so
/// rendering and the delta99 extraction see fine-derived data with zero
/// changes to any consumer.
/// @param fine      Fine lattice view (post-collision, current src).
/// @param coarse    Coarse lattice view to overwrite (the t+1 buffer).
/// @param box       Patch box in coarse cells.
/// @param factor    Refinement factor m (2..kMaxRefineFactor).
/// @param tauCoarse Coarse base relaxation time this step.
/// @param tauFine   Fine base relaxation time.
/// @param macroRho  Optional coarse density array (unpadded); may be null.
/// @param macroU    Optional coarse x-velocity output, as macroRho.
/// @param macroV    Optional y-velocity output.
/// @param macroW    Optional z-velocity output.
cudaError_t launchFineToCoarseRestrict(DeviceLatticeView fine,
                                       DeviceLatticeView coarse,
                                       PatchBox box, int factor,
                                       float tauCoarse, float tauFine,
                                       float* macroRho, float* macroU,
                                       float* macroV, float* macroW,
                                       cudaStream_t stream);

/// @brief Trilinear upsample of macroscopic fields from a small grid onto a
/// larger one (the mesh-sequencing presolve, plan M-refine part 2). x/y
/// sampling is clamped at the borders; z wraps periodically. Arrays are
/// unpadded, cellCount() floats each, on the SAME device.
/// @param srcRho/srcU/srcV/srcW Source arrays (coarse presolve grid).
/// @param srcDims               Source grid dimensions.
/// @param dstRho/dstU/dstV/dstW Destination arrays (full-resolution grid).
/// @param dstDims               Destination grid dimensions.
cudaError_t launchUpsampleMacro(const float* srcRho, const float* srcU,
                                const float* srcV, const float* srcW,
                                GridDims srcDims,
                                float* dstRho, float* dstU,
                                float* dstV, float* dstW,
                                GridDims dstDims, cudaStream_t stream);

} // namespace foilcfd
