// Device half of the iMEM slip-velocity wall function (Asmuth et al., Phys.
// Fluids 33:105111): once per stepN batch, a small kernel walks the wall-cell
// list (built host-side in wall_model.h), samples the tangential velocity a
// few cells off the wall, Newton-solves the Reichardt law of the wall for the
// friction velocity, and writes the slip velocity u_w that makes the bounce-
// back links of each wall cell exchange exactly the modeled wall shear
// stress. The stream-collide and force-reduction kernels consume the result
// through WallSlipView (lbm_core.cuh).
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

#include "lbm_core.cuh"

namespace foilcfd {

// ===========================================================================
// Law-of-the-wall constants (Reichardt's all-y+ profile):
//   u+ = ln(1 + kappa y+)/kappa + C (1 - e^(-y+/11) - (y+/11) e^(-y+/3))
// Smooth from the viscous sublayer through the log layer, which matters here
// because first-cell y+ spans ~1 (leading edge) to ~300 (mid-chord coarse).
// ===========================================================================

/// von Karman constant.
inline constexpr float kWallModelKappa = 0.41f;
/// Reichardt outer coefficient; yields log-law intercept B ~ 5.6.
inline constexpr float kWallModelReichardtC = 7.8f;

/// EMA factor for the friction-velocity update: u_tau is a slow quantity and
/// the u_tau <-> u_w loop must not ring, so each per-batch solve only moves
/// the stored value a quarter of the way to the new solution.
inline constexpr float kWallModelEmaAlpha = 0.25f;

/// Low-y+ fade band: when the SAMPLED y+ sits below kWallModelFadeY0 the
/// viscous sublayer is genuinely resolved and the plain bounce-back stress
/// is already the truth — a wall function there can only disagree with the
/// resolved flow by its own discretization error, so the slip fades to zero.
/// Full model strength returns by kWallModelFadeY1 (buffer layer and up).
inline constexpr float kWallModelFadeY0 = 3.0f;
inline constexpr float kWallModelFadeY1 = 6.0f;

/// @brief Device view of the wall-cell list (mirrors wall_model.h's SoA
/// layout, uploaded once per geometry change). uTau is the only mutable
/// member: it persists across updates as the EMA state.
struct WallCellListView {
    const long long*     cellIdx    = nullptr; ///< Unpadded wall-cell index.
    const long long*     sampleIdx  = nullptr; ///< Unpadded sample-cell index.
    const float*         normalX    = nullptr; ///< Unit fluid-pointing normal.
    const float*         normalY    = nullptr;
    const float*         normalZ    = nullptr;
    const float*         sampleDist = nullptr; ///< y_s in cells of this level.
    const std::uint32_t* linkMask   = nullptr; ///< Bit q: pull link q is solid.
    float*               uTau       = nullptr; ///< Persistent friction velocity.
    int                  count      = 0;       ///< Listed wall cells.
};

/// @brief Per-update tunables, all in the LEVEL's lattice units.
struct WallModelParams {
    /// Molecular lattice viscosity of the level INCLUDING the startup ramp:
    /// the law of the wall must see the viscosity the flow actually runs at
    /// or early u_tau estimates land in the wrong regime.
    float nuLat = 0.05f;
    /// Tangential speeds below this read as stagnation/separation: u_tau and
    /// u_w drop to zero and the cell degrades to plain bounce-back, exactly
    /// where the equilibrium wall law stops being meaningful anyway.
    /// Typically 1e-3 * u_lat.
    float utCutoff = 1e-4f;
    /// EMA factor (kWallModelEmaAlpha unless experimenting).
    float emaAlpha = kWallModelEmaAlpha;
};

/// @brief Device accumulator for the UI readout, zeroed by the launch
/// wrapper each update. maxYplusBits holds a positive float's bit pattern
/// (monotone under integer atomicMax).
struct WallModelDeviceStats {
    float sumYplus    = 0.0f; ///< Sum of y+ over active cells (mean = /active).
    int   maxYplusBits = 0;   ///< Bit pattern of the max y+ seen.
    int   activeCells = 0;    ///< Cells with a live u_tau this update.
    int   clampedCells = 0;   ///< Cells whose slip hit the safety clamp.
};

/// @brief Run one wall-model update over the listed cells: sample moments
/// from @p lattice's f, solve Reichardt for u_tau (Newton, warm-seeded from
/// the stored value), EMA-blend, then solve the LINEAR iMEM balance for the
/// slip velocity and scatter it into the half-precision slip arrays.
/// Unlisted cells are never written — the solver zero-fills the slip field
/// once at (re)build, which this kernel preserves.
/// @param list    Device wall-cell list view.
/// @param lattice CURRENT source buffer (post-collision populations).
/// @param uwx     Mutable slip-field x array (half bits, cellCount entries).
/// @param uwy     Slip-field y array.
/// @param uwz     Slip-field z array.
/// @param params  Level-specific tunables (viscosity, cutoff, EMA).
/// @param d_stats Device stats block (zeroed here before the launch).
cudaError_t launchWallModelUpdate(WallCellListView list,
                                  DeviceLatticeView lattice,
                                  std::uint16_t* uwx, std::uint16_t* uwy,
                                  std::uint16_t* uwz,
                                  const WallModelParams& params,
                                  WallModelDeviceStats* d_stats,
                                  cudaStream_t stream);

} // namespace foilcfd
