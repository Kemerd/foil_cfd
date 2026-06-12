// D3Q19 lattice constants (FROZEN ordering), cell flags, device field views,
// and the host-callable launch wrappers for every solver kernel: init,
// fused stream-collide (TRT + Smagorinsky), NaN watchdog, momentum-exchange
// force reduction, and the particle-advection entry point (plan section 4.1).
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector_types.h> // float4 for the particle-buffer declarations

namespace foilcfd {

// ===========================================================================
// D3Q19 lattice — Krueger et al. ("The Lattice Boltzmann Method", Springer
// 2017, Table 3.5) ordering. THIS ORDERING IS FROZEN: VRAM and disk snapshots
// (snapshot.h) store raw f arrays indexed by these q values, so reordering
// silently corrupts every cached state. Never change it.
//
//   q  :  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
//   cx :  0  +1  -1   0   0   0   0  +1  -1  +1  -1   0   0  +1  -1  +1  -1   0   0
//   cy :  0   0   0  +1  -1   0   0  +1  -1   0   0  +1  -1  -1  +1   0   0  +1  -1
//   cz :  0   0   0   0   0  +1  -1   0   0  +1  -1  +1  -1   0   0  -1  +1  -1  +1
//   w  : 1/3, 1/18 x6 (q=1..6), 1/36 x12 (q=7..18)
//
// Opposite pairs (needed for bounce-back and TRT symmetric/antisymmetric
// decomposition): (1,2) (3,4) (5,6) (7,8) (9,10) (11,12) (13,14) (15,16) (17,18).
// ===========================================================================

/// Number of discrete velocities in the D3Q19 lattice.
inline constexpr int kQ = 19;

/// Lattice velocity x-components, indexed by q (frozen Krueger ordering).
inline constexpr int kCx[kQ] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0};
/// Lattice velocity y-components, indexed by q.
inline constexpr int kCy[kQ] = {0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1};
/// Lattice velocity z-components, indexed by q.
inline constexpr int kCz[kQ] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1};

/// Index of the opposite direction for each q: kCx[kOpp[q]] == -kCx[q], etc.
/// Note the pair layout: opposites are consecutive (1,2)(3,4)...(17,18), which
/// the TRT collision exploits to walk pairs as (a, a+1) for odd a.
inline constexpr int kOpp[kQ] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};

/// Index of the y-mirrored direction for each q: (cx, -cy, cz). Used by the
/// free-slip top/bottom boundary: a population arriving at cell C with
/// velocity c_q after specular reflection off a y-normal wall must have left
/// cell (x - cx, y, z - cz) travelling with the mirrored direction — the y
/// half-steps before/after the bounce cancel, so only the tangential (x,z)
/// displacement survives. Derived from the frozen table above; verify any
/// edit against kCx/kCy/kCz by hand.
inline constexpr int kMirY[kQ] = {0, 1, 2, 4, 3, 5, 6, 13, 14, 9, 10, 18, 17, 7, 8, 15, 16, 12, 11};

/// Index of the z-mirrored direction for each q: (cx, cy, -cz). Used by the
/// free-slip z side walls in STL mode (plan 7.4: spanwise-periodic BC may be
/// physically wrong for a full 3D object, so the import modal can switch the
/// z faces to specular walls). Same derivation as kMirY with the roles of y
/// and z swapped: the reflected population left cell (x - cx, y - cy, z) with
/// the mirrored direction — the z half-steps around the bounce cancel.
/// Derived from the frozen table above; verify any edit against kCx/kCy/kCz.
inline constexpr int kMirZ[kQ] = {0, 1, 2, 3, 4, 6, 5, 7, 8, 15, 16, 17, 18, 13, 14, 9, 10, 11, 12};

/// Quadrature weights, indexed by q.
inline constexpr float kW[kQ] = {
    1.0f / 3.0f,
    1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f};

/// Lattice speed of sound squared: cs^2 = 1/3 for D3Q19.
inline constexpr float kCs2 = 1.0f / 3.0f;

// ===========================================================================
// Population storage type + accessors (plan section 11, Lehmann PRE 106
// 015308): the solver is memory-bandwidth bound, so FP16 *storage* with FP32
// *compute* is the single biggest future win (~2x). Every kernel goes through
// load_f()/store_f(); switching to half precision later is this typedef plus
// conversion code in the two accessors — no kernel rewrites.
// ===========================================================================

/// Storage type of one f population in device memory. FP32 today; the FP16
/// upgrade swaps this to __half (or a custom posit-style 16-bit pack) and
/// adjusts only load_f/store_f below.
using FPop = float;

#if defined(__CUDACC__)
/// @brief Load one population from storage, widened to FP32 for compute.
__device__ __forceinline__ float load_f(const FPop* __restrict__ f, long long idx) {
    return f[idx]; // FP32 storage: identity. FP16 later: __half2float(f[idx]).
}

/// @brief Store one FP32 compute value back to population storage.
__device__ __forceinline__ void store_f(FPop* __restrict__ f, long long idx, float v) {
    f[idx] = v; // FP32 storage: identity. FP16 later: __float2half_rn(v).
}
#endif // __CUDACC__

// ===========================================================================
// Cell flags (plan section 4.1). uint8 per cell; boundaries are handled by
// flag lookup on the *neighbor* during the pull, never per-q branching.
// ===========================================================================

/// @brief Per-cell type flag. Values are stored in snapshots' flag-hash
/// computation, so the numeric assignments are frozen alongside the lattice.
enum class CellFlag : std::uint8_t {
    Fluid      = 0, ///< Ordinary bulk cell: full stream + collide.
    Solid      = 1, ///< Inside foil/VG/STL: half-way bounce-back target.
    Inlet      = 2, ///< x=0 plane: equilibrium Dirichlet at u_in.
    Outlet     = 3, ///< x=nx-1 plane: zero-gradient copy from neighbor column.
    SlipTop    = 4, ///< y=ny-1 plane: free-slip (specular reflection).
    SlipBottom = 5, ///< y=0 plane: free-slip (specular reflection).
    SlipFront  = 6, ///< z=0 plane: free-slip wall (STL mode, plan 7.4).
    SlipBack   = 7, ///< z=nz-1 plane: free-slip wall (STL mode, plan 7.4).
};

// ===========================================================================
// Grid geometry and device-memory views. Views are POD structs of raw device
// pointers — cheap to pass by value into kernels, no ownership semantics.
// ===========================================================================

/// @brief Domain dimensions in cells. Linear cell index = x + nx*(y + ny*z)
/// (x fastest — matches the SoA coalescing requirement of plan section 11).
struct GridDims {
    int nx = 0;
    int ny = 0;
    int nz = 0;

    /// @brief Total cell count as 64-bit (default grid is 23.6M cells; any
    /// intermediate q*ncells index overflows 32-bit).
    long long cellCount() const {
        return static_cast<long long>(nx) * static_cast<long long>(ny)
             * static_cast<long long>(nz);
    }

    /// @brief Cell count INCLUDING the two spanwise ghost planes (see the
    /// padded-layout note on DeviceLatticeView). f and flag buffers are sized
    /// by this; macroscopic buffers stay at cellCount().
    long long paddedCellCount() const {
        return static_cast<long long>(nx) * static_cast<long long>(ny)
             * (static_cast<long long>(nz) + 2);
    }
};

/// @brief Read-only device view of the macroscopic velocity field, consumed
/// by the particle-advection and slice-rendering kernels. Components are
/// separate planar arrays (SoA) of cellCount() floats in lattice units.
struct DeviceVelocityField {
    const float* u = nullptr; ///< x-velocity per cell (lattice units).
    const float* v = nullptr; ///< y-velocity per cell.
    const float* w = nullptr; ///< z-velocity per cell.
    GridDims     dims;        ///< Grid the pointers are sized for.
};

/// @brief Mutable device view of the full solver state for one buffer of the
/// ping-pong pair. f is SoA: f[q*ncellsPad + paddedCell], cell fastest
/// (plan section 11).
///
/// PADDED SPANWISE LAYOUT (plan 11: "padded ghosts in z are simplest"): the
/// f and flag buffers carry one extra nx*ny ghost plane at EACH z end, so the
/// pull never wraps an index in the hot loop:
///
///   padded index(x, y, z) = x + nx*(y + ny*(z+1)),   z in [-1, nz]
///   buffer length         = dims.paddedCellCount()   (per q for f)
///
/// Ghost plane z=-1 mirrors real plane z=nz-1 and ghost z=nz mirrors real
/// z=0 (periodic span). The solver refreshes the destination buffer's ghost
/// planes after every stream-collide launch (launchRefreshGhostZ); flag
/// ghosts refresh once per flag upload. The real-domain region of a padded
/// buffer is contiguous and starts nx*ny elements in, which is how
/// LBMSolver::deviceFlags() hands external consumers an unpadded-indexable
/// pointer without a second copy of the flag field.
struct DeviceLatticeView {
    FPop*               f     = nullptr; ///< kQ * paddedCellCount() populations (padded base).
    const std::uint8_t* flags = nullptr; ///< paddedCellCount() CellFlag bytes (padded base).
    GridDims            dims;            ///< REAL grid dims (excluding ghosts).
};

// ===========================================================================
// Per-step kernel parameters, grouped so the launch wrappers stay stable as
// the collision model gains terms.
// ===========================================================================

/// @brief Collision/boundary parameters for one fused stream-collide step.
struct StepParams {
    float tau         = 0.6f;   ///< Relaxation time this step (startup ramp varies it).
    float magicLambda = 3.0f / 16.0f; ///< TRT magic parameter (units.h kTRTMagicLambda).
    float smagorinskyCs = 0.12f; ///< LES constant; 0 disables the eddy-viscosity term.
    float uInlet      = 0.08f;  ///< Inlet x-velocity in lattice units (u_lat).
    bool  writeMacro  = true;   ///< Store rho/u/v/w this step (only when rendering needs it).
};

/// @brief Result slot of the momentum-exchange force reduction: total lattice
/// force on all SOLID-adjacent links, accumulated on device, read back by host.
struct DeviceForceAccumulator {
    float* d_force = nullptr; ///< Device buffer of 3 floats: Fx, Fy, Fz (lattice units).
};

// ===========================================================================
// Host-callable launch wrappers (implemented in lbm_core.cu / particles.cu).
// All take an explicit stream; the app runs a single CUDA stream (plan 9.3).
// Every wrapper returns the cudaGetLastError() of its launch so the solver
// can surface failures without sprinkling error checks in the hot loop.
// ===========================================================================

/// @brief Initialize a lattice buffer to equilibrium at uniform inflow
/// (rho = 1, u = (uInlet, 0, 0)); SOLID cells get zero-velocity equilibrium.
/// Used for cold starts and for the compact-snapshot equilibrium re-init.
/// @param lattice Destination buffer view (written in full).
/// @param uInlet  Lattice inflow speed.
/// @param stream  CUDA stream to launch on.
cudaError_t launchInitEquilibrium(DeviceLatticeView lattice, float uInlet,
                                  cudaStream_t stream);

/// @brief Add the spanwise "speck" perturbation to a freshly-initialized
/// field: a tiny deterministic z-dependent velocity ripple that breaks the
/// quasi-2D coherence a uniform init would otherwise lock in (the spanwise-
/// periodic domain has no other symmetry-breaking source).
/// @param lattice   Buffer to perturb in place (post-init, pre-run).
/// @param amplitude Perturbation amplitude in lattice velocity units
///                  (typically 1e-3 * u_lat).
/// @param seed      Deterministic seed so reruns are reproducible.
cudaError_t launchSpanwisePerturbation(DeviceLatticeView lattice, float amplitude,
                                       unsigned int seed, cudaStream_t stream);

/// @brief One fused pull-scheme step: read neighbor post-collision values,
/// compute rho/u, TRT collide with Smagorinsky eddy viscosity, write to dst.
/// Boundary handling is branchless via neighbor-flag lookup (plan 4.1/4.2).
/// @param src       Source buffer (read).
/// @param dst       Destination buffer (written); same dims/flags as src.
/// @param params    Relaxation/boundary parameters for this step.
/// @param macroRho  Optional device array (cellCount floats) for density; may
///                  be nullptr when params.writeMacro is false.
/// @param macroU    Optional x-velocity output, as macroRho.
/// @param macroV    Optional y-velocity output.
/// @param macroW    Optional z-velocity output.
cudaError_t launchStreamCollide(DeviceLatticeView src, DeviceLatticeView dst,
                                const StepParams& params,
                                float* macroRho, float* macroU, float* macroV,
                                float* macroW, cudaStream_t stream);

/// @brief NaN watchdog (plan 4.5): checks a strided sample of cells (~1/4096)
/// in the given buffer and sets *d_nanFlag to nonzero if any sampled f is
/// NaN/Inf. Host polls the flag every ~200 steps and pauses the sim on trip.
/// @param lattice   Buffer to sample.
/// @param d_nanFlag Device int (single value); caller zeroes it beforehand.
cudaError_t launchNaNWatchdog(DeviceLatticeView lattice, int* d_nanFlag,
                              cudaStream_t stream);

/// @brief Momentum-exchange force reduction (plan 4.4): for every fluid cell
/// with a SOLID neighbor, accumulate f_q + f_qbar momentum transfer across
/// the link into acc.d_force (3 floats, zeroed by the wrapper before launch).
/// Block-level reduction + atomics — fine at this scale.
/// @param lattice Post-collision buffer to read populations from.
/// @param acc     Device accumulator receiving (Fx, Fy, Fz) in lattice units.
cudaError_t launchForceReduction(DeviceLatticeView lattice,
                                 DeviceForceAccumulator acc, cudaStream_t stream);

/// @brief Refresh the two spanwise ghost planes of a padded f buffer (all kQ
/// population slices): ghost z=-1 <- real z=nz-1, ghost z=nz <- real z=0.
/// Must run after every kernel that rewrites the buffer (stream-collide, init
/// variants run over the padded domain with z-periodic formulas and do NOT
/// need it — see lbm_core.cu notes).
/// @param f    Padded f buffer base (kQ * paddedCellCount() values).
/// @param dims Real grid dimensions.
cudaError_t launchRefreshGhostZ(FPop* f, GridDims dims, cudaStream_t stream);

/// @brief Refresh the two spanwise ghost planes of the padded flag buffer.
/// Runs once per flag upload (flags are static between geometry edits).
/// @param flags Padded flag buffer base (paddedCellCount() bytes).
cudaError_t launchRefreshGhostZFlags(std::uint8_t* flags, GridDims dims,
                                     cudaStream_t stream);

/// @brief Set f = equilibrium(rho, u) from UNPADDED macroscopic device arrays
/// (the compact-snapshot restore path, plan section 8): every padded cell gets
/// the equilibrium of its (z-wrapped) macroscopic sample; SOLID cells get the
/// rest state. Smagorinsky state needs no re-init — the eddy term rebuilds
/// from non-equilibrium populations within one step.
/// @param lattice Destination buffer view (written in full, ghosts included).
/// @param rho     Device density array, cellCount() floats (unpadded).
/// @param u       Device x-velocity, as rho.
/// @param v       Device y-velocity.
/// @param w       Device z-velocity.
cudaError_t launchInitFromMacroscopic(DeviceLatticeView lattice,
                                      const float* rho, const float* u,
                                      const float* v, const float* w,
                                      cudaStream_t stream);

/// @brief Warm-restart flag-edit fixup (plan section 8 VG-edit flow), applied
/// AFTER the new flags are uploaded and their ghosts refreshed: cells that
/// just became SOLID get all populations zeroed; cells that just became FLUID
/// get the equilibrium of the averaged (rho, u) of their still-fluid
/// neighbors (falling back to rest equilibrium when fully enclosed). Both
/// ping-pong buffers are patched so the next pull is sane regardless of
/// parity. Caller refreshes f ghosts afterwards (edits may touch z edges).
/// @param lattice    Current source buffer view with the NEW flags.
/// @param fOther     The other ping-pong buffer (padded base), patched too.
/// @param d_editMask Device byte per UNPADDED cell: 0 = unchanged,
///                   1 = newly solid, 2 = newly fluid.
cudaError_t launchApplyFlagEdits(DeviceLatticeView lattice, FPop* fOther,
                                 const std::uint8_t* d_editMask,
                                 cudaStream_t stream);

/// @brief Particle advection entry point (plan 9.1): RK2-advect `count`
/// particles through the trilinearly-sampled velocity field, handle inlet
/// respawn / solid-entry respawn / age fade. Position buffer is the CUDA-
/// mapped GL VBO; layout and the kernel itself live in render/particles.cuh —
/// this declaration exists so the solver module can drive advection without
/// including GL headers.
/// @param positions   Mapped GL buffer: float4 per particle (xyz = lattice-
///                    space position, w = normalized age in [0,1]).
/// @param count       Number of particles.
/// @param vel         Velocity field view to sample.
/// @param flags       Cell flags (for solid-entry respawn detection).
/// @param dtSteps     Advection time in lattice steps (sim steps this frame).
/// @param seed        Per-frame RNG seed for respawn jitter.
cudaError_t launchAdvectParticles(float4* positions, int count,
                                  DeviceVelocityField vel,
                                  const std::uint8_t* flags,
                                  float dtSteps, unsigned int seed,
                                  cudaStream_t stream);

} // namespace foilcfd
