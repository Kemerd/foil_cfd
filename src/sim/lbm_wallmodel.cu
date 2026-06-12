// iMEM wall-model update kernel (see lbm_wallmodel.cuh for the contract):
// per listed wall cell — sample moments off the wall, Newton-solve Reichardt
// for u_tau, EMA-blend, then the closed-form slip velocity that makes the
// bounce-back links carry exactly the modeled wall shear stress.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include "lbm_wallmodel.cuh"

#include <cuda_fp16.h>

namespace foilcfd {

namespace {

// Per-TU copies of the frozen D3Q19 constants (single-TU compilation: each
// .cu owns its constant-memory tables; values must match lbm_core.cuh).
__constant__ int   wm_cx[kQ] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0};
__constant__ int   wm_cy[kQ] = {0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1};
__constant__ int   wm_cz[kQ] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1};
__constant__ int   wm_opp[kQ] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};
__constant__ float wm_w[kQ] = {
    1.0f / 3.0f,
    1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f};

/// One warp-multiple block; the kernel is tiny (O(surface) threads).
constexpr int kBlock = 128;

/// @brief Reichardt's all-y+ law: u+ as a function of y+.
__device__ __forceinline__ float reichardtUPlus(float yp) {
    return __logf(1.0f + kWallModelKappa * yp) / kWallModelKappa
         + kWallModelReichardtC
               * (1.0f - __expf(-yp / 11.0f)
                  - (yp / 11.0f) * __expf(-yp / 3.0f));
}

/// @brief d(u+)/d(y+) of the Reichardt profile (for the Newton derivative).
__device__ __forceinline__ float reichardtDuPlus(float yp) {
    return 1.0f / (1.0f + kWallModelKappa * yp)
         + kWallModelReichardtC
               * ((1.0f / 11.0f) * __expf(-yp / 11.0f)
                  - (1.0f / 11.0f) * __expf(-yp / 3.0f)
                  + (yp / 33.0f) * __expf(-yp / 3.0f));
}

/// @brief Solve u_t = u_tau * u+(y_s u_tau / nu) for u_tau by Newton's
/// method. g(u_tau) = u_tau u+(y+) - u_t is monotone increasing (u+ > 0,
/// du+/dy+ > 0), so the iteration is globally well-behaved; six iterations
/// take any sane seed to float precision.
/// @param ut   Tangential speed at the sample point (> cutoff).
/// @param ys   Sample wall distance in cells.
/// @param nu   Molecular lattice viscosity.
/// @param seed Previous u_tau (warm start) or <= 0 for a fresh seed.
__device__ float solveUTau(float ut, float ys, float nu, float seed) {
    // Fresh seed: the larger of a log-layer-typical fraction of u_t and the
    // exact viscous-sublayer solution u_tau = sqrt(nu u_t / y) — whichever
    // regime the sample sits in, the seed lands within Newton's basin.
    float utau = (seed > 0.0f)
        ? seed
        : fmaxf(0.04f * ut, sqrtf(nu * ut / ys));
#pragma unroll
    for (int it = 0; it < 6; ++it) {
        const float yp = fmaxf(ys * utau / nu, 1e-8f);
        const float up = reichardtUPlus(yp);
        const float g  = utau * up - ut;
        // dg/du_tau = u+ + y+ du+/dy+ (chain rule through y+ = ys u_tau/nu).
        const float dg = up + yp * reichardtDuPlus(yp);
        utau -= g / fmaxf(dg, 1e-6f);
        utau = fmaxf(utau, 0.0f);
    }
    return utau;
}

/// @brief The update kernel: one thread per listed wall cell.
__global__ void wallModelUpdateKernel(WallCellListView list,
                                      const FPop* __restrict__ f,
                                      long long ncellsPad, long long nxny,
                                      std::uint16_t* __restrict__ uwx,
                                      std::uint16_t* __restrict__ uwy,
                                      std::uint16_t* __restrict__ uwz,
                                      WallModelParams p,
                                      WallModelDeviceStats* __restrict__ stats) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= list.count) return;

    const long long cell = list.cellIdx[i];
    const float nX = list.normalX[i];
    const float nY = list.normalY[i];
    const float nZ = list.normalZ[i];

    // ---- moments at the sample cell (padded index = unpadded + nx*ny) -----
    const long long ps = list.sampleIdx[i] + nxny;
    float rho = 0.0f, jx = 0.0f, jy = 0.0f, jz = 0.0f;
#pragma unroll
    for (int q = 0; q < kQ; ++q) {
        const float fq = load_f(f, static_cast<long long>(q) * ncellsPad + ps);
        rho += fq;
        jx += wm_cx[q] * fq;
        jy += wm_cy[q] * fq;
        jz += wm_cz[q] * fq;
    }
    rho = fmaxf(rho, 0.05f); // same floor as the collision kernel
    const float inv = 1.0f / rho;
    const float ux = jx * inv, uy = jy * inv, uz = jz * inv;

    // ---- tangential projection -------------------------------------------
    const float un = ux * nX + uy * nY + uz * nZ;
    const float tx = ux - un * nX;
    const float ty = uy - un * nY;
    const float tz = uz - un * nZ;
    const float ut = sqrtf(tx * tx + ty * ty + tz * tz);

    // Stagnation / separation / reattachment: no meaningful tangential flow,
    // no meaningful equilibrium wall law. Zero the state so the cell behaves
    // as plain no-slip bounce-back until the flow re-attaches over it.
    if (ut < p.utCutoff) {
        list.uTau[i] = 0.0f;
        const std::uint16_t zero = __half_as_ushort(__float2half_rn(0.0f));
        uwx[cell] = zero; uwy[cell] = zero; uwz[cell] = zero;
        return;
    }
    const float itx = tx / ut, ity = ty / ut, itz = tz / ut;

    // ---- friction velocity: Newton on Reichardt, EMA against the stored
    // value so the u_tau <-> u_w feedback loop cannot ring ------------------
    const float prev = list.uTau[i];
    const float solved = solveUTau(ut, list.sampleDist[i], p.nuLat, prev);
    const float utau = (prev > 0.0f)
        ? prev + p.emaAlpha * (solved - prev)
        : solved;
    list.uTau[i] = utau;

    // ---- iMEM slip velocity: the balance is LINEAR in u_w ----------------
    // Plain moving-wall bounce-back hands the wall, per solid link q' (the
    // OUTGOING direction; our mask stores pull directions, so q' = opp(q)
    // and c_q' = -c_q):  (c_q' . t) (2 f*_q' - 6 w (c_q' . u_w)).
    // Summed: F_t = M0 - u_w D. Setting F_t = rho u_tau^2 A_w (the modeled
    // shear force on this cell's stair-step area) and solving for u_w needs
    // no iteration — and no eddy viscosity, the property that makes iMEM fit
    // this solver (Asmuth et al. 2021).
    float m0 = 0.0f, dcoef = 0.0f;
    const std::uint32_t mask = list.linkMask[i];
    const long long pc = cell + nxny;
#pragma unroll
    for (int q = 1; q < kQ; ++q) {
        if (!(mask & (1u << q))) continue;
        const int o = wm_opp[q]; // outgoing direction, c_o = -c_q
        const float ct = -(wm_cx[q] * itx + wm_cy[q] * ity + wm_cz[q] * itz);
        const float fo =
            load_f(f, static_cast<long long>(o) * ncellsPad + pc);
        m0 += 2.0f * fo * ct;
        dcoef += 6.0f * wm_w[q] * ct * ct;
    }

    // Stair-step area of the cell's wall facet: a 45-degree facet exposes
    // sqrt(2) the axis-aligned area, and the normal encodes exactly that.
    const float maxN = fmaxf(fabsf(nX), fmaxf(fabsf(nY), fabsf(nZ)));
    const float aw = 1.0f / fmaxf(maxN, 0.1f);

    float uw = 0.0f;
    if (dcoef > 1e-6f) {
        // Target the modeled stress with the SIGN of the current exchange:
        // the wall model rescales the wall friction, never reverses it.
        const float target = rho * utau * utau * aw;
        const float sgn = (m0 >= 0.0f) ? 1.0f : -1.0f;
        uw = (m0 - sgn * target) / dcoef;
    }

    // Low-y+ fade: below kWallModelFadeY0 the sublayer is resolved and the
    // resolved stress is already right — the model steps aside instead of
    // imposing its (coarser) estimate of the same number. Linear ramp to
    // full strength at kWallModelFadeY1.
    const float ypSample = list.sampleDist[i] * utau / p.nuLat;
    const float fade = fminf(fmaxf((ypSample - kWallModelFadeY0)
                                       / (kWallModelFadeY1 - kWallModelFadeY0),
                                   0.0f), 1.0f);
    uw *= fade;

    // Safety clamp: the slip is a numerical device, not a physical velocity;
    // letting it approach the sample speed (or the collision limiter's cap)
    // would mean the model is fighting the resolved flow.
    const float cap = fminf(0.5f * ut, 0.5f * kMaxSimSpeed);
    const float uwClamped = fminf(fmaxf(uw, -cap), cap);

    uwx[cell] = __half_as_ushort(__float2half_rn(uwClamped * itx));
    uwy[cell] = __half_as_ushort(__float2half_rn(uwClamped * ity));
    uwz[cell] = __half_as_ushort(__float2half_rn(uwClamped * itz));

    // ---- UI diagnostics ----------------------------------------------------
    const float yplus = list.sampleDist[i] * utau / p.nuLat;
    atomicAdd(&stats->sumYplus, yplus);
    atomicMax(&stats->maxYplusBits, __float_as_int(fmaxf(yplus, 0.0f)));
    atomicAdd(&stats->activeCells, 1);
    if (uwClamped != uw) atomicAdd(&stats->clampedCells, 1);
}

} // namespace

cudaError_t launchWallModelUpdate(WallCellListView list,
                                  DeviceLatticeView lattice,
                                  std::uint16_t* uwx, std::uint16_t* uwy,
                                  std::uint16_t* uwz,
                                  const WallModelParams& params,
                                  WallModelDeviceStats* d_stats,
                                  cudaStream_t stream) {
    if (list.count <= 0 || !list.cellIdx || !list.uTau || !lattice.f
        || !uwx || !uwy || !uwz || !d_stats)
        return cudaErrorInvalidValue;

    // Stats are per-update; the wrapper owns the zeroing.
    if (auto err = cudaMemsetAsync(d_stats, 0, sizeof(WallModelDeviceStats),
                                   stream);
        err != cudaSuccess)
        return err;

    const long long nxny =
        static_cast<long long>(lattice.dims.nx) * lattice.dims.ny;
    const unsigned int blocks =
        static_cast<unsigned int>((list.count + kBlock - 1) / kBlock);
    wallModelUpdateKernel<<<blocks, kBlock, 0, stream>>>(
        list, lattice.f, lattice.dims.paddedCellCount(), nxny,
        uwx, uwy, uwz, params, d_stats);
    return cudaGetLastError();
}

} // namespace foilcfd
