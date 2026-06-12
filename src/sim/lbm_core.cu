// D3Q19 kernel implementations: init, fused TRT-Smagorinsky stream-collide
// (pull scheme, branchless boundaries), spanwise speck perturbation, NaN
// watchdog, momentum-exchange force reduction, ghost-plane refresh, and the
// warm-restart helpers (equilibrium-from-macroscopic, flag-edit fixup).
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include "lbm_core.cuh"

#include <cmath>

namespace foilcfd {

namespace {

// ===========================================================================
// Device copies of the frozen lattice constants. Host constexpr arrays are
// not visible to device code, so they are duplicated into __constant__ memory
// — values MUST match lbm_core.cuh exactly (snapshots depend on the order).
// ===========================================================================

__constant__ int   d_cx[kQ] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0};
__constant__ int   d_cy[kQ] = {0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1};
__constant__ int   d_cz[kQ] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1};
__constant__ int   d_opp[kQ]  = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};
__constant__ int   d_mirY[kQ] = {0, 1, 2, 4, 3, 5, 6, 13, 14, 9, 10, 18, 17, 7, 8, 15, 16, 12, 11};
__constant__ int   d_mirZ[kQ] = {0, 1, 2, 3, 4, 6, 5, 7, 8, 15, 16, 17, 18, 13, 14, 9, 10, 11, 12};
__constant__ float d_w[kQ] = {
    1.0f / 3.0f,
    1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f};

/// One thread block per 256 cells everywhere: the fused kernel is bandwidth
/// bound, so block size only needs to keep the SMs occupied (plan 4.1).
constexpr int kBlock = 256;

/// Raw flag byte values, named once so kernels stay readable without the
/// scoped-enum casts (numeric values frozen alongside CellFlag).
constexpr std::uint8_t kFlagFluid  = 0;
constexpr std::uint8_t kFlagSolid  = 1;
constexpr std::uint8_t kFlagInlet  = 2;
constexpr std::uint8_t kFlagOutlet = 3;
constexpr std::uint8_t kFlagSlipT  = 4;
constexpr std::uint8_t kFlagSlipB  = 5;
constexpr std::uint8_t kFlagSlipF  = 6; // z=0 free-slip wall (STL mode)
constexpr std::uint8_t kFlagSlipK  = 7; // z=nz-1 free-slip wall (STL mode)

// ===========================================================================
// Index helpers. Layout contract (lbm_core.cuh): real cell (x,y,z) lives at
// padded index x + nx*(y + ny*(z+1)); ghost planes sit at z=-1 and z=nz.
// Per-q population: q * ncellsPad + paddedCell.
// ===========================================================================

/// @brief Padded linear index of lattice site (x, y, z) with z in [-1, nz].
__device__ __forceinline__ long long pIdx(int x, int y, int z, int nx,
                                          long long nxny) {
    return static_cast<long long>(x) + static_cast<long long>(nx) * y
         + nxny * (z + 1);
}

/// @brief D3Q19 equilibrium for all 19 directions at (rho, u). Second-order
/// expansion: feq_q = w_q rho (1 + 3 c.u + 4.5 (c.u)^2 - 1.5 u.u), cs^2=1/3.
__device__ __forceinline__ void equilibrium19(float rho, float ux, float uy,
                                              float uz, float feq[kQ]) {
    const float usq = 1.5f * (ux * ux + uy * uy + uz * uz);
#pragma unroll
    for (int q = 0; q < kQ; ++q) {
        const float cu = 3.0f * (d_cx[q] * ux + d_cy[q] * uy + d_cz[q] * uz);
        feq[q] = d_w[q] * rho * (1.0f + cu + 0.5f * cu * cu - usq);
    }
}

/// @brief Decompose an UNPADDED linear cell index into (x, y, z).
__device__ __forceinline__ void unpackCell(long long cell, int nx, long long nxny,
                                           int& x, int& y, int& z) {
    z = static_cast<int>(cell / nxny);
    const long long rem = cell - static_cast<long long>(z) * nxny;
    y = static_cast<int>(rem / nx);
    x = static_cast<int>(rem - static_cast<long long>(y) * nx);
}

/// @brief Decompose a PADDED linear index into (x, y, z) with z in [-1, nz].
__device__ __forceinline__ void unpackPadded(long long pcell, int nx, long long nxny,
                                             int& x, int& y, int& z) {
    const int zp = static_cast<int>(pcell / nxny);
    const long long rem = pcell - static_cast<long long>(zp) * nxny;
    y = static_cast<int>(rem / nx);
    x = static_cast<int>(rem - static_cast<long long>(y) * nx);
    z = zp - 1; // padded plane 0 is the low ghost (z = -1)
}

// ===========================================================================
// Init kernels. Both run over the PADDED domain with z-periodic formulas so
// the ghost planes come out consistent by construction (no refresh needed) —
// flag ghosts must already be valid (refreshed at flag upload).
// ===========================================================================

/// @brief Equilibrium init: rho = 1, u = (uInlet,0,0) everywhere except SOLID
/// cells, which get the rest state so bounce-back starts from a clean field.
__global__ void initEquilibriumKernel(FPop* __restrict__ f,
                                      const std::uint8_t* __restrict__ flags,
                                      long long ncellsPad, float uInlet) {
    const long long pcell =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pcell >= ncellsPad) return;

    // Solid cells hold quiescent equilibrium; everything else gets inflow.
    const bool solid = flags[pcell] == kFlagSolid;
    const float ux = solid ? 0.0f : uInlet;

    float feq[kQ];
    equilibrium19(1.0f, ux, 0.0f, 0.0f, feq);
#pragma unroll
    for (int q = 0; q < kQ; ++q)
        store_f(f, static_cast<long long>(q) * ncellsPad + pcell, feq[q]);
}

/// @brief Spanwise "speck" perturbation: quasi-2D LBM around an extruded foil
/// never breaks spanwise coherence on its own (the periodic span has no
/// symmetry-breaking source), so stall never develops 3D structure. Inject a
/// tiny deterministic z-velocity ripple — two incommensurate integer spanwise
/// modes, gently modulated along x — into every fluid cell of a fresh field.
///
/// The momentum is added linearly: df_q = 3 w_q c_z[q] w'. This shifts the
/// first velocity moment by exactly w' (sum w_q c_z^2 = cs^2 = 1/3) while
/// leaving density untouched (sum w_q c_z = 0). Integer mode counts keep the
/// field exactly z-periodic, so the ghost planes stay consistent.
__global__ void spanwisePerturbKernel(FPop* __restrict__ f,
                                      const std::uint8_t* __restrict__ flags,
                                      long long ncellsPad, long long nxny,
                                      int nx, int nz, float amplitude,
                                      float phase1, float phase2, float phaseX) {
    const long long pcell =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pcell >= ncellsPad) return;
    if (flags[pcell] != kFlagFluid) return; // only bulk cells; boundary cells
                                            // are rewritten every step anyway.
    int x, y, z;
    unpackPadded(pcell, nx, nxny, x, y, z);

    // Two spanwise modes (3 and 7 wavelengths across the span) so the
    // perturbation has no single dominant wavelength to lock onto; the slow
    // x modulation de-correlates chordwise stations. All angles are exactly
    // periodic in z, so the z=-1 ghost evaluates identical to z=nz-1.
    const float tz = 6.2831853f * (static_cast<float>(z) + 0.5f) / nz;
    const float tx = 6.2831853f * (static_cast<float>(x) + 0.5f) / nx;
    const float wz = amplitude
                   * (0.6f * __sinf(3.0f * tz + phase1)
                    + 0.4f * __sinf(7.0f * tz + phase2))
                   * (1.0f + 0.3f * __sinf(2.0f * tx + phaseX));

#pragma unroll
    for (int q = 0; q < kQ; ++q) {
        if (d_cz[q] == 0) continue; // c_z = 0 directions carry no z momentum
        const long long idx = static_cast<long long>(q) * ncellsPad + pcell;
        store_f(f, idx, load_f(f, idx) + 3.0f * d_w[q] * d_cz[q] * wz);
    }
}

/// @brief f = equilibrium(rho, u) from unpadded macroscopic arrays (compact
/// snapshot restore). Ghost cells sample the z-wrapped macroscopic cell so
/// the result is consistent without a separate ghost refresh.
__global__ void initFromMacroKernel(FPop* __restrict__ f,
                                    const std::uint8_t* __restrict__ flags,
                                    const float* __restrict__ rho,
                                    const float* __restrict__ u,
                                    const float* __restrict__ v,
                                    const float* __restrict__ w,
                                    long long ncellsPad, long long nxny,
                                    int nx, int nz) {
    const long long pcell =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pcell >= ncellsPad) return;
    int x, y, z;
    unpackPadded(pcell, nx, nxny, x, y, z);

    // z-wrap the macroscopic sample for the two ghost planes (periodic span).
    const int zw = (z < 0) ? (nz - 1) : ((z >= nz) ? 0 : z);
    const long long cell = static_cast<long long>(x)
                         + static_cast<long long>(nx) * y + nxny * zw;

    // Solid cells get the rest state; everything else re-equilibrates at the
    // stored macroscopic values. The short transient this causes (~1000
    // steps) is the documented cost of the compact variant (plan section 8).
    const bool solid = flags[pcell] == kFlagSolid;
    const float r  = solid ? 1.0f : rho[cell];
    const float ux = solid ? 0.0f : u[cell];
    const float uy = solid ? 0.0f : v[cell];
    const float uz = solid ? 0.0f : w[cell];

    float feq[kQ];
    equilibrium19(r, ux, uy, uz, feq);
#pragma unroll
    for (int q = 0; q < kQ; ++q)
        store_f(f, static_cast<long long>(q) * ncellsPad + pcell, feq[q]);
}

// ===========================================================================
// The fused stream-collide kernel (plan 4.1): ONE kernel per step, pull
// scheme. Each thread owns one REAL cell, reads its 19 upstream neighbors'
// post-collision values from src (boundary types resolved branchlessly per
// link by the NEIGHBOR's flag), computes rho/u, TRT-collides with the
// Smagorinsky eddy viscosity, and writes the post-collision state to dst at
// its own cell. dst ghost planes are refreshed by a separate tiny kernel.
// ===========================================================================

__global__ void streamCollideKernel(const FPop* __restrict__ src,
                                    FPop* __restrict__ dst,
                                    const std::uint8_t* __restrict__ flags,
                                    float* __restrict__ macroRho,
                                    float* __restrict__ macroU,
                                    float* __restrict__ macroV,
                                    float* __restrict__ macroW,
                                    long long ncells, long long ncellsPad,
                                    long long nxny, int nx,
                                    float tau0, float magicLambda,
                                    float smagPrefactor, float uInlet,
                                    int writeMacro) {
    const long long cell =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (cell >= ncells) return;

    int x, y, z;
    unpackCell(cell, nx, nxny, x, y, z);
    // Real cell (x,y,z) sits exactly one ghost plane (nx*ny) into the padded
    // buffer: pIdx(x,y,z) = cell + nxny. Cheapest identity in the file —
    // and the most important one to get right.
    const long long pcell = cell + nxny;

    const std::uint8_t myFlag = flags[pcell];

    // ---- boundary-cell paths (small, warp-coherent populations) ----------
    if (myFlag != kFlagFluid) {
        if (myFlag == kFlagInlet) {
            // Equilibrium Dirichlet at u_in (plan 4.2). Rewritten every step
            // so downstream pulls always see the prescribed inflow.
            float feq[kQ];
            equilibrium19(1.0f, uInlet, 0.0f, 0.0f, feq);
#pragma unroll
            for (int q = 0; q < kQ; ++q)
                store_f(dst, static_cast<long long>(q) * ncellsPad + pcell, feq[q]);
            if (writeMacro) {
                macroRho[cell] = 1.0f;
                macroU[cell] = uInlet; macroV[cell] = 0.0f; macroW[cell] = 0.0f;
            }
        } else if (myFlag == kFlagOutlet) {
            // Zero-gradient outlet: copy the upstream column's populations
            // (one step stale — the standard pull-scheme convention; the
            // wake convects out cleanly). x-1 is always in range: outlet
            // cells live only at x = nx-1.
            float r = 0.0f, jx = 0.0f, jy = 0.0f, jz = 0.0f;
#pragma unroll
            for (int q = 0; q < kQ; ++q) {
                const float fq =
                    load_f(src, static_cast<long long>(q) * ncellsPad + pcell - 1);
                store_f(dst, static_cast<long long>(q) * ncellsPad + pcell, fq);
                r += fq;
                jx += d_cx[q] * fq; jy += d_cy[q] * fq; jz += d_cz[q] * fq;
            }
            if (writeMacro) {
                const float inv = 1.0f / r;
                macroRho[cell] = r;
                macroU[cell] = jx * inv; macroV[cell] = jy * inv; macroW[cell] = jz * inv;
            }
        } else {
            // SOLID and the free-slip marker planes: their storage is never
            // read (bounce-back pulls from the puller's own opposite slot,
            // slip pulls from a mirrored FLUID cell), so no write needed.
            // Just keep the render fields clean.
            if (writeMacro) {
                macroRho[cell] = 1.0f;
                macroU[cell] = 0.0f; macroV[cell] = 0.0f; macroW[cell] = 0.0f;
            }
        }
        return;
    }

    // ---- fluid hot path ---------------------------------------------------
    // Pull all 19 populations. Per link, the upstream neighbor's flag picks
    // ONE of three (q-slice, cell) source pairs via predicated selects — a
    // single coalesced-by-default load per q, never an if/else ladder:
    //   normal   : src[q]        at the upstream cell
    //   solid    : src[opp(q)]   at THIS cell (half-way bounce-back)
    //   slip y   : src[mirY(q)]  at the tangentially-displaced cell (same y)
    //   slip z   : src[mirZ(q)]  at the tangentially-displaced cell (same z;
    //                            STL-mode free-slip side walls, plan 7.4)
    // Fluid cells satisfy x in [1,nx-2], y in [1,ny-2] (boundary faces are
    // all flagged), so xn/yn never leave the grid; zn in [-1,nz] is covered
    // by the ghost planes — no modulo anywhere in this loop (plan 11). When
    // the z faces are flagged SlipFront/SlipBack instead of being periodic,
    // fluid cells satisfy z in [1,nz-2] too, so the ghosts are never read.
    float fpop[kQ];
#pragma unroll
    for (int q = 0; q < kQ; ++q) {
        const int xn = x - d_cx[q];
        const int yn = y - d_cy[q];
        const int zn = z - d_cz[q];
        const long long pn = pIdx(xn, yn, zn, nx, nxny);

        const std::uint8_t nf = flags[pn];
        const bool solid = (nf == kFlagSolid);
        const bool slipY = (nf == kFlagSlipT) || (nf == kFlagSlipB);
        const bool slipZ = (nf == kFlagSlipF) || (nf == kFlagSlipK);

        // Specular reflection off a y-normal wall departs from
        // (x - cx, y, z - cz): the two y half-steps around the bounce cancel
        // (see kMirY derivation). The z-normal wall case is identical with
        // y and z roles swapped: source is (x - cx, y - cy, z).
        const long long pslipY = pIdx(xn, y, zn, nx, nxny);
        const long long pslipZ = pIdx(xn, yn, z, nx, nxny);

        const int qs = solid ? d_opp[q]
                             : (slipY ? d_mirY[q] : (slipZ ? d_mirZ[q] : q));
        const long long cs = solid ? pcell
                                   : (slipY ? pslipY : (slipZ ? pslipZ : pn));
        fpop[q] = load_f(src, static_cast<long long>(qs) * ncellsPad + cs);
    }

    // Macroscopic moments from the freshly streamed populations.
    float rho = 0.0f, jx = 0.0f, jy = 0.0f, jz = 0.0f;
#pragma unroll
    for (int q = 0; q < kQ; ++q) {
        rho += fpop[q];
        jx += d_cx[q] * fpop[q];
        jy += d_cy[q] * fpop[q];
        jz += d_cz[q] * fpop[q];
    }
    // Density floor: a pathological transient can drag a cell's mass toward
    // (or below) zero, and 1/rho would then catapult the velocity to infinity
    // in one step. fmaxf also scrubs a NaN rho to the floor value (IEEE: the
    // non-NaN operand wins), cutting that propagation path too.
    rho = fmaxf(rho, 0.05f);
    const float invRho = 1.0f / rho;
    float ux = jx * invRho, uy = jy * invRho, uz = jz * invRho;

    // Local stability limiter: the second-order equilibrium below is only
    // valid for |u| well under cs (~0.577); once a cell overshoots ~0.3 at
    // the tau clamp, the feq error feeds back through the collision and the
    // cell runs away to NaN within a few steps (seen as the deterministic
    // ~step-3600 divergence on aggressive sections at coarse grids). Rescale
    // the velocity VECTOR (direction preserved) onto the validity bound; the
    // collision then relaxes the populations toward this bounded equilibrium,
    // dissipating the spike instead of amplifying it. Normal flow (|u| ~ 0.1)
    // never engages this branch, so resolved physics is untouched.
    const float u2 = ux * ux + uy * uy + uz * uz;
    if (u2 > kMaxSimSpeed * kMaxSimSpeed) {
        const float s = kMaxSimSpeed * rsqrtf(u2);
        ux *= s; uy *= s; uz *= s;
    }

    float feq[kQ];
    equilibrium19(rho, ux, uy, uz, feq);

    // Smagorinsky LES from non-equilibrium populations (plan 4.1): the
    // deviatoric momentum flux Pi_ab = sum_q c_a c_b (f_q - feq_q) encodes
    // the resolved strain rate without finite differences. With
    // Qbar = |Pi|_F, nu_t = (Cs dx)^2 |S| folds into a closed-form effective
    // relaxation time (Hou et al. 1996):
    //   tau_eff = (tau0 + sqrt(tau0^2 + 18 sqrt(2) Cs^2 Qbar / rho)) / 2
    // smagPrefactor carries 18*sqrt(2)*Cs^2 (0 disables the model).
    float pxx = 0.0f, pyy = 0.0f, pzz = 0.0f;
    float pxy = 0.0f, pxz = 0.0f, pyz = 0.0f;
#pragma unroll
    for (int q = 0; q < kQ; ++q) {
        const float fneq = fpop[q] - feq[q];
        pxx += d_cx[q] * d_cx[q] * fneq;
        pyy += d_cy[q] * d_cy[q] * fneq;
        pzz += d_cz[q] * d_cz[q] * fneq;
        pxy += d_cx[q] * d_cy[q] * fneq;
        pxz += d_cx[q] * d_cz[q] * fneq;
        pyz += d_cy[q] * d_cz[q] * fneq;
    }
    float tauEff = tau0;
    if (smagPrefactor > 0.0f) {
        const float qbar = sqrtf(pxx * pxx + pyy * pyy + pzz * pzz
                                 + 2.0f * (pxy * pxy + pxz * pxz + pyz * pyz));
        tauEff = 0.5f * (tau0 + sqrtf(tau0 * tau0 + smagPrefactor * qbar * invRho));
    }

    // TRT relaxation rates: the even (viscous) rate carries the eddy
    // viscosity; the odd rate follows from the magic parameter
    // Lambda = (tau+ - 1/2)(tau- - 1/2), which pins the bounce-back wall to
    // the half-way plane for Lambda = 3/16 (plan 4.1).
    const float omP  = 1.0f / tauEff;
    const float tauM = 0.5f + magicLambda / (tauEff - 0.5f);
    const float omM  = 1.0f / tauM;

    // Collide and write. q=0 has no antisymmetric part; the 9 opposite pairs
    // are consecutive (a, a+1) for odd a (kOpp layout), letting symmetric /
    // antisymmetric halves be formed once per pair.
    store_f(dst, pcell, fpop[0] - omP * (fpop[0] - feq[0]));
#pragma unroll
    for (int a = 1; a < kQ; a += 2) {
        const int b = a + 1;
        const float symRelax  = omP * (0.5f * (fpop[a] + fpop[b]) - 0.5f * (feq[a] + feq[b]));
        const float asymRelax = omM * (0.5f * (fpop[a] - fpop[b]) - 0.5f * (feq[a] - feq[b]));
        store_f(dst, static_cast<long long>(a) * ncellsPad + pcell,
                fpop[a] - symRelax - asymRelax);
        store_f(dst, static_cast<long long>(b) * ncellsPad + pcell,
                fpop[b] - symRelax + asymRelax);
    }

    // Macroscopic store only when a render mode needs it this frame
    // (plan 11: skip the 16 B/cell store otherwise).
    if (writeMacro) {
        macroRho[cell] = rho;
        macroU[cell] = ux; macroV[cell] = uy; macroW[cell] = uz;
    }
}

// ===========================================================================
// Ghost-plane refresh: copy real edge planes into the opposite ghost planes
// (periodic span). ~2 * kQ * nx * ny float copies — well under 1% of a step's
// traffic at the default grid, and far simpler than wrap arithmetic in the
// 19-load hot loop.
// ===========================================================================

__global__ void refreshGhostZKernel(FPop* __restrict__ f, long long nxny,
                                    long long ncellsPad, int nz) {
    const long long t =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = nxny * kQ;
    if (t >= total) return;
    const int q = static_cast<int>(t / nxny);
    const long long i = t - static_cast<long long>(q) * nxny;
    const long long base = static_cast<long long>(q) * ncellsPad;
    // Padded plane p holds real z = p-1: low ghost (p=0) <- z=nz-1 (p=nz),
    // high ghost (p=nz+1) <- z=0 (p=1).
    store_f(f, base + i, load_f(f, base + i + nxny * nz));
    store_f(f, base + i + nxny * (nz + 1), load_f(f, base + i + nxny));
}

__global__ void refreshGhostZFlagsKernel(std::uint8_t* __restrict__ flags,
                                         long long nxny, int nz) {
    const long long i =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nxny) return;
    flags[i] = flags[i + nxny * nz];                  // low ghost <- z = nz-1
    flags[i + nxny * (nz + 1)] = flags[i + nxny];     // high ghost <- z = 0
}

// ===========================================================================
// NaN watchdog (plan 4.5): strided ~1/4093 sample; the sum of all 19
// populations is NaN/Inf iff any member is (NaN propagates; +/-Inf either
// survives or collapses to NaN), so one isfinite() covers the whole cell.
//
// The stride MUST NOT share factors with the grid dimensions: a power-of-two
// stride aliases with power-of-two nx*ny (e.g. on a 64^3 grid every multiple
// of 4096 lands on the x=0,y=0 inlet column, which is rewritten to clean
// equilibrium each step — the watchdog could never trip). 4093 is prime, so
// the sample set sweeps every x/y column for any realistic grid dimension.
// A per-launch offset additionally rotates which residue class is sampled,
// so successive checks cover different cells over time.
// ===========================================================================

/// Prime sampling stride for the watchdog — coprime to any practical grid
/// dimension, so the strided sample cannot collapse onto boundary columns.
constexpr long long kWatchdogStride = 4093;

__global__ void nanWatchdogKernel(const FPop* __restrict__ f,
                                  long long ncells, long long ncellsPad,
                                  long long nxny, long long offset,
                                  int* __restrict__ d_flag) {
    const long long t =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long cell = t * kWatchdogStride + offset;
    if (cell >= ncells) return;
    const long long pcell = cell + nxny;
    float s = 0.0f, jx = 0.0f, jy = 0.0f, jz = 0.0f;
#pragma unroll
    for (int q = 0; q < kQ; ++q) {
        const float fq = load_f(f, static_cast<long long>(q) * ncellsPad + pcell);
        s += fq;
        jx += d_cx[q] * fq;
        jy += d_cy[q] * fq;
        jz += d_cz[q] * fq;
    }
    // Two failure modes, two flag values (benign races: any writer wins):
    //   1 = non-finite populations (classic NaN/Inf divergence);
    //   2 = velocity runaway — the cell sits at (or beyond) the collision
    //       limiter's cap, meaning the limiter is the only thing standing
    //       between this state and NaN. A healthy flow never reaches the cap,
    //       so a sampled cell pinned there is a diverged field even though
    //       every number in it is still finite (the "whole domain turns red"
    //       state). 0.95x: catch cells the limiter clamped EXACTLY to the cap.
    if (!isfinite(s)) {
        *d_flag = 1;
    } else {
        const float speed2 = (jx * jx + jy * jy + jz * jz) / fmaxf(s * s, 1e-12f);
        const float cap = 0.95f * kMaxSimSpeed;
        if (speed2 > cap * cap) *d_flag = 2;
    }
}

// ===========================================================================
// Momentum-exchange force (plan 4.4): for every fluid cell, every link whose
// neighbor is SOLID transfers 2 f_q^post c_q to the wall per step (half-way
// bounce-back returns the same post-collision population, so the exchanged
// momentum is exactly twice the outgoing one). Block-level shared reduction,
// then 3 atomics per block — microseconds at this scale.
// ===========================================================================

__global__ void forceReductionKernel(const FPop* __restrict__ f,
                                     const std::uint8_t* __restrict__ flags,
                                     long long ncells, long long ncellsPad,
                                     long long nxny, int nx,
                                     float* __restrict__ d_force) {
    __shared__ float sFx[kBlock], sFy[kBlock], sFz[kBlock];
    const long long cell =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;

    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    if (cell < ncells) {
        const long long pcell = cell + nxny;
        if (flags[pcell] == kFlagFluid) {
            int x, y, z;
            unpackCell(cell, nx, nxny, x, y, z);
#pragma unroll
            for (int q = 1; q < kQ; ++q) {
                // Link pointing INTO the candidate wall: neighbor at +c_q.
                // Fluid-cell coordinate bounds keep x+cx / y+cy in range;
                // z+cz rides the ghost planes.
                const long long pn =
                    pIdx(x + d_cx[q], y + d_cy[q], z + d_cz[q], nx, nxny);
                if (flags[pn] == kFlagSolid) {
                    const float fq =
                        load_f(f, static_cast<long long>(q) * ncellsPad + pcell);
                    fx += 2.0f * d_cx[q] * fq;
                    fy += 2.0f * d_cy[q] * fq;
                    fz += 2.0f * d_cz[q] * fq;
                }
            }
        }
    }

    // Standard tree reduction in shared memory, then one atomic triplet.
    const int t = threadIdx.x;
    sFx[t] = fx; sFy[t] = fy; sFz[t] = fz;
    __syncthreads();
    for (int s = kBlock / 2; s > 0; s >>= 1) {
        if (t < s) { sFx[t] += sFx[t + s]; sFy[t] += sFy[t + s]; sFz[t] += sFz[t + s]; }
        __syncthreads();
    }
    if (t == 0 && (sFx[0] != 0.0f || sFy[0] != 0.0f || sFz[0] != 0.0f)) {
        atomicAdd(&d_force[0], sFx[0]);
        atomicAdd(&d_force[1], sFy[0]);
        atomicAdd(&d_force[2], sFz[0]);
    }
}

// ===========================================================================
// Warm-restart flag-edit fixup (plan section 8 VG-edit flow). Cold path —
// runs once per VG edit on a handful of cells; clarity over throughput.
// ===========================================================================

__global__ void applyFlagEditsKernel(FPop* __restrict__ fA, FPop* __restrict__ fB,
                                     const std::uint8_t* __restrict__ flags,
                                     const std::uint8_t* __restrict__ editMask,
                                     long long ncells, long long ncellsPad,
                                     long long nxny, int nx, int ny, int nz) {
    const long long cell =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (cell >= ncells) return;
    const std::uint8_t edit = editMask[cell];
    if (edit == 0) return; // overwhelmingly common: untouched cell

    const long long pcell = cell + nxny;

    if (edit == 1) {
        // Newly SOLID: zero every population in both ping-pong buffers so no
        // stale fluid state can ever leak out of the new vane.
#pragma unroll
        for (int q = 0; q < kQ; ++q) {
            store_f(fA, static_cast<long long>(q) * ncellsPad + pcell, 0.0f);
            store_f(fB, static_cast<long long>(q) * ncellsPad + pcell, 0.0f);
        }
        return;
    }

    // Newly FLUID: average rho/u over surrounding cells that were fluid
    // BEFORE the edit (new flag Fluid, mask 0) and equilibrium-fill. The
    // restored field is wrong only here — exactly where the solver fixes it
    // first (plan section 8).
    int x, y, z;
    unpackCell(cell, nx, nxny, x, y, z);

    float rho = 0.0f, ux = 0.0f, uy = 0.0f, uz = 0.0f;
    int found = 0;
    for (int q = 1; q < kQ; ++q) {
        const int xn = x + d_cx[q];
        const int yn = y + d_cy[q];
        int zn = z + d_cz[q];
        if (xn < 0 || xn >= nx || yn < 0 || yn >= ny) continue;
        // The unpadded edit mask needs an explicit z wrap (cold path only).
        const int znw = (zn < 0) ? (nz - 1) : ((zn >= nz) ? 0 : zn);
        const long long ncell = static_cast<long long>(xn)
                              + static_cast<long long>(nx) * yn + nxny * znw;
        if (flags[ncell + nxny] != kFlagFluid || editMask[ncell] != 0) continue;

        // Moments of the (still valid) neighbor from the source buffer.
        float r = 0.0f, jx = 0.0f, jy = 0.0f, jz = 0.0f;
#pragma unroll
        for (int p = 0; p < kQ; ++p) {
            const float fq =
                load_f(fA, static_cast<long long>(p) * ncellsPad + ncell + nxny);
            r += fq;
            jx += d_cx[p] * fq; jy += d_cy[p] * fq; jz += d_cz[p] * fq;
        }
        rho += r;
        ux += jx / r; uy += jy / r; uz += jz / r;
        ++found;
    }
    if (found > 0) {
        const float inv = 1.0f / found;
        rho *= inv; ux *= inv; uy *= inv; uz *= inv;
    } else {
        // Fully enclosed (e.g. a hollow ramp interior just opened): rest
        // state is the only defensible choice.
        rho = 1.0f; ux = uy = uz = 0.0f;
    }

    float feq[kQ];
    equilibrium19(rho, ux, uy, uz, feq);
#pragma unroll
    for (int q = 0; q < kQ; ++q) {
        store_f(fA, static_cast<long long>(q) * ncellsPad + pcell, feq[q]);
        store_f(fB, static_cast<long long>(q) * ncellsPad + pcell, feq[q]);
    }
}

// ===========================================================================
// Generic float fill: cudaMemset can only splat bytes, but the macroscopic
// density must start at the LBM rest value 1.0 (a rho = 0 init renders a
// bogus pressure field and, worse, poisons compact snapshots captured before
// the first step — equilibrium(rho = 0) is an all-zero population set that
// divides by zero on the very next collide).
// ===========================================================================

__global__ void fillFloatKernel(float* __restrict__ dst, long long n,
                                float value) {
    const long long i =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = value;
}

/// @brief 1D grid size for @p n threads at the standard block size.
inline unsigned int gridFor(long long n) {
    return static_cast<unsigned int>((n + kBlock - 1) / kBlock);
}

} // namespace

// ===========================================================================
// Host-callable launch wrappers (declared in lbm_core.cuh). Each validates
// pointers cheaply and returns the launch's cudaGetLastError().
// ===========================================================================

cudaError_t launchInitEquilibrium(DeviceLatticeView lattice, float uInlet,
                                  cudaStream_t stream) {
    const long long npad = lattice.dims.paddedCellCount();
    if (npad <= 0 || !lattice.f || !lattice.flags) return cudaErrorInvalidValue;
    initEquilibriumKernel<<<gridFor(npad), kBlock, 0, stream>>>(
        lattice.f, lattice.flags, npad, uInlet);
    return cudaGetLastError();
}

cudaError_t launchSpanwisePerturbation(DeviceLatticeView lattice, float amplitude,
                                       unsigned int seed, cudaStream_t stream) {
    const long long npad = lattice.dims.paddedCellCount();
    if (npad <= 0 || !lattice.f || !lattice.flags) return cudaErrorInvalidValue;
    const long long nxny = static_cast<long long>(lattice.dims.nx) * lattice.dims.ny;

    // Deterministic phases from the seed via a Weyl-style integer hash —
    // identical seed reproduces the identical field, per the header contract.
    auto phaseOf = [seed](unsigned int salt) {
        unsigned int h = (seed + salt) * 2654435761u;
        h ^= h >> 16;
        return 6.2831853f * static_cast<float>(h & 0xFFFFFFu) / 16777216.0f;
    };
    spanwisePerturbKernel<<<gridFor(npad), kBlock, 0, stream>>>(
        lattice.f, lattice.flags, npad, nxny, lattice.dims.nx, lattice.dims.nz,
        amplitude, phaseOf(1u), phaseOf(2u), phaseOf(3u));
    return cudaGetLastError();
}

cudaError_t launchStreamCollide(DeviceLatticeView src, DeviceLatticeView dst,
                                const StepParams& params,
                                float* macroRho, float* macroU, float* macroV,
                                float* macroW, cudaStream_t stream) {
    const long long ncells = src.dims.cellCount();
    if (ncells <= 0 || !src.f || !dst.f || !src.flags) return cudaErrorInvalidValue;
    const bool wantMacro = params.writeMacro && macroRho && macroU && macroV && macroW;

    const long long nxny = static_cast<long long>(src.dims.nx) * src.dims.ny;
    // 18*sqrt(2)*Cs^2 precomputed on host; 0 disables the LES term entirely.
    const float smagPre = (params.smagorinskyCs > 0.0f)
        ? 18.0f * 1.41421356f * params.smagorinskyCs * params.smagorinskyCs
        : 0.0f;

    streamCollideKernel<<<gridFor(ncells), kBlock, 0, stream>>>(
        src.f, dst.f, src.flags, macroRho, macroU, macroV, macroW,
        ncells, src.dims.paddedCellCount(), nxny, src.dims.nx,
        params.tau, params.magicLambda, smagPre, params.uInlet,
        wantMacro ? 1 : 0);
    return cudaGetLastError();
}

cudaError_t launchRefreshGhostZ(FPop* f, GridDims dims, cudaStream_t stream) {
    if (!f || dims.cellCount() <= 0) return cudaErrorInvalidValue;
    const long long nxny = static_cast<long long>(dims.nx) * dims.ny;
    refreshGhostZKernel<<<gridFor(nxny * kQ), kBlock, 0, stream>>>(
        f, nxny, dims.paddedCellCount(), dims.nz);
    return cudaGetLastError();
}

cudaError_t launchRefreshGhostZFlags(std::uint8_t* flags, GridDims dims,
                                     cudaStream_t stream) {
    if (!flags || dims.cellCount() <= 0) return cudaErrorInvalidValue;
    const long long nxny = static_cast<long long>(dims.nx) * dims.ny;
    refreshGhostZFlagsKernel<<<gridFor(nxny), kBlock, 0, stream>>>(
        flags, nxny, dims.nz);
    return cudaGetLastError();
}

cudaError_t launchInitFromMacroscopic(DeviceLatticeView lattice,
                                      const float* rho, const float* u,
                                      const float* v, const float* w,
                                      cudaStream_t stream) {
    const long long npad = lattice.dims.paddedCellCount();
    if (npad <= 0 || !lattice.f || !lattice.flags || !rho || !u || !v || !w)
        return cudaErrorInvalidValue;
    const long long nxny = static_cast<long long>(lattice.dims.nx) * lattice.dims.ny;
    initFromMacroKernel<<<gridFor(npad), kBlock, 0, stream>>>(
        lattice.f, lattice.flags, rho, u, v, w, npad, nxny,
        lattice.dims.nx, lattice.dims.nz);
    return cudaGetLastError();
}

cudaError_t launchApplyFlagEdits(DeviceLatticeView lattice, FPop* fOther,
                                 const std::uint8_t* d_editMask,
                                 cudaStream_t stream) {
    const long long ncells = lattice.dims.cellCount();
    if (ncells <= 0 || !lattice.f || !fOther || !lattice.flags || !d_editMask)
        return cudaErrorInvalidValue;
    const long long nxny = static_cast<long long>(lattice.dims.nx) * lattice.dims.ny;
    applyFlagEditsKernel<<<gridFor(ncells), kBlock, 0, stream>>>(
        lattice.f, fOther, lattice.flags, d_editMask, ncells,
        lattice.dims.paddedCellCount(), nxny,
        lattice.dims.nx, lattice.dims.ny, lattice.dims.nz);
    return cudaGetLastError();
}

cudaError_t launchNaNWatchdog(DeviceLatticeView lattice, int* d_nanFlag,
                              long long sampleOffset, cudaStream_t stream) {
    const long long ncells = lattice.dims.cellCount();
    if (ncells <= 0 || !lattice.f || !d_nanFlag) return cudaErrorInvalidValue;
    const long long nxny = static_cast<long long>(lattice.dims.nx) * lattice.dims.ny;
    // Reduce the caller-supplied offset into [0, stride): the kernel's bounds
    // guard then keeps every sampled cell inside the interior region.
    const long long offset =
        ((sampleOffset % kWatchdogStride) + kWatchdogStride) % kWatchdogStride;
    const long long samples = (ncells + kWatchdogStride - 1) / kWatchdogStride;
    nanWatchdogKernel<<<gridFor(samples), kBlock, 0, stream>>>(
        lattice.f, ncells, lattice.dims.paddedCellCount(), nxny, offset,
        d_nanFlag);
    return cudaGetLastError();
}

cudaError_t launchFillFloat(float* d_dst, long long count, float value,
                            cudaStream_t stream) {
    if (!d_dst || count <= 0) return cudaErrorInvalidValue;
    fillFloatKernel<<<gridFor(count), kBlock, 0, stream>>>(d_dst, count, value);
    return cudaGetLastError();
}

cudaError_t launchForceReduction(DeviceLatticeView lattice,
                                 DeviceForceAccumulator acc, cudaStream_t stream) {
    const long long ncells = lattice.dims.cellCount();
    if (ncells <= 0 || !lattice.f || !lattice.flags || !acc.d_force)
        return cudaErrorInvalidValue;
    // The wrapper owns zeroing the accumulator (header contract).
    if (auto err = cudaMemsetAsync(acc.d_force, 0, 3 * sizeof(float), stream);
        err != cudaSuccess)
        return err;
    const long long nxny = static_cast<long long>(lattice.dims.nx) * lattice.dims.ny;
    forceReductionKernel<<<gridFor(ncells), kBlock, 0, stream>>>(
        lattice.f, lattice.flags, ncells, lattice.dims.paddedCellCount(), nxny,
        lattice.dims.nx, acc.d_force);
    return cudaGetLastError();
}

} // namespace foilcfd
