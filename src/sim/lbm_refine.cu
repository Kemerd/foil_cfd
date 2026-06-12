// Two-level coupling kernels (plan M-refine): coarse-to-fine interface fill
// (trilinear space + two-level time interpolation of populations, with the
// Dupuis-Chopard non-equilibrium rescale), fine-to-coarse restriction
// (8-child average, inverse rescale, optional coarse macro write), and the
// macroscopic trilinear upsample for the mesh-sequencing presolve.
//
// Every kernel here is a COLD-path or once-per-step pass over a shell or a
// sub-box — none of them touch the fused stream-collide hot loop, which runs
// unmodified on both levels (it is fully dims/flag-parametric).
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include "lbm_refine.cuh"

namespace foilcfd {

namespace {

// Device copies of the frozen D3Q19 constants (host constexpr arrays are not
// visible to device code; values MUST match lbm_core.cuh exactly).
__constant__ int   rf_cx[kQ] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0};
__constant__ int   rf_cy[kQ] = {0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1};
__constant__ int   rf_cz[kQ] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1};
__constant__ float rf_w[kQ] = {
    1.0f / 3.0f,
    1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f};

constexpr int kBlock = 256;

constexpr std::uint8_t kFlagFluid     = 0;
constexpr std::uint8_t kFlagSolid     = 1;
constexpr std::uint8_t kFlagInterface = 8;

/// @brief Padded linear index of lattice site (x, y, z) with z in [-1, nz]
/// (identical layout contract to lbm_core.cu).
__device__ __forceinline__ long long pIdx(int x, int y, int z, int nx,
                                          long long nxny) {
    return static_cast<long long>(x) + static_cast<long long>(nx) * y
         + nxny * (z + 1);
}

/// @brief D3Q19 second-order equilibrium for all 19 directions.
__device__ __forceinline__ void equilibrium19(float rho, float ux, float uy,
                                              float uz, float feq[kQ]) {
    const float usq = 1.5f * (ux * ux + uy * uy + uz * uz);
#pragma unroll
    for (int q = 0; q < kQ; ++q) {
        const float cu = 3.0f * (rf_cx[q] * ux + rf_cy[q] * uy + rf_cz[q] * uz);
        feq[q] = rf_w[q] * rho * (1.0f + cu + 0.5f * cu * cu - usq);
    }
}

// Lattice sound speed (D3Q19): cs^2 = 1/3, so the Hermite normalization in the
// regularization below is w_q / (2*cs^4) = w_q / (2/9) = 4.5 * w_q.
constexpr float kCs2     = 1.0f / 3.0f;
constexpr float kRegNorm = 4.5f; // 1 / (2 * cs^4)

/// @brief Latt-Chopard regularization of a non-equilibrium population set.
///
/// The interpolated/averaged deviation @p fneqIn carries both the hydrodynamic
/// stress AND the non-hydrodynamic ("ghost") moments. At a grid-refinement
/// interface those ghost moments are exactly the spurious-vortex / speckle
/// signature (aliasing energy dumped into modes the coarse grid cannot carry;
/// Astoul et al. 2020). Projecting fneq onto the second-order Hermite (stress)
/// space discards them while preserving mass and momentum exactly:
///   Pi_neq_ab = sum_q c_qa c_qb fneq_q
///   fneq_q    = (w_q / 2 cs^4) * (c_qa c_qb - cs^2 d_ab) : Pi_neq_ab
/// (Latt & Chopard 2006, Eq. 10). The result replaces @p fneqIn in place.
__device__ __forceinline__ void regularizeFneq(float fneq[kQ]) {
    // Second-order non-equilibrium momentum-flux tensor (symmetric: 6 terms).
    float pxx = 0.0f, pyy = 0.0f, pzz = 0.0f;
    float pxy = 0.0f, pxz = 0.0f, pyz = 0.0f;
#pragma unroll
    for (int q = 0; q < kQ; ++q) {
        const float f = fneq[q];
        const float cx = static_cast<float>(rf_cx[q]);
        const float cy = static_cast<float>(rf_cy[q]);
        const float cz = static_cast<float>(rf_cz[q]);
        pxx += cx * cx * f; pyy += cy * cy * f; pzz += cz * cz * f;
        pxy += cx * cy * f; pxz += cx * cz * f; pyz += cy * cz * f;
    }
    // Project back: fneq_q = w_q/(2cs^4) * Q_q:Pi, Q_q = c_q c_q - cs^2 I.
    // The diagonal trace term carries the -cs^2 I contraction once.
    const float trace = kCs2 * (pxx + pyy + pzz);
#pragma unroll
    for (int q = 0; q < kQ; ++q) {
        const float cx = static_cast<float>(rf_cx[q]);
        const float cy = static_cast<float>(rf_cy[q]);
        const float cz = static_cast<float>(rf_cz[q]);
        // Q_q : Pi = (c c):Pi - cs^2 tr(Pi); off-diagonals counted twice.
        const float qPi = cx * cx * pxx + cy * cy * pyy + cz * cz * pzz
                        + 2.0f * (cx * cy * pxy + cx * cz * pxz + cy * cz * pyz)
                        - trace;
        fneq[q] = kRegNorm * rf_w[q] * qPi;
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

// ===========================================================================
// Coarse-to-fine fill.
//
// A fine cell (xf, yf, zf) has its center at coarse coordinates
//   xc = box.x0 + (xf + 0.5)/2,   yc = box.y0 + (yf + 0.5)/2,
//   zc = (zf + 0.5)/2
// The 19 populations are sampled there by trilinear interpolation over the
// 8 surrounding coarse CELL CENTERS (grid coordinate g = c - 0.5), blended
// between the two coarse time levels, then re-split into feq + fneq via the
// interpolated moments so the non-equilibrium part can be rescaled to the
// fine level. The patch clearance contract (>= 3 coarse cells of fluid
// between the shell and any domain face or solid) keeps the x/y stencil in
// range and solid-free; the z stencil rides the coarse ghost planes.
// ===========================================================================

__global__ void coarseToFineFillKernel(
    const FPop* __restrict__ fcT0, const FPop* __restrict__ fcT1,
    const std::uint8_t* __restrict__ coarseFlags,
    const std::uint8_t* __restrict__ fineFlags, FPop* __restrict__ ff,
    long long fineNcells, long long fineNxny, long long fineNcellsPad,
    int fineNx, int fineNy,
    long long coarseNxny, long long coarseNcellsPad, int coarseNx, int coarseNz,
    int boxX0, int boxY0, float invFactor,
    float timeWeight, float neqScale, int fullVolume) {
    const long long cell =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (cell >= fineNcells) return;

    int xf, yf, zf;
    unpackCell(cell, fineNx, fineNxny, xf, yf, zf);
    const long long pf = cell + fineNxny; // fine padded index of this cell

    // Target selection: shell-only in the per-step path; everything that is
    // not solid in the seeding path.
    const std::uint8_t flag = fineFlags[pf];
    if (fullVolume) {
        if (flag == kFlagSolid) return;
    } else {
        if (flag != kFlagInterface) return;
    }

    // ---- locate in coarse cell-center grid coordinates (factor-generic) --
    const float gx = static_cast<float>(boxX0)
                   + (static_cast<float>(xf) + 0.5f) * invFactor - 0.5f;
    const float gy = static_cast<float>(boxY0)
                   + (static_cast<float>(yf) + 0.5f) * invFactor - 0.5f;
    const float gz = (static_cast<float>(zf) + 0.5f) * invFactor - 0.5f;

    const int ix = static_cast<int>(floorf(gx));
    const int iy = static_cast<int>(floorf(gy));
    const int iz = static_cast<int>(floorf(gz)); // may be -1: ghost plane
    const float tx = gx - static_cast<float>(ix);
    const float ty = gy - static_cast<float>(iy);
    const float tz = gz - static_cast<float>(iz);

    // Trilinear corner weights, written once — reused for all 19 q slices.
    float wgt[8];
    wgt[0] = (1 - tx) * (1 - ty) * (1 - tz);
    wgt[1] = tx * (1 - ty) * (1 - tz);
    wgt[2] = (1 - tx) * ty * (1 - tz);
    wgt[3] = tx * ty * (1 - tz);
    wgt[4] = (1 - tx) * (1 - ty) * tz;
    wgt[5] = tx * (1 - ty) * tz;
    wgt[6] = (1 - tx) * ty * tz;
    wgt[7] = tx * ty * tz;

    // Padded indices of the 8 coarse corners. iz in [-1, nz-1] and iz+1 in
    // [0, nz] are both inside the padded buffer (ghost planes), so no wrap
    // arithmetic is needed anywhere.
    long long corner[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        const int cx = ix + (c & 1);
        const int cy = iy + ((c >> 1) & 1);
        const int cz = iz + ((c >> 2) & 1);
        corner[c] = pIdx(cx, cy, cz, coarseNx, coarseNxny);
    }

    // Solid-aware reweighting: a corner that lands INSIDE a parent solid (a VG
    // vane, or a stair-step of the foil) holds bounce-back state, NOT a valid
    // fluid distribution — interpolating through it injects garbage fneq that a
    // near-floor-tau child grid amplifies into runaway. This bites the nested
    // VG patch, whose shell hugs the vanes; the foil-bbox patch was solid-free
    // by its clearance margin and is unaffected. We drop solid corners and
    // renormalize the weights over the fluid corners so the shell is filled
    // from fluid neighbours only — the vane itself stays SOLID and is resolved
    // by the child grid's own bounce-back, exactly as intended.
    float wsum = 0.0f;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        if (coarseFlags[corner[c]] == kFlagSolid) wgt[c] = 0.0f;
        wsum += wgt[c];
    }
    // All 8 corners solid (the target sits deep inside a parent solid): nothing
    // fluid to pull from — leave the cell's current populations untouched. For
    // a real shell/fluid target at least one fluid corner always exists.
    if (wsum < 1e-6f) return;
    const float invWsum = 1.0f / wsum;
#pragma unroll
    for (int c = 0; c < 8; ++c) wgt[c] *= invWsum;

    // ---- interpolate the 19 populations (space, then time) ---------------
    float fi[kQ];
    float rho = 0.0f, jx = 0.0f, jy = 0.0f, jz = 0.0f;
#pragma unroll
    for (int q = 0; q < kQ; ++q) {
        const long long qBase = static_cast<long long>(q) * coarseNcellsPad;
        float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            s0 += wgt[c] * fcT0[qBase + corner[c]];
            s1 += wgt[c] * fcT1[qBase + corner[c]];
        }
        const float fq = s0 + timeWeight * (s1 - s0);
        fi[q] = fq;
        rho += fq;
        jx += rf_cx[q] * fq;
        jy += rf_cy[q] * fq;
        jz += rf_cz[q] * fq;
    }
    rho = fmaxf(rho, 0.05f); // same floor as the collision kernel
    const float invRho = 1.0f / rho;
    float ux = jx * invRho, uy = jy * invRho, uz = jz * invRho;

    // Same velocity-validity cap as the fused collision kernel: in deep-stall
    // transients an interpolated state can momentarily exceed the equilibrium
    // expansion's validity bound, and a near-floor-tau fine grid amplifies
    // that into runaway within a few sub-steps. Direction-preserving rescale.
    const float u2 = ux * ux + uy * uy + uz * uz;
    if (u2 > kMaxSimSpeed * kMaxSimSpeed) {
        const float sc = kMaxSimSpeed * rsqrtf(u2);
        ux *= sc; uy *= sc; uz *= sc;
    }

    // ---- split feq + fneq, REGULARIZE, rescale, store ---------------------
    // Raw (fi - feq) carries non-hydrodynamic ghost moments that the resolution
    // jump turns into the interface speckle. Project fneq onto the stress
    // tensor (Latt-Chopard) so only the hydrodynamic part crosses the
    // interface, THEN apply the Dupuis-Chopard tau rescale. Mass and momentum
    // are untouched (the projection is traceless in the conserved moments).
    float feq[kQ], fneq[kQ];
    equilibrium19(rho, ux, uy, uz, feq);
#pragma unroll
    for (int q = 0; q < kQ; ++q) fneq[q] = fi[q] - feq[q];
    regularizeFneq(fneq);
#pragma unroll
    for (int q = 0; q < kQ; ++q) {
        const float v = feq[q] + neqScale * fneq[q];
        ff[static_cast<long long>(q) * fineNcellsPad + pf] = v;
    }
}

// ===========================================================================
// Fine-to-coarse restriction. One thread per coarse cell of the interior
// sub-box; averages the populations of its 8 fine children (Solid children
// near stair-step walls excluded), applies the inverse fneq rescale, and
// overwrites the coarse post-collision value. Optionally also writes the
// coarse macroscopic arrays so every macro consumer (renderer, particles,
// delta99 extraction) sees the fine-derived solution with zero code changes.
// ===========================================================================

/// @brief Child-averaged non-equilibrium populations of ONE coarse cell, for
/// the Lagrava interface filter. Averages the m^3 fine children (skipping solid
/// children), derives rho/u, and returns fneq = childAvg - feq(rho,u) plus the
/// cell's own rho/u/velocity-capped state. Returns false when the cell has no
/// fluid children (fully solid) so the caller can drop it from the stencil.
__device__ __forceinline__ bool childAvgFneq(
    const FPop* __restrict__ ff, const std::uint8_t* __restrict__ fineFlags,
    long long fineNxny, long long fineNcellsPad, int fineNx,
    int fxc, int fyc, int fzc, int factor, float fneqOut[kQ],
    float& rhoOut, float& uxOut, float& uyOut, float& uzOut) {
    float fi[kQ];
#pragma unroll
    for (int q = 0; q < kQ; ++q) fi[q] = 0.0f;
    int nFluid = 0;
    for (int k = 0; k < factor; ++k)
        for (int j = 0; j < factor; ++j)
            for (int i = 0; i < factor; ++i) {
                const long long pfc =
                    pIdx(fxc + i, fyc + j, fzc + k, fineNx, fineNxny);
                if (fineFlags[pfc] == kFlagSolid) continue;
                ++nFluid;
#pragma unroll
                for (int q = 0; q < kQ; ++q)
                    fi[q] += ff[static_cast<long long>(q) * fineNcellsPad + pfc];
            }
    if (nFluid == 0) return false;

    const float invN = 1.0f / static_cast<float>(nFluid);
    float rho = 0.0f, jx = 0.0f, jy = 0.0f, jz = 0.0f;
#pragma unroll
    for (int q = 0; q < kQ; ++q) {
        fi[q] *= invN;
        rho += fi[q]; jx += rf_cx[q] * fi[q];
        jy += rf_cy[q] * fi[q]; jz += rf_cz[q] * fi[q];
    }
    rho = fmaxf(rho, 0.05f);
    const float invRho = 1.0f / rho;
    float ux = jx * invRho, uy = jy * invRho, uz = jz * invRho;
    const float u2 = ux * ux + uy * uy + uz * uz;
    if (u2 > kMaxSimSpeed * kMaxSimSpeed) {
        const float sc = kMaxSimSpeed * rsqrtf(u2);
        ux *= sc; uy *= sc; uz *= sc;
    }
    float feq[kQ];
    equilibrium19(rho, ux, uy, uz, feq);
#pragma unroll
    for (int q = 0; q < kQ; ++q) fneqOut[q] = fi[q] - feq[q];
    rhoOut = rho; uxOut = ux; uyOut = uy; uzOut = uz;
    return true;
}

__global__ void fineToCoarseRestrictKernel(
    const FPop* __restrict__ ff, const std::uint8_t* __restrict__ fineFlags,
    FPop* __restrict__ fc, const std::uint8_t* __restrict__ coarseFlags,
    long long fineNxny, long long fineNcellsPad, int fineNx,
    long long coarseNxny, long long coarseNcellsPad, int coarseNx, int coarseNz,
    int rx0, int ry0, int rx1, int ry1,   // restricted coarse sub-box
    int boxX0, int boxY0, int factor,
    float neqScaleInv,
    float* __restrict__ macroRho, float* __restrict__ macroU,
    float* __restrict__ macroV, float* __restrict__ macroW, int writeMacro) {
    const int rw = rx1 - rx0;
    const int rh = ry1 - ry0;
    const long long total = static_cast<long long>(rw) * rh * coarseNz;
    const long long t =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= total) return;

    // Decompose the sub-box-local index into coarse coordinates.
    const int zc = static_cast<int>(t / (static_cast<long long>(rw) * rh));
    const long long rem = t - static_cast<long long>(zc) * rw * rh;
    const int yc = ry0 + static_cast<int>(rem / rw);
    const int xc = rx0 + static_cast<int>(rem - static_cast<long long>(rem / rw) * rw);

    const long long cCell = static_cast<long long>(xc)
                          + static_cast<long long>(coarseNx) * yc
                          + coarseNxny * zc;
    const long long pc = cCell + coarseNxny;
    if (coarseFlags[pc] != kFlagFluid) return; // solids/markers keep their state

    // Edge blend weight: 0 at the band edge ramping to 1 over
    // kRestrictBlendCoarse cells inward. A hard hand-off leaves a velocity
    // kink the Q-criterion renders as a spurious vortex sheet standing on
    // the patch faces; the ramp hands the levels over smoothly instead.
    const int dEdge = min(min(xc - rx0, rx1 - 1 - xc),
                          min(yc - ry0, ry1 - 1 - yc));
    const float blend = fminf(1.0f,
        static_cast<float>(dEdge + 1)
            / static_cast<float>(kRestrictBlendCoarse + 1));

    // ---- center cell: child-averaged fneq + its own rho/u ----------------
    // Stair-step walls differ slightly between levels; childAvgFneq averages
    // over the fluid children only (skipping fine-solid children at the wall).
    const int fx = factor * (xc - boxX0);
    const int fy = factor * (yc - boxY0);
    const int fz = factor * zc;
    float fneqC[kQ], rho, ux, uy, uz;
    if (!childAvgFneq(ff, fineFlags, fineNxny, fineNcellsPad, fineNx,
                      fx, fy, fz, factor, fneqC, rho, ux, uy, uz))
        return; // fully solid at the fine level: leave coarse value

    // ---- Lagrava 1/19 box filter on fneq (Lagrava et al. 2012, Eq. 33) ----
    // The small fine-grid structures otherwise pollute the coarse grid and
    // trigger the high-ratio instability. Average fneq over this node and its
    // 18 lattice neighbours (fneq ONLY — never rho/u, which would over-dissipate
    // and inflate the interface viscosity). Neighbours outside the patch or with
    // no fluid children simply drop from the average. q=0 is the node itself.
    float fneqF[kQ];
#pragma unroll
    for (int q = 0; q < kQ; ++q) fneqF[q] = fneqC[q];
    int nStencil = 1;
    for (int n = 1; n < kQ; ++n) {
        const int nxc = xc + rf_cx[n];
        const int nyc = yc + rf_cy[n];
        const int nzc = zc + rf_cz[n];
        // Stay inside the restricted sub-box in x/y; z wraps periodically.
        if (nxc < rx0 || nxc >= rx1 || nyc < ry0 || nyc >= ry1) continue;
        const int nzw = (nzc + coarseNz) % coarseNz;
        float fneqN[kQ], rn, un, vn, wn;
        if (!childAvgFneq(ff, fineFlags, fineNxny, fineNcellsPad, fineNx,
                          factor * (nxc - boxX0), factor * (nyc - boxY0),
                          factor * nzw, factor, fneqN, rn, un, vn, wn))
            continue;
#pragma unroll
        for (int q = 0; q < kQ; ++q) fneqF[q] += fneqN[q];
        ++nStencil;
    }
    const float invS = 1.0f / static_cast<float>(nStencil);
#pragma unroll
    for (int q = 0; q < kQ; ++q) fneqF[q] *= invS;

    // ---- regularize the filtered fneq (Latt-Chopard), then inverse-rescale
    // and edge-blend with the coarse-evolved state ------------------------
    regularizeFneq(fneqF);
    float feq[kQ];
    equilibrium19(rho, ux, uy, uz, feq);
    float br = 0.0f, bjx = 0.0f, bjy = 0.0f, bjz = 0.0f;
#pragma unroll
    for (int q = 0; q < kQ; ++q) {
        const long long idx = static_cast<long long>(q) * coarseNcellsPad + pc;
        const float restricted = feq[q] + neqScaleInv * fneqF[q];
        const float existing   = fc[idx]; // coarse post-collision value
        const float blended    = existing + blend * (restricted - existing);
        fc[idx] = blended;
        // Macro moments of the BLENDED state, so the displayed field matches
        // the populations exactly (moments are linear in f).
        br  += blended;
        bjx += rf_cx[q] * blended;
        bjy += rf_cy[q] * blended;
        bjz += rf_cz[q] * blended;
    }

    if (writeMacro) {
        br = fmaxf(br, 0.05f);
        const float bInv = 1.0f / br;
        macroRho[cCell] = br;
        macroU[cCell] = bjx * bInv;
        macroV[cCell] = bjy * bInv;
        macroW[cCell] = bjz * bInv;
    }
}

// ===========================================================================
// Macroscopic trilinear upsample (presolve seeding). One thread per
// destination cell; x/y sampling clamps at the borders, z wraps (periodic
// span). Arrays are unpadded, so the z wrap is explicit here.
// ===========================================================================

__global__ void upsampleMacroKernel(
    const float* __restrict__ sRho, const float* __restrict__ sU,
    const float* __restrict__ sV, const float* __restrict__ sW,
    int snx, int sny, int snz,
    float* __restrict__ dRho, float* __restrict__ dU,
    float* __restrict__ dV, float* __restrict__ dW,
    long long dNcells, long long dNxny, int dnx, int dny, int dnz) {
    const long long cell =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (cell >= dNcells) return;

    int x, y, z;
    unpackCell(cell, dnx, dNxny, x, y, z);

    // Map the destination cell center into source cell-center coordinates.
    const float gx = (static_cast<float>(x) + 0.5f) * snx / dnx - 0.5f;
    const float gy = (static_cast<float>(y) + 0.5f) * sny / dny - 0.5f;
    const float gz = (static_cast<float>(z) + 0.5f) * snz / dnz - 0.5f;

    int ix = static_cast<int>(floorf(gx));
    int iy = static_cast<int>(floorf(gy));
    const int iz = static_cast<int>(floorf(gz));
    float tx = gx - static_cast<float>(ix);
    float ty = gy - static_cast<float>(iy);
    const float tz = gz - static_cast<float>(iz);

    // Clamp x/y to the border sample (collapses to nearest at the edges).
    if (ix < 0) { ix = 0; tx = 0.0f; }
    if (iy < 0) { iy = 0; ty = 0.0f; }
    int ix1 = ix + 1, iy1 = iy + 1;
    if (ix1 > snx - 1) { ix1 = snx - 1; tx = 0.0f; }
    if (iy1 > sny - 1) { iy1 = sny - 1; ty = 0.0f; }

    // z wraps periodically (unpadded arrays — explicit modulo, cold path).
    const int iz0 = ((iz % snz) + snz) % snz;
    const int iz1 = (iz0 + 1) % snz;

    const long long sNxny = static_cast<long long>(snx) * sny;
    const long long c000 = ix  + static_cast<long long>(snx) * iy  + sNxny * iz0;
    const long long c100 = ix1 + static_cast<long long>(snx) * iy  + sNxny * iz0;
    const long long c010 = ix  + static_cast<long long>(snx) * iy1 + sNxny * iz0;
    const long long c110 = ix1 + static_cast<long long>(snx) * iy1 + sNxny * iz0;
    const long long c001 = ix  + static_cast<long long>(snx) * iy  + sNxny * iz1;
    const long long c101 = ix1 + static_cast<long long>(snx) * iy  + sNxny * iz1;
    const long long c011 = ix  + static_cast<long long>(snx) * iy1 + sNxny * iz1;
    const long long c111 = ix1 + static_cast<long long>(snx) * iy1 + sNxny * iz1;

    const float w000 = (1 - tx) * (1 - ty) * (1 - tz);
    const float w100 = tx * (1 - ty) * (1 - tz);
    const float w010 = (1 - tx) * ty * (1 - tz);
    const float w110 = tx * ty * (1 - tz);
    const float w001 = (1 - tx) * (1 - ty) * tz;
    const float w101 = tx * (1 - ty) * tz;
    const float w011 = (1 - tx) * ty * tz;
    const float w111 = tx * ty * tz;

    auto lerp8 = [&](const float* __restrict__ s) {
        return w000 * s[c000] + w100 * s[c100] + w010 * s[c010] + w110 * s[c110]
             + w001 * s[c001] + w101 * s[c101] + w011 * s[c011] + w111 * s[c111];
    };
    dRho[cell] = lerp8(sRho);
    dU[cell]   = lerp8(sU);
    dV[cell]   = lerp8(sV);
    dW[cell]   = lerp8(sW);
}

/// @brief 1D grid size for @p n threads at the standard block size.
inline unsigned int gridFor(long long n) {
    return static_cast<unsigned int>((n + kBlock - 1) / kBlock);
}

} // namespace

// ===========================================================================
// Launch wrappers.
// ===========================================================================

cudaError_t launchCoarseToFineFill(DeviceLatticeView coarseT0,
                                   const FPop* coarseT1F,
                                   DeviceLatticeView fine,
                                   PatchBox box, int factor, float timeWeight,
                                   float tauCoarse, float tauFine,
                                   bool fullVolume, cudaStream_t stream) {
    const long long fineNcells = fine.dims.cellCount();
    if (fineNcells <= 0 || !coarseT0.f || !coarseT0.flags || !coarseT1F
        || !fine.f || !fine.flags
        || !box.valid() || factor < 2 || factor > kMaxRefineFactor)
        return cudaErrorInvalidValue;

    const long long fineNxny =
        static_cast<long long>(fine.dims.nx) * fine.dims.ny;
    const long long coarseNxny =
        static_cast<long long>(coarseT0.dims.nx) * coarseT0.dims.ny;

    // fneq rescale coarse -> fine: tau_f / (m * tau_c).
    const float neqScale = tauFine / (static_cast<float>(factor) * tauCoarse);

    coarseToFineFillKernel<<<gridFor(fineNcells), kBlock, 0, stream>>>(
        coarseT0.f, coarseT1F, coarseT0.flags, fine.flags, fine.f,
        fineNcells, fineNxny, fine.dims.paddedCellCount(),
        fine.dims.nx, fine.dims.ny,
        coarseNxny, coarseT0.dims.paddedCellCount(), coarseT0.dims.nx,
        coarseT0.dims.nz,
        box.x0, box.y0, 1.0f / static_cast<float>(factor),
        timeWeight, neqScale, fullVolume ? 1 : 0);
    return cudaGetLastError();
}

cudaError_t launchFineToCoarseRestrict(DeviceLatticeView fine,
                                       DeviceLatticeView coarse,
                                       PatchBox box, int factor,
                                       float tauCoarse, float tauFine,
                                       float* macroRho, float* macroU,
                                       float* macroV, float* macroW,
                                       cudaStream_t stream) {
    if (!fine.f || !fine.flags || !coarse.f || !coarse.flags || !box.valid()
        || factor < 2 || factor > kMaxRefineFactor)
        return cudaErrorInvalidValue;

    // Interior sub-box: skip the band feeding the next interface fill.
    const int rx0 = box.x0 + kRestrictBandCoarse;
    const int ry0 = box.y0 + kRestrictBandCoarse;
    const int rx1 = box.x1 - kRestrictBandCoarse;
    const int ry1 = box.y1 - kRestrictBandCoarse;
    if (rx1 <= rx0 || ry1 <= ry0) return cudaErrorInvalidValue;

    const long long total = static_cast<long long>(rx1 - rx0)
                          * (ry1 - ry0) * coarse.dims.nz;
    const long long fineNxny =
        static_cast<long long>(fine.dims.nx) * fine.dims.ny;
    const long long coarseNxny =
        static_cast<long long>(coarse.dims.nx) * coarse.dims.ny;

    // fneq rescale fine -> coarse: (m * tau_c) / tau_f, the exact inverse.
    const float neqScaleInv =
        (static_cast<float>(factor) * tauCoarse) / tauFine;
    const bool wantMacro = macroRho && macroU && macroV && macroW;

    fineToCoarseRestrictKernel<<<gridFor(total), kBlock, 0, stream>>>(
        fine.f, fine.flags, coarse.f, coarse.flags,
        fineNxny, fine.dims.paddedCellCount(), fine.dims.nx,
        coarseNxny, coarse.dims.paddedCellCount(), coarse.dims.nx,
        coarse.dims.nz,
        rx0, ry0, rx1, ry1, box.x0, box.y0, factor, neqScaleInv,
        macroRho, macroU, macroV, macroW, wantMacro ? 1 : 0);
    return cudaGetLastError();
}

cudaError_t launchUpsampleMacro(const float* srcRho, const float* srcU,
                                const float* srcV, const float* srcW,
                                GridDims srcDims,
                                float* dstRho, float* dstU,
                                float* dstV, float* dstW,
                                GridDims dstDims, cudaStream_t stream) {
    const long long dNcells = dstDims.cellCount();
    if (dNcells <= 0 || srcDims.cellCount() <= 0 || !srcRho || !dstRho)
        return cudaErrorInvalidValue;
    const long long dNxny = static_cast<long long>(dstDims.nx) * dstDims.ny;

    upsampleMacroKernel<<<gridFor(dNcells), kBlock, 0, stream>>>(
        srcRho, srcU, srcV, srcW, srcDims.nx, srcDims.ny, srcDims.nz,
        dstRho, dstU, dstV, dstW, dNcells, dNxny,
        dstDims.nx, dstDims.ny, dstDims.nz);
    return cudaGetLastError();
}

} // namespace foilcfd
