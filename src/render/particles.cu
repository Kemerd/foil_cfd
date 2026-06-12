// Particle advection kernels (RK2 over trilinear velocity), pool seeding,
// slice-plane field rasterization (surface-object writes), the Q-criterion
// volume fill, and the solver-facing launchAdvectParticles forwarder from
// sim/lbm_core.cuh. All GL specifics stay in viz.cpp — these kernels only see
// raw mapped pointers and surface objects.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include "particles.cuh"

namespace foilcfd {

namespace {

// ===========================================================================
// Small device utilities shared by every kernel in this TU.
// ===========================================================================

/// Linear cell index for the frozen x-fastest layout (lbm_core.cuh GridDims).
/// 64-bit on purpose: the default grid is 23.6M cells, so x + nx*(y + ny*z)
/// products overflow 32-bit on Fine/Ultra presets.
__device__ __forceinline__ long long cellIndex(int x, int y, int z,
                                               const GridDims& d) {
    return static_cast<long long>(x)
         + static_cast<long long>(d.nx)
               * (static_cast<long long>(y)
                  + static_cast<long long>(d.ny) * static_cast<long long>(z));
}

/// Integer clamp without the std:: machinery (device-friendly).
__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/// Component-wise lerp for float3 (used by the trilinear blend and palettes).
__device__ __forceinline__ float3 lerp3(float3 a, float3 b, float t) {
    return make_float3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t,
                       a.z + (b.z - a.z) * t);
}

/// Component-wise fused multiply-add for float3 (Horner palette evaluation).
__device__ __forceinline__ float3 fma3(float3 a, float t, float3 b) {
    return make_float3(a.x * t + b.x, a.y * t + b.y, a.z * t + b.z);
}

/// Lowbias32 integer hash (Chris Wellons) — excellent avalanche for the cost,
/// which matters when 1M threads each derive several decorrelated draws from
/// (seed ^ index).
__device__ __forceinline__ unsigned int hashU32(unsigned int x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

/// Stateful uniform draw in [0,1): advances the RNG state by re-hashing.
__device__ __forceinline__ float rand01(unsigned int& state) {
    state = hashU32(state);
    // 24 mantissa-safe bits -> exactly representable uniform floats.
    return static_cast<float>(state & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

/// Velocity at a cell (no interpolation) with clamped indices — the building
/// block for both the trilinear sample corners and the finite-difference
/// stencils below.
__device__ __forceinline__ float3 velocityAtCell(const DeviceVelocityField& vel,
                                                 int x, int y, int z) {
    const long long c = cellIndex(clampi(x, 0, vel.dims.nx - 1),
                                  clampi(y, 0, vel.dims.ny - 1),
                                  clampi(z, 0, vel.dims.nz - 1), vel.dims);
    return make_float3(vel.u[c], vel.v[c], vel.w[c]);
}

/// Trilinear velocity sample at a continuous lattice-space position. Cell
/// centers sit at integer+0.5 (cell i spans [i, i+1)), so the interpolation
/// coordinate is p - 0.5; edges clamp, which freezes the gradient at the
/// boundary — fine for tracers that respawn there anyway.
__device__ float3 sampleVelocity(const DeviceVelocityField& vel, float3 p) {
    const float gx = fminf(fmaxf(p.x - 0.5f, 0.0f),
                           static_cast<float>(vel.dims.nx) - 1.0001f);
    const float gy = fminf(fmaxf(p.y - 0.5f, 0.0f),
                           static_cast<float>(vel.dims.ny) - 1.0001f);
    const float gz = fminf(fmaxf(p.z - 0.5f, 0.0f),
                           static_cast<float>(vel.dims.nz) - 1.0001f);
    const int x0 = static_cast<int>(gx);
    const int y0 = static_cast<int>(gy);
    const int z0 = static_cast<int>(gz);
    const float fx = gx - static_cast<float>(x0);
    const float fy = gy - static_cast<float>(y0);
    const float fz = gz - static_cast<float>(z0);

    // Eight-corner gather; the planar SoA arrays give coalesced-ish reads for
    // particles that cluster in cells (streaks do exactly that).
    const float3 c000 = velocityAtCell(vel, x0,     y0,     z0);
    const float3 c100 = velocityAtCell(vel, x0 + 1, y0,     z0);
    const float3 c010 = velocityAtCell(vel, x0,     y0 + 1, z0);
    const float3 c110 = velocityAtCell(vel, x0 + 1, y0 + 1, z0);
    const float3 c001 = velocityAtCell(vel, x0,     y0,     z0 + 1);
    const float3 c101 = velocityAtCell(vel, x0 + 1, y0,     z0 + 1);
    const float3 c011 = velocityAtCell(vel, x0,     y0 + 1, z0 + 1);
    const float3 c111 = velocityAtCell(vel, x0 + 1, y0 + 1, z0 + 1);

    // Standard trilinear blend, fused per component with lerp identities.
    const float3 x00 = lerp3(c000, c100, fx);
    const float3 x10 = lerp3(c010, c110, fx);
    const float3 x01 = lerp3(c001, c101, fx);
    const float3 x11 = lerp3(c011, c111, fx);
    const float3 y0v = lerp3(x00, x10, fy);
    const float3 y1v = lerp3(x01, x11, fy);
    return lerp3(y0v, y1v, fz);
}

/// Full vorticity vector at a cell via clamped central differences:
/// omega = curl(u) = (dw/dy - dv/dz, du/dz - dw/dx, dv/dx - du/dy).
/// Lattice spacing is 1, so the central difference is just 0.5 * delta.
__device__ float3 vorticityAtCell(const DeviceVelocityField& vel,
                                  int x, int y, int z) {
    const float3 xp = velocityAtCell(vel, x + 1, y, z);
    const float3 xm = velocityAtCell(vel, x - 1, y, z);
    const float3 yp = velocityAtCell(vel, x, y + 1, z);
    const float3 ym = velocityAtCell(vel, x, y - 1, z);
    const float3 zp = velocityAtCell(vel, x, y, z + 1);
    const float3 zm = velocityAtCell(vel, x, y, z - 1);
    return make_float3(0.5f * ((yp.z - ym.z) - (zp.y - zm.y)),
                       0.5f * ((zp.x - zm.x) - (xp.z - xm.z)),
                       0.5f * ((xp.y - xm.y) - (yp.x - ym.x)));
}

/// True when any component is NaN/Inf — the solver may hand us a field mid-
/// blowup (the watchdog only samples every 200 steps), and tracer positions
/// must never go non-finite or they poison the VBO until a reseed.
__device__ __forceinline__ bool finite3(float3 v) {
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

// ===========================================================================
// Colormaps. Plan 9.1 bans rainbow/jet: viridis (sequential, perceptually
// uniform) + coolwarm (diverging) only. The viridis fit is Matt Zucker's
// public-domain degree-6 polynomial (shadertoy WlfXRN) — max error well under
// 1% across the ramp, far cheaper than a LUT texture for these kernels.
// ===========================================================================

__device__ float3 viridisColor(float t) {
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    const float3 c0 = make_float3( 0.2777273f,  0.0054073f,  0.3340998f);
    const float3 c1 = make_float3( 0.1050930f,  1.4046135f,  1.3845902f);
    const float3 c2 = make_float3(-0.3308618f,  0.2148476f,  0.0950952f);
    const float3 c3 = make_float3(-4.6342305f, -5.7991010f, -19.3324410f);
    const float3 c4 = make_float3( 6.2282699f, 14.1799334f,  56.6905526f);
    const float3 c5 = make_float3( 4.7763850f, -13.7451454f, -65.3530326f);
    const float3 c6 = make_float3(-5.4354559f,  4.6458526f,  26.3124352f);
    // Horner evaluation, vectorized by hand over the three channels.
    float3 r = c6;
    r = fma3(r, t, c5);
    r = fma3(r, t, c4);
    r = fma3(r, t, c3);
    r = fma3(r, t, c2);
    r = fma3(r, t, c1);
    r = fma3(r, t, c0);
    return make_float3(fminf(fmaxf(r.x, 0.0f), 1.0f),
                       fminf(fmaxf(r.y, 0.0f), 1.0f),
                       fminf(fmaxf(r.z, 0.0f), 1.0f));
}

/// Two-segment approximation of Moreland's coolwarm diverging map: cool blue
/// through near-white to warm red. Smoothstep on each half keeps the midpoint
/// from forming a visible crease.
__device__ float3 coolwarmColor(float t) {
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    const float3 blue  = make_float3(0.230f, 0.299f, 0.754f);
    const float3 white = make_float3(0.865f, 0.865f, 0.865f);
    const float3 red   = make_float3(0.706f, 0.016f, 0.150f);
    if (t < 0.5f) {
        const float s = t * 2.0f;
        return lerp3(blue, white, s * s * (3.0f - 2.0f * s));
    }
    const float s = (t - 0.5f) * 2.0f;
    return lerp3(white, red, s * s * (3.0f - 2.0f * s));
}

// ===========================================================================
// Respawn helpers (plan 9.1: inlet respawn rate-balanced + volume re-seeding).
// ===========================================================================

/// Fresh particle at the inlet plane: just downstream of the x=0 inlet column,
/// uniformly spread over the (y,z) face, with a small streamwise jitter band
/// so a whole frame's worth of respawns doesn't form a visible sheet.
__device__ float3 inletSpawn(const GridDims& d, unsigned int& rng) {
    return make_float3(0.5f + 1.5f * rand01(rng),
                       static_cast<float>(d.ny) * rand01(rng),
                       static_cast<float>(d.nz) * rand01(rng));
}

/// Fresh particle anywhere in the fluid: uniform over the box with a few
/// retries to dodge SOLID cells; falls back to the inlet when unlucky (deep
/// inside the foil every retry — rare, and the inlet is always fluid).
__device__ float3 volumeSpawn(const GridDims& d, const std::uint8_t* flags,
                              unsigned int& rng) {
    for (int attempt = 0; attempt < 4; ++attempt) {
        const float3 p = make_float3(static_cast<float>(d.nx) * rand01(rng),
                                     static_cast<float>(d.ny) * rand01(rng),
                                     static_cast<float>(d.nz) * rand01(rng));
        if (!flags) return p;
        const long long c = cellIndex(clampi(static_cast<int>(p.x), 0, d.nx - 1),
                                      clampi(static_cast<int>(p.y), 0, d.ny - 1),
                                      clampi(static_cast<int>(p.z), 0, d.nz - 1), d);
        if (flags[c] != static_cast<std::uint8_t>(CellFlag::Solid)) return p;
    }
    return inletSpawn(d, rng);
}

// ===========================================================================
// Kernels
// ===========================================================================

/// Hero-mode advection: RK2 midpoint step, death classification, respawn,
/// age fade, color scalar. One thread per particle.
__global__ void advectKernel(float4* positions, float* colorKeys, int count,
                             DeviceVelocityField vel, const std::uint8_t* flags,
                             ParticleAdvectParams prm) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    // Decorrelate threads: golden-ratio stride over the index, mixed with the
    // per-frame seed, then hashed. Each rand01() advances the state further.
    unsigned int rng = hashU32(prm.seed ^ (static_cast<unsigned int>(i) * 0x9E3779B9u));

    const float4 packed = positions[i];
    float3 p   = make_float3(packed.x, packed.y, packed.z);
    float  age = packed.w;

    const GridDims d = vel.dims;
    const float nzf = static_cast<float>(d.nz);

    // --- RK2 midpoint advection (plan 9.1) -------------------------------
    // u1 = u(x); xm = x + dt/2 * u1; x' = x + dt * u(xm). Velocities are in
    // cells/step and dt is in steps, so positions stay in cell space.
    bool blewUp = !finite3(p);
    if (!blewUp && prm.dtSteps > 0.0f) {
        const float3 u1 = sampleVelocity(vel, p);
        float3 pm = make_float3(p.x + 0.5f * prm.dtSteps * u1.x,
                                p.y + 0.5f * prm.dtSteps * u1.y,
                                p.z + 0.5f * prm.dtSteps * u1.z);
        // Spanwise-periodic midpoint wrap keeps the second sample valid.
        if (pm.z < 0.0f) pm.z += nzf; else if (pm.z >= nzf) pm.z -= nzf;
        const float3 u2 = sampleVelocity(vel, pm);
        p = make_float3(p.x + prm.dtSteps * u2.x, p.y + prm.dtSteps * u2.y,
                        p.z + prm.dtSteps * u2.z);
        blewUp = !finite3(p) || !finite3(u2);
        // Aging is tied to advected time so pausing the sim freezes the fade.
        age -= prm.ageRate * prm.dtSteps;
    }

    // --- death classification ---------------------------------------------
    // z is periodic (plan 4.2): wrap, never kill.
    if (!blewUp) {
        if (p.z < 0.0f) p.z += nzf; else if (p.z >= nzf) p.z -= nzf;
    }
    const bool outflow = blewUp || p.x >= static_cast<float>(d.nx) - 1.0f;
    const bool oob = !outflow && (p.x < 0.0f || p.y < 0.0f
                                  || p.y >= static_cast<float>(d.ny));
    int cx = clampi(static_cast<int>(p.x), 0, d.nx - 1);
    int cy = clampi(static_cast<int>(p.y), 0, d.ny - 1);
    int cz = clampi(static_cast<int>(p.z), 0, d.nz - 1);
    const bool solid = !outflow && !oob && flags
                    && flags[cellIndex(cx, cy, cz, d)]
                           == static_cast<std::uint8_t>(CellFlag::Solid);

    if (outflow || oob || solid || age <= 0.0f) {
        // Rate balance (plan 9.1): every particle the flow carries OUT comes
        // back IN at the inlet, so inlet seeding exactly tracks outflow loss.
        // Lifetime/solid deaths split inlet vs volume by the bias knob so the
        // wake keeps tracers without starving the inlet sheet.
        const bool toInlet = outflow || oob
                          || rand01(rng) < prm.inletSpawnFraction;
        p = toInlet ? inletSpawn(d, rng) : volumeSpawn(d, flags, rng);
        // Newborn age jittered below 1 so one frame's respawns don't fade in
        // lockstep and create a strobing population wave.
        age = 0.6f + 0.4f * rand01(rng);
        cx = clampi(static_cast<int>(p.x), 0, d.nx - 1);
        cy = clampi(static_cast<int>(p.y), 0, d.ny - 1);
        cz = clampi(static_cast<int>(p.z), 0, d.nz - 1);
    }

    // --- color scalar -------------------------------------------------------
    // Normalized here (not in the shader) so the GLSL side stays a pure
    // palette lookup and both palettes share one [0,1] contract.
    if (colorKeys) {
        float value;
        if (prm.colorBy == 1) {
            const float3 w = vorticityAtCell(vel, cx, cy, cz);
            value = sqrtf(w.x * w.x + w.y * w.y + w.z * w.z);
        } else {
            const float3 u = sampleVelocity(vel, p);
            value = sqrtf(u.x * u.x + u.y * u.y + u.z * u.z);
        }
        if (!isfinite(value)) value = 0.0f;
        colorKeys[i] = fminf(value / fmaxf(prm.colorScale, 1e-6f), 1.0f);
    }

    positions[i] = make_float4(p.x, p.y, p.z, age);
}

/// Pool seeding: uniform positions over the whole box (flags are not
/// available at renderer init; particles seeded inside the foil die into a
/// respawn on the very first advect pass), ages randomized across the full
/// fade range so the population reaches steady state immediately.
__global__ void seedKernel(float4* positions, float* colorKeys, int count,
                           GridDims dims, unsigned int seed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned int rng = hashU32(seed ^ (static_cast<unsigned int>(i) * 0x9E3779B9u));
    positions[i] = make_float4(static_cast<float>(dims.nx) * rand01(rng),
                               static_cast<float>(dims.ny) * rand01(rng),
                               static_cast<float>(dims.nz) * rand01(rng),
                               rand01(rng));
    if (colorKeys) colorKeys[i] = 0.0f;
}

/// Slice rasterizer: one thread per texel; derives the field value at the
/// texel's cell, colormaps it, and writes RGBA8 through the surface object.
__global__ void sliceFillKernel(cudaSurfaceObject_t surface,
                                DeviceVelocityField vel, const float* rho,
                                const std::uint8_t* flags,
                                SliceFillParams prm) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y * blockDim.y + threadIdx.y;
    if (s >= prm.width || t >= prm.height) return;

    // Texel -> cell mapping (contract documented in particles.cuh; the quad
    // the renderer draws uses the same (s,t) -> world correspondence).
    int x, y, z;
    if (prm.axis == 0)      { x = prm.cell; y = t;        z = s; }
    else if (prm.axis == 1) { x = s;        y = prm.cell; z = t; }
    else                    { x = s;        y = t;        z = prm.cell; }

    const long long c = cellIndex(x, y, z, vel.dims);

    // Solid cells: flat dark slate so the foil cross-section reads as a
    // silhouette instead of palette noise from the zeroed wall velocity.
    if (flags && flags[c] == static_cast<std::uint8_t>(CellFlag::Solid)) {
        surf2Dwrite(make_uchar4(40, 42, 48, 255), surface,
                    s * static_cast<int>(sizeof(uchar4)), t);
        return;
    }

    // Field value -> normalized palette coordinate. Sequential fields map
    // [0, scale] -> [0,1]; signed fields map [-scale, +scale] -> [0,1] so the
    // palette midpoint is exactly zero (diverging-map convention, plan 9.1).
    float tNorm;
    if (prm.field == 1) {
        // Vorticity-z: omega_z = dv/dx - du/dy by central differences.
        const float3 xp = velocityAtCell(vel, x + 1, y, z);
        const float3 xm = velocityAtCell(vel, x - 1, y, z);
        const float3 yp = velocityAtCell(vel, x, y + 1, z);
        const float3 ym = velocityAtCell(vel, x, y - 1, z);
        const float omegaZ = 0.5f * ((xp.y - xm.y) - (yp.x - ym.x));
        tNorm = 0.5f + 0.5f * omegaZ / fmaxf(prm.scale, 1e-9f);
    } else if (prm.field == 2) {
        // Pressure deviation: p = cs^2 * (rho - 1) in lattice units.
        const float pDev = rho ? kCs2 * (rho[c] - 1.0f) : 0.0f;
        tNorm = 0.5f + 0.5f * pDev / fmaxf(prm.scale, 1e-9f);
    } else {
        // Speed magnitude.
        const float3 u = velocityAtCell(vel, x, y, z);
        tNorm = sqrtf(u.x * u.x + u.y * u.y + u.z * u.z)
              / fmaxf(prm.scale, 1e-9f);
    }
    if (!isfinite(tNorm)) tNorm = 0.0f;

    const float3 rgb = (prm.colormap == 1) ? coolwarmColor(tNorm)
                                           : viridisColor(tNorm);
    surf2Dwrite(make_uchar4(static_cast<unsigned char>(rgb.x * 255.0f + 0.5f),
                            static_cast<unsigned char>(rgb.y * 255.0f + 0.5f),
                            static_cast<unsigned char>(rgb.z * 255.0f + 0.5f),
                            255),
                surface, s * static_cast<int>(sizeof(uchar4)), t);
}

/// Q-criterion volume fill: one thread per cell. Q = 0.5*(|Omega|^2 - |S|^2)
/// where S/Omega are the symmetric/antisymmetric parts of the velocity
/// gradient tensor; positive Q marks rotation-dominated cores (the classic
/// vortex visualization). Output is normalized R8 for the raymarcher.
__global__ void qCriterionKernel(cudaSurfaceObject_t surface,
                                 DeviceVelocityField vel,
                                 const std::uint8_t* flags, float qScale) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const GridDims d = vel.dims;
    if (x >= d.nx || y >= d.ny || z >= d.nz) return;

    unsigned char out = 0;
    // Boundary ring and solids stay 0: the one-sided differences there are
    // garbage, and a lit domain shell would dominate the raymarch.
    const bool interior = x > 0 && x < d.nx - 1 && y > 0 && y < d.ny - 1
                       && z > 0 && z < d.nz - 1;
    const bool isSolid = flags && flags[cellIndex(x, y, z, d)]
                                      == static_cast<std::uint8_t>(CellFlag::Solid);
    if (interior && !isSolid) {
        // Velocity gradient tensor J_ij = du_i/dx_j via central differences.
        const float3 xp = velocityAtCell(vel, x + 1, y, z);
        const float3 xm = velocityAtCell(vel, x - 1, y, z);
        const float3 yp = velocityAtCell(vel, x, y + 1, z);
        const float3 ym = velocityAtCell(vel, x, y - 1, z);
        const float3 zp = velocityAtCell(vel, x, y, z + 1);
        const float3 zm = velocityAtCell(vel, x, y, z - 1);
        const float ux = 0.5f * (xp.x - xm.x), uy = 0.5f * (yp.x - ym.x), uz = 0.5f * (zp.x - zm.x);
        const float vx = 0.5f * (xp.y - xm.y), vy = 0.5f * (yp.y - ym.y), vz = 0.5f * (zp.y - zm.y);
        const float wx = 0.5f * (xp.z - xm.z), wy = 0.5f * (yp.z - ym.z), wz = 0.5f * (zp.z - zm.z);

        // ||S||^2 and ||Omega||^2 from the symmetric/antisymmetric split.
        // Diagonal terms belong to S only; off-diagonals split as
        // S_ij = (J_ij + J_ji)/2, O_ij = (J_ij - J_ji)/2 (each counted twice
        // in the Frobenius norm by symmetry).
        const float sxy = 0.5f * (uy + vx), sxz = 0.5f * (uz + wx), syz = 0.5f * (vz + wy);
        const float oxy = 0.5f * (uy - vx), oxz = 0.5f * (uz - wx), oyz = 0.5f * (vz - wy);
        const float s2 = ux * ux + vy * vy + wz * wz
                       + 2.0f * (sxy * sxy + sxz * sxz + syz * syz);
        const float o2 = 2.0f * (oxy * oxy + oxz * oxz + oyz * oyz);
        const float q = 0.5f * (o2 - s2);

        if (isfinite(q) && q > 0.0f) {
            const float tn = fminf(q / fmaxf(qScale, 1e-12f), 1.0f);
            out = static_cast<unsigned char>(tn * 255.0f + 0.5f);
        }
    }
    surf3Dwrite(out, surface, x * static_cast<int>(sizeof(unsigned char)), y, z);
}

} // namespace

// ===========================================================================
// Host launch wrappers
// ===========================================================================

cudaError_t launchParticleAdvectRK2(float4* positions, float* colorKeys,
                                    int count, DeviceVelocityField vel,
                                    const std::uint8_t* flags,
                                    const ParticleAdvectParams& params,
                                    cudaStream_t stream) {
    if (!positions || count <= 0 || !vel.u || !vel.v || !vel.w
        || vel.dims.cellCount() <= 0) {
        return cudaErrorInvalidValue;
    }
    const int block = 256;
    const int grid = (count + block - 1) / block;
    advectKernel<<<grid, block, 0, stream>>>(positions, colorKeys, count, vel,
                                             flags, params);
    return cudaGetLastError();
}

cudaError_t launchParticleSeed(float4* positions, float* colorKeys, int count,
                               GridDims dims, unsigned int seed,
                               cudaStream_t stream) {
    if (!positions || count <= 0 || dims.cellCount() <= 0) {
        return cudaErrorInvalidValue;
    }
    const int block = 256;
    const int grid = (count + block - 1) / block;
    seedKernel<<<grid, block, 0, stream>>>(positions, colorKeys, count, dims,
                                           seed);
    return cudaGetLastError();
}

cudaError_t launchSliceFill(cudaSurfaceObject_t surface, DeviceVelocityField vel,
                            const float* rho, const std::uint8_t* flags,
                            const SliceFillParams& params, cudaStream_t stream) {
    if (!surface || !vel.u || params.width <= 0 || params.height <= 0) {
        return cudaErrorInvalidValue;
    }
    const dim3 block(16, 16, 1);
    const dim3 grid((params.width + block.x - 1) / block.x,
                    (params.height + block.y - 1) / block.y, 1);
    sliceFillKernel<<<grid, block, 0, stream>>>(surface, vel, rho, flags, params);
    return cudaGetLastError();
}

cudaError_t launchQCriterionVolume(cudaSurfaceObject_t volumeSurface,
                                   DeviceVelocityField vel,
                                   const std::uint8_t* flags, float qScale,
                                   cudaStream_t stream) {
    if (!volumeSurface || !vel.u || vel.dims.cellCount() <= 0) {
        return cudaErrorInvalidValue;
    }
    const dim3 block(8, 8, 4);
    const dim3 grid((vel.dims.nx + block.x - 1) / block.x,
                    (vel.dims.ny + block.y - 1) / block.y,
                    (vel.dims.nz + block.z - 1) / block.z);
    qCriterionKernel<<<grid, block, 0, stream>>>(volumeSurface, vel, flags,
                                                 qScale);
    return cudaGetLastError();
}

// ===========================================================================
// Solver-facing entry point declared in sim/lbm_core.cuh: forwards into the
// same advection kernel so the sim module never includes render headers.
// ===========================================================================

cudaError_t launchAdvectParticles(float4* positions, int count,
                                  DeviceVelocityField vel,
                                  const std::uint8_t* flags, float dtSteps,
                                  unsigned int seed, cudaStream_t stream) {
    ParticleAdvectParams params;
    params.dtSteps = dtSteps;
    params.seed = seed;
    // No color VBO on the solver path: nullptr skips the color writes
    // (advection-only use by the sim module).
    return launchParticleAdvectRK2(positions, nullptr, count, vel, flags,
                                   params, stream);
}

} // namespace foilcfd
