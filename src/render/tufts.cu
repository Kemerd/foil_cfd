// Cotton-tuft advection kernel (render/tufts.cuh). A grid of short flexible
// strands taped to the suction surface; the local lattice velocity drags the
// free nodes downstream while a spring-to-rest + damping mini cloth sim keeps
// each strand coherent. Per strand we derive an "attachment" scalar from how
// far the relaxed strand departs from its surface tangent (reversal / strong
// cross-flow = separated) and write it to every node so the shader can paint
// the strand green (attached) -> red (separated) — the stall line, made legible.
//
// Velocity sampling reuses the SAME trilinear convention as the particle
// advector (positions in lattice cell space, cell centers at integer+0.5, z
// periodic). Kept self-contained here rather than sharing a header so the
// particle TU stays untouched.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include "tufts.cuh"

namespace foilcfd {

namespace {

// ===========================================================================
// Device math helpers (mirrors the small utilities in particles.cu; tufts are
// a separate TU so they carry their own copies rather than exporting them).
// ===========================================================================

/// Linear cell index for the frozen x-fastest GridDims layout. 64-bit because
/// the default grid exceeds 32-bit cell counts (same reasoning as particles.cu).
__device__ __forceinline__ long long cellIndex(int x, int y, int z,
                                               const GridDims& d) {
    return static_cast<long long>(x)
         + static_cast<long long>(d.nx)
               * (static_cast<long long>(y)
                  + static_cast<long long>(d.ny) * static_cast<long long>(z));
}

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ float3 add3(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ float3 sub3(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ float3 scale3(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ __forceinline__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ float len3(float3 a) { return sqrtf(dot3(a, a)); }

__device__ __forceinline__ float3 lerp3(float3 a, float3 b, float t) {
    return make_float3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t,
                       a.z + (b.z - a.z) * t);
}

/// Periodic wrap of an integer z index (the span is periodic — see particles.cu).
__device__ __forceinline__ int wrapZ(int z, int nz) {
    if (z < 0) z += nz;
    else if (z >= nz) z -= nz;
    return z;
}

__device__ __forceinline__ bool finite3(float3 v) {
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

/// Velocity at an integer cell: x/y clamp at the box (freezing the boundary
/// gradient is fine for tufts that live on the body), z wraps periodically.
__device__ __forceinline__ float3 velocityAtCell(const DeviceVelocityField& vel,
                                                 int x, int y, int z) {
    const long long c = cellIndex(clampi(x, 0, vel.dims.nx - 1),
                                  clampi(y, 0, vel.dims.ny - 1),
                                  wrapZ(z, vel.dims.nz), vel.dims);
    return make_float3(vel.u[c], vel.v[c], vel.w[c]);
}

/// Trilinear velocity sample at a continuous lattice-space position (cell
/// centers at integer+0.5 -> interpolation coordinate p - 0.5; x/y clamp, z
/// blends against the periodic image). Identical convention to particles.cu so
/// tufts and tracers see the same flow.
__device__ float3 sampleVelocity(const DeviceVelocityField& vel, float3 p) {
    const float gx = fminf(fmaxf(p.x - 0.5f, 0.0f),
                           static_cast<float>(vel.dims.nx) - 1.0001f);
    const float gy = fminf(fmaxf(p.y - 0.5f, 0.0f),
                           static_cast<float>(vel.dims.ny) - 1.0001f);
    float gz = p.z - 0.5f;
    if (gz < 0.0f) gz += static_cast<float>(vel.dims.nz);

    const int x0 = static_cast<int>(gx);
    const int y0 = static_cast<int>(gy);
    const int z0 = static_cast<int>(gz);
    const float fx = gx - static_cast<float>(x0);
    const float fy = gy - static_cast<float>(y0);
    const float fz = gz - static_cast<float>(z0);

    // Eight-corner gather + standard trilinear blend.
    const float3 c000 = velocityAtCell(vel, x0,     y0,     z0);
    const float3 c100 = velocityAtCell(vel, x0 + 1, y0,     z0);
    const float3 c010 = velocityAtCell(vel, x0,     y0 + 1, z0);
    const float3 c110 = velocityAtCell(vel, x0 + 1, y0 + 1, z0);
    const float3 c001 = velocityAtCell(vel, x0,     y0,     z0 + 1);
    const float3 c101 = velocityAtCell(vel, x0 + 1, y0,     z0 + 1);
    const float3 c011 = velocityAtCell(vel, x0,     y0 + 1, z0 + 1);
    const float3 c111 = velocityAtCell(vel, x0 + 1, y0 + 1, z0 + 1);

    const float3 x00 = lerp3(c000, c100, fx);
    const float3 x10 = lerp3(c010, c110, fx);
    const float3 x01 = lerp3(c001, c101, fx);
    const float3 x11 = lerp3(c011, c111, fx);
    const float3 y0v = lerp3(x00, x10, fy);
    const float3 y1v = lerp3(x01, x11, fy);
    return lerp3(y0v, y1v, fz);
}

// ===========================================================================
// Reset kernel: lay every strand straight along its surface tangent so a
// freshly seeded grid draws sensibly before any advection runs.
// ===========================================================================

__global__ void tuftResetKernel(float4* nodes, const TuftAnchor* anchors,
                                 int tuftCount) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tuftCount) return;

    const TuftAnchor a = anchors[t];
    const int base = t * kTuftNodes;
    // Rest pose stands the strand UP along the outward surface normal: taped at
    // the root, free end pointing off the skin (like a real cotton tuft before
    // the tunnel starts). The flow then bends it over toward the tangent — so a
    // tuft never starts buried in the geometry. Attachment scalar starts at 0.
    for (int i = 0; i < kTuftNodes; ++i) {
        const float3 p = add3(a.root, scale3(a.normal,
                                             a.segLen * static_cast<float>(i)));
        nodes[base + i] = make_float4(p.x, p.y, p.z, 0.0f);
    }
}

// ===========================================================================
// Advection kernel: one thread per STRAND walks the chain root -> tip. The
// root is pinned; each free node is dragged by the sampled flow then pulled
// back to its rest distance from the previous node (length-preserving spring).
// A short chain integrated this way naturally lies flat in steady downstream
// flow and curls / reverses where the flow separates.
// ===========================================================================

__global__ void tuftAdvectKernel(float4* nodes, const TuftAnchor* anchors,
                                  int tuftCount, DeviceVelocityField vel,
                                  TuftAdvectParams prm) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tuftCount) return;

    const TuftAnchor a = anchors[t];
    const int base = t * kTuftNodes;

    // Pin the root exactly on the skin every frame (no drift).
    float3 prev = a.root;
    nodes[base] = make_float4(prev.x, prev.y, prev.z, 0.0f);

    // Reference flow at the root for the attachment metric — sampled a
    // sampleFloor off the skin, NOT at the root itself: the halfway bounce-back
    // wall + voxel stair-step leave a ~zero-velocity dead band right at the
    // surface, which would make the reversal test read noise.
    float3 rootFlow = sampleVelocity(
        vel, add3(a.root, scale3(a.normal, prm.sampleFloor)));
    if (!finite3(rootFlow)) rootFlow = make_float3(0.0f, 0.0f, 0.0f);
    const float rootSpeed = len3(rootFlow);

    // Walk the free nodes. Each node integrates from its current VBO position
    // (so the strand keeps temporal memory and flutters) toward where the flow
    // wants it, then the link is re-projected to its EXACT rest length.
    float3 tip = prev;       // running position of the node just placed
    for (int i = 1; i < kTuftNodes; ++i) {
        // Previous-frame position of this node (temporal state lives in the VBO).
        const float4 cur = nodes[base + i];
        float3 p = make_float3(cur.x, cur.y, cur.z);
        if (!finite3(p)) {
            // Recover a poisoned node onto the upright rest line (along normal).
            p = add3(a.root, scale3(a.normal, a.segLen * static_cast<float>(i)));
        }

        // --- flow drag: nudge the node along the locally sampled velocity ---
        // Wall-proximity lift: a node hugging the skin sits in the dead band
        // (halfway wall / voxel solid reads ~zero), so its drag velocity is
        // read from at least sampleFloor off the surface. Without this, low
        // nodes freeze while the free tip rides fast air — the strand shears,
        // stretches, and folds back over itself (the "two lines" glitch).
        const float reach0 = dot3(sub3(p, a.root), a.normal);
        const float3 ps = (reach0 < prm.sampleFloor)
            ? add3(p, scale3(a.normal, prm.sampleFloor - reach0))
            : p;
        float3 flow = sampleVelocity(vel, ps);
        if (!finite3(flow)) flow = make_float3(0.0f, 0.0f, 0.0f);
        const float3 target = add3(p, scale3(flow, prm.flowScale * prm.dtSteps));
        // Damping blends previous position toward the flow target — low damping
        // keeps the lazy cotton lag/flutter, high snaps to the streamlines.
        p = lerp3(p, target, prm.damping);

        // --- hard inextensible link (follow-the-leader / PBD rope) ----------
        // The node is snapped to EXACTLY segLen from its parent: a tuft is an
        // inextensible string, so links may only rotate, never stretch. The
        // earlier soft spring let the chain stretch under near-wall shear and
        // the lagging strand folded into a zigzag.
        float3 d = sub3(p, prev);
        float dl = len3(d);
        if (dl < 1e-5f) { d = a.normal; dl = 1.0f; } // degenerate -> stand up
        p = add3(prev, scale3(d, a.segLen / dl));

        // --- surface collision: keep the node ON or ABOVE the skin -----------
        // The root sits on the smooth surface; the outward normal defines the
        // half-space the strand must stay in. Without this, the flow bending a
        // strand over drags its nodes straight through the (zero-velocity) wall
        // and they get stuck under the skin where no flow can recover them.
        // Project any node that has dipped below the root's tangent plane back
        // onto it, leaving a thin floor so it visibly lies flat ON the surface
        // rather than vanishing into it. (This can stretch the link by a hair;
        // invisible at strand scale, and the next frame re-projects anyway.)
        const float floor = 0.25f * a.segLen; // sit a hair proud of the skin
        const float reach = dot3(sub3(p, a.root), a.normal); // signed wall dist
        if (reach < floor) {
            // Push straight out along the normal by the shortfall — slides the
            // node along the surface instead of letting it penetrate.
            p = add3(p, scale3(a.normal, floor - reach));
        }

        nodes[base + i] = make_float4(p.x, p.y, p.z, 0.0f); // .w filled below

        prev = p;
        tip = p;
    }

    // --- attachment metric -> per-strand color scalar in [0,1] -------------
    // Compare the strand's overall direction (root -> tip) against the surface
    // tangent. Attached flow lays the tuft DOWN-tangent (alignment ~ +1);
    // separation reverses or splays it (alignment <= 0). We also fold in the
    // root flow's streamwise reversal (negative tangential velocity is the
    // classic separation signature) so the metric fires even before the strand
    // has fully curled.
    float3 chord = sub3(tip, a.root);
    const float chordLen = len3(chord);
    const float3 chordDir = (chordLen > 1e-5f) ? scale3(chord, 1.0f / chordLen)
                                               : a.tangent;
    const float align = dot3(chordDir, a.tangent);  // +1 attached .. -1 reversed

    // Streamwise (tangential) component of the root flow, normalized by the
    // root speed: < 0 means the near-wall flow is moving UPSTREAM (separated).
    const float tangVel = dot3(rootFlow, a.tangent);
    const float revFrac = (rootSpeed > 1e-6f)
                              ? fmaxf(0.0f, -tangVel / rootSpeed) : 0.0f;

    // Map alignment in [+1, -1] -> [0, 1] (1 - align)/2, then bias upward by
    // the measured reversal fraction scaled by the user's sepScale knob. The
    // result clamps to [0,1]: green where the tuft streams flat, red where it
    // reverses or flutters off-tangent.
    float sep = 0.5f * (1.0f - align) + revFrac / fmaxf(prm.sepScale, 1e-3f) * 0.5f;
    sep = fminf(fmaxf(sep, 0.0f), 1.0f);

    // Stamp the strand color into every node's .w (root included) so the whole
    // GL_LINE_STRIP reads one color.
    for (int i = 0; i < kTuftNodes; ++i) {
        float4 n = nodes[base + i];
        n.w = sep;
        nodes[base + i] = n;
    }
}

} // namespace

// ===========================================================================
// Launch wrappers (host-callable; viz.cpp maps the VBO and calls these).
// ===========================================================================

cudaError_t launchTuftAdvect(float4* nodes, const TuftAnchor* anchors,
                             int tuftCount, DeviceVelocityField vel,
                             const TuftAdvectParams& params,
                             cudaStream_t stream) {
    if (tuftCount <= 0) return cudaSuccess;
    const int block = 128;
    const int grid = (tuftCount + block - 1) / block;
    tuftAdvectKernel<<<grid, block, 0, stream>>>(nodes, anchors, tuftCount,
                                                 vel, params);
    return cudaGetLastError();
}

cudaError_t launchTuftReset(float4* nodes, const TuftAnchor* anchors,
                            int tuftCount, cudaStream_t stream) {
    if (tuftCount <= 0) return cudaSuccess;
    const int block = 128;
    const int grid = (tuftCount + block - 1) / block;
    tuftResetKernel<<<grid, block, 0, stream>>>(nodes, anchors, tuftCount);
    return cudaGetLastError();
}

} // namespace foilcfd
