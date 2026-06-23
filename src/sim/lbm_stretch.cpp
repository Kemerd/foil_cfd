// ISLBM stretched-mesh host builder: turn a wall-distance field into per-axis
// smooth spacing profiles, the per-link interpolation foot LUTs, and the
// per-cell relaxation field, then upload the device mirror. Pure host math +
// cudaMalloc/Memcpy — no kernels. See lbm_stretch.h for the physics contract.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include "lbm_stretch.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "units.h"            // LatticeScaling, kMinTau
#include "geom/voxelizer.h"   // GridDims

namespace foilcfd {

namespace {

// ---------------------------------------------------------------------------
// One-axis spacing profile. Input: per-index "distance from the wall in cells"
// (the min over the transverse extent so the profile is a clean 1-D function of
// the index along this axis). Output: dx(i) physical spacing and X(i) node
// positions, finest (dxMin) at the wall-nearest plane and growing geometrically
// outward at <= growthCap per cell, saturating at dxMax. The returned profile
// integrates to whatever physical length the index count * dxMin would imply at
// uniform spacing scaled by the average stretch — the caller does NOT need the
// integral to equal a fixed L (the grid keeps its index count; stretch only
// redistributes where resolution sits). growth achieved is reported back.
struct AxisProfile {
    std::vector<float> dx;   ///< Spacing per index [m].
    float growth = 1.0f;     ///< Achieved per-cell growth ratio.
};

AxisProfile buildAxisProfile(const std::vector<float>& wallDistCells, int n,
                             float dxMin, float dxMax, float growthCap) {
    AxisProfile p;
    p.dx.assign(static_cast<std::size_t>(n), dxMin);
    if (n <= 1 || dxMax <= dxMin) return p; // degenerate / no room to stretch

    // dx grows geometrically with distance from the wall: dx(d) = dxMin*g^d,
    // clamped at dxMax. g is the per-cell growth; cap it so no neighbour pair
    // ever jumps more than growthCap (the accuracy guarantee). The distance is
    // measured RELATIVE to the finest plane (the global minimum along this
    // axis), so the closest-to-wall index gets g^0 = dxMin and a uniform
    // distance field maps to a uniform dxMin grid (the gather then collapses to
    // the exact integer pull — the m6_stretch uniform-collapse invariant).
    float dMin = std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i)
        dMin = std::min(dMin, std::max(0.0f,
                                       wallDistCells[static_cast<std::size_t>(i)]));
    if (!std::isfinite(dMin)) dMin = 0.0f;
    const float g = std::min(growthCap, dxMax / dxMin); // never exceed the cap
    p.growth = g;
    const float lng = std::log(g);
    for (int i = 0; i < n; ++i) {
        const float d = std::max(0.0f, wallDistCells[static_cast<std::size_t>(i)])
                      - dMin; // cells beyond the finest plane
        // dxMin * g^d, but computed as exp(d*ln g) and clamped to dxMax.
        const float dxi = dxMin * std::exp(d * lng);
        p.dx[static_cast<std::size_t>(i)] = std::min(dxi, dxMax);
    }

    // Smooth the profile once (3-point box) so the geometric->plateau corner has
    // no curvature kink — the quadratic gather is most accurate on a C1 dx(i).
    if (n >= 3) {
        std::vector<float> s = p.dx;
        for (int i = 1; i < n - 1; ++i)
            p.dx[static_cast<std::size_t>(i)] =
                0.25f * s[static_cast<std::size_t>(i - 1)]
              + 0.50f * s[static_cast<std::size_t>(i)]
              + 0.25f * s[static_cast<std::size_t>(i + 1)];
    }
    return p;
}

// Reduce the full 3-D wall-distance field to a 1-D "distance from wall" per
// index along one axis: the MIN distance over the transverse plane at that
// index, so a single foil row/column near the surface pulls the whole plane to
// the finest spacing. axis 0 = x, 1 = y.
std::vector<float> reduceAxisDistance(const std::vector<float>& dist,
                                      const GridDims& dims, int axis) {
    const int nx = dims.nx, ny = dims.ny, nz = dims.nz;
    const int n = (axis == 0) ? nx : ny;
    std::vector<float> out(static_cast<std::size_t>(n),
                           std::numeric_limits<float>::infinity());
    auto idx = [nx, ny](int x, int y, int z) {
        return static_cast<std::size_t>(x)
             + static_cast<std::size_t>(nx) * (y + static_cast<long long>(ny) * z);
    };
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                const float d = dist[idx(x, y, z)];
                const int i = (axis == 0) ? x : y;
                out[static_cast<std::size_t>(i)] =
                    std::min(out[static_cast<std::size_t>(i)], d);
            }
    // Any index that never saw a finite distance (no solid in its plane) maps to
    // 0 -> coarsest spacing is selected for it by the large distance instead;
    // keep it large so it lands at dxMax.
    for (float& v : out)
        if (!std::isfinite(v)) v = 1e9f;
    return out;
}

// Per-link interpolation foot in INDEX space for one axis. For a population that
// travels |c|=1 cell physically per step, the index distance it covers at cell i
// is (c_phys = dxMin physical step / dx(i)). On the finest cells dx==dxMin so it
// is exactly 1 index cell (foot lands on the integer neighbour, frac 0); on
// coarser cells dx>dxMin so it covers LESS than one index cell (foot between the
// home cell and its neighbour). We pull from upstream, so the foot for sign s
// (s=0 -> link goes -1 in index, s=1 -> +1) sits at i - s_dir * shift where
// shift = dxMin/dx(i) in [0,1]. base = round(-s_dir*shift), frac = (-s_dir*shift)
// - base, stored so the kernel's quadratic weights {f(f-1)/2, 1-f^2, f(f+1)/2}
// reconstruct the off-node value. dxMin is the physical step a finest-cell link
// takes; using it as the reference keeps the wall exact (frac 0 there).
void buildFootMaps(const std::vector<float>& dxAxis, int n, float dxMin,
                   std::vector<float>& footFrac, std::vector<std::int8_t>& footBase) {
    footFrac.assign(static_cast<std::size_t>(2 * n), 0.0f);
    footBase.assign(static_cast<std::size_t>(2 * n), 0);
    for (int s = 0; s < 2; ++s) {
        const float dir = (s == 0) ? -1.0f : 1.0f; // pull-from side
        for (int i = 0; i < n; ++i) {
            const float dxi = std::max(dxAxis[static_cast<std::size_t>(i)], 1e-12f);
            const float shift = dir * (dxMin / dxi); // index cells travelled
            const int   base  = static_cast<int>(std::lround(shift));
            const float frac  = shift - static_cast<float>(base);
            const std::size_t k = static_cast<std::size_t>(s) * n + i;
            footFrac[k] = frac;
            // base is small (|shift| <= 1), so int8 is ample; clamp defensively.
            footBase[k] = static_cast<std::int8_t>(std::clamp(base, -2, 2));
        }
    }
}

} // namespace

bool buildStretchMesh(StretchMesh& mesh, const GridDims& dims,
                      const LatticeScaling& scaling,
                      const std::vector<float>& wallDist, cudaStream_t stream,
                      float nearWallFactor, std::string* error) {
    mesh.free(); // replace any previous mesh
    const int nx = dims.nx, ny = dims.ny, nz = dims.nz;
    const long long ncells = dims.cellCount();
    if (ncells <= 0
        || wallDist.size() != static_cast<std::size_t>(ncells)) {
        if (error) *error = "wall-distance field size does not match the grid";
        return false;
    }

    // Sub-base refinement factor k (>= 1): the finest cell is k-times finer than
    // the base grid. k == 1 reproduces the legacy base-wall behavior exactly.
    const float k = std::max(1.0f, nearWallFactor);
    mesh.nearWallFactor = k;

    // Finest spacing = base/k. This REMAINS the global minimum spacing (the
    // profile builder measures distance from the finest plane), so every foot
    // ratio dxMin/dxi stays <= 1 and the kernel/foot-map/collar are untouched.
    // The gradient still only coarsens OUTWARD from this finest plane.
    const float dxMin = scaling.dx / k;

    // Choose the far-field coarsest spacing so the far-field lattice viscosity
    // (nu_lat ~ (dxMin/dx)^2 * nu_lat_wall) keeps tau >= kMinTau. tau falls as
    // dx grows: tau(dx) = 0.5 + 3*nu_lat_wall*(dxMin/dx)^2. The floor on dx/dxMin
    // is where that hits kMinTau; clamp dxMax to it (report when clamped).
    //
    // ACOUSTIC RE-ANCHOR (refine-capable): nuWall is the lattice viscosity AT
    // the finest cell. Refining the reference by k is acoustic scaling (dx/=k,
    // dt/=k, nu_lat *= k — LINEAR, exactly as the cascade's refinedScaling does),
    // NOT diffusive (k^2). The two distinct exponents, kept straight:
    //   * ACROSS cells of ONE mesh (shared dt): nu_lat ~ (dxMin/dx)^2  [squared,
    //     applied per-cell below — UNCHANGED and correct].
    //   * RE-ANCHORING the reference (dxMin: base->base/k WITH dt: base->base/k):
    //     nuWall scales LINEARLY in k.
    // Derivation: nu_lat_ref = nu_phys * dt_global / dxMin^2 with dt_global =
    // dt_base/k and dxMin = base/k gives nu_lat_ref = k * (nu_phys*dt_base/base^2)
    // = k * nuBase. The earlier k^2 (and leaving dt unchanged) silently ran the
    // sim at U/k, i.e. Re/k — which laminarized the wake. The paired dt/=k anchor
    // (applied in the solver's scaling) is what keeps U, Re, and the fine-cell
    // sound speed all invariant.
    const float nuBase = (scaling.tau - 0.5f) / 3.0f;       // == scaling.nu_lat
    const float nuWall = nuBase * k;
    const float nuFloor = (kMinTau - 0.5f) / 3.0f;
    // tau floor reached when (dxMin/dx)^2 = nuFloor/nuWall -> dx/dxMin = sqrt(nuWall/nuFloor).
    float ratioCap = (nuFloor > 0.0f && nuWall > nuFloor)
        ? std::sqrt(nuWall / nuFloor) : 1.0f;
    // Also bound the coarsening so a tiny domain doesn't ask for an absurd ratio;
    // the growth cap will limit how fast we approach it anyway.
    ratioCap = std::min(ratioCap, 8.0f);
    const float dxMaxUnclamped = dxMin * std::max(1.0f, ratioCap);
    mesh.tauFloorClamped = (ratioCap < 8.0f && nuWall > nuFloor); // floor was the binder
    const float dxMax = dxMaxUnclamped;

    // Per-axis distance-from-wall (1-D) and spacing profiles. Z stays UNIFORM
    // (periodic span): never stretch it, so no foot map for z.
    const std::vector<float> distX = reduceAxisDistance(wallDist, dims, 0);
    const std::vector<float> distY = reduceAxisDistance(wallDist, dims, 1);
    const AxisProfile profX =
        buildAxisProfile(distX, nx, dxMin, dxMax, kMaxStretchGrowth);
    const AxisProfile profY =
        buildAxisProfile(distY, ny, dxMin, dxMax, kMaxStretchGrowth);

    // Per-link foot LUTs (host).
    std::vector<float>        hFootFracX, hFootFracY;
    std::vector<std::int8_t>  hFootBaseX, hFootBaseY;
    buildFootMaps(profX.dx, nx, dxMin, hFootFracX, hFootBaseX);
    buildFootMaps(profY.dx, ny, dxMin, hFootFracY, hFootBaseY);

    // Per-cell tau: nu_lat(cell) = nuWall * (dxMin / dxEff(cell))^2, where the
    // effective in-plane cell size is the geometric mean of the two stretched
    // spacings (z is uniform = dxMin, so it drops out of the in-plane mean).
    // tau falls toward kMinTau in the coarse far field; clamp defensively.
    std::vector<float> hTau(static_cast<std::size_t>(ncells));
    float tauFarMin = scaling.tau;
    float achievedDxMax = dxMin; // largest cell size actually present (diagnostic)
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y) {
            const float dxy = profY.dx[static_cast<std::size_t>(y)];
            for (int x = 0; x < nx; ++x) {
                const float dxx = profX.dx[static_cast<std::size_t>(x)];
                const float dxEff = std::sqrt(dxx * dxy); // in-plane geo mean
                const float r = dxMin / std::max(dxEff, 1e-12f);
                const float nu = nuWall * r * r;
                const float tau = std::max(kMinTau, 0.5f + 3.0f * nu);
                const std::size_t c = static_cast<std::size_t>(x)
                    + static_cast<std::size_t>(nx) * (y + static_cast<long long>(ny) * z);
                hTau[c] = tau;
                tauFarMin = std::min(tauFarMin, tau);
                achievedDxMax = std::max(achievedDxMax, dxEff);
            }
        }

    // ---- allocate + upload the device mirror -----------------------------
    auto fail = [&](const char* what, cudaError_t err) {
        if (error) *error = std::string(what) + ": " + cudaGetErrorString(err);
        mesh.free();
        return false;
    };
    const std::size_t fxBytes = static_cast<std::size_t>(2 * nx) * sizeof(float);
    const std::size_t bxBytes = static_cast<std::size_t>(2 * nx) * sizeof(std::int8_t);
    const std::size_t fyBytes = static_cast<std::size_t>(2 * ny) * sizeof(float);
    const std::size_t byBytes = static_cast<std::size_t>(2 * ny) * sizeof(std::int8_t);
    const std::size_t tBytes  = static_cast<std::size_t>(ncells) * sizeof(float);
    if (auto e = cudaMalloc(reinterpret_cast<void**>(&mesh.dFootFracX), fxBytes);
        e != cudaSuccess)
        return fail("stretch footFracX alloc failed", e);
    if (auto e = cudaMalloc(reinterpret_cast<void**>(&mesh.dFootBaseX), bxBytes);
        e != cudaSuccess)
        return fail("stretch footBaseX alloc failed", e);
    if (auto e = cudaMalloc(reinterpret_cast<void**>(&mesh.dFootFracY), fyBytes);
        e != cudaSuccess)
        return fail("stretch footFracY alloc failed", e);
    if (auto e = cudaMalloc(reinterpret_cast<void**>(&mesh.dFootBaseY), byBytes);
        e != cudaSuccess)
        return fail("stretch footBaseY alloc failed", e);
    if (auto e = cudaMalloc(reinterpret_cast<void**>(&mesh.dTauField), tBytes);
        e != cudaSuccess)
        return fail("stretch tauField alloc failed", e);

    cudaMemcpyAsync(mesh.dFootFracX, hFootFracX.data(), fxBytes,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(mesh.dFootBaseX, hFootBaseX.data(), bxBytes,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(mesh.dFootFracY, hFootFracY.data(), fyBytes,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(mesh.dFootBaseY, hFootBaseY.data(), byBytes,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(mesh.dTauField, hTau.data(), tBytes,
                    cudaMemcpyHostToDevice, stream);
    if (auto e = cudaStreamSynchronize(stream); e != cudaSuccess)
        return fail("stretch upload sync failed", e);

    // ---- fill diagnostics + activate -------------------------------------
    mesh.nx = nx; mesh.ny = ny; mesh.nz = nz;
    // dxMax is the ACHIEVED coarsest cell present (not the tau-floor TARGET),
    // so the resolution view + dxMax/dxMin readout reflect the real gradient
    // rather than a far-field target the growth cap never reached.
    mesh.dxMin = dxMin; mesh.dxMax = achievedDxMax;
    mesh.growthX = profX.growth; mesh.growthY = profY.growth;
    // The wall (finest cell) carries nuWall, the acoustic re-anchored (k-scaled)
    // viscosity, so tauWall is its tau — not scaling.tau.
    mesh.tauWall = 0.5f + 3.0f * nuWall; mesh.tauFar = tauFarMin;
    // Rough "what the gradient buys": fraction of cells coarser than ~1.5*dxMin.
    long long coarser = 0;
    for (float t : hTau) if (t < 0.5f + 3.0f * nuWall * (1.0f / (1.5f * 1.5f))) ++coarser;
    mesh.fluidCellSaving =
        static_cast<double>(coarser) / static_cast<double>(ncells);
    mesh.active = true;
    return true;
}

} // namespace foilcfd
