// Shared helpers for the GPU physics tests (m0/m1/m2): boundary-flag stamping
// that mirrors the solver's domain contract, a host-side D3Q19 equilibrium
// (frozen Krueger ordering from lbm_core.cuh), f-buffer download, and moment
// (rho, u) reconstruction so tests stay independent of the solver's optional
// macroscopic-array write path. Host-only code; links against CUDA::cudart.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_runtime_api.h>

#include "sim/lbm_core.cuh"
#include "sim/lbm_solver.h"

namespace foilcfd::testutil {

/// @brief Linear cell index for the frozen layout: x fastest (lbm_core.cuh).
inline long long cellIndex(const GridDims& d, int x, int y, int z) {
    return static_cast<long long>(x)
         + static_cast<long long>(d.nx)
               * (static_cast<long long>(y)
                  + static_cast<long long>(d.ny) * static_cast<long long>(z));
}

/// @brief Build the standard open-domain flag field the solver is contracted
/// for (plan 4.2): Inlet at x=0, Outlet at x=nx-1, SlipBottom/SlipTop at the
/// y faces, Fluid elsewhere, z periodic (no z-face flags). Inlet/outlet win
/// at the y-face corners so those columns are uniform top to bottom —
/// matching the convention buildBoundaryFlags() established in the scaffold.
/// Re-implemented locally so the GPU tests link only against sim/ sources.
inline std::vector<std::uint8_t> openDomainFlags(const GridDims& dims) {
    std::vector<std::uint8_t> flags(
        static_cast<std::size_t>(dims.cellCount()),
        static_cast<std::uint8_t>(CellFlag::Fluid));
    for (int z = 0; z < dims.nz; ++z) {
        for (int y = 0; y < dims.ny; ++y) {
            // Stamp the y faces first; x-face stamps below overwrite corners.
            if (y == 0 || y == dims.ny - 1) {
                const std::uint8_t face = static_cast<std::uint8_t>(
                    y == 0 ? CellFlag::SlipBottom : CellFlag::SlipTop);
                for (int x = 0; x < dims.nx; ++x)
                    flags[static_cast<std::size_t>(cellIndex(dims, x, y, z))] = face;
            }
            flags[static_cast<std::size_t>(cellIndex(dims, 0, y, z))] =
                static_cast<std::uint8_t>(CellFlag::Inlet);
            flags[static_cast<std::size_t>(cellIndex(dims, dims.nx - 1, y, z))] =
                static_cast<std::uint8_t>(CellFlag::Outlet);
        }
    }
    return flags;
}

/// @brief Second-order D3Q19 equilibrium distribution for direction q:
///   f_q^eq = w_q * rho * (1 + 3 c.u + 4.5 (c.u)^2 - 1.5 u.u)
/// (cs^2 = 1/3 absorbed into the integer-velocity dot products). Must match
/// the device-side equilibrium so a host-built field restored through
/// activeDeviceF() is a legitimate "snapshot" the solver can continue from.
inline float equilibrium(int q, float rho, float ux, float uy, float uz) {
    const float cu = static_cast<float>(kCx[q]) * ux
                   + static_cast<float>(kCy[q]) * uy
                   + static_cast<float>(kCz[q]) * uz;
    const float uu = ux * ux + uy * uy + uz * uz;
    return kW[q] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * uu);
}

/// @brief Host-side macroscopic field reconstructed from a downloaded f buffer.
struct HostMoments {
    std::vector<float> rho; ///< Density per cell.
    std::vector<float> u;   ///< x-velocity per cell (lattice units).
    std::vector<float> v;   ///< y-velocity per cell.
    std::vector<float> w;   ///< z-velocity per cell.
};

// ---------------------------------------------------------------------------
// Device f-buffer layout adapter. The lbm_solver.h contract describes
// activeDeviceF() as kQ * cellCount floats; the kernel implementation stores
// each direction with two extra z ghost planes (periodic span, plan 11:
// "padded ghosts in z are simplest"), making the per-direction stride
// nx*ny*(nz+2) with real cell 0 one plane in. Rather than hard-code either
// convention, the tests size the layout from the solver's own fBufferBytes()
// and handle both — whichever the integrated solver publishes works.
// ---------------------------------------------------------------------------

/// @brief Per-direction layout of the solver's device f buffer.
struct FLayout {
    std::size_t stride = 0;     ///< Floats per direction block.
    std::size_t realOffset = 0; ///< Index of real cell 0 inside a block.
};

/// @brief Derive the f layout from the solver's reported buffer size.
inline FLayout fLayout(const LBMSolver& solver) {
    const GridDims d = solver.dims();
    const std::size_t plane = static_cast<std::size_t>(d.nx)
                            * static_cast<std::size_t>(d.ny);
    const std::size_t stride =
        solver.fBufferBytes() / sizeof(float) / static_cast<std::size_t>(kQ);
    // z-padded layout: ghost planes at both ends of every direction block.
    if (stride == plane * static_cast<std::size_t>(d.nz + 2))
        return FLayout{stride, plane};
    // Header-contract unpadded layout (or anything else: treat as flat).
    return FLayout{stride, 0};
}

/// @brief Upload a host-built UNPADDED f field (kQ * cellCount floats, SoA,
/// frozen Krueger ordering) into the solver's current source buffer, staging
/// through the solver's actual device layout, then tell the solver its state
/// was replaced (full-state restore semantics — the snapshot path, plan 8).
/// Ghost planes, when present, are filled by periodic z-wrap so the field is
/// consistent without relying on a solver-side refresh.
/// @param solver Initialized solver to write into.
/// @param f      kQ * cellCount floats, unpadded.
/// @return cudaSuccess or the copy error.
inline cudaError_t uploadF(LBMSolver& solver, const std::vector<float>& f) {
    const GridDims d = solver.dims();
    const FLayout layout = fLayout(solver);
    const std::size_t n = static_cast<std::size_t>(d.cellCount());
    const std::size_t plane = static_cast<std::size_t>(d.nx)
                            * static_cast<std::size_t>(d.ny);

    cudaError_t err;
    if (layout.realOffset == 0 && layout.stride == n) {
        // Layout matches the host build exactly: single direct copy.
        err = cudaMemcpy(solver.activeDeviceF(), f.data(),
                         solver.fBufferBytes(), cudaMemcpyHostToDevice);
    } else {
        // Stage into the padded layout: real cells one plane in, low ghost
        // mirrors z = nz-1, high ghost mirrors z = 0 (periodic span).
        std::vector<float> staged(static_cast<std::size_t>(kQ) * layout.stride,
                                  0.0f);
        for (int q = 0; q < kQ; ++q) {
            const float* src = f.data() + static_cast<std::size_t>(q) * n;
            float* dst = staged.data() + static_cast<std::size_t>(q) * layout.stride;
            for (std::size_t c = 0; c < n; ++c) dst[layout.realOffset + c] = src[c];
            for (std::size_t i = 0; i < plane; ++i) {
                dst[i] = src[(static_cast<std::size_t>(d.nz) - 1) * plane + i];
                dst[layout.realOffset + n + i] = src[i];
            }
        }
        err = cudaMemcpy(solver.activeDeviceF(), staged.data(),
                         staged.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
    if (err == cudaSuccess) solver.notifySnapshotRestored(/*fullState=*/true);
    return err;
}

/// @brief Reconstruct rho and u from raw populations: rho = sum_q f_q,
/// rho*u = sum_q c_q f_q. Valid for any post-collision buffer because both
/// moments are collision invariants of BGK/TRT. Computed on host so the tests
/// do not depend on whether the solver wrote its macroscopic arrays that step.
/// @param f      f buffer, SoA with the given per-direction layout.
/// @param dims   Grid the buffer covers (outputs are UNPADDED cellCount arrays).
/// @param layout Per-direction stride/offset; default = tightly packed.
inline HostMoments momentsFromF(const std::vector<float>& f, const GridDims& dims,
                                FLayout layout = FLayout{}) {
    const std::size_t n = static_cast<std::size_t>(dims.cellCount());
    if (layout.stride == 0) layout.stride = n; // tightly packed default
    HostMoments m;
    m.rho.assign(n, 0.0f);
    m.u.assign(n, 0.0f);
    m.v.assign(n, 0.0f);
    m.w.assign(n, 0.0f);
    // Accumulate per direction across all cells (SoA-friendly traversal).
    for (int q = 0; q < kQ; ++q) {
        const float* fq = f.data() + static_cast<std::size_t>(q) * layout.stride
                        + layout.realOffset;
        const float cx = static_cast<float>(kCx[q]);
        const float cy = static_cast<float>(kCy[q]);
        const float cz = static_cast<float>(kCz[q]);
        for (std::size_t i = 0; i < n; ++i) {
            const float v = fq[i];
            m.rho[i] += v;
            m.u[i] += cx * v;
            m.v[i] += cy * v;
            m.w[i] += cz * v;
        }
    }
    // Momentum -> velocity. Guard rho ~ 0 (zeroed solid cells) to avoid Inf.
    for (std::size_t i = 0; i < n; ++i) {
        const float invRho = (m.rho[i] > 1e-12f) ? 1.0f / m.rho[i] : 0.0f;
        m.u[i] *= invRho;
        m.v[i] *= invRho;
        m.w[i] *= invRho;
    }
    return m;
}

/// @brief Download the solver's current f state and reconstruct macroscopic
/// moments in one step, honoring whatever device layout the solver reports.
/// Synchronous; ~20 MB at 64^3 — fine at the per-1000-steps test cadence.
/// @param solver Initialized solver to read from.
/// @param out    Receives UNPADDED cellCount moment arrays.
/// @return cudaSuccess or the copy error.
inline cudaError_t downloadMoments(LBMSolver& solver, HostMoments& out) {
    std::vector<float> f(solver.fBufferBytes() / sizeof(float));
    const cudaError_t err =
        cudaMemcpy(f.data(), solver.activeDeviceF(), solver.fBufferBytes(),
                   cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;
    out = momentsFromF(f, solver.dims(), fLayout(solver));
    return cudaSuccess;
}

/// @brief Total kinetic energy sum(0.5 * rho * |u|^2) over FLUID cells only
/// (boundary-flagged cells carry imposed values, not evolved flow). Returned
/// as double: 260k float summands deserve a wide accumulator.
/// @param m     Reconstructed moments.
/// @param flags Host flag field matching m's grid.
inline double kineticEnergy(const HostMoments& m,
                            const std::vector<std::uint8_t>& flags) {
    double ke = 0.0;
    for (std::size_t i = 0; i < m.rho.size(); ++i) {
        if (flags[i] != static_cast<std::uint8_t>(CellFlag::Fluid)) continue;
        const double uu = static_cast<double>(m.u[i]) * m.u[i]
                        + static_cast<double>(m.v[i]) * m.v[i]
                        + static_cast<double>(m.w[i]) * m.w[i];
        ke += 0.5 * static_cast<double>(m.rho[i]) * uu;
    }
    return ke;
}

} // namespace foilcfd::testutil
