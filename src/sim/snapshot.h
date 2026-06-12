// Warm-start cache (plan section 8): the VRAM-resident full-f clean-foil
// snapshot for instant VG-edit restarts, the compact macroscopic variant for
// disk persistence, the exact cache-key contract, and the LRU disk cache.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "lbm_core.cuh"
#include "units.h"

namespace foilcfd {

class LBMSolver;

/// Version stamp baked into every snapshot key and file header. Bump whenever
/// anything that changes the meaning of stored f/macroscopic data changes:
/// lattice ordering (frozen, but still), collision model, flag semantics.
inline constexpr int kSolverVersion = 1;

// ===========================================================================
// Cache keying (plan section 8, exact contract):
//   (airfoil id or STL hash, AoA bucket of 0.5 deg, u_lat, grid dims,
//    solver version)
// Airspeed is deliberately ABSENT: changing airspeed rescales via units.h and
// restarts the ramp but does not invalidate the cached flow (plan section 13).
// ===========================================================================

/// @brief Identity of a cached flow state. Two runs with equal keys may share
/// a snapshot; anything that voids that guarantee must be part of the key.
struct SnapshotKey {
    std::string geometryId;     ///< Airfoil identity: "naca:2412", "dat:<filename>",
                                ///< or "stl:<16-hex content hash>" for STL imports.
    int   aoaBucket  = 0;       ///< AoA quantized to 0.5 deg buckets:
                                ///< bucket = lround(aoa_deg * 2).
    float u_lat      = 0.0f;    ///< Lattice inflow speed (exact match required).
    GridDims dims;              ///< Grid dimensions.
    int   solverVersion = kSolverVersion; ///< Format/physics version stamp.

    /// @brief Quantize an angle of attack into the 0.5-degree bucket index.
    static int bucketForAoA(float aoa_deg) {
        return static_cast<int>(std::lround(aoa_deg * 2.0f));
    }

    /// @brief Exact-match equality — the cache never fuzzy-matches.
    bool operator==(const SnapshotKey& r) const {
        return geometryId == r.geometryId && aoaBucket == r.aoaBucket
            && u_lat == r.u_lat && dims.nx == r.dims.nx && dims.ny == r.dims.ny
            && dims.nz == r.dims.nz && solverVersion == r.solverVersion;
    }

    /// @brief Stable 64-bit hash of the key, used as the disk cache filename
    /// stem so lookups are a directory stat, not a scan.
    std::uint64_t hash() const;

    /// @brief Filesystem-safe printable form, e.g.
    /// "naca2412_a16.0_u0.0800_768x320x96_v1" (debugging + cache filenames).
    std::string toString() const;
};

/// @brief FNV-1a content hash of a flag field; stored with snapshots so a
/// restore can verify the geometry it was captured against (plan section 8:
/// "full f + flags hash + scaling params").
std::uint64_t hashFlags(const std::uint8_t* flags, std::size_t count);

// ===========================================================================
// VRAM snapshot: one full-f copy kept on device. 1.8 GB at default grid, so
// exactly ONE clean-foil slot exists; restore is a device-to-device memcpy
// (milliseconds) — this is what makes VG editing feel instant.
// ===========================================================================

/// @brief Device-resident exact snapshot (full f, plan section 8).
class VramSnapshot {
public:
    VramSnapshot();
    ~VramSnapshot();
    VramSnapshot(const VramSnapshot&) = delete;
    VramSnapshot& operator=(const VramSnapshot&) = delete;

    /// @brief Capture the solver's current f field (D2D copy) plus the key,
    /// flags hash, and scaling. Allocates the device buffer on first use;
    /// re-captures reuse it when sizes match, else reallocate.
    /// @param solver Source solver (must be initialized).
    /// @param key    Identity to store alongside the field.
    /// @param stream Stream to issue the copy on.
    /// @param error  On failure (OOM is the realistic one), the reason.
    /// @return True on success.
    bool capture(LBMSolver& solver, const SnapshotKey& key,
                 cudaStream_t stream, std::string* error);

    /// @brief Restore the captured f into the solver (D2D copy) and call
    /// LBMSolver::notifySnapshotRestored(true). Caller then applies edited
    /// flags via LBMSolver::applyEditedFlags() — see plan section 8 flow.
    /// @return False if empty or the solver's grid no longer matches.
    bool restore(LBMSolver& solver, cudaStream_t stream) const;

    /// @brief True once capture() has succeeded.
    bool hasData() const;

    /// @brief Key of the captured state (meaningful only when hasData()).
    const SnapshotKey& key() const;

    /// @brief Flags hash captured with the state (verify before restoring
    /// onto supposedly-identical geometry).
    std::uint64_t flagsHash() const;

    /// @brief Free the device buffer (e.g. before a bigger grid re-init).
    void release();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ===========================================================================
// Compact snapshot: macroscopic-only (rho,u,v,w = 16 B/cell, ~380 MB at the
// default grid; smaller on disk with compression). Restore = equilibrium
// re-init from (rho,u) + a short settling transient (~1000 steps) — far
// cheaper than a cold start, cheap enough to persist between app runs.
// ===========================================================================

/// @brief Host/disk macroscopic snapshot (plan section 8 compact variant).
class CompactSnapshot {
public:
    CompactSnapshot();
    ~CompactSnapshot();
    CompactSnapshot(const CompactSnapshot&) = delete;
    CompactSnapshot& operator=(const CompactSnapshot&) = delete;

    /// @brief Download the solver's macroscopic fields (rho,u,v,w) to host
    /// memory together with key, flags hash, and scaling parameters.
    /// @return True on success.
    bool capture(LBMSolver& solver, const SnapshotKey& key,
                 cudaStream_t stream, std::string* error);

    /// @brief Upload (rho,u) and set f = equilibrium(rho,u), then call
    /// LBMSolver::notifySnapshotRestored(false) so the solver schedules the
    /// settling transient and keeps the force gate closed through it.
    /// @return False if empty or grid mismatch.
    bool restore(LBMSolver& solver, cudaStream_t stream) const;

    /// @brief Serialize to disk (header: magic, version, key, flags hash,
    /// scaling; payload: the four field arrays, compressed). Safe to run on a
    /// worker thread — touches only host state (plan: "saved on a worker
    /// thread").
    /// @param path Destination file (conventionally <key.hash()>.fcs in the
    ///             cache directory).
    bool saveToFile(const std::filesystem::path& path, std::string* error) const;

    /// @brief Load from disk; rejects mismatched magic/version. The decoded
    /// key is available via key() for cache validation.
    bool loadFromFile(const std::filesystem::path& path, std::string* error);

    /// @brief True once capture() or loadFromFile() has succeeded.
    bool hasData() const;

    /// @brief Key of the held state.
    const SnapshotKey& key() const;

    /// @brief Flags hash captured with the state.
    std::uint64_t flagsHash() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ===========================================================================
// Disk cache: cache/ directory next to the exe, LRU-evicted against a
// user-configurable byte budget (default 10 GB), keyed by SnapshotKey::hash().
// ===========================================================================

/// @brief LRU disk cache of CompactSnapshot files (plan section 8).
class DiskSnapshotCache {
public:
    /// @brief Open (and create if missing) the cache directory and scan the
    /// existing entries to seed the LRU bookkeeping (uses file mtimes).
    /// @param directory   Cache root, conventionally "cache/" next to the exe.
    /// @param budgetBytes Maximum total bytes before LRU eviction
    ///                    (default 10 GB).
    explicit DiskSnapshotCache(std::filesystem::path directory,
                               std::uint64_t budgetBytes = 10ull << 30);
    ~DiskSnapshotCache();
    DiskSnapshotCache(const DiskSnapshotCache&) = delete;
    DiskSnapshotCache& operator=(const DiskSnapshotCache&) = delete;

    /// @brief Look up a snapshot file for @p key. A hit refreshes the entry's
    /// LRU recency.
    /// @return Path to the cached file, or nullopt on miss.
    std::optional<std::filesystem::path> find(const SnapshotKey& key);

    /// @brief Store @p snapshot under its own key, then evict least-recently-
    /// used entries until the total size fits the budget. Intended to be
    /// called from a worker thread; internally serialized.
    /// @return True on success (eviction failures only log; the store wins).
    bool store(const CompactSnapshot& snapshot, std::string* error);

    /// @brief Enqueue @p snapshot for storage on the cache's internal worker
    /// thread and return immediately (plan section 8: "saved on a worker
    /// thread") — the frame loop never blocks on disk I/O. Failures are
    /// logged to stderr. The shared_ptr keeps the (hundreds of MB) snapshot
    /// alive until the write completes without copying it.
    void storeAsync(std::shared_ptr<const CompactSnapshot> snapshot);

    /// @brief Change the byte budget at runtime (UI setting); evicts
    /// immediately if the new budget is smaller than current usage.
    void setBudgetBytes(std::uint64_t budgetBytes);

    /// @brief Current total size of all cached files in bytes.
    std::uint64_t usedBytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace foilcfd
