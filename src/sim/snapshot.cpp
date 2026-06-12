// Warm-start snapshot implementations (plan section 8): VRAM-resident full-f
// snapshot (instant D2D restore for VG edits), compact macroscopic disk
// variant (equilibrium re-init on restore), key/flag hashing, and the LRU
// disk cache with its background store worker.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include "snapshot.h"

#include <cuda_runtime.h>

#include <condition_variable>
#include <cstdio>
#include <deque>
#include <fstream>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>

#include "lbm_solver.h"

namespace foilcfd {

namespace {

// FNV-1a 64-bit: tiny, dependency-free, good enough for cache keys (not
// security). Shared by key hashing and flag-field hashing.
constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime  = 1099511628211ull;

std::uint64_t fnv1a(const void* data, std::size_t bytes,
                    std::uint64_t seed = kFnvOffset) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    std::uint64_t h = seed;
    for (std::size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= kFnvPrime;
    }
    return h;
}

// ---------------------------------------------------------------------------
// Compact snapshot disk format. Everything little-endian (x64-only project),
// written field-by-field — never raw structs, so padding bytes can't leak
// nondeterminism into files.
// ---------------------------------------------------------------------------

/// File magic "FCS1" and the container revision. kSolverVersion (physics/
/// lattice semantics) is stored separately inside the key.
constexpr std::uint32_t kFileMagic = 0x31534346u; // 'F','C','S','1'
constexpr std::uint32_t kFileFormatVersion = 1;

/// Payload compression tag. 0 = raw FP32 arrays. The header reserves the
/// field so a future compressed writer stays readable by this loader.
constexpr std::uint32_t kCompressionRaw = 0;

/// @brief Write one trivially-copyable value to a binary stream.
template <typename T>
void writePod(std::ofstream& os, const T& v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

/// @brief Read one trivially-copyable value from a binary stream.
template <typename T>
bool readPod(std::ifstream& is, T& v) {
    is.read(reinterpret_cast<char*>(&v), sizeof(T));
    return is.good();
}

/// @brief Disk-cache filename for a key: 16-hex-digit hash + extension, so a
/// lookup is a single directory stat (snapshot.h contract).
std::string cacheFileName(const SnapshotKey& key) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%016llx.fcs",
                  static_cast<unsigned long long>(key.hash()));
    return buf;
}

} // namespace

// ===========================================================================
// Key hashing
// ===========================================================================

std::uint64_t SnapshotKey::hash() const {
    // Chain every key field through FNV so equal keys hash equal and the
    // result is stable across runs (it names files on disk).
    std::uint64_t h = fnv1a(geometryId.data(), geometryId.size());
    h = fnv1a(&aoaBucket, sizeof aoaBucket, h);
    h = fnv1a(&u_lat, sizeof u_lat, h);
    h = fnv1a(&dims.nx, sizeof dims.nx, h);
    h = fnv1a(&dims.ny, sizeof dims.ny, h);
    h = fnv1a(&dims.nz, sizeof dims.nz, h);
    h = fnv1a(&solverVersion, sizeof solverVersion, h);
    return h;
}

std::string SnapshotKey::toString() const {
    char buf[256];
    std::snprintf(buf, sizeof buf, "%s_a%.1f_u%.4f_%dx%dx%d_v%d",
                  geometryId.c_str(), static_cast<float>(aoaBucket) * 0.5f,
                  u_lat, dims.nx, dims.ny, dims.nz, solverVersion);
    // Filesystem-safe: geometry ids may contain ':' (e.g. "naca:2412").
    for (char& c : buf) {
        if (c == '\0') break;
        if (c == ':' || c == '\\' || c == '/') c = '-';
    }
    return buf;
}

std::uint64_t hashFlags(const std::uint8_t* flags, std::size_t count) {
    return fnv1a(flags, count);
}

// ===========================================================================
// VramSnapshot: one device-resident full-f copy (1.8+ GB at default grid —
// exactly one slot exists by design). Capture/restore are single D2D copies,
// which is what makes VG editing feel instant (plan section 8).
// ===========================================================================

struct VramSnapshot::Impl {
    SnapshotKey   key{};
    std::uint64_t flagsHash = 0;
    bool          hasData = false;
    float*        d_f = nullptr;   ///< Device copy of the full padded f buffer.
    std::size_t   bytes = 0;       ///< Allocation size (solver fBufferBytes).

    void releaseBuffer() {
        cudaFree(d_f);
        d_f = nullptr;
        bytes = 0;
        hasData = false;
    }
};

VramSnapshot::VramSnapshot() : impl_(std::make_unique<Impl>()) {}
VramSnapshot::~VramSnapshot() { impl_->releaseBuffer(); }

bool VramSnapshot::capture(LBMSolver& solver, const SnapshotKey& key,
                           cudaStream_t stream, std::string* error) {
    Impl& s = *impl_;
    const std::size_t need = solver.fBufferBytes();
    if (need == 0) {
        if (error) *error = "solver holds no field to capture";
        return false;
    }
    // Reuse the buffer across re-captures of the same grid; reallocate only
    // when the size changed (resolution preset switch).
    if (s.d_f == nullptr || s.bytes != need) {
        s.releaseBuffer();
        if (auto err = cudaMalloc(&s.d_f, need); err != cudaSuccess) {
            if (error)
                *error = std::string("VRAM snapshot allocation failed (")
                       + cudaGetErrorString(err)
                       + ") — consider a smaller resolution preset";
            s.d_f = nullptr;
            return false;
        }
        s.bytes = need;
    }
    if (auto err = cudaMemcpyAsync(s.d_f, solver.activeDeviceF(), need,
                                   cudaMemcpyDeviceToDevice, stream);
        err != cudaSuccess) {
        if (error) *error = std::string("VRAM snapshot copy failed: ")
                          + cudaGetErrorString(err);
        return false;
    }
    // Identity bookkeeping: key + the geometry hash this field was solved on,
    // so a restore can verify it lands on the same clean-foil mask.
    s.key = key;
    const auto& flags = solver.hostFlags();
    s.flagsHash = hashFlags(flags.data(), flags.size());
    s.hasData = true;
    return true;
}

bool VramSnapshot::restore(LBMSolver& solver, cudaStream_t stream) const {
    const Impl& s = *impl_;
    if (!s.hasData) return false;
    // The grid must be byte-identical: same dims AND same (padded) buffer
    // size — a resolution change invalidates the snapshot entirely.
    const GridDims d = solver.dims();
    if (solver.fBufferBytes() != s.bytes || d.nx != s.key.dims.nx
        || d.ny != s.key.dims.ny || d.nz != s.key.dims.nz)
        return false;
    if (cudaMemcpyAsync(solver.activeDeviceF(), s.d_f, s.bytes,
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
        return false;
    // Caller applies edited flags next (plan section 8 VG-edit flow); the
    // solver just needs to know its field was replaced wholesale.
    solver.notifySnapshotRestored(true);
    return true;
}

bool VramSnapshot::hasData() const { return impl_->hasData; }
const SnapshotKey& VramSnapshot::key() const { return impl_->key; }
std::uint64_t VramSnapshot::flagsHash() const { return impl_->flagsHash; }
void VramSnapshot::release() { impl_->releaseBuffer(); }

// ===========================================================================
// CompactSnapshot: host-side macroscopic state (rho, u, v, w — 16 B/cell).
// Restore = equilibrium re-init (LBMSolver::restoreFromMacroscopic), costing
// the documented short transient; cheap enough to persist between app runs.
// ===========================================================================

struct CompactSnapshot::Impl {
    SnapshotKey    key{};
    std::uint64_t  flagsHash = 0;
    LatticeScaling scaling{};
    bool           hasData = false;
    std::vector<float> rho, u, v, w; ///< cellCount() floats each, unpadded.
};

CompactSnapshot::CompactSnapshot() : impl_(std::make_unique<Impl>()) {}
CompactSnapshot::~CompactSnapshot() = default;

bool CompactSnapshot::capture(LBMSolver& solver, const SnapshotKey& key,
                              cudaStream_t stream, std::string* error) {
    Impl& s = *impl_;
    const GridDims d = solver.dims();
    const long long ncells = d.cellCount();
    if (ncells <= 0) {
        if (error) *error = "solver holds no field to capture";
        return false;
    }
    s.rho.resize(static_cast<std::size_t>(ncells));
    s.u.resize(static_cast<std::size_t>(ncells));
    s.v.resize(static_cast<std::size_t>(ncells));
    s.w.resize(static_cast<std::size_t>(ncells));

    // Download the four macroscopic arrays. They were refreshed on the last
    // stepped frame (writeMacro on the batch's final step), which is exactly
    // the state the user sees on screen when they hit "save".
    const DeviceVelocityField vel = solver.velocityField();
    const std::size_t bytes = static_cast<std::size_t>(ncells) * sizeof(float);
    struct { const float* src; float* dst; } downloads[] = {
        {solver.deviceRho(), s.rho.data()}, {vel.u, s.u.data()},
        {vel.v, s.v.data()},                {vel.w, s.w.data()}};
    for (const auto& dl : downloads) {
        if (auto err = cudaMemcpyAsync(dl.dst, dl.src, bytes,
                                       cudaMemcpyDeviceToHost, stream);
            err != cudaSuccess) {
            if (error) *error = std::string("macroscopic download failed: ")
                              + cudaGetErrorString(err);
            return false;
        }
    }
    // The host buffers must be complete before this object is handed to the
    // disk worker — synchronize here, off the hot path.
    if (auto err = cudaStreamSynchronize(stream); err != cudaSuccess) {
        if (error) *error = std::string("stream sync failed: ")
                          + cudaGetErrorString(err);
        return false;
    }
    s.key = key;
    s.scaling = solver.scaling();
    const auto& flags = solver.hostFlags();
    s.flagsHash = hashFlags(flags.data(), flags.size());
    s.hasData = true;
    return true;
}

bool CompactSnapshot::restore(LBMSolver& solver, cudaStream_t stream) const {
    (void)stream; // the solver runs the upload on its own (same) stream
    const Impl& s = *impl_;
    if (!s.hasData) return false;
    const GridDims d = solver.dims();
    if (d.nx != s.key.dims.nx || d.ny != s.key.dims.ny || d.nz != s.key.dims.nz)
        return false;
    // restoreFromMacroscopic uploads, equilibrium-re-inits both buffers, and
    // runs the notifySnapshotRestored(false) bookkeeping internally.
    std::string err;
    if (!solver.restoreFromMacroscopic(s.rho.data(), s.u.data(), s.v.data(),
                                       s.w.data(), &err)) {
        std::fprintf(stderr, "[snapshot] compact restore failed: %s\n",
                     err.c_str());
        return false;
    }
    return true;
}

bool CompactSnapshot::saveToFile(const std::filesystem::path& path,
                                 std::string* error) const {
    const Impl& s = *impl_;
    if (!s.hasData) {
        if (error) *error = "no snapshot data to save";
        return false;
    }
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os) {
        if (error) *error = "cannot open " + path.string() + " for writing";
        return false;
    }

    // --- header: identity first, so a loader can validate cheaply ---------
    writePod(os, kFileMagic);
    writePod(os, kFileFormatVersion);
    const std::uint32_t idLen = static_cast<std::uint32_t>(s.key.geometryId.size());
    writePod(os, idLen);
    os.write(s.key.geometryId.data(), idLen);
    writePod(os, s.key.aoaBucket);
    writePod(os, s.key.u_lat);
    writePod(os, s.key.dims.nx);
    writePod(os, s.key.dims.ny);
    writePod(os, s.key.dims.nz);
    writePod(os, s.key.solverVersion);
    writePod(os, s.flagsHash);

    // --- scaling echo (field-by-field; restorers display effective Re) ----
    writePod(os, s.scaling.phys.chord_m);
    writePod(os, s.scaling.phys.airspeed_ms);
    writePod(os, s.scaling.phys.nu_m2s);
    writePod(os, s.scaling.chordCells);
    writePod(os, s.scaling.u_lat);
    writePod(os, s.scaling.dx);
    writePod(os, s.scaling.dt);
    writePod(os, s.scaling.nu_lat_target);
    writePod(os, s.scaling.nu_lat);
    writePod(os, s.scaling.tau);
    writePod(os, static_cast<std::uint8_t>(s.scaling.tauClamped ? 1 : 0));
    writePod(os, s.scaling.reTarget);
    writePod(os, s.scaling.reEffective);

    // --- payload -----------------------------------------------------------
    writePod(os, kCompressionRaw);
    const std::uint64_t ncells = s.rho.size();
    writePod(os, ncells);
    for (const std::vector<float>* arr : {&s.rho, &s.u, &s.v, &s.w}) {
        os.write(reinterpret_cast<const char*>(arr->data()),
                 static_cast<std::streamsize>(arr->size() * sizeof(float)));
    }
    os.flush();
    if (!os.good()) {
        if (error) *error = "write failed (disk full?) at " + path.string();
        return false;
    }
    return true;
}

bool CompactSnapshot::loadFromFile(const std::filesystem::path& path,
                                   std::string* error) {
    Impl& s = *impl_;
    s.hasData = false;
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        if (error) *error = "cannot open " + path.string();
        return false;
    }
    auto reject = [&](const char* why) {
        if (error) *error = std::string(why) + ": " + path.string();
        return false;
    };

    std::uint32_t magic = 0, fmt = 0;
    if (!readPod(is, magic) || magic != kFileMagic) return reject("bad magic");
    if (!readPod(is, fmt) || fmt != kFileFormatVersion)
        return reject("unsupported format version");

    std::uint32_t idLen = 0;
    if (!readPod(is, idLen) || idLen > 4096) return reject("corrupt geometry id");
    s.key.geometryId.resize(idLen);
    is.read(s.key.geometryId.data(), idLen);
    if (!readPod(is, s.key.aoaBucket)) return reject("truncated header");
    if (!readPod(is, s.key.u_lat)) return reject("truncated header");
    if (!readPod(is, s.key.dims.nx)) return reject("truncated header");
    if (!readPod(is, s.key.dims.ny)) return reject("truncated header");
    if (!readPod(is, s.key.dims.nz)) return reject("truncated header");
    if (!readPod(is, s.key.solverVersion)) return reject("truncated header");
    if (s.key.solverVersion != kSolverVersion)
        return reject("solver version mismatch");
    if (!readPod(is, s.flagsHash)) return reject("truncated header");

    std::uint8_t clamped = 0;
    bool ok = readPod(is, s.scaling.phys.chord_m)
           && readPod(is, s.scaling.phys.airspeed_ms)
           && readPod(is, s.scaling.phys.nu_m2s)
           && readPod(is, s.scaling.chordCells)
           && readPod(is, s.scaling.u_lat)
           && readPod(is, s.scaling.dx)
           && readPod(is, s.scaling.dt)
           && readPod(is, s.scaling.nu_lat_target)
           && readPod(is, s.scaling.nu_lat)
           && readPod(is, s.scaling.tau)
           && readPod(is, clamped)
           && readPod(is, s.scaling.reTarget)
           && readPod(is, s.scaling.reEffective);
    if (!ok) return reject("truncated scaling block");
    s.scaling.tauClamped = (clamped != 0);

    std::uint32_t compression = 0;
    std::uint64_t ncells = 0;
    if (!readPod(is, compression) || compression != kCompressionRaw)
        return reject("unsupported payload compression");
    if (!readPod(is, ncells)) return reject("truncated payload header");
    // Cross-check payload size against the key's own grid dims — a corrupt
    // count would otherwise drive a multi-GB allocation below.
    const std::uint64_t expect =
        static_cast<std::uint64_t>(s.key.dims.cellCount());
    if (ncells == 0 || ncells != expect) return reject("payload size mismatch");

    for (std::vector<float>* arr : {&s.rho, &s.u, &s.v, &s.w}) {
        arr->resize(static_cast<std::size_t>(ncells));
        is.read(reinterpret_cast<char*>(arr->data()),
                static_cast<std::streamsize>(ncells * sizeof(float)));
        if (!is.good()) return reject("truncated field payload");
    }
    s.hasData = true;
    return true;
}

bool CompactSnapshot::hasData() const { return impl_->hasData; }
const SnapshotKey& CompactSnapshot::key() const { return impl_->key; }
std::uint64_t CompactSnapshot::flagsHash() const { return impl_->flagsHash; }

// ===========================================================================
// DiskSnapshotCache: cache/ directory, LRU by file mtime (touched on every
// hit), byte-budget eviction, and a background worker for storeAsync so the
// frame loop never blocks on a ~380 MB write.
// ===========================================================================

struct DiskSnapshotCache::Impl {
    std::filesystem::path dir;
    std::uint64_t budget = 0;

    // All filesystem state below is guarded by m (store may run on the
    // worker while the UI thread calls find/setBudgetBytes).
    mutable std::mutex m;
    std::uint64_t used = 0;

    // Background store worker (lazy-started on first storeAsync).
    std::thread worker;
    std::mutex qm;
    std::condition_variable qcv;
    std::deque<std::shared_ptr<const CompactSnapshot>> queue;
    bool stop = false;
    bool workerStarted = false;

    /// @brief Recompute total cache size from the directory (startup scan).
    void scanLocked() {
        used = 0;
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
            if (!e.is_regular_file(ec)) continue;
            if (e.path().extension() != ".fcs") continue;
            used += static_cast<std::uint64_t>(e.file_size(ec));
        }
    }

    /// @brief Evict least-recently-used .fcs files (oldest mtime first) until
    /// the total fits the budget; never evicts @p protect (the file that was
    /// just stored must win — snapshot.h contract).
    void evictLocked(const std::filesystem::path& protect) {
        while (used > budget) {
            std::error_code ec;
            std::filesystem::path oldest;
            std::filesystem::file_time_type oldestTime =
                std::filesystem::file_time_type::max();
            std::uint64_t oldestSize = 0;
            for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
                if (!e.is_regular_file(ec)) continue;
                if (e.path().extension() != ".fcs") continue;
                if (e.path() == protect) continue;
                const auto t = e.last_write_time(ec);
                if (t < oldestTime) {
                    oldestTime = t;
                    oldest = e.path();
                    oldestSize = static_cast<std::uint64_t>(e.file_size(ec));
                }
            }
            if (oldest.empty()) break; // nothing evictable (one giant entry)
            if (!std::filesystem::remove(oldest, ec)) {
                // Locked by another process? Log and stop rather than spin.
                std::fprintf(stderr, "[cache] eviction of %s failed\n",
                             oldest.string().c_str());
                break;
            }
            used = (used > oldestSize) ? used - oldestSize : 0;
        }
    }

    /// @brief Worker loop: drain the queue, write each snapshot via store().
    void workerLoop(DiskSnapshotCache* owner) {
        for (;;) {
            std::shared_ptr<const CompactSnapshot> snap;
            {
                std::unique_lock<std::mutex> lk(qm);
                qcv.wait(lk, [this] { return stop || !queue.empty(); });
                if (stop && queue.empty()) return;
                snap = std::move(queue.front());
                queue.pop_front();
            }
            std::string err;
            if (!owner->store(*snap, &err))
                std::fprintf(stderr, "[cache] background store failed: %s\n",
                             err.c_str());
        }
    }
};

DiskSnapshotCache::DiskSnapshotCache(std::filesystem::path directory,
                                     std::uint64_t budgetBytes)
    : impl_(std::make_unique<Impl>()) {
    impl_->dir = std::move(directory);
    impl_->budget = budgetBytes;
    std::error_code ec;
    std::filesystem::create_directories(impl_->dir, ec);
    std::lock_guard<std::mutex> lk(impl_->m);
    impl_->scanLocked(); // seed LRU bookkeeping from existing entries
}

DiskSnapshotCache::~DiskSnapshotCache() {
    // Orderly worker shutdown: let queued writes finish (a snapshot the user
    // asked to save should not vanish because the app closed quickly).
    {
        std::lock_guard<std::mutex> lk(impl_->qm);
        impl_->stop = true;
    }
    impl_->qcv.notify_all();
    if (impl_->worker.joinable()) impl_->worker.join();
}

std::optional<std::filesystem::path> DiskSnapshotCache::find(
    const SnapshotKey& key) {
    std::lock_guard<std::mutex> lk(impl_->m);
    const std::filesystem::path path = impl_->dir / cacheFileName(key);
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) return std::nullopt;
    // A hit refreshes LRU recency: bump the mtime to "now".
    std::filesystem::last_write_time(
        path, std::filesystem::file_time_type::clock::now(), ec);
    return path;
}

bool DiskSnapshotCache::store(const CompactSnapshot& snapshot,
                              std::string* error) {
    if (!snapshot.hasData()) {
        if (error) *error = "snapshot holds no data";
        return false;
    }
    std::lock_guard<std::mutex> lk(impl_->m);
    const std::string name = cacheFileName(snapshot.key());
    const std::filesystem::path finalPath = impl_->dir / name;
    const std::filesystem::path tmpPath = impl_->dir / (name + ".tmp");

    // Write to a temp file then rename: a crash mid-write can never leave a
    // truncated .fcs that a later run would trust.
    if (!snapshot.saveToFile(tmpPath, error)) {
        std::error_code ec;
        std::filesystem::remove(tmpPath, ec);
        return false;
    }
    std::error_code ec;
    std::uint64_t replacedSize = 0;
    if (std::filesystem::exists(finalPath, ec))
        replacedSize = static_cast<std::uint64_t>(
            std::filesystem::file_size(finalPath, ec));
    std::filesystem::rename(tmpPath, finalPath, ec);
    if (ec) {
        if (error) *error = "rename to " + finalPath.string()
                          + " failed: " + ec.message();
        std::filesystem::remove(tmpPath, ec);
        return false;
    }
    const std::uint64_t newSize = static_cast<std::uint64_t>(
        std::filesystem::file_size(finalPath, ec));
    impl_->used = impl_->used - std::min(impl_->used, replacedSize) + newSize;

    // Budget enforcement: evict oldest-first, never the entry just stored.
    impl_->evictLocked(finalPath);
    return true;
}

void DiskSnapshotCache::storeAsync(
    std::shared_ptr<const CompactSnapshot> snapshot) {
    if (!snapshot) return;
    {
        std::lock_guard<std::mutex> lk(impl_->qm);
        if (!impl_->workerStarted) {
            // Lazy start: apps that never save snapshots never pay a thread.
            impl_->worker = std::thread([this] { impl_->workerLoop(this); });
            impl_->workerStarted = true;
        }
        impl_->queue.push_back(std::move(snapshot));
    }
    impl_->qcv.notify_one();
}

void DiskSnapshotCache::setBudgetBytes(std::uint64_t budgetBytes) {
    std::lock_guard<std::mutex> lk(impl_->m);
    impl_->budget = budgetBytes;
    // Shrinking the budget evicts immediately (snapshot.h contract); pass an
    // empty protect path so everything is fair game.
    impl_->evictLocked({});
}

std::uint64_t DiskSnapshotCache::usedBytes() const {
    std::lock_guard<std::mutex> lk(impl_->m);
    return impl_->used;
}

} // namespace foilcfd
