// OS-specific shim implementation: high-resolution timer, executable
// directory, and native file dialogs (Win32 IFileOpenDialog/IFileSaveDialog).
// The only file outside this directory allowed to include Win32 headers is
// nothing — this is the quarantine zone. Non-Windows builds compile the stub
// dialog paths (return nullopt) and the chrono timer fallback.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include "platform.h"

#include <chrono>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shobjidl.h> // IFileOpenDialog / IFileSaveDialog (Vista+ common dialogs)
// COM plumbing for the dialogs; pulled in here so CMake stays untouched.
#pragma comment(lib, "ole32.lib")
#endif

namespace foilcfd::platform {

#ifdef _WIN32
namespace {

/// @brief UTF-8 -> UTF-16 conversion for dialog strings (titles, filters).
/// The app is UTF-8 throughout; Win32 COM dialogs speak wide chars only.
std::wstring widen(const std::string& utf8) {
    if (utf8.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(),
                                      static_cast<int>(utf8.size()), nullptr, 0);
    std::wstring out(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(),
                        static_cast<int>(utf8.size()), out.data(), n);
    return out;
}

/// @brief Scoped COM init: the dialogs need an apartment-threaded COM context
/// on the calling (main/UI) thread. RPC_E_CHANGED_MODE means somebody already
/// initialized COM differently — the dialog still works, just don't uninit.
struct ComScope {
    HRESULT hr;
    ComScope()
        : hr(CoInitializeEx(nullptr,
                            COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE)) {}
    ~ComScope() {
        if (SUCCEEDED(hr)) CoUninitialize();
    }
    bool usable() const { return SUCCEEDED(hr) || hr == RPC_E_CHANGED_MODE; }
};

/// @brief Shared body for open/save: configure an IFileDialog, show it modal,
/// and extract the chosen filesystem path. Returns nullopt on cancel or any
/// COM failure (a failed dialog must never crash the app — plan section 1
/// portability posture treats dialogs as best-effort).
/// @param clsid         CLSID_FileOpenDialog or CLSID_FileSaveDialog.
/// @param title         Dialog title (UTF-8).
/// @param filters       File-type filter list.
/// @param suggestedName Pre-filled filename (save dialog only; empty = none).
std::optional<std::filesystem::path> runFileDialog(
    REFCLSID clsid, const std::string& title,
    const std::vector<FileFilter>& filters, const std::string& suggestedName) {
    ComScope com;
    if (!com.usable()) return std::nullopt;

    // IFileDialog is the common base of both open and save variants, so one
    // configuration path serves both CLSIDs.
    IFileDialog* dlg = nullptr;
    if (FAILED(CoCreateInstance(clsid, nullptr, CLSCTX_INPROC_SERVER,
                                IID_PPV_ARGS(&dlg)))) {
        return std::nullopt;
    }

    const std::wstring wtitle = widen(title);
    dlg->SetTitle(wtitle.c_str());

    // Build the COMDLG_FILTERSPEC array. The wide strings must outlive Show(),
    // so keep them in vectors that stay in scope.
    std::vector<std::wstring> names, patterns;
    std::vector<COMDLG_FILTERSPEC> specs;
    names.reserve(filters.size());
    patterns.reserve(filters.size());
    for (const FileFilter& f : filters) {
        names.push_back(widen(f.description));
        patterns.push_back(widen(f.pattern));
        specs.push_back({names.back().c_str(), patterns.back().c_str()});
    }
    if (!specs.empty()) {
        dlg->SetFileTypes(static_cast<UINT>(specs.size()), specs.data());
        dlg->SetFileTypeIndex(1); // 1-based: preselect the first filter.
        // Default extension from the first pattern ("*.png" -> "png") so the
        // save dialog appends it when the user types a bare name.
        const std::string& p0 = filters.front().pattern;
        if (const auto dot = p0.find('.');
            dot != std::string::npos && dot + 1 < p0.size()) {
            dlg->SetDefaultExtension(widen(p0.substr(dot + 1)).c_str());
        }
    }
    if (!suggestedName.empty()) {
        dlg->SetFileName(widen(suggestedName).c_str());
    }
    // Filesystem results only — no library/shell virtual locations.
    DWORD opts = 0;
    if (SUCCEEDED(dlg->GetOptions(&opts))) {
        dlg->SetOptions(opts | FOS_FORCEFILESYSTEM);
    }

    std::optional<std::filesystem::path> result;
    // nullptr owner: GLFW window handles are not plumbed through this shim;
    // the dialog is still application-modal in practice for a single window.
    if (SUCCEEDED(dlg->Show(nullptr))) {
        IShellItem* item = nullptr;
        if (SUCCEEDED(dlg->GetResult(&item))) {
            PWSTR ws = nullptr;
            if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &ws))) {
                result = std::filesystem::path(ws);
                CoTaskMemFree(ws);
            }
            item->Release();
        }
    }
    dlg->Release();
    return result;
}

} // namespace
#endif // _WIN32

std::optional<std::filesystem::path> openFileDialog(
    const std::string& title, const std::vector<FileFilter>& filters) {
#ifdef _WIN32
    return runFileDialog(CLSID_FileOpenDialog, title, filters, std::string());
#else
    // Non-Windows stub: no native dialog dependency is worth carrying for the
    // untested Linux path; callers fall back to drag-and-drop (plan 7.1).
    (void)title;
    (void)filters;
    return std::nullopt;
#endif
}

std::optional<std::filesystem::path> saveFileDialog(
    const std::string& title, const std::vector<FileFilter>& filters,
    const std::string& suggestedName) {
#ifdef _WIN32
    return runFileDialog(CLSID_FileSaveDialog, title, filters, suggestedName);
#else
    (void)title;
    (void)filters;
    (void)suggestedName;
    return std::nullopt;
#endif
}

double timerSeconds() {
#ifdef _WIN32
    // QueryPerformanceCounter: monotonic, sub-microsecond on modern Windows.
    static const double invFreq = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return 1.0 / static_cast<double>(f.QuadPart);
    }();
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return static_cast<double>(t.QuadPart) * invFreq;
#else
    // Portable fallback for the (untested) Linux build path.
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double>(now).count();
#endif
}

std::filesystem::path executableDirectory() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH] = {};
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
        return std::filesystem::path(buf).parent_path();
    }
#endif
    // Fallback: current working directory (good enough for dev runs).
    return std::filesystem::current_path();
}

int gpuUtilizationPercent() {
#ifdef _WIN32
    // The driver's management library (nvml.dll, installed system-wide with
    // the display driver) is loaded dynamically so the app keeps running on
    // machines without it — no import-table dependency, no SDK lib to link.
    // Only the three documented C entry points needed for the utilization
    // query are resolved; minimal ABI mirror types are declared locally.
    using NvmlReturn = int;                       // 0 == NVML_SUCCESS
    using NvmlDevice = void*;
    struct NvmlUtilization { unsigned int gpu; unsigned int memory; };
    using FnInit       = NvmlReturn (*)();
    using FnGetHandle  = NvmlReturn (*)(unsigned int, NvmlDevice*);
    using FnGetUtil    = NvmlReturn (*)(NvmlDevice, NvmlUtilization*);

    // One-time resolve, cached for the process lifetime. A failed resolve
    // latches the device handle null so every later call is a cheap early-out.
    static FnGetUtil  getUtil = nullptr;
    static NvmlDevice device  = nullptr;
    static bool tried = false;
    if (!tried) {
        tried = true;
        if (HMODULE lib = LoadLibraryW(L"nvml.dll")) {
            const auto init = reinterpret_cast<FnInit>(
                GetProcAddress(lib, "nvmlInit_v2"));
            const auto handle = reinterpret_cast<FnGetHandle>(
                GetProcAddress(lib, "nvmlDeviceGetHandleByIndex_v2"));
            getUtil = reinterpret_cast<FnGetUtil>(
                GetProcAddress(lib, "nvmlDeviceGetUtilizationRates"));
            // Device 0 matches the app's single-GPU CUDA usage.
            if (!init || !handle || !getUtil || init() != 0
                || handle(0u, &device) != 0) {
                getUtil = nullptr;
                device = nullptr;
            }
        }
    }
    if (getUtil && device) {
        NvmlUtilization u{};
        if (getUtil(device, &u) == 0) {
            return static_cast<int>(u.gpu);
        }
    }
#endif
    return -1; // unavailable: caller hides the readout
}

} // namespace foilcfd::platform
