// The ONLY header whose implementation may touch OS-specific APIs (plan
// section 1: Windows-first, keep Linux-portable by quarantining Win32 here).
// Surface: native file dialogs and a high-resolution monotonic timer.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace foilcfd::platform {

/// @brief One file-type filter for the dialogs, e.g. {"Airfoil data", "*.dat"}.
struct FileFilter {
    std::string description; ///< Human-readable label.
    std::string pattern;     ///< Glob pattern ("*.dat", "*.stl").
};

/// @brief Show a native modal "Open File" dialog.
/// @param title   Dialog title.
/// @param filters File-type filters (first is preselected).
/// @return Chosen path, or nullopt if the user cancelled.
std::optional<std::filesystem::path> openFileDialog(
    const std::string& title, const std::vector<FileFilter>& filters);

/// @brief Show a native modal "Save File" dialog.
/// @param title            Dialog title.
/// @param filters          File-type filters.
/// @param suggestedName    Pre-filled filename (e.g. "foilcfd_screenshot.png").
/// @return Chosen path, or nullopt if the user cancelled.
std::optional<std::filesystem::path> saveFileDialog(
    const std::string& title, const std::vector<FileFilter>& filters,
    const std::string& suggestedName);

/// @brief Monotonic high-resolution timestamp in seconds (QueryPerformance-
/// Counter on Windows). Used for the adaptive steps-per-frame pacer and the
/// MLUPS readout — std::chrono::steady_clock is acceptable but this shim
/// keeps the precision contract explicit and swappable.
double timerSeconds();

/// @brief Directory of the running executable — the snapshot disk cache
/// lives at <exeDir>/cache (plan section 8) regardless of the CWD the app
/// was launched from.
std::filesystem::path executableDirectory();

} // namespace foilcfd::platform
