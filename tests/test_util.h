// Minimal hand-rolled test harness shared by every FoilCFD CTest executable:
// CHECK macros that record failures (with file:line and the failed expression)
// without aborting, approximate-comparison helpers, and the exit-code summary.
// Deliberately dependency-free so test TUs build instantly under MSVC.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <cmath>
#include <cstdio>

namespace foilcfd::testutil {

// Global pass/fail tally for the current test executable. Plain ints are fine:
// every test runs single-threaded and each executable owns its own process.
inline int g_checksRun    = 0;
inline int g_checksFailed = 0;

/// @brief Record one check result; print a diagnostic line on failure.
/// Kept as a function (not buried in the macro) so the macro stays tiny and
/// the failure formatting lives in exactly one place.
/// @param ok    Whether the check passed.
/// @param expr  Stringified expression for the log.
/// @param file  Source file of the check site.
/// @param line  Source line of the check site.
inline void recordCheck(bool ok, const char* expr, const char* file, int line) {
    ++g_checksRun;
    if (!ok) {
        ++g_checksFailed;
        std::printf("FAIL %s:%d: %s\n", file, line, expr);
    }
}

/// @brief True when |a - b| <= tol (absolute tolerance compare for floats).
inline bool approxEq(double a, double b, double tol) {
    return std::fabs(a - b) <= tol;
}

/// @brief True when a and b agree to a relative tolerance (guards b ~ 0 by
/// falling back to absolute comparison against the same tolerance).
inline bool approxRel(double a, double b, double relTol) {
    const double scale = std::fabs(b);
    return std::fabs(a - b) <= relTol * (scale > 1.0 ? scale : 1.0);
}

/// @brief Print the summary line and produce the process exit code for main().
/// CTest treats nonzero exit as failure, so this is the single funnel point.
/// @param testName Executable name for the log (e.g. "test_units").
/// @return 0 when every check passed, 1 otherwise.
inline int finish(const char* testName) {
    if (g_checksFailed == 0) {
        std::printf("%s: PASS (%d checks)\n", testName, g_checksRun);
        return 0;
    }
    std::printf("%s: FAIL (%d of %d checks failed)\n", testName,
                g_checksFailed, g_checksRun);
    return 1;
}

} // namespace foilcfd::testutil

// Non-fatal check: failures are tallied and the test keeps running so one
// broken assertion still shows the full failure picture in the ctest log.
#define TCHECK(expr) \
    ::foilcfd::testutil::recordCheck(static_cast<bool>(expr), #expr, __FILE__, __LINE__)

// Check with a custom printf-style message appended on failure (the message
// arguments are only evaluated when the check fails).
#define TCHECK_MSG(expr, ...)                                                   \
    do {                                                                        \
        const bool ok_ = static_cast<bool>(expr);                               \
        ::foilcfd::testutil::recordCheck(ok_, #expr, __FILE__, __LINE__);       \
        if (!ok_) { std::printf("     "); std::printf(__VA_ARGS__); std::printf("\n"); } \
    } while (0)

// Fatal variant: when the condition fails, later checks would crash or be
// meaningless (e.g. solver init failed), so finish immediately with FAIL.
#define TREQUIRE(expr)                                                          \
    do {                                                                        \
        const bool ok_ = static_cast<bool>(expr);                               \
        ::foilcfd::testutil::recordCheck(ok_, #expr, __FILE__, __LINE__);       \
        if (!ok_) {                                                             \
            std::printf("     fatal: cannot continue past this failure\n");     \
            return ::foilcfd::testutil::finish("(aborted early)");              \
        }                                                                       \
    } while (0)
