# FoilCFD build system. Single configure command (see BUILDING.md):
#   cmake -B build -G "Visual Studio 17 2022" -A x64 -T cuda=12.9
#   cmake --build build --config Release
# FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
cmake_minimum_required(VERSION 3.24)

# ---------------------------------------------------------------------------
# CUDA architectures must be decided before project() enables the language.
# Default "native" targets the dev machine's card (sm_120 on the RTX 5090);
# release/distribution builds override with the explicit fallback list.
# ---------------------------------------------------------------------------
set(FOILCFD_CUDA_ARCHS "native" CACHE STRING
    "CUDA architectures (use \"86;89;120\" for release builds)")
set(CMAKE_CUDA_ARCHITECTURES "${FOILCFD_CUDA_ARCHS}")

project(FoilCFD LANGUAGES CXX CUDA)

# C++20 host code, CUDA C++ kernels (plan section 1).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Single-TU kernels: separable compilation stays OFF (plan section 2).
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

# MSVC: source and execution charset UTF-8 everywhere, including through nvcc.
if(MSVC)
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()

# If nvcc rejects a too-new MSVC (17.14+ vs the toolkit's tested ceiling),
# uncomment per BUILDING.md:
# string(APPEND CMAKE_CUDA_FLAGS " --allow-unsupported-compiler")

find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Vendored dependencies (plan section 2): GLFW 3.4 + Dear ImGui docking
# branch via FetchContent. No other third-party code.
# ---------------------------------------------------------------------------
include(FetchContent)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE)

FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        docking
    GIT_SHALLOW    TRUE)

FetchContent_MakeAvailable(glfw imgui)

# GL loader: the glad2 header bundled with GLFW (header-only; exactly one TU
# defines GLAD_GL_IMPLEMENTATION — src/platform/gl_loader.cpp). If a future
# GLFW drop removes it, the fallback is a hand-rolled WGL loader in
# src/platform/ exposing the GL 4.6 entry points the renderer uses.
set(FOILCFD_GLAD_HEADER "${glfw_SOURCE_DIR}/deps/glad/gl.h")
if(NOT EXISTS "${FOILCFD_GLAD_HEADER}")
    message(FATAL_ERROR
        "Bundled glad header not found at ${FOILCFD_GLAD_HEADER}. "
        "Implement the WGL fallback loader in src/platform/ (see CMakeLists).")
endif()

# Dear ImGui ships no CMakeLists — build core + the two backends we use as a
# static lib. The OpenGL3 backend uses its own embedded loader, so it does
# not conflict with glad.
add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
target_include_directories(imgui_lib PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui_lib PUBLIC glfw)

# ---------------------------------------------------------------------------
# The application.
# ---------------------------------------------------------------------------
add_executable(FoilCFD
    src/main.cpp
    src/app/aircraft_manifest.cpp
    src/app/camera.cpp
    src/app/ui.cpp
    src/geom/airfoil.cpp
    src/geom/stl.cpp
    src/geom/vg.cpp
    src/geom/vg_audit.cpp
    src/geom/voxelizer.cpp
    src/geom/voxelizer_stl.cu
    src/platform/gl_loader.cpp
    src/platform/platform.cpp
    src/render/particles.cu
    src/render/tufts.cu
    src/render/viz.cpp
    src/sim/lbm_core.cu
    src/sim/lbm_refine.cu
    src/sim/lbm_solver.cpp
    src/sim/lbm_wallmodel.cu
    src/sim/snapshot.cpp
    src/sim/wall_model.cpp)

target_include_directories(FoilCFD PRIVATE
    src
    ${glfw_SOURCE_DIR}/deps)   # <glad/gl.h>

target_link_libraries(FoilCFD PRIVATE
    glfw
    imgui_lib
    CUDA::cudart
    $<$<PLATFORM_ID:Windows>:opengl32>)

# Console subsystem so --selftest output and watchdog logs are visible.
if(MSVC)
    set_target_properties(FoilCFD PROPERTIES WIN32_EXECUTABLE OFF)
endif()

# ---------------------------------------------------------------------------
# Tests (CTest). Eight real tests: four host-only (units math, .dat parsing,
# voxelizer parity, aircraft manifest) and four labelled "gpu" (cuda_smoke
# toolchain canary plus the M0 Taylor-Green / M1 Ghia cavity / M2 cylinder-
# Strouhal physics milestones, plan section 10). `ctest -LE gpu` skips the
# device-bound set on machines without a CUDA card.
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
