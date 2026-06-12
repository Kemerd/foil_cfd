# FoilCFD test suite (plan section 10 milestones + section 3 test list).
#
# Host tests (always runnable):
#   test_units     - units.h scaling/round-trip/clamp/ramp contract
#   test_airfoil   - .dat loader (Lednicer ls413 + Selig a18), NACA generator
#   test_voxelizer - boundary flags, diamond parity voxelization, TE closure
#   test_manifest  - aircraft_manifest.csv parse/resolve (plan 15.5), quoted
#                    commas, malformed-row skip-with-reason
#   test_wallmodel - wall-model cell list: normals, masks, VG exclusion
# GPU tests (labeled "gpu" so CI-less machines exclude them: ctest -LE gpu):
#   cuda_smoke      - toolchain canary (real kernel on the installed card)
#   m0_taylor_green - 64^3 TG decay: monotone KE, no NaN over 50k steps
#   m1_cavity       - Re 1000 lid-driven cavity vs Ghia et al. (1982)
#   m2_cylinder     - D=40 cylinder at Re 150: lift-spectrum Strouhal in band
# FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

# ---------------------------------------------------------------------------
# Solver objects shared by the three GPU milestone tests. Built once as a
# static lib so lbm_core.cu goes through nvcc a single time; deliberately
# limited to the sim/ module so geometry churn cannot break the GPU tests.
# ---------------------------------------------------------------------------
add_library(foilcfd_test_sim STATIC
    ${CMAKE_SOURCE_DIR}/src/sim/lbm_core.cu
    ${CMAKE_SOURCE_DIR}/src/sim/lbm_refine.cu
    ${CMAKE_SOURCE_DIR}/src/sim/lbm_solver.cpp
    ${CMAKE_SOURCE_DIR}/src/sim/lbm_wallmodel.cu
    ${CMAKE_SOURCE_DIR}/src/sim/wall_model.cpp)
target_include_directories(foilcfd_test_sim PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(foilcfd_test_sim PUBLIC CUDA::cudart)

# ---------------------------------------------------------------------------
# Host-only tests.
# ---------------------------------------------------------------------------

# units.h is header-only; no module sources needed.
add_executable(test_units test_units.cpp)
target_include_directories(test_units PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_test(NAME test_units COMMAND test_units)

# Airfoil parsing/generation. The absolute airfoils/ path is injected so the
# test finds the in-tree reference files from any working directory.
add_executable(test_airfoil
    test_airfoil.cpp
    ${CMAKE_SOURCE_DIR}/src/geom/airfoil.cpp)
target_include_directories(test_airfoil PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(test_airfoil PRIVATE
    FOILCFD_AIRFOIL_DIR="${CMAKE_SOURCE_DIR}/airfoils")
add_test(NAME test_airfoil COMMAND test_airfoil)

# Voxelizer parity + TE closure. voxelizer.h pulls lbm_core.cuh for the flag
# enum/grid types, so the CUDA runtime headers must be on the include path
# (CUDA::cudart carries them); no kernels are launched by this test.
add_executable(test_voxelizer
    test_voxelizer.cpp
    ${CMAKE_SOURCE_DIR}/src/geom/voxelizer.cpp
    ${CMAKE_SOURCE_DIR}/src/geom/vg.cpp  # voxelizer.cpp's deriveVGPatchBoxFine
    ${CMAKE_SOURCE_DIR}/src/geom/airfoil.cpp)
target_include_directories(test_voxelizer PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_voxelizer PRIVATE CUDA::cudart)
add_test(NAME test_voxelizer COMMAND test_voxelizer)

# q-LIBB link-list geometry (host: cut fraction, clamp, ffFluid, densify).
add_executable(test_qlibb
    test_qlibb.cpp
    ${CMAKE_SOURCE_DIR}/src/geom/vg.cpp
    ${CMAKE_SOURCE_DIR}/src/geom/voxelizer.cpp
    ${CMAKE_SOURCE_DIR}/src/geom/airfoil.cpp)
target_include_directories(test_qlibb PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_qlibb PRIVATE CUDA::cudart)
add_test(NAME test_qlibb COMMAND test_qlibb)

# Wall-model cell list builder (wall_model.h): selection/link masks, stair-
# step normals (flat, 45-degree wedge, sphere), periodic z, determinism, VG
# exclusion, sample fallback. Header-includes lbm_core.cuh for the flag enum,
# so CUDA runtime headers are needed on the include path; no kernels launch.
add_executable(test_wallmodel
    test_wallmodel.cpp
    ${CMAKE_SOURCE_DIR}/src/sim/wall_model.cpp)
target_include_directories(test_wallmodel PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_wallmodel PRIVATE CUDA::cudart)
add_test(NAME test_wallmodel COMMAND test_wallmodel)

# VG vortex-audit math (geom/vg_audit.h): Lamb-Oseen circulation recovery +
# Wendt-correlation spot/invariant checks. Links the sim lib because the
# audit orchestration references LBMSolver (no kernels run in this test).
add_executable(test_vgaudit
    test_vgaudit.cpp
    ${CMAKE_SOURCE_DIR}/src/geom/vg_audit.cpp)
target_link_libraries(test_vgaudit PRIVATE foilcfd_test_sim)
add_test(NAME test_vgaudit COMMAND test_vgaudit)

# Aircraft manifest (plan 15.5): CSV parse + case-insensitive dat resolution
# against the REAL airfoils/ tree, so the same path injection as test_airfoil.
# aircraft_manifest.cpp only needs headers from geom/ and platform/ (both
# inline at the symbols it uses), so no further module sources are linked.
add_executable(test_manifest
    test_manifest.cpp
    ${CMAKE_SOURCE_DIR}/src/app/aircraft_manifest.cpp)
target_include_directories(test_manifest PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(test_manifest PRIVATE
    FOILCFD_AIRFOIL_DIR="${CMAKE_SOURCE_DIR}/airfoils")
add_test(NAME test_manifest COMMAND test_manifest)

# ---------------------------------------------------------------------------
# GPU tests.
# ---------------------------------------------------------------------------

# Toolchain smoke test: launches a real kernel, verifies results on host.
add_executable(cuda_smoke cuda_smoke.cu)
target_link_libraries(cuda_smoke PRIVATE CUDA::cudart)
add_test(NAME cuda_smoke COMMAND cuda_smoke)

# Physics milestones M0-M3 (plan section 10 + plan M-refine). Host TUs
# driving the solver's public API; all device work happens in the sim module.
foreach(gpu_test IN ITEMS m0_taylor_green m1_cavity m2_cylinder m3_refine
        m4_finer m4_wallmodel)
    add_executable(${gpu_test} ${gpu_test}.cpp)
    target_link_libraries(${gpu_test} PRIVATE foilcfd_test_sim)
    add_test(NAME ${gpu_test} COMMAND ${gpu_test})
endforeach()

# M5 integration protocol: full app pipeline (geometry + patch + wall model
# + vortex audit) on a real airfoil — needs geom sources alongside the sim
# lib. Minutes of wall clock (the 4x-patch case dominates at ~m^4 cost);
# labeled "slow" so quick loops can exclude it: ctest -LE slow.
add_executable(m5_integration
    m5_integration.cpp
    ${CMAKE_SOURCE_DIR}/src/geom/airfoil.cpp
    ${CMAKE_SOURCE_DIR}/src/geom/vg.cpp
    ${CMAKE_SOURCE_DIR}/src/geom/vg_audit.cpp
    ${CMAKE_SOURCE_DIR}/src/geom/voxelizer.cpp)
target_link_libraries(m5_integration PRIVATE foilcfd_test_sim)
add_test(NAME m5_integration COMMAND m5_integration)

# Manual validation tool (deliberately NOT registered with CTest): the
# LS(1)-0413 lift-polar sweep against the digitized NASA TM X-72843 curves,
# wall model on/off. ~15 minutes of GPU time; run from the repo root so the
# CSV lands in validation/.
add_executable(polar_ls413
    polar_ls413.cpp
    ${CMAKE_SOURCE_DIR}/src/geom/airfoil.cpp
    ${CMAKE_SOURCE_DIR}/src/geom/voxelizer.cpp
    ${CMAKE_SOURCE_DIR}/src/geom/vg.cpp) # voxelizer.cpp's deriveVGPatchBoxFine
target_link_libraries(polar_ls413 PRIVATE foilcfd_test_sim)
target_compile_definitions(polar_ls413 PRIVATE
    FOILCFD_AIRFOIL_DIR="${CMAKE_SOURCE_DIR}/airfoils")

# The "gpu" label lets machines without an NVIDIA card (or CI) skip every
# device-touching test in one flag: ctest -LE gpu.
set_tests_properties(cuda_smoke m0_taylor_green m1_cavity m2_cylinder m3_refine
    m4_finer m4_wallmodel
    PROPERTIES LABELS "gpu")
set_tests_properties(m5_integration PROPERTIES LABELS "gpu;slow" TIMEOUT 900)

# Per-test wall-clock budgets (seconds). M0 is specified < 3 min; M2 < 5 min.
set_tests_properties(test_units test_airfoil test_voxelizer test_manifest
    test_wallmodel test_vgaudit test_qlibb
    PROPERTIES TIMEOUT 60)
set_tests_properties(cuda_smoke PROPERTIES TIMEOUT 120)
set_tests_properties(m0_taylor_green PROPERTIES TIMEOUT 180)
set_tests_properties(m1_cavity PROPERTIES TIMEOUT 300)
set_tests_properties(m2_cylinder PROPERTIES TIMEOUT 300)
set_tests_properties(m3_refine PROPERTIES TIMEOUT 180)
set_tests_properties(m4_finer PROPERTIES TIMEOUT 240)
set_tests_properties(m4_wallmodel PROPERTIES TIMEOUT 300)
