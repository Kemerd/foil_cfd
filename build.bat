@echo off
setlocal EnableDelayedExpansion

:: ===========================================================================
:: FoilCFD build script for Windows
:: Usage:
::   build.bat           -- configure (if needed) + build Release
::   build.bat clean     -- wipe build\ and reconfigure + build
::   build.bat run       -- build then launch the app
::   build.bat test      -- build then run all CTests
::   build.bat selftest  -- build then run --selftest (MLUPS benchmark + screenshot)
::   build.bat release   -- build with multi-arch CUDA (86;89;120) for distribution
:: ===========================================================================

set BUILD_DIR=build
set CONFIG=Release
set EXE=%BUILD_DIR%\%CONFIG%\FoilCFD.exe
set GENERATOR=Visual Studio 17 2022
set CUDA_TOOLSET=cuda=12.9

:: Parse optional first argument
set ACTION=build
if not "%~1"=="" set ACTION=%~1

:: ---------------------------------------------------------------------------
:: Handle 'clean': wipe the build directory so CMake starts fresh.
:: This is needed if you change the CUDA toolkit or switch CUDA architectures.
:: ---------------------------------------------------------------------------
if /i "%ACTION%"=="clean" (
    echo [FoilCFD] Cleaning build directory...
    if exist "%BUILD_DIR%" (
        rmdir /s /q "%BUILD_DIR%"
        echo [FoilCFD] Removed %BUILD_DIR%\
    ) else (
        echo [FoilCFD] Nothing to clean.
    )
    :: Fall through to configure + build after clean
    set ACTION=build
)

:: ---------------------------------------------------------------------------
:: Determine CUDA architecture flags.
:: 'release' action targets a broad GPU family (Ampere/Ada/Blackwell).
:: All other actions use 'native' to compile only for the local GPU (fastest).
:: ---------------------------------------------------------------------------
set CUDA_ARCHS=native
if /i "%ACTION%"=="release" (
    set CUDA_ARCHS=86;89;120
    echo [FoilCFD] Distribution build: targeting CUDA archs 86 89 120
)

:: ---------------------------------------------------------------------------
:: Configure step — only run cmake configure if CMakeCache.txt is missing,
:: which means either first run or after a clean. This avoids re-running
:: FetchContent network fetches on every build.
:: ---------------------------------------------------------------------------
if not exist "%BUILD_DIR%\CMakeCache.txt" (
    echo [FoilCFD] Configuring with CMake...
    echo   Generator : %GENERATOR%
    echo   Toolset   : %CUDA_TOOLSET%
    echo   CUDA Archs: %CUDA_ARCHS%
    echo.
    cmake -B "%BUILD_DIR%" ^
          -G "%GENERATOR%" ^
          -A x64 ^
          -T "%CUDA_TOOLSET%" ^
          -DFOILCFD_CUDA_ARCHS="%CUDA_ARCHS%"
    if errorlevel 1 (
        echo.
        echo [FoilCFD] ERROR: CMake configure failed.
        echo   - Ensure CUDA 12.9 is installed with the VS 2022 integration.
        echo   - The nvcc on PATH may be CUDA 10.1 -- that is OK, -T cuda=12.9 overrides it.
        echo   - If the configure log does NOT say "CUDA compiler identification is NVIDIA 12.9.x",
        echo     delete build\ and retry.
        exit /b 1
    )
    echo.
)

:: ---------------------------------------------------------------------------
:: Build step — always invoked (cmake --build is incremental via MSBuild).
:: ---------------------------------------------------------------------------
echo [FoilCFD] Building %CONFIG%...
cmake --build "%BUILD_DIR%" --config %CONFIG% --parallel
if errorlevel 1 (
    echo.
    echo [FoilCFD] ERROR: Build failed. Check compiler output above.
    exit /b 1
)

echo.
echo [FoilCFD] Build succeeded: %EXE%
echo.

:: ---------------------------------------------------------------------------
:: Post-build actions based on ACTION.
:: ---------------------------------------------------------------------------

if /i "%ACTION%"=="run" (
    echo [FoilCFD] Launching app...
    start "" "%EXE%"
    goto :done
)

if /i "%ACTION%"=="selftest" (
    echo [FoilCFD] Running selftest ^(200 steps, outputs screenshots\selftest.png^)...
    echo.
    "%EXE%" --selftest
    if errorlevel 1 (
        echo [FoilCFD] selftest FAILED.
        exit /b 1
    )
    goto :done
)

if /i "%ACTION%"=="test" (
    echo [FoilCFD] Running CTest suite...
    echo.
    ctest --test-dir "%BUILD_DIR%" -C %CONFIG% --output-on-failure
    if errorlevel 1 (
        echo [FoilCFD] One or more tests FAILED.
        exit /b 1
    )
    goto :done
)

:: Default and 'release' just build -- already done above.

:done
endlocal
