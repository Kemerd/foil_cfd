# Building FoilCFD

Verified 2026-06-11 on the primary dev machine (Windows 11 Pro, VS 2022
Enterprise 17.14 / MSVC 19.44, CMake 3.31.7, RTX 5090 driver 596.36).

## Exact verified commands

From the repository root, in PowerShell:

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64 -T cuda=12.9
cmake --build build --config Release
```

Run the app:

```powershell
.\build\Release\FoilCFD.exe              # interactive
.\build\Release\FoilCFD.exe --selftest   # 200 steps + screenshots\selftest.png, prints MLUPS + PASS
```

Run the tests:

```powershell
ctest --test-dir build -C Release
```

All seven tests must pass. Three are host-only (`test_units`, `test_airfoil`,
`test_voxelizer`); four carry the `gpu` CTest label and need a physical CUDA
card (`cuda_smoke`, `m0_taylor_green`, `m1_cavity`, `m2_cylinder` — the plan
§10 physics milestones; observed runtimes on the dev 5090: ~0.1 s / 1.7 s /
2 s / 12 s). On a machine without a GPU, run the host set only:

```powershell
ctest --test-dir build -C Release -LE gpu
```

## Critical environment facts (this machine)

- **The `nvcc` on PATH is CUDA 10.1 — never use it.** The `-T cuda=12.9`
  toolset argument makes MSBuild use CUDA 12.9 from
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9`. The configure
  log must say `CUDA compiler identification is NVIDIA 12.9.x`.
- MSVC 19.44 (VS 17.14) is accepted by nvcc 12.9 as-is —
  `--allow-unsupported-compiler` was NOT needed. If a future VS update breaks
  this, uncomment the `CMAKE_CUDA_FLAGS` line near the top of `CMakeLists.txt`.
- CUDA architectures default to `native` (compiles sm_120 on the 5090). For
  release/distribution builds configure with:

  ```powershell
  cmake -B build -G "Visual Studio 17 2022" -A x64 -T cuda=12.9 -DFOILCFD_CUDA_ARCHS="86;89;120"
  ```

- First configure clones GLFW 3.4 and Dear ImGui (docking) from GitHub via
  FetchContent — network access is required once; afterwards the sources are
  cached in `build/_deps/`.
- The GL loader is the glad2 header bundled with GLFW
  (`build/_deps/glfw-src/deps/glad/gl.h`), instantiated in exactly one TU:
  `src/platform/gl_loader.cpp`. Note: this bundled header exposes core GL
  entry points **up to 3.3** (plus extensions). The context itself is 4.6
  core; see `notes/integration_scaffold.md` before calling newer GL functions.

## Troubleshooting

- *"No CUDA toolset found"*: the `-T cuda=12.9` flag is missing, or the CUDA
  12.9 Visual Studio integration is not installed for VS 2022.
- *Configure picks the wrong CUDA*: delete `build/` entirely and reconfigure —
  CMake caches the compiler choice.
- *TDR timeouts on huge grids*: the solver caps per-present work at ~10 ms,
  but if you push extreme grids consider raising the `TdrDelay` registry key
  (HKLM\System\CurrentControlSet\Control\GraphicsDrivers).
