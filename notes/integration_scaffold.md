# Integration notes — scaffold (build system + public headers)

## Facts downstream agents must know (not proposals — already true in the tree)

### GL loader covers core GL 3.3, context is 4.6
The glad2 header bundled with GLFW 3.4 (`${glfw_SOURCE_DIR}/deps/glad/gl.h`,
instantiated in `src/platform/gl_loader.cpp`) generates loaders for core GL
**1.0–3.3** plus extensions only. The app still creates a 4.6 core context
(main.cpp window hints) and ImGui compiles `#version 460` shaders fine.

Render agent: the plan's renderer (VBO particles, GLSL programs, textures,
glReadPixels) fits entirely inside GL 3.3 entry points. If you need a >3.3
function (e.g. DSA, glTexStorage*), either fetch it yourself via
`glfwGetProcAddress` in your own TU, or propose a loader extension here and
the integration agent will grow `src/platform/gl_loader.cpp` into the
hand-rolled 4.6 loader the build plan reserved for this case.

### Stub conventions
Every non-header body file carries the single marker line
`// STUB: implemented by module agent` and is intended to be REPLACED
wholesale by its module agent. Exceptions (complete, do not replace):
- `src/main.cpp` (frame loop + --selftest plumbing; extend, don't rewrite)
- `src/platform/gl_loader.cpp` (the one GLAD_GL_IMPLEMENTATION TU)
- `tests/cuda_smoke.cu` (toolchain canary; keep passing)

Functional bits already inside stubs you may keep or replace freely:
- `viz.cpp` contains a dependency-free PNG encoder (uncompressed zlib) used
  by `--selftest`; replacing it is fine as long as screenshotPNG still works.
- `voxelizer.cpp` `buildBoundaryFlags()` is real and selftest-load-bearing.
- `snapshot.cpp` key hashing (FNV-1a) is real; disk format is yours.
- `camera.cpp`, `ui.cpp` backend plumbing, `platform.cpp` timer/exe-dir are real.

### Selftest contract
`FoilCFD.exe --selftest` must keep: init GL+CUDA, default NACA 2412 domain,
200 sim steps, write `screenshots/selftest.png`, print `PASS`, exit 0. Module
agents must not regress this — it is the build-verification hook.

Status: INFORMATIONAL
