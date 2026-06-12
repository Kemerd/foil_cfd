// The single translation unit that instantiates the glad2 GL loader bundled
// with GLFW (deps/glad/gl.h is header-only; exactly one TU may define
// GLAD_GL_IMPLEMENTATION or the link breaks with duplicate symbols).
// Every other TU includes <glad/gl.h> plainly and gets declarations only.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#define GLAD_GL_IMPLEMENTATION
#include <glad/gl.h>
