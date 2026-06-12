// Slice plane vertex stage (plan 9.1 mode 2): a single textured quad whose
// corners are computed host-side in lattice cell space; the texture content
// is written by the CUDA slice kernel through a surface object.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#version 330 core

layout(location = 0) in vec3 aPos; // lattice cell space
layout(location = 1) in vec2 aUV;  // matches the kernel's texel->cell mapping

uniform mat4 uViewProj;

out vec2 vUV;

void main() {
    gl_Position = uViewProj * vec4(aPos, 1.0);
    vUV = aUV;
}
