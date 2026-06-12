// Particle hero mode, vertex stage (plan 9.1 mode 1): pulls the lattice-space
// position + age from the CUDA-advected VBO and the normalized color key from
// its companion buffer; renders as GL_POINTS with program point size.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#version 330 core

// Binding contract with render/particles.cuh: xyz = position in lattice cell
// space, w = age in [0,1] (1 = newborn -> alpha fade), key already normalized.
layout(location = 0) in vec4 aPosAge;
layout(location = 1) in float aColorKey;

uniform mat4  uViewProj;   // camera view * projection (cell-space world)
uniform float uPointSize;  // UI particle size, pixels

out float vAge;  // age fade factor for the fragment stage
out float vKey;  // colormap coordinate in [0,1]

void main() {
    gl_Position  = uViewProj * vec4(aPosAge.xyz, 1.0);
    gl_PointSize = uPointSize;
    vAge = clamp(aPosAge.w, 0.0, 1.0);
    vKey = clamp(aColorKey, 0.0, 1.0);
}
