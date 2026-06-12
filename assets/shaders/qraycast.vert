// Q-criterion raycast vertex stage (plan 9.1 mode 3, stretch): draws the
// domain box (unit cube scaled to grid dims); the fragment stage marches the
// eye ray through the 3D Q texture the CUDA kernel fills.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#version 330 core

layout(location = 0) in vec3 aUnit; // unit-cube corner in [0,1]^3

uniform mat4 uViewProj;
uniform vec3 uDims; // grid dimensions (nx, ny, nz) in cells

out vec3 vWorldPos;

void main() {
    vec3 world = aUnit * uDims;
    gl_Position = uViewProj * vec4(world, 1.0);
    vWorldPos = world;
}
