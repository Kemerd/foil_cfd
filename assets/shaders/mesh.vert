// Foil + VG mesh vertex stage (plan 9.1): the extruded outline prism and the
// VG vane boxes, flat-shaded via per-face normals baked into duplicated
// vertices (no marching cubes — the mesh comes straight from the 2D polygon).
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#version 330 core

layout(location = 0) in vec3 aPos;    // lattice cell space
layout(location = 1) in vec3 aNormal; // face normal (flat shading)
layout(location = 2) in vec3 aColor;  // per-part tint (foil vs VG vanes)

uniform mat4 uViewProj;

out vec3 vNormal;
out vec3 vWorldPos;
out vec3 vColor;

void main() {
    gl_Position = uViewProj * vec4(aPos, 1.0);
    vNormal   = aNormal;
    vWorldPos = aPos;
    vColor    = aColor;
}
