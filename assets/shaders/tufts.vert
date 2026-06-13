// Cotton-tuft mode, vertex stage: pulls each strand node's lattice-space
// position + the strand's attachment scalar from the CUDA-advected VBO. Nodes
// are drawn as GL_LINE_STRIP per strand (the renderer issues one draw slice
// per tuft), so this stage is just the standard view-projection transform plus
// passing the color key through.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#version 330 core

// Binding contract with render/tufts.cuh: xyz = node position in lattice cell
// space, w = attachment scalar in [0,1] (0 = attached, 1 = separated).
layout(location = 0) in vec4 aPosKey;

uniform mat4 uViewProj;  // camera view * projection (cell-space world)

out float vKey;  // attachment scalar in [0,1] for the fragment stage

void main() {
    gl_Position = uViewProj * vec4(aPosKey.xyz, 1.0);
    vKey = clamp(aPosKey.w, 0.0, 1.0);
}
