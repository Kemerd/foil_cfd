// Slice plane fragment stage: pure texture fetch — the field derivation and
// colormapping already happened on the CUDA side (render/particles.cu
// sliceFillKernel), so the plane costs nothing at draw time.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#version 330 core

in vec2 vUV;

uniform sampler2D uField;

out vec4 fragColor;

void main() {
    fragColor = vec4(texture(uField, vUV).rgb, 1.0);
}
