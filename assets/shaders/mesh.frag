// Foil + VG mesh fragment stage: flat shading from the baked face normal,
// fixed key light + ambient, and a subtle rim light (plan 9.1) so the dark
// foil silhouette separates from the dark background without washing out the
// additive particle streaks drawn over it.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#version 330 core

in vec3 vNormal;
in vec3 vWorldPos;
in vec3 vColor;

uniform vec3 uEye;    // camera position, lattice cell space
uniform float uAlpha; // mesh/wireframe opacity [0,1]; 1 = opaque (CPU side
                      // enables blending only when below 1, so the common
                      // opaque path keeps its blend-free fill rate)

out vec4 fragColor;

void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uEye - vWorldPos);
    // Light the side the camera sees: flat-shaded prisms are closed, so
    // flipping toward the viewer avoids pitch-black backfacing caps when the
    // camera crosses the span planes.
    if (dot(N, V) < 0.0) N = -N;

    // Fixed key light from upper-front-left in cell space; aesthetics only.
    vec3 L = normalize(vec3(-0.45, 0.80, 0.40));
    float diffuse = max(dot(N, L), 0.0);

    // Subtle rim: brightens grazing silhouettes (plan 9.1 "subtle rim light").
    float rim = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 3.0);

    vec3 rgb = vColor * (0.28 + 0.62 * diffuse) + vec3(0.35, 0.42, 0.50) * (0.30 * rim);
    fragColor = vec4(rgb, uAlpha);
}
