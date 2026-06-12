// Q-criterion raycast fragment stage: fixed-step march of the eye ray through
// the normalized Q volume (R8 3D texture); cells above the threshold light up
// with gradient-normal shading and front-to-back alpha accumulation. The box
// is drawn with FRONT faces culled (back faces rasterized) so the march also
// works with the camera inside the domain.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#version 330 core

in vec3 vWorldPos; // back-face hit point, lattice cell space

uniform sampler3D uVolume;     // normalized Q in [0,1]
uniform vec3  uDims;           // grid dimensions in cells
uniform vec3  uEye;            // camera position, cell space
uniform float uThreshold;      // iso threshold on normalized Q

out vec4 fragColor;

// Slab-method ray/AABB intersection against the [0, uDims] domain box.
vec2 intersectBox(vec3 ro, vec3 rd) {
    vec3 inv = 1.0 / rd;
    vec3 t0 = (vec3(0.0) - ro) * inv;
    vec3 t1 = (uDims - ro) * inv;
    vec3 tmin3 = min(t0, t1);
    vec3 tmax3 = max(t0, t1);
    return vec2(max(max(tmin3.x, tmin3.y), tmin3.z),
                min(min(tmax3.x, tmax3.y), tmax3.z));
}

void main() {
    vec3 rd = normalize(vWorldPos - uEye);
    vec2 hit = intersectBox(uEye, rd);
    // Clamp the entry to the eye itself when the camera sits inside the box.
    float tNear = max(hit.x, 0.0);
    float tFar  = min(hit.y, length(vWorldPos - uEye));
    if (tFar <= tNear) discard;

    // Fixed step count keeps the cost bounded on huge grids; ~1.5-cell steps
    // at the default domain. Threshold crossings get a soft shoulder rather
    // than a binary cut so the pseudo-isosurface doesn't alias.
    const int   kSteps = 160;
    float stepLen = (tFar - tNear) / float(kSteps);
    vec3  texel   = 1.0 / uDims;

    vec3  accumRgb = vec3(0.0);
    float accumA   = 0.0;
    vec3  L = normalize(vec3(-0.45, 0.80, 0.40)); // matches the mesh key light

    for (int i = 0; i < kSteps; ++i) {
        vec3 p = uEye + rd * (tNear + (float(i) + 0.5) * stepLen);
        vec3 uvw = p / uDims;
        float q = texture(uVolume, uvw).r;
        if (q > uThreshold) {
            // Gradient of the Q field = surface normal of the pseudo-iso.
            float gx = texture(uVolume, uvw + vec3(texel.x, 0, 0)).r
                     - texture(uVolume, uvw - vec3(texel.x, 0, 0)).r;
            float gy = texture(uVolume, uvw + vec3(0, texel.y, 0)).r
                     - texture(uVolume, uvw - vec3(0, texel.y, 0)).r;
            float gz = texture(uVolume, uvw + vec3(0, 0, texel.z)).r
                     - texture(uVolume, uvw - vec3(0, 0, texel.z)).r;
            vec3 g = vec3(gx, gy, gz);
            vec3 N = (dot(g, g) > 1e-12) ? normalize(-g) : -rd;
            if (dot(N, rd) > 0.0) N = -N;

            float diffuse = max(dot(N, L), 0.0);
            float rim = pow(1.0 - clamp(dot(N, -rd), 0.0, 1.0), 2.0);
            // Strength of the surface rises with how far above threshold the
            // sample sits — soft shoulder over ~10% of the normalized range.
            float density = smoothstep(uThreshold,
                                       min(uThreshold + 0.1, 1.0), q);
            // Pale cyan vortex cores: distinct from both palettes in use.
            vec3 shade = vec3(0.55, 0.85, 0.95) * (0.25 + 0.75 * diffuse)
                       + vec3(0.25) * rim;

            // Front-to-back "over" compositing with early exit when opaque.
            float a = density * 0.35;
            accumRgb += (1.0 - accumA) * a * shade;
            accumA   += (1.0 - accumA) * a;
            if (accumA > 0.95) break;
        }
    }

    if (accumA <= 0.003) discard;
    fragColor = vec4(accumRgb, accumA);
}
