// Q-criterion isosurface fragment stage: first-HIT raycast through the
// normalized Q volume rendered as an OPAQUE surface — march until the ray
// crosses the iso threshold, bisection-refine the crossing to sub-step
// precision, shade with the gradient normal, and write the hit's true depth
// to gl_FragDepth so the z-buffer composites the filaments against the foil
// mesh (and everything else) exactly like rasterized geometry would.
//
// This replaces the old translucent front-to-back fog compositing: vortex
// tubes now read as crisp solid surfaces on the black background (the classic
// wind-tunnel hero shot), and the depth write makes occlusion correct from
// every angle with zero extra depth-texture plumbing.
//
// Coloring: either the fixed pale-cyan core tint, or (default) the local air
// speed sampled from the companion speed volume pushed through the selected
// palette — rainbow by default, so slow air shades blue, freestream green,
// accelerated air red.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#version 330 core

in vec3 vWorldPos; // back-face hit point, lattice cell space

uniform sampler3D uVolume;       // normalized Q in [0,1]
uniform sampler3D uSpeedVolume;  // normalized |u| in [0,1] (velocity coloring)
uniform mat4  uViewProj;         // world -> clip, for the depth write
uniform vec3  uDims;             // grid dimensions in cells
uniform vec3  uEye;              // camera position, cell space
uniform float uThreshold;        // iso threshold on normalized Q
uniform int   uColorByVel;       // 0 = fixed cyan cores, 1 = speed colormap
uniform int   uColormap;         // palette selector when coloring by velocity

out vec4 fragColor;

// ---- palettes (kept in sync with particles.frag / the CUDA kernels) -------
vec3 viridis(float t) {
    const vec3 c0 = vec3( 0.2777273,  0.0054073,  0.3340998);
    const vec3 c1 = vec3( 0.1050930,  1.4046135,  1.3845902);
    const vec3 c2 = vec3(-0.3308618,  0.2148476,  0.0950952);
    const vec3 c3 = vec3(-4.6342305, -5.7991010, -19.3324410);
    const vec3 c4 = vec3( 6.2282699, 14.1799334,  56.6905526);
    const vec3 c5 = vec3( 4.7763850, -13.7451454, -65.3530326);
    const vec3 c6 = vec3(-5.4354559,  4.6458526,  26.3124352);
    return clamp(c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6))))), 0.0, 1.0);
}
vec3 coolwarm(float t) {
    const vec3 blue  = vec3(0.230, 0.299, 0.754);
    const vec3 white = vec3(0.865, 0.865, 0.865);
    const vec3 red   = vec3(0.706, 0.016, 0.150);
    return (t < 0.5) ? mix(blue, white, smoothstep(0.0, 1.0, t * 2.0))
                     : mix(white, red,  smoothstep(0.0, 1.0, (t - 0.5) * 2.0));
}
vec3 inferno(float t) {
    const vec3 c0 = vec3(-0.00021,  0.00016, -0.01976);
    const vec3 c1 = vec3( 0.10677,  0.56329,  3.93245);
    const vec3 c2 = vec3(11.60249, -3.97282, -15.94254);
    const vec3 c3 = vec3(-41.70399, 17.43639,  44.35414);
    const vec3 c4 = vec3(77.16296, -33.40235, -81.80730);
    const vec3 c5 = vec3(-71.31899, 32.62606,  73.20951);
    const vec3 c6 = vec3(25.13112, -12.24266, -23.07032);
    return clamp(c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6))))), 0.0, 1.0);
}
// Full-saturation HSV hue sweep, blue (slow) -> green -> red (fast). The
// classic wind-tunnel palette: not perceptually uniform, but unbeatable for
// reading speed bands on vortex structures at a glance.
vec3 rainbow(float t) {
    float h = (1.0 - clamp(t, 0.0, 1.0)) * 4.0;  // hue in sextants: 0=red..4=blue
    vec3 k = mod(vec3(5.0, 3.0, 1.0) + h, 6.0);  // standard HSV channel offsets
    return 1.0 - clamp(min(min(k, 4.0 - k), vec3(1.0)), 0.0, 1.0);
}
vec3 palette(int which, float t) {
    return (which == 3) ? rainbow(t)
         : (which == 2) ? inferno(t)
         : (which == 1) ? coolwarm(t)
                        : viridis(t);
}

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

float sampleQ(vec3 p) { return texture(uVolume, p / uDims).r; }

void main() {
    vec3 rd = normalize(vWorldPos - uEye);
    vec2 hit = intersectBox(uEye, rd);
    float tNear = max(hit.x, 0.0);
    float tFar  = min(hit.y, length(vWorldPos - uEye));
    if (tFar <= tNear) discard;

    // Fixed-step march with early exit on the first crossing; 256 steps keeps
    // thin filaments from being skipped at the default domain size while the
    // first-hit exit keeps the common case cheap.
    const int kSteps = 256;
    float stepLen = (tFar - tNear) / float(kSteps);

    float tHit  = -1.0;
    float qPrev = sampleQ(uEye + rd * tNear);
    if (qPrev > uThreshold) {
        tHit = tNear; // camera (or box entry) already inside a core
    } else {
        float tPrev = tNear;
        for (int i = 1; i <= kSteps; ++i) {
            float t = tNear + float(i) * stepLen;
            float q = sampleQ(uEye + rd * t);
            if (q > uThreshold) {
                // Bisection-refine the crossing inside [tPrev, t]: six halvings
                // give sub-1/64-step precision — silhouettes stay smooth even
                // when the march stride is ~1.5 cells.
                float a = tPrev, b = t;
                for (int j = 0; j < 6; ++j) {
                    float m = 0.5 * (a + b);
                    if (sampleQ(uEye + rd * m) > uThreshold) b = m;
                    else                                     a = m;
                }
                tHit = 0.5 * (a + b);
                break;
            }
            tPrev = t;
            qPrev = q;
        }
    }
    if (tHit < 0.0) discard;

    vec3 p   = uEye + rd * tHit;
    vec3 uvw = p / uDims;
    vec3 texel = 1.0 / uDims;

    // Surface normal from the Q-field gradient (central differences in
    // texture space). Fallback to the view direction on flat gradients.
    float gx = texture(uVolume, uvw + vec3(texel.x, 0, 0)).r
             - texture(uVolume, uvw - vec3(texel.x, 0, 0)).r;
    float gy = texture(uVolume, uvw + vec3(0, texel.y, 0)).r
             - texture(uVolume, uvw - vec3(0, texel.y, 0)).r;
    float gz = texture(uVolume, uvw + vec3(0, 0, texel.z)).r
             - texture(uVolume, uvw - vec3(0, 0, texel.z)).r;
    vec3 g = vec3(gx, gy, gz);
    vec3 N = (dot(g, g) > 1e-12) ? normalize(-g) : -rd;
    if (dot(N, rd) > 0.0) N = -N; // always face the camera

    // Base color: speed palette (the velocity field painted onto the vortex
    // skins) or the legacy fixed cyan.
    vec3 base = (uColorByVel == 1)
                  ? palette(uColormap, texture(uSpeedVolume, uvw).r)
                  : vec3(0.55, 0.85, 0.95);

    // Saturated-but-shaded: strong ambient floor keeps the palette readable
    // (the color IS the data), diffuse + a small specular give the tubes
    // their roundness against the black background.
    vec3  L = normalize(vec3(-0.45, 0.80, 0.40)); // matches the mesh key light
    float diffuse = max(dot(N, L), 0.0);
    float spec = pow(max(dot(reflect(-L, N), -rd), 0.0), 24.0);
    vec3 rgb = base * (0.45 + 0.55 * diffuse) + vec3(0.18) * spec;

    // True depth for the hit point: project to clip space and convert NDC z
    // to the [0,1] window range. Writing gl_FragDepth disables early-z for
    // this draw, so the hardware test runs AFTER the shader with this value —
    // exactly what makes foil-vs-filament occlusion per-pixel correct.
    vec4 clip = uViewProj * vec4(p, 1.0);
    gl_FragDepth = clamp(clip.z / clip.w * 0.5 + 0.5, 0.0, 1.0);

    fragColor = vec4(rgb, 1.0);
}
