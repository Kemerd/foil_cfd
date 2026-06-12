// Velocity-volume fragment stage: fixed-step march of the eye ray through the
// normalized speed volume (single-channel float 3D texture), colormapped on
// the GPU with the quiet freestream culled to a faint haze. This is the hero
// "wind-tunnel smoke" look: fast / wake air glows hot, undisturbed air stays
// nearly transparent so you can see the structure floating in space.
//
// Two correctness pieces beyond a plain volume render:
//   1. Slow-air opacity is a SLIDER (uSlowOpacity), not a hard cull — slow
//      cells keep a small floor alpha so the body and the field underneath
//      stay faintly visible instead of vanishing into black.
//   2. The march terminates at the opaque scene (foil/VG mesh) by reading the
//      depth buffer copied into uSceneDepth, so geometry correctly occludes
//      vortices that sit BEHIND it along the view ray (the old bug: the foil
//      appeared to float over the volume from every angle).
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#version 330 core

in vec3 vWorldPos; // back-face hit point, lattice cell space

uniform sampler3D uVolume;       // normalized speed |u| in [0,1]
uniform sampler2D uSceneDepth;   // copied depth buffer of the opaque pass
uniform vec3  uDims;             // grid dimensions in cells
uniform vec3  uEye;              // camera position, cell space
uniform mat4  uInvViewProj;      // clip -> world, to un-project the depth fetch
uniform vec2  uViewport;         // framebuffer size (pixels), for depth fetch
uniform int   uColormap;         // palette selector (2 = inferno default)
uniform float uSlowOpacity;      // floor alpha for quiet/freestream air [0,1]
uniform float uDensity;          // overall opacity gain for fast air
uniform float uFreestream;       // normalized freestream speed (u_inf/scale):
                                 // the "calm air" baseline disturbance is
                                 // measured FROM, so it reads as haze even
                                 // though it is not literally zero speed.

out vec4 fragColor;

// ---- palette (kept in sync with particles.frag / the CUDA kernels) --------
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
vec3 palette(int which, float t) {
    return (which == 2) ? inferno(t)
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

// Un-project a window-space depth sample at this fragment into a WORLD-space
// position, so its distance from the eye is exact along the view ray (no
// forward-axis / cosine bookkeeping needed). Returns the world point.
vec3 worldFromDepth(vec2 uv, float depth) {
    vec4 ndc = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 world = uInvViewProj * ndc;
    return world.xyz / world.w;
}

void main() {
    vec3 rd = normalize(vWorldPos - uEye);
    vec2 hit = intersectBox(uEye, rd);
    float tNear = max(hit.x, 0.0);
    float tFar  = min(hit.y, length(vWorldPos - uEye));

    // ---- occlude against the opaque scene (depth-buffer copy) -------------
    // Reconstruct the foil's world position from its depth sample and clamp
    // the march to its distance from the eye, so solid geometry correctly
    // hides vortices behind it at ANY view angle (the old floating-foil bug).
    vec2 uv = gl_FragCoord.xy / uViewport;
    float sceneD = texture(uSceneDepth, uv).r;
    if (sceneD < 1.0) {
        float tFoil = length(worldFromDepth(uv, sceneD) - uEye);
        tFar = min(tFar, tFoil);
    }
    if (tFar <= tNear) discard;

    // Fixed step count keeps the cost bounded on huge grids; ~1.5-cell steps
    // at the default domain.
    const int kSteps = 192;
    float stepLen = (tFar - tNear) / float(kSteps);

    vec3  accumRgb = vec3(0.0);
    float accumA   = 0.0;

    for (int i = 0; i < kSteps; ++i) {
        vec3 p = uEye + rd * (tNear + (float(i) + 0.5) * stepLen);
        vec3 uvw = p / uDims;
        float s = texture(uVolume, uvw).r;   // normalized speed in [0,1]
        // Exact-zero cells are void (solid interior / boundary ring) — skip so
        // the foil shell and domain walls never tint the haze.
        if (s <= 0.0) continue;

        // Opacity curve: quiet air (speed near the freestream baseline) keeps
        // only the user's floor alpha (uSlowOpacity) so it reads as faint
        // haze; air that departs from freestream — the wake slowdown and the
        // suction-peak speedup alike — ramps up toward uDensity. Disturbance
        // is the fractional deviation from the freestream level, normalized so
        // a full stop (s -> 0) or a doubling both saturate.
        float base = max(uFreestream, 1e-3);
        float dev = (s - base) / base;            // signed fractional deviation
        // Speed-ups read a touch stronger than slow-downs (acceleration over
        // the suction surface is the headline feature), but both light up.
        float disturb = clamp(abs(dev) * 1.1 + max(dev, 0.0) * 0.4, 0.0, 1.0);
        float a = mix(uSlowOpacity, uDensity, disturb);

        // Per-step alpha scaled by step length so the look is resolution
        // independent (more steps != denser fog).
        a = 1.0 - pow(1.0 - clamp(a, 0.0, 1.0), stepLen);

        vec3 shade = palette(uColormap, s);
        accumRgb += (1.0 - accumA) * a * shade;
        accumA   += (1.0 - accumA) * a;
        if (accumA > 0.98) break;
    }

    if (accumA <= 0.002) discard;
    fragColor = vec4(accumRgb, accumA);
}
