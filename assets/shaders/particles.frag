// Particle hero mode, fragment stage: round soft sprite, viridis or coolwarm
// palette (plan 9.1 explicitly bans rainbow/jet), age-fade alpha for additive
// blending. The palette polynomials mirror the device versions in
// render/particles.cu so slices and particles agree on color.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#version 330 core

in float vAge;
in float vKey;

uniform int   uColormap;   // 0 = viridis, 1 = coolwarm, 2 = inferno
uniform float uAlphaScale; // overall brightness for the additive blend

out vec4 fragColor;

// Matt Zucker's public-domain degree-6 polynomial fit of viridis
// (shadertoy WlfXRN) — identical coefficients to the CUDA kernels.
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

// Two-segment smoothstep approximation of Moreland's coolwarm diverging map.
vec3 coolwarm(float t) {
    const vec3 blue  = vec3(0.230, 0.299, 0.754);
    const vec3 white = vec3(0.865, 0.865, 0.865);
    const vec3 red   = vec3(0.706, 0.016, 0.150);
    return (t < 0.5) ? mix(blue, white, smoothstep(0.0, 1.0, t * 2.0))
                     : mix(white, red,  smoothstep(0.0, 1.0, (t - 0.5) * 2.0));
}

// Inferno (matplotlib) degree-6 fit — identical coefficients to the CUDA
// kernels' infernoColor. Black -> purple -> red -> orange -> pale yellow.
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

// Palette dispatch shared by particles and the volume raymarch (kept in sync
// with the CUDA-side selector: 2 = inferno, 1 = coolwarm, else viridis).
vec3 palette(int which, float t) {
    return (which == 2) ? inferno(t)
         : (which == 1) ? coolwarm(t)
                        : viridis(t);
}

void main() {
    // Round sprite: kill corners of the point square, soften the rim so 1-2px
    // points don't shimmer.
    vec2 d = gl_PointCoord - vec2(0.5);
    float r2 = dot(d, d);
    if (r2 > 0.25) discard;
    float soft = 1.0 - smoothstep(0.15, 0.25, r2);

    vec3 rgb = palette(uColormap, vKey);

    // Age fade (plan 9.1): newborns bright, dying particles vanish. Slight
    // ease-in keeps freshly respawned inlet sheets from flashing.
    float alpha = uAlphaScale * soft * vAge * vAge;
    fragColor = vec4(rgb, alpha);
}
