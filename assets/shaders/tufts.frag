// Cotton-tuft mode, fragment stage: paints each strand by its attachment
// scalar on a green -> yellow -> red ramp so the stall line reads instantly.
// Attached flow (key ~ 0) is calm green; the transition band is yellow; fully
// separated / reversed flow (key ~ 1) is hot red — the classic stall-survey
// color language. Lines are drawn opaque over the foil mesh with a user
// opacity multiplier for the over-blend.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#version 330 core

in float vKey;

uniform float uAlpha;  // overall line opacity (the over-blend multiplier)

out vec4 fragColor;

// Attachment ramp: green (attached) -> yellow (transition) -> red (separated).
// Two linear segments through a saturated yellow midpoint keep the transition
// band wide and legible rather than a muddy brown crossover.
vec3 attachment(float t) {
    const vec3 green  = vec3(0.15, 0.85, 0.25);  // attached, calm
    const vec3 yellow = vec3(0.95, 0.90, 0.15);  // marginal / transition
    const vec3 red    = vec3(0.95, 0.15, 0.10);  // separated, reversed
    return (t < 0.5) ? mix(green, yellow, smoothstep(0.0, 1.0, t * 2.0))
                     : mix(yellow, red,   smoothstep(0.0, 1.0, (t - 0.5) * 2.0));
}

void main() {
    vec3 rgb = attachment(vKey);
    fragColor = vec4(rgb, clamp(uAlpha, 0.0, 1.0));
}
