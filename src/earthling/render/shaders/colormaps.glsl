// Colour ramps for procedural layers. Inputs t in [0, 1]; outputs sRGB.

vec3 ramp_grayscale(float t) { return vec3(t); }

// Hypsometric tints (lowland green -> brown -> rock -> snow)
vec3 ramp_hypsometric(float t) {
    const vec3 c[6] = vec3[](vec3(0.33, 0.55, 0.30), vec3(0.62, 0.72, 0.42), vec3(0.87, 0.80, 0.55),
                             vec3(0.72, 0.55, 0.38), vec3(0.62, 0.58, 0.56), vec3(0.98, 0.98, 1.00));
    float x = clamp(t, 0.0, 1.0) * 5.0;
    int i = int(min(floor(x), 4.0));
    return mix(c[i], c[i + 1], x - float(i));
}

// Viridis (polynomial fit, Matt Zucker)
vec3 ramp_viridis(float t) {
    t = clamp(t, 0.0, 1.0);
    const vec3 c0 = vec3(0.2777, 0.0054, 0.3340), c1 = vec3(0.1050, 1.4046, 1.3845);
    const vec3 c2 = vec3(-0.3308, 0.2148, 0.0950), c3 = vec3(-4.6342, -5.7991, -19.3324);
    const vec3 c4 = vec3(6.2282, 14.1799, 56.6905), c5 = vec3(4.7763, -13.7451, -65.3530);
    const vec3 c6 = vec3(-5.4355, 4.6459, 26.3124);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

// Turbo (polynomial approximation, Google)
vec3 ramp_turbo(float t) {
    t = clamp(t, 0.0, 1.0);
    const vec4 kr = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const vec4 kg = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const vec4 kb = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const vec2 kr2 = vec2(-152.94239396, 59.28637943);
    const vec2 kg2 = vec2(4.27729857, 2.82956604);
    const vec2 kb2 = vec2(-89.90310912, 27.34824973);
    vec4 v4 = vec4(1.0, t, t * t, t * t * t);
    vec2 v2 = v4.zw * v4.z;
    return clamp(vec3(dot(v4, kr) + dot(v2, kr2), dot(v4, kg) + dot(v2, kg2),
                      dot(v4, kb) + dot(v2, kb2)), 0.0, 1.0);
}

vec3 colormap(int ramp, float t) {
    if (ramp == 1) return ramp_viridis(t);
    if (ramp == 2) return ramp_turbo(t);
    if (ramp == 3) return ramp_grayscale(t);
    return ramp_hypsometric(t);
}

// Avalanche slope classes (degrees): < 30 none, 30-35 yellow, 35-40 orange, 40-45 red, > 45 violet
vec3 slope_classes(float deg) {
    if (deg < 30.0) return vec3(0.92, 0.92, 0.90);
    if (deg < 35.0) return vec3(0.98, 0.86, 0.20);
    if (deg < 40.0) return vec3(0.96, 0.55, 0.12);
    if (deg < 45.0) return vec3(0.86, 0.15, 0.15);
    return vec3(0.55, 0.25, 0.65);
}

vec3 hsv_to_rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}
