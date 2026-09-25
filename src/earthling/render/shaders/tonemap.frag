in vec2 v_uv;
out vec4 f_color;

uniform sampler2D u_hdr;
uniform float u_exposure = 1.0;  // linear multiplier (2^EV)
uniform int u_tonemap = 0;       // 0 ACES, 1 Reinhard, 2 none (clamp)
uniform sampler2D u_bloom;
uniform bool u_has_bloom = false;
uniform float u_bloom_intensity = 1.0;

vec3 aces(vec3 x) {
    // Narkowicz 2015 fit
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 linear_to_srgb(vec3 c) {
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
}

float dither(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453) - 0.5;
}

void main() {
    vec3 hdr = max(texture(u_hdr, v_uv).rgb, 0.0);
    if (u_has_bloom) hdr += texture(u_bloom, v_uv).rgb * u_bloom_intensity;
    hdr *= u_exposure;
    vec3 mapped;
    if (u_tonemap == 0) mapped = aces(hdr);
    else if (u_tonemap == 1) mapped = hdr / (1.0 + hdr);
    else mapped = clamp(hdr, 0.0, 1.0);
    vec3 srgb = linear_to_srgb(mapped);
    srgb += dither(gl_FragCoord.xy) / 255.0;  // hide banding in dark gradients
    f_color = vec4(srgb, 1.0);
}
