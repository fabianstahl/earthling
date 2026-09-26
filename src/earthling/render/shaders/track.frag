#include "logdepth.glsl"
#include "atmosphere_lut.glsl"
#include "fog.glsl"
in float v_side;
in float v_dist;
in float v_width_px;
in vec3 v_world;
in float v_log_z;
out vec4 f_color;

uniform vec3 u_color = vec3(1.0, 0.2, 0.05);  // linear
uniform float u_track_length;
uniform float u_track_opacity = 1.0;
uniform float u_track_emissive = 0.5;     // 0 lit by the scene .. 1 self-illuminated
uniform float u_track_outline = 1.0;      // outline width in pixels
uniform vec3 u_sun_radiance = vec3(1.4);
uniform vec3 u_sky_ambient = vec3(0.2, 0.24, 0.35);
uniform float u_track_glow = 2.0;  // emission into the glow (bloom) buffer
uniform float u_track_offset = 0.0;  // global distance of this track's start
uniform float u_head_m = 1e12;       // visible range on the global distance axis
uniform float u_tail_m = 0.0;
uniform float u_head_fade = 150.0;   // metres over which the line fades in behind the head
uniform float u_head_boost = 1.5;    // extra brightness right at the head
uniform float u_dash_px = 0.0;       // dash length on screen, 0 = solid
uniform float u_track_casing = 0.0;  // soft border (pixels each side, see track.vert)
uniform vec3 u_casing_color = vec3(0.03, 0.03, 0.05);  // sRGB
uniform float u_casing_opacity = 0.75;
uniform int u_part = 1;              // 0 the casings (drawn first, below all lines), 1 the lines

// Dashes of about u_dash_px on screen at any zoom: the period snaps to powers of two metres
// and crossfades between neighbouring octaves, so dashes neither crawl nor pop.
float dash_mask(float dist) {
    float m_per_px = max(fwidth(dist), 1e-5);
    float level = log2(u_dash_px * 2.0 * m_per_px);
    float l0 = floor(level);
    float f = level - l0;
    float on0 = 1.0 - step(0.5, fract(dist / exp2(l0)));
    float on1 = 1.0 - step(0.5, fract(dist / exp2(l0 + 1.0)));
    return mix(on0, on1, f);
}

void main() {
    write_log_depth(v_log_z);
    float px = abs(v_side) * v_width_px * 0.5;  // pixels from the centre line
    float outer = v_width_px * 0.5 - 1.0;       // 1 px AA margin
    float half_w = outer - u_track_casing;      // visible half width of the line itself
    float a = min(px / max(half_w + 1.0, 1e-3), 1.0);
    float gd = v_dist + u_track_offset;
    if (gd > u_head_m || gd < u_tail_m) discard;
    float along = smoothstep(u_tail_m, u_tail_m + 5.0, gd);
#ifndef GLOW_PASS
    if (u_part == 0) {
        if (u_track_casing <= 0.0) discard;
        float x = clamp((px - half_w) / max(u_track_casing, 1e-3), 0.0, 1.0);
        float c = (1.0 - x * x) * (1.0 - smoothstep(outer - 0.5, outer + 0.5, px));
        if (u_dash_px > 0.0) c *= mix(0.5, 1.0, dash_mask(v_dist));  // dashed lines: gaps show
        c *= along * u_casing_opacity * u_track_opacity;
        if (c <= 0.0) discard;
        f_color = vec4(pow(u_casing_color, vec3(2.2)), c);
        return;
    }
#endif
    float alpha = 1.0 - smoothstep(half_w - 0.5, half_w + 0.5, px);
    alpha *= along;
    if (u_dash_px > 0.0) alpha *= dash_mask(v_dist);
    // the freshly drawn part near the head is brighter
    float near_head = u_head_m < 1e11 ? 1.0 - smoothstep(0.0, max(u_head_fade, 1.0), u_head_m - gd) : 0.0;
    if (alpha <= 0.0) discard;
    // tube shading: fake cylinder normal across the ribbon, light from above
    float nz = sqrt(max(0.0, 1.0 - a * a));
#ifdef GLOW_PASS
    float boost = 1.0 + u_head_boost * near_head;
    f_color = vec4(u_color * u_track_glow * boost * (0.5 + 0.5 * nz) * alpha * u_track_opacity, 1.0);
    return;
#endif
    float shade = 0.55 + 0.45 * nz;
    vec3 lit = u_color * (u_sky_ambient + u_sun_radiance * 0.8) * shade;
    vec3 self_lit = u_color * mix(0.6, 1.2, nz);
    vec3 color = mix(lit, self_lit, u_track_emissive) * (1.0 + u_head_boost * near_head);
    // dark outline for readability on bright ground
    float outline = smoothstep(half_w - u_track_outline - 0.5, half_w - u_track_outline + 0.5, px);
    color = mix(color, color * 0.25, outline * step(0.01, u_track_outline));
    float dist = length(v_world);
    color = apply_atmosphere(color, v_world / max(dist, 1e-3), dist);
    f_color = vec4(color, alpha * u_track_opacity);
}
