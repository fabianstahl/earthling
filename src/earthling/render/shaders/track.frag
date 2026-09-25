#include "logdepth.glsl"
#include "atmosphere.glsl"
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

void main() {
    write_log_depth(v_log_z);
    float a = abs(v_side);
    float px = a * v_width_px * 0.5;          // pixels from the centre line
    float half_w = v_width_px * 0.5 - 1.0;    // visible half width (1 px AA margin)
    float alpha = 1.0 - smoothstep(half_w - 0.5, half_w + 0.5, px);
    if (alpha <= 0.0) discard;
    // tube shading: fake cylinder normal across the ribbon, light from above
    float nz = sqrt(max(0.0, 1.0 - a * a));
#ifdef GLOW_PASS
    f_color = vec4(u_color * u_track_glow * (0.5 + 0.5 * nz) * alpha * u_track_opacity, 1.0);
    return;
#endif
    float shade = 0.55 + 0.45 * nz;
    vec3 lit = u_color * (u_sky_ambient + u_sun_radiance * 0.8) * shade;
    vec3 self_lit = u_color * mix(0.6, 1.2, nz);
    vec3 color = mix(lit, self_lit, u_track_emissive);
    // dark outline for readability on bright ground
    float outline = smoothstep(half_w - u_track_outline - 0.5, half_w - u_track_outline + 0.5, px);
    color = mix(color, color * 0.25, outline * step(0.01, u_track_outline));
    float dist = length(v_world);
    color = apply_atmosphere(color, v_world / max(dist, 1e-3), dist);
    f_color = vec4(color, alpha * u_track_opacity);
}
