#include "terrain_common.glsl"
#include "logdepth.glsl"
#include "atmosphere.glsl"
#include "fog.glsl"
#include "shadows.glsl"
in vec2 v_hm_uv;
in vec2 v_tile_uv;
in float v_height;
in vec3 v_world;
in float v_log_z;
out vec4 f_color;

uniform sampler2D u_imagery;
uniform bool u_has_imagery;
uniform bool u_debug_lod;
uniform bool u_show_imagery = true;
uniform vec3 u_sun_radiance = vec3(1.4);         // linear, 0 below the horizon
uniform vec3 u_sky_ambient = vec3(0.2, 0.24, 0.35);
uniform vec3 u_ground_ambient = vec3(0.1, 0.08, 0.06);
uniform int u_zoom;
uniform float u_shadow_strength = 1.0;

vec3 srgb_to_linear(vec3 c) { return pow(c, vec3(2.2)); }

vec3 elevation_ramp(float h) {
    const vec3 c0 = vec3(0.18, 0.35, 0.18);  //  500 m
    const vec3 c1 = vec3(0.45, 0.55, 0.30);  // 1500 m
    const vec3 c2 = vec3(0.55, 0.45, 0.35);  // 2500 m
    const vec3 c3 = vec3(0.95, 0.95, 0.97);  // 3500 m
    float t = clamp((h - 500.0) / 1000.0, 0.0, 3.0);
    if (t < 1.0) return mix(c0, c1, t);
    if (t < 2.0) return mix(c1, c2, t - 1.0);
    return mix(c2, c3, t - 2.0);
}

vec3 zoom_color(int z) {
    const vec3 colors[6] = vec3[](vec3(1, 0.3, 0.3), vec3(1, 0.7, 0.2), vec3(0.9, 1, 0.3),
                                  vec3(0.3, 1, 0.5), vec3(0.3, 0.7, 1), vec3(0.8, 0.4, 1));
    return colors[z % 6];
}

void main() {
    if (valid_at(v_hm_uv) < 0.5) discard;
#ifdef SHADOW_PASS
    return;  // depth only
#else
    write_log_depth(v_log_z);
    vec3 n = terrain_normal(v_hm_uv);
    vec3 sun = normalize(u_sun_dir);
    float diffuse = max(dot(n, sun), 0.0);
    if (diffuse > 0.0) diffuse *= mix(1.0, sun_shadow(v_world, n, sun), u_shadow_strength);
    vec3 albedo = (u_has_imagery && u_show_imagery) ? srgb_to_linear(texture(u_imagery, v_tile_uv).rgb)
                                : srgb_to_linear(elevation_ramp(v_height));
    if (u_debug_lod) {
        float edge = step(min(v_tile_uv.x, v_tile_uv.y), 0.004) + step(0.996, max(v_tile_uv.x, v_tile_uv.y));
        albedo = mix(albedo, srgb_to_linear(zoom_color(u_zoom)), 0.55);
        albedo = mix(albedo, vec3(0.0), clamp(edge, 0.0, 1.0));
    }
    // Hemispherical ambient (sky from above, bounce light from below) + direct sun.
    vec3 ambient = mix(u_ground_ambient, u_sky_ambient, n.z * 0.5 + 0.5);
    vec3 color = albedo * (ambient + u_sun_radiance * diffuse);
    float dist = length(v_world);
    color = apply_atmosphere(color, v_world / max(dist, 1e-3), dist);
    f_color = vec4(color, 1.0);  // linear HDR radiance
#endif
}
