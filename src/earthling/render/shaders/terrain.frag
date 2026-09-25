#include "terrain_common.glsl"
#include "logdepth.glsl"
in vec2 v_uv;
in vec2 v_tile_uv;
in float v_height;
in vec3 v_world;
in float v_log_z;
out vec4 f_color;

uniform sampler2D u_imagery;
uniform bool u_has_imagery;

vec3 srgb_to_linear(vec3 c) { return pow(c, vec3(2.2)); }
vec3 linear_to_srgb(vec3 c) { return pow(max(c, 0.0), vec3(1.0 / 2.2)); }

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

void main() {
    write_log_depth(v_log_z);
    vec3 n = terrain_normal(v_uv);
    vec3 sun = normalize(vec3(-0.5, 0.6, 0.6));
    float diffuse = max(dot(n, sun), 0.0);
    vec3 albedo = u_has_imagery ? srgb_to_linear(texture(u_imagery, v_tile_uv).rgb)
                                : srgb_to_linear(elevation_ramp(v_height));
    // Imagery already contains baked-in shading; only add moderate relief lighting.
    vec3 color = albedo * (0.55 + 0.75 * diffuse);
    f_color = vec4(linear_to_srgb(color), 1.0);
}
