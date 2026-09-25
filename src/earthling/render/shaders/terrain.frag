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
uniform vec3 u_sun_radiance = vec3(1.4);         // linear, 0 below the horizon
uniform vec3 u_sky_ambient = vec3(0.2, 0.24, 0.35);
uniform vec3 u_ground_ambient = vec3(0.1, 0.08, 0.06);
uniform int u_zoom;
uniform float u_shadow_strength = 1.0;
uniform int u_layer_a = 0;
uniform int u_layer_b = 1;
uniform float u_layer_mix = 0.0;

// Inputs available to every texture layer (see render/layers.py).
struct LayerInput {
    vec2 tile_uv;       // 0..1 across the LOD node
    float height;       // metres above sea level (without exaggeration)
    vec3 normal;        // ENU
    vec3 normal_local;  // east/north/up tangent frame of the node
    float slope;        // degrees
    float aspect;       // degrees clockwise from north (direction the slope faces)
    vec3 imagery;       // sRGB of the node's imagery texture (fallback: hypsometric tint)
};

#include "layers_generated.glsl"

vec3 srgb_to_linear(vec3 c) { return pow(max(c, 0.0), vec3(2.2)); }

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

    LayerInput li;
    li.tile_uv = v_tile_uv;
    li.height = v_height;
    li.normal = n;
    li.normal_local = transpose(u_tangent) * n;
    li.slope = degrees(acos(clamp(li.normal_local.z, -1.0, 1.0)));
    li.aspect = mod(degrees(atan(li.normal_local.x, li.normal_local.y)) + 360.0, 360.0);
    li.imagery = u_has_imagery ? texture(u_imagery, v_tile_uv).rgb
                               : ramp_hypsometric((v_height - 500.0) / 4000.0);

    // Hemispherical ambient (sky from above, bounce light from below) + direct sun.
    vec3 ambient = mix(u_ground_ambient, u_sky_ambient, n.z * 0.5 + 0.5);
    vec3 lit = ambient + u_sun_radiance * diffuse;
    // "flat" presentation: same overall brightness as the scene, without relief
    vec3 flat_light = mix(u_ground_ambient, u_sky_ambient, 0.75) + u_sun_radiance * 0.65;

    vec3 albedo_a = srgb_to_linear(layer_albedo(u_layer_a, li)) * layer_gain(u_layer_a);
    vec3 color = albedo_a * mix(flat_light, lit, layer_shading(u_layer_a));
    if (u_layer_mix > 0.0) {
        vec3 albedo_b = srgb_to_linear(layer_albedo(u_layer_b, li)) * layer_gain(u_layer_b);
        vec3 color_b = albedo_b * mix(flat_light, lit, layer_shading(u_layer_b));
        color = mix(color, color_b, u_layer_mix);
    }
    if (u_debug_lod) {
        float edge = step(min(v_tile_uv.x, v_tile_uv.y), 0.004)
                   + step(0.996, max(v_tile_uv.x, v_tile_uv.y));
        color = mix(color, srgb_to_linear(zoom_color(u_zoom)) * lit, 0.55);
        color = mix(color, vec3(0.0), clamp(edge, 0.0, 1.0));
    }
    float dist = length(v_world);
    color = apply_atmosphere(color, v_world / max(dist, 1e-3), dist);
    f_color = vec4(color, 1.0);  // linear HDR radiance
#endif
}
