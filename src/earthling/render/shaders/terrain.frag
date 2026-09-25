#include "terrain_common.glsl"
#include "logdepth.glsl"
#include "atmosphere.glsl"
#include "fog.glsl"
#include "shadows.glsl"
in vec2 v_hm_uv;
in vec2 v_tile_uv;
in float v_skirt;  // 1 on the skirts hiding cracks between nodes
in float v_height;
in vec3 v_world;
in float v_log_z;
out vec4 f_color;

uniform sampler2D u_imagery;
uniform bool u_has_imagery;
uniform sampler2D u_topo;
uniform bool u_has_topo;
uniform sampler2D u_borders;  // distance to border lines in node uv units (r: countries, g: regions)
uniform bool u_has_borders;
uniform bool u_borders_overlay = false;
uniform bool u_borders_regions = true;
uniform float u_border_width = 2.5;
uniform vec3 u_border_color = vec3(1.0, 0.85, 0.3);
uniform float u_border_glow = 12.0;
uniform float u_border_glow_strength = 0.6;
uniform float u_region_width = 1.2;
uniform vec3 u_region_color = vec3(0.9, 0.9, 0.9);
uniform bool u_contours_overlay = false;
uniform float u_contour_interval = 100.0;
uniform float u_contour_major = 5.0;
uniform float u_contour_width = 1.0;
uniform vec3 u_contour_color = vec3(0.35, 0.2, 0.1);
uniform float u_contour_opacity = 0.8;
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
    vec3 topo;          // sRGB of the topographic map texture (fallback: light grey)
    float border_px;    // screen-pixel distance to the nearest country border (large if none)
    float region_px;    // screen-pixel distance to the nearest regional border
    float border_reach; // 1 near a border .. 0 at the distance-field clamp (no border nearby)
};

const float BORDER_MAX_UV = 0.15;  // must match data/borders.py MAX_DISTANCE_UV

// Anti-aliased line coverage for a pixel distance and a line width in pixels.
float line_coverage(float dist_px, float width_px) {
    return 1.0 - smoothstep(width_px * 0.5 - 0.6, width_px * 0.5 + 0.6, dist_px);
}

// Anti-aliased contour line coverage (0..1) for a height in metres.
float contour_lines(float height, float interval, float major_every, float width_px) {
    float f = height / max(interval, 0.1);
    float fw = max(fwidth(f), 1e-5);
    float d = abs(fract(f - 0.5) - 0.5) / fw;  // distance to the nearest line in pixels
    float major = abs(fract(f / max(major_every, 1.0) - 0.5 / max(major_every, 1.0)) - 0.5);
    float is_major = step(major * max(major_every, 1.0), 0.5);
    float w = width_px * mix(0.6, 1.4, is_major);
    // lines fade out where they would get denser than ~3 px
    float fade = 1.0 - smoothstep(0.25, 0.4, fw);
    return (1.0 - smoothstep(w * 0.5 - 0.5, w * 0.5 + 0.5, d)) * fade;
}

#include "layers_generated.glsl"

vec3 srgb_to_linear(vec3 c) { return pow(max(c, 0.0), vec3(2.2)); }

vec3 zoom_color(int z) {
    const vec3 colors[6] = vec3[](vec3(1, 0.3, 0.3), vec3(1, 0.7, 0.2), vec3(0.9, 1, 0.3),
                                  vec3(0.3, 1, 0.5), vec3(0.3, 0.7, 1), vec3(0.8, 0.4, 1));
    return colors[z % 6];
}

// Close-range detail normals: the imagery's luminance as a few metres of relief, applied with
// screen-space derivative bump mapping (Mikkelsen 2010, no tangents needed). Adds rock and
// forest structure where the heightmap is too coarse; fades out with distance.
uniform float u_detail_normals = 0.0;       // strength
uniform float u_detail_distance_km = 3.0;   // fully faded out at this distance
const float DETAIL_RELIEF_M = 6.0;          // relief of luminance 0 -> 1

float luminance(vec2 uv) {
    return dot(texture(u_imagery, uv).rgb, vec3(0.299, 0.587, 0.114));
}

vec3 detail_normal(vec3 n, float strength) {
    // relief gradient in texture space (finite differences over the pixel footprint), then
    // into screen space with the uv derivatives: no seams where 2x2 pixel quads straddle
    // node borders, unlike derivatives of the sampled luminance itself
    vec2 uv = v_tile_uv;
    vec2 texel = 1.0 / vec2(textureSize(u_imagery, 0));
    vec2 fw = fwidth(uv);
    float d = max(max(fw.x, fw.y), max(texel.x, texel.y));
    float scale = DETAIL_RELIEF_M * strength / (2.0 * d);
    float dh_du = (luminance(uv + vec2(d, 0.0)) - luminance(uv - vec2(d, 0.0))) * scale;
    float dh_dv = (luminance(uv + vec2(0.0, d)) - luminance(uv - vec2(0.0, d))) * scale;
    vec2 duvx = dFdx(uv);
    vec2 duvy = dFdy(uv);
    float dhx = dh_du * duvx.x + dh_dv * duvx.y;
    float dhy = dh_du * duvy.x + dh_dv * duvy.y;
    vec3 dpdx = dFdx(v_world);
    vec3 dpdy = dFdy(v_world);
    vec3 r1 = cross(dpdy, n);
    vec3 r2 = cross(n, dpdx);
    float det = dot(dpdx, r1);
    vec3 grad = sign(det) * (dhx * r1 + dhy * r2);
    vec3 r = abs(det) * n - grad;
    float len2 = dot(r, r);
    // degenerate screen-space frames (edge-on slivers) keep the plain normal
    return (abs(det) > 1e-12 && len2 > 1e-24) ? r * inversesqrt(len2) : n;
}

void main() {
    if (valid_at(v_hm_uv) < 0.5) discard;
#ifdef SHADOW_PASS
    return;  // depth only
#else
    write_log_depth(v_log_z);
    vec3 n = morphed_normal(v_hm_uv, v_tile_uv);
    if (u_detail_normals > 0.0 && u_has_imagery && v_skirt < 0.5) {
        float reach = u_detail_distance_km * 1000.0;
        float fade = 1.0 - smoothstep(0.35 * reach, reach, length(v_world));
        if (fade > 0.0) n = detail_normal(n, u_detail_normals * fade);
    }
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
    li.topo = u_has_topo ? texture(u_topo, v_tile_uv).rgb : vec3(0.85);
    {
        // node uv -> screen pixels (uses the larger footprint of anisotropic texels)
        float px_per_uv = 1.0 / max(max(fwidth(v_tile_uv.x), fwidth(v_tile_uv.y)), 1e-7);
        vec2 d = u_has_borders ? texture(u_borders, v_tile_uv).rg : vec2(1.0);
        li.border_px = d.r * px_per_uv;
        li.region_px = u_borders_regions ? d.g * px_per_uv : 1e9;
        li.border_reach = 1.0 - smoothstep(0.5 * BORDER_MAX_UV, 0.95 * BORDER_MAX_UV, d.r);
    }

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
    if (u_borders_overlay) {
        float region = line_coverage(li.region_px, u_region_width);
        color = mix(color, srgb_to_linear(u_region_color) * flat_light * 0.8, region * 0.8);
        float border = line_coverage(li.border_px, u_border_width);
        vec3 border_rgb = srgb_to_linear(u_border_color);
        color = mix(color, border_rgb * flat_light, border);
        // glow: emitted light, visible at night too
        float glow = exp(-li.border_px / max(u_border_glow, 0.1)) * u_border_glow_strength
                   * li.border_reach;
        color += border_rgb * glow * (0.15 + 0.35 * max(flat_light.g, 0.02));
    }
    if (u_contours_overlay) {
        float c = contour_lines(v_height, u_contour_interval, u_contour_major, u_contour_width);
        vec3 line = srgb_to_linear(u_contour_color) * mix(flat_light, lit, 0.5) * 0.5;
        color = mix(color, line, c * u_contour_opacity);
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
