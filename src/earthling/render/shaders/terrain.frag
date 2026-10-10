#include "terrain_common.glsl"
#include "logdepth.glsl"
#include "atmosphere_lut.glsl"
#include "fog.glsl"
#include "shadows.glsl"
#include "weather.glsl"
uniform vec3 u_camera_enu;
uniform float u_wetness = 0.0;  // rain: darker, more saturated ground
uniform float u_snow_cover = 0.0;        // fresh snow above the snow line (0 = none)
uniform float u_snow_line = 2500.0;      // metres above sea level
uniform float u_snow_transition = 200.0; // height range of the patchy edge
uniform float u_snow_max_slope = 50.0;   // steeper rock stays free (degrees)
// views from far up: every point lit by its own sun elevation (see lighting.py)
uniform float u_local_light = 0.0;       // 0 the camera's light everywhere .. 1 local light
uniform vec3 u_sun_base = vec3(1.4);     // sunlight before the atmosphere (linear)
uniform vec3 u_sky_day = vec3(0.2);      // ambient parts by day and at night
uniform vec3 u_sky_night = vec3(0.01);
uniform vec3 u_ground_day = vec3(0.1);
uniform vec3 u_ground_night = vec3(0.004);
uniform float u_cam_day = 1.0;           // the camera's daylight factor
uniform sampler2D u_night;               // city lights (NASA Black Marble)
uniform bool u_has_night = false;
uniform float u_night_lights = 0.0;      // strength (0 = off)
uniform vec3 u_night_color = vec3(1.0, 0.7, 0.4);
in vec2 v_hm_uv;
in vec2 v_tile_uv;
in float v_skirt;  // 1 on the skirts hiding cracks between nodes
in float v_height;
in vec3 v_world;
in float v_log_z;
in vec3 v_up;
out vec4 f_color;

uniform sampler2D u_imagery;
uniform bool u_has_imagery;
uniform sampler2D u_parent_imagery;     // geomorphing: blend in from the parent's imagery
uniform bool u_has_parent_imagery = false;
uniform sampler2D u_topo;
uniform bool u_has_topo;
uniform sampler2D u_borders;  // distance to border lines in node uv units (r: countries, g: regions)
uniform bool u_has_borders;
uniform bool u_borders_overlay = false;
uniform bool u_borders_regions = true;
uniform float u_borders_opacity = 1.0;
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

#include "water.glsl"

float snow_hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float snow_noise(vec2 p) {  // value noise, 0..1
    vec2 i = floor(p), f = fract(p);
    vec2 w = f * f * (3.0 - 2.0 * f);
    return mix(mix(snow_hash(i), snow_hash(i + vec2(1, 0)), w.x),
               mix(snow_hash(i + vec2(0, 1)), snow_hash(i + vec2(1, 1)), w.x), w.y);
}

// Fresh snow: above the snow line (with a patchy, noisy edge), not on steep rock. A partial
// cover leaves snow only in the patches the noise favours.
float snow_amount(float height, float slope) {
    if (u_snow_cover <= 0.0) return 0.0;
    vec2 xy = (v_world + u_camera_enu).xy;
    // every octave fades to its mean once a pixel covers its period (no speckles from afar)
    float fp = max(fwidth(xy).x, fwidth(xy).y);
    float n90 = mix(snow_noise(xy / 90.0), 0.5, smoothstep(15.0, 60.0, fp));
    float n23 = mix(snow_noise(xy / 23.0), 0.5, smoothstep(4.0, 15.0, fp));
    float n = n90 * 0.65 + n23 * 0.35;
    float h = height + (n - 0.5) * u_snow_transition * 1.2;
    float above = smoothstep(u_snow_line - u_snow_transition, u_snow_line + u_snow_transition, h);
    float flat_enough = 1.0 - smoothstep(u_snow_max_slope - 8.0, u_snow_max_slope + 4.0, slope);
    float patches = smoothstep(0.0, 0.3, u_snow_cover * 1.3 - n * (1.0 - u_snow_cover));
    // small rocks and ridges poke through (the bare slopes stay dark in the imagery)
    patches *= mix(1.0, smoothstep(0.08, 0.3, dot(texture(u_imagery, v_tile_uv).rgb,
                                                  vec3(0.299, 0.587, 0.114))), 0.5);
    return clamp(above * flat_enough * patches, 0.0, 1.0);
}

void main() {
    if (valid_at(v_hm_uv) < 0.5) discard;
#ifdef SHADOW_PASS
    return;  // depth only
#else
    write_log_depth(v_log_z);
    mat3 tangent = tangent_at(normalize(v_up));
    vec3 n = morphed_normal(v_hm_uv, v_tile_uv, tangent);
    if (u_detail_normals > 0.0 && u_has_imagery && v_skirt < 0.5) {
        float reach = u_detail_distance_km * 1000.0;
        float fade = 1.0 - smoothstep(0.35 * reach, reach, length(v_world));
        if (fade > 0.0) n = detail_normal(n, u_detail_normals * fade);
    }
    vec3 sun = normalize(u_sun_dir);
    // the sun elevation at this point (the planet's centre is R below the frame origin)
    vec3 up_here = normalize(v_world + u_camera_enu + vec3(0.0, 0.0, PLANET_RADIUS));
    float sin_sun = dot(up_here, sun);
    vec3 sun_radiance = u_sun_radiance;
    vec3 sky_ambient = u_sky_ambient;
    vec3 ground_ambient = u_ground_ambient;
    if (u_local_light > 0.0) {
        float day_here = smoothstep(-0.139, 0.139, sin_sun);  // +-8 degrees, like lighting.py
        vec3 sky_cam = u_sky_day * u_cam_day + u_sky_night * (1.0 - u_cam_day);
        // (the bounce light needs the sun itself: its visibility at each point, not the camera's)
        float vis_here = smoothstep(-0.017, 0.035, sin_sun);
        // (as a ratio: the weather's dimming and lightning flashes stay in)
        vec3 sky_here = u_sky_ambient * (u_sky_day * day_here + u_sky_night * (1.0 - day_here))
                        / max(sky_cam, vec3(1e-6));
        vec3 dim = u_sky_ambient / max(sky_cam, vec3(1e-6));  // weather dimming, flashes
        vec3 ground_here = dim * (u_ground_day * (0.3 + 0.7 * vis_here) * day_here
                                  + u_ground_night * (1.0 - day_here));
        float r_here = PLANET_RADIUS + max(v_height, 0.0);
        vec3 sun_here = u_sun_base * transmittance_to_top(r_here, sin_sun)
                        * smoothstep(-0.017, 0.035, sin_sun);  // the disc sets over ~1 degree
        sky_ambient = mix(u_sky_ambient, sky_here, u_local_light);
        ground_ambient = mix(u_ground_ambient, ground_here, u_local_light);
        sun_radiance = mix(u_sun_radiance, sun_here, u_local_light);
    }
    float diffuse = max(dot(n, sun), 0.0);
    float sun_vis = 1.0;  // terrain and cloud shadows (also for the glint on the water)
    if (diffuse > 0.0 || u_water_enabled) {
        sun_vis = mix(1.0, sun_shadow(v_world, n, sun), u_shadow_strength);
        if (u_layer_count > 0) sun_vis *= cloud_shadow(v_world, u_camera_enu, u_camera_height, sun);
    }
    diffuse *= sun_vis;

    LayerInput li;
    li.tile_uv = v_tile_uv;
    li.height = v_height;
    li.normal = n;
    li.normal_local = transpose(tangent) * n;
    li.slope = degrees(acos(clamp(li.normal_local.z, -1.0, 1.0)));
    li.aspect = mod(degrees(atan(li.normal_local.x, li.normal_local.y)) + 360.0, 360.0);
    li.imagery = u_has_imagery ? texture(u_imagery, v_tile_uv).rgb
                               : ramp_hypsometric((v_height - 500.0) / 4000.0);
    if (u_morph < 1.0 && u_has_parent_imagery) {
        vec3 parent_rgb = texture(u_parent_imagery, u_parent_uv_offset + v_tile_uv * 0.5).rgb;
        li.imagery = mix(parent_rgb, li.imagery, u_has_imagery ? u_morph : 0.0);
    }
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
    vec3 ambient = mix(ground_ambient, sky_ambient, li.normal_local.z * 0.5 + 0.5);
    vec3 lit = ambient + sun_radiance * diffuse;
    // "flat" presentation: same overall brightness as the scene, without relief
    vec3 flat_light = mix(ground_ambient, sky_ambient, 0.75) + sun_radiance * 0.65;

    vec3 wet = mix(vec3(1.0), vec3(0.55, 0.58, 0.62), u_wetness);
    float snow = snow_amount(v_height, li.slope);
    const vec3 SNOW = vec3(0.78, 0.8, 0.84);  // linear albedo of fresh snow
    vec3 albedo_a = srgb_to_linear(layer_albedo(u_layer_a, li)) * layer_gain(u_layer_a) * wet;
    // on the imagery: the snow keeps a little of its brightness detail (no plastic look)
    vec3 snow_albedo = SNOW * (0.82 + 0.35 * dot(li.imagery, vec3(0.299, 0.587, 0.114)));
    if (u_layer_a == 0) albedo_a = mix(albedo_a, snow_albedo, snow);
    vec3 color = albedo_a * mix(flat_light, lit, layer_shading(u_layer_a));
    if (u_layer_mix > 0.0) {
        vec3 albedo_b = srgb_to_linear(layer_albedo(u_layer_b, li)) * layer_gain(u_layer_b) * wet;
        if (u_layer_b == 0) albedo_b = mix(albedo_b, snow_albedo, snow);
        vec3 color_b = albedo_b * mix(flat_light, lit, layer_shading(u_layer_b));
        color = mix(color, color_b, u_layer_mix);
    }
    if (u_borders_overlay && u_borders_opacity > 0.0) {
        float region = line_coverage(li.region_px, u_region_width) * u_borders_opacity;
        color = mix(color, srgb_to_linear(u_region_color) * flat_light * 0.8, region * 0.8);
        float border = line_coverage(li.border_px, u_border_width) * u_borders_opacity;
        vec3 border_rgb = srgb_to_linear(u_border_color);
        color = mix(color, border_rgb * flat_light, border);
        // glow: emitted light, visible at night too
        float glow = exp(-li.border_px / max(u_border_glow, 0.1)) * u_border_glow_strength
                   * li.border_reach * u_borders_opacity;
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
    // the water surface (on the imagery): sky reflection, sun glint, waves
    float water_layer = (u_layer_a == 0 ? 1.0 - u_layer_mix : 0.0) + (u_layer_b == 0 ? u_layer_mix : 0.0);
    if (u_water_enabled && water_layer > 0.0 && v_skirt < 0.5) {
        // the OpenStreetMap water mask, or (without it) the sea level of the DEM
        vec2 wm = u_has_water ? texture(u_water, v_tile_uv).rg
                              : vec2(1.0 - smoothstep(0.5, 3.0, v_height));
        if (wm.r > 0.002) {
            vec3 shaded = water_shade(color, v_world, up_here, wm.g, sun_radiance * sun_vis);
            color = mix(color, shaded, wm.r * water_layer);
        }
    }
    if (u_night_lights > 0.0 && u_has_night) {
        // city lights where it is night here (they come on in the dusk)
        float dark = 1.0 - smoothstep(-0.105, 0.05, sin_sun);
        vec3 lights = texture(u_night, v_tile_uv).rgb;
        float lum = dot(lights, vec3(0.299, 0.587, 0.114));
        float glow = max(lum - 0.12, 0.0) / 0.88;  // (the dark land of the mosaic stays dark)
        color += u_night_color * glow * u_night_lights * dark * 0.08;
    }
    float dist = length(v_world);
    color = apply_atmosphere(color, v_world / max(dist, 1e-3), dist);
    f_color = vec4(color, 1.0);  // linear HDR radiance
#endif
}
