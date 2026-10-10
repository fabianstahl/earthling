// Volumetric cloud / fog layers shared by the cloud pass and the terrain (cloud shadows).
// Coordinates: x, y = ENU metres of the frame (world anchored), h = metres above sea level.
#define MAX_CLOUD_LAYERS 4
uniform sampler3D u_cloud_noise;
uniform sampler2D u_weather_map;
uniform int u_layer_count = 0;
uniform vec4 u_layer_shape[MAX_CLOUD_LAYERS];  // base (m), thickness (m), coverage, extinction (1/m)
uniform vec4 u_layer_noise[MAX_CLOUD_LAYERS];  // feature size (m), wind offset x, y (m), detail
uniform vec4 u_layer_color[MAX_CLOUD_LAYERS];  // albedo rgb, brightness
uniform float u_layer_vary[MAX_CLOUD_LAYERS];   // 0 .. 1: how much base and top vary
uniform float u_weather_time = 0.0;             // seconds (noise evolution)
uniform float u_cloud_shadow_strength = 0.0;

// ``lod``: noise mip level from the sample footprint (negative when magnified: the finer
// detail noise then still gets its full resolution); ``detail``: 0 .. 1 fine erosion
float cloud_density_lod(int i, vec2 xy, float h, float lod, float detail_amount) {
    vec4 shape = u_layer_shape[i];
    float hf = (h - shape.x) / max(shape.y, 1.0);
    if (hf <= 0.0 || hf >= 1.0) return 0.0;
    vec4 nz = u_layer_noise[i];
    vec2 p = xy - nz.yz;
    // a slow warp of the pattern (no row of similar round blobs)
    vec2 warp = textureLod(u_cloud_noise, vec3(p / (nz.x * 3.0), 0.37), 1.0).ag - 0.5;
    p += warp * nz.x * 0.45;
    // base and top of each cloud vary (not all clouds at one height)
    float vary = u_layer_vary[i];
    if (vary > 0.0) {
        float lo = texture(u_weather_map, p / (nz.x * 2.3) + 0.31).r;
        float hi = texture(u_weather_map, p / (nz.x * 3.1) + 0.67).r;
        float bottom = vary * 0.55 * lo;
        float top = 1.0 - vary * 0.45 * (1.0 - hi);
        hf = (hf - bottom) / max(top - bottom, 0.05);
        if (hf <= 0.0 || hf >= 1.0) return 0.0;
    }
    // clusters and gaps at two scales (fields of cloud and wide clear areas between them);
    // no modulation at coverage 0 (clear) and 1 (overcast)
    float cover = mix(texture(u_weather_map, p / (nz.x * 9.0)).r,
                      texture(u_weather_map, p / (nz.x * 40.0) + 0.53).r, 0.6);
    cover = clamp((cover - 0.5) * 1.5 + 0.5, 0.0, 1.0);  // (the mix is narrower than 0..1)
    // the local coverage: up to 3 x the set one in the cloud fields, none in the gaps (mean
    // about the set coverage; mirrored for a mostly covered sky: holes in the deck)
    float sparse = shape.z * 3.0 * cover * cover;
    float dense = 1.0 - (1.0 - shape.z) * 3.0 * (1.0 - cover) * (1.0 - cover);
    float c = clamp(mix(sparse, dense, smoothstep(0.3, 0.7, shape.z)), 0.0, 1.0);
    if (c <= 0.001) return 0.0;
    float profile = smoothstep(0.0, 0.25, hf) * smoothstep(1.0, 0.45, hf);
    vec3 uvw = vec3(p, h + u_weather_time * 0.3) / nz.x;
    vec4 n = textureLod(u_cloud_noise, uvw, max(lod, 0.0));
    // a coarser octave: big and small clouds side by side, not one size everywhere
    float big = textureLod(u_cloud_noise, uvw * 0.37 + vec3(0.21, 0.63, 0.0), max(lod - 1.4, 0.0)).r;
    float base = mix(n.r, big, 0.35) * profile;
    float d = clamp((base - (1.0 - c)) / max(c, 1e-3), 0.0, 1.0);
    if (d <= 0.0) return 0.0;
    if (detail_amount > 0.0) {
        float detail = textureLod(u_cloud_noise, uvw * 3.7, max(lod + 1.9, 0.0)).g;
        d = clamp(d - (1.0 - detail) * nz.w * 0.35 * detail_amount, 0.0, 1.0);
    }
    return d * shape.w;
}

float cloud_density(int i, vec2 xy, float h) {
    return cloud_density_lod(i, xy, h, 0.0, 1.0);
}

float weather_height(vec3 rel, float camera_height) {
    // camera-relative ENU offset -> height above sea level (earth curvature included)
    vec3 l = u_atm_frame * rel;  // the camera's local vertical
    return camera_height + l.z + dot(l.xy, l.xy) / (2.0 * 6371e3);
}

// Optical depth towards the sun through the layers above height h (one sample per layer).
// ``above_only``: skip layers the point is inside of (for cloud samples, whose own layer is
// raymarched separately).
float layers_optical_depth(vec2 xy, float h, vec3 sun, bool above_only) {
    float tau = 0.0;
    // continuous down to the horizon (a cut-off at low sun made clouds pop at sunrise)
    float sz = max((u_atm_frame * sun).z, 0.02);
    for (int i = 0; i < u_layer_count; ++i) {
        vec4 shape = u_layer_shape[i];
        float mid = shape.x + 0.5 * shape.y;
        if (h >= mid || (above_only && h >= shape.x)) continue;
        float t = (mid - h) / sz;
        vec2 p = xy + sun.xy * t;
        float slant = shape.y / max(sz, 0.25);
        tau += cloud_density_lod(i, p, mid, 1.5, 0.0) * slant * 0.6;
    }
    return tau;
}

// Transmittance of the sun light through all layers above a point (ground shadows).
float cloud_shadow(vec3 rel, vec3 camera_enu, float camera_height, vec3 sun) {
    if (u_layer_count == 0 || u_cloud_shadow_strength <= 0.0 || sun.z <= -0.1) return 1.0;
    float h = weather_height(rel, camera_height);
    float tau = layers_optical_depth(camera_enu.xy + rel.xy, h, sun, false);
    return mix(1.0, exp(-tau), u_cloud_shadow_strength);
}

// Sun transmittance through the layers entirely above a cloud sample (e.g. fog under the
// rain deck is not lit by the sun).
float shadow_from_layers_above(vec3 rel, vec3 camera_enu, float camera_height, vec3 sun) {
    if (u_layer_count < 2 || sun.z <= -0.1) return 1.0;
    float h = weather_height(rel, camera_height);
    return exp(-layers_optical_depth(camera_enu.xy + rel.xy, h, sun, true));
}
