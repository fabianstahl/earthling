// Cascaded shadow map lookup (atlas of 2x2 cascades, hardware depth comparison).
uniform bool u_shadows_enabled = false;
uniform sampler2DShadow u_shadow_map;
uniform mat4 u_shadow_matrix[4];   // camera-relative world -> atlas uv + depth
uniform vec4 u_cascade_splits;     // far distance of each cascade (view depth)
uniform vec4 u_shadow_texel;       // world size of a shadow texel per cascade
uniform float u_shadow_atlas_texel;  // 1 / atlas size
uniform vec3 u_camera_forward;
uniform float u_shadow_softness = 1.5;  // PCF radius in texels
uniform float u_shadow_bias = 1.0;

// 1 = fully lit, 0 = in shadow. `world` is camera-relative, `n` the surface normal (ENU).
float sun_shadow(vec3 world, vec3 n, vec3 sun_dir) {
    if (!u_shadows_enabled) return 1.0;
    float depth = dot(world, u_camera_forward);
    int c = 0;
    if (depth > u_cascade_splits.x) c = 1;
    if (depth > u_cascade_splits.y) c = 2;
    if (depth > u_cascade_splits.z) c = 3;
    if (depth > u_cascade_splits.w) return 1.0;
    float texel = u_shadow_texel[c];
    // normal offset + slope scaled bias against acne
    float cos_l = clamp(dot(n, sun_dir), 0.0, 1.0);
    float slope = min(sqrt(1.0 - cos_l * cos_l) / max(cos_l, 0.05), 3.0);
    vec3 p = world + n * texel * (1.0 + slope) * u_shadow_bias;
    vec4 s = u_shadow_matrix[c] * vec4(p, 1.0);
    vec3 sc = s.xyz / s.w;
    // keep PCF taps inside the cascade's quadrant
    vec2 quad_min = vec2(float(c % 2), float(c / 2)) * 0.5;
    float margin = (u_shadow_softness + 1.0) * u_shadow_atlas_texel;
    if (any(lessThan(sc.xy, quad_min + margin)) || any(greaterThan(sc.xy, quad_min + 0.5 - margin)))
        return 1.0;
    float lit = 0.0;
    float r = u_shadow_softness * u_shadow_atlas_texel;
    for (int i = -1; i <= 1; ++i)
        for (int j = -1; j <= 1; ++j)
            lit += texture(u_shadow_map, vec3(sc.xy + vec2(i, j) * r, sc.z - 0.0002 * u_shadow_bias));
    return lit / 9.0;
}
