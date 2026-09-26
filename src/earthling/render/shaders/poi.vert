// POI geometry: every vertex carries its 3D anchor (camera-relative) plus a pixel offset.
// The anchor is tested against the scene depth (3 x 3 taps) for occlusion.
in vec3 in_anchor;
in vec2 in_offset;  // output pixels, y down
in vec2 in_uv;
in vec3 in_params;  // alpha, mode (0 icon, 1 SDF text, 2 dark, 3 light), glyph scale

uniform mat4 u_view_proj;
uniform vec2 u_viewport;
uniform sampler2D u_depth;
uniform vec2 u_depth_size;
uniform float u_log_depth_coef;

out vec2 v_uv;
out float v_alpha;
out float v_mode;
out float v_scale;

float scene_view_depth(vec2 uv) {
    float d = texture(u_depth, uv).r;
    if (d >= 0.999999) return 1e30;
    return exp2(d / u_log_depth_coef) - 1.0;
}

void main() {
    vec4 clip = u_view_proj * vec4(in_anchor, 1.0);
    float visible = 0.0;
    vec2 ndc = vec2(2.0);  // off screen
    if (clip.w > 0.0) {
        ndc = clip.xy / clip.w;
        vec2 uv = ndc * 0.5 + 0.5;
        vec2 texel = 2.0 / u_depth_size;
        // the drawn terrain is a coarser LOD surface than the heights the anchor sits on (it
        // can lie tens of metres higher far away): tolerance grows with the distance
        float tolerance = 10.0 + 0.015 * clip.w;
        for (int j = -1; j <= 1; ++j) {
            for (int i = -1; i <= 1; ++i) {
                vec2 q = uv + vec2(i, j) * texel;
                if (any(lessThan(q, vec2(0.0))) || any(greaterThan(q, vec2(1.0)))) continue;
                if (clip.w <= scene_view_depth(q) * 1.002 + tolerance) visible += 1.0;
            }
        }
        visible /= 9.0;
    }
    gl_Position = vec4(ndc + in_offset * vec2(2.0, -2.0) / u_viewport, 0.0, 1.0);
    v_uv = in_uv;
    v_alpha = in_params.x * visible;
    v_mode = in_params.y;
    v_scale = in_params.z;
}
