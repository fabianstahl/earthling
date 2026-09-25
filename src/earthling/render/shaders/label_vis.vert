// Label occlusion test: one point per label. The anchor is compared with the scene's log depth
// at 3x3 taps around its screen position; the visible fraction goes to texel i of an N x 1
// target that is read back for decluttering.
in vec4 in_anchor;  // camera-relative position (xyz), depth tolerance in metres (w)

uniform mat4 u_view_proj;
uniform sampler2D u_depth;
uniform vec2 u_depth_size;
uniform float u_log_depth_coef;
uniform float u_count;

out float v_visible;

float scene_view_depth(vec2 uv) {
    float d = texture(u_depth, uv).r;
    if (d >= 0.999999) return 1e30;  // sky
    return exp2(d / u_log_depth_coef) - 1.0;
}

void main() {
    vec4 clip = u_view_proj * vec4(in_anchor.xyz, 1.0);
    float visible = 0.0;
    if (clip.w > 0.0) {
        vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
        vec2 texel = 2.0 / u_depth_size;
        float taps = 0.0;
        for (int j = -1; j <= 1; ++j) {
            for (int i = -1; i <= 1; ++i) {
                vec2 q = uv + vec2(i, j) * texel;
                taps += 1.0;
                if (any(lessThan(q, vec2(0.0))) || any(greaterThan(q, vec2(1.0)))) continue;
                float scene = scene_view_depth(q);
                float view_z = clip.w;  // (the log depth stores 1 + w)
                if (view_z <= scene * 1.002 + in_anchor.w) visible += 1.0;
            }
        }
        visible /= taps;
    }
    v_visible = visible;
    gl_Position = vec4((float(gl_VertexID) + 0.5) / u_count * 2.0 - 1.0, 0.0, 0.0, 1.0);
}
