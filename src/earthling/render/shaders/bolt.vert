// Lightning bolt segments as screen-space ribbons. Every vertex carries both segment ends
// (same order for all six vertices), which end it sits on and its side (+-1), so the normal
// of the projected segment is consistent across the quad.
in vec3 in_a;          // camera-relative segment start
in vec3 in_b;          // segment end
in float in_end;       // 0 = at a, 1 = at b
in float in_side;
in float in_strength;

uniform mat4 u_view_proj;
uniform vec2 u_viewport;
uniform float u_width_px = 2.0;

out float v_side;
out float v_strength;
out float v_log_z;

void main() {
    vec4 a = u_view_proj * vec4(in_a, 1.0);
    vec4 b = u_view_proj * vec4(in_b, 1.0);
    vec2 sa = a.xy / max(a.w, 1e-3) * u_viewport;
    vec2 sb = b.xy / max(b.w, 1e-3) * u_viewport;
    vec2 d = normalize(sb - sa + vec2(1e-6));
    vec2 n = vec2(-d.y, d.x);
    vec4 p = in_end < 0.5 ? a : b;
    float width = u_width_px * (0.6 + 0.8 * min(in_strength, 1.5));
    p.xy += n * in_side * width / u_viewport * p.w;
    gl_Position = p;
    v_side = in_side;
    v_strength = in_strength;
    v_log_z = 1.0 + p.w;
}
