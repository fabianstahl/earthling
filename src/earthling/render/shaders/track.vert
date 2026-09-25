// Screen-space extruded track ribbon (miter joins).
in vec3 in_pos;   // relative to the track origin
in vec3 in_prev;
in vec3 in_next;
in float in_side; // -1 left, +1 right
in float in_dist; // metres along the track

uniform mat4 u_view_proj;
uniform vec3 u_offset;      // track origin - camera position
uniform vec2 u_viewport;    // pixels
uniform float u_focal_px;   // pixels per radian (for widths in metres)
uniform float u_track_width = 4.0;
uniform int u_track_width_mode = 0;  // 0 pixels, 1 metres
uniform float u_track_min_px = 1.5;  // metre widths never get thinner than this

out float v_side;
out float v_dist;
out float v_width_px;
out vec3 v_world;
out float v_log_z;

vec2 to_screen(vec4 clip) {
    return clip.xy / max(clip.w, 1e-4) * 0.5 * u_viewport;
}

void main() {
    vec3 p = in_pos + u_offset;
    vec4 clip = u_view_proj * vec4(p, 1.0);
    vec4 clip_prev = u_view_proj * vec4(in_prev + u_offset, 1.0);
    vec4 clip_next = u_view_proj * vec4(in_next + u_offset, 1.0);
    vec2 s = to_screen(clip);
    // neighbours behind the camera: fall back to the other side's direction
    vec2 d_in = clip_prev.w > 0.0 ? s - to_screen(clip_prev) : to_screen(clip_next) - s;
    vec2 d_out = clip_next.w > 0.0 ? to_screen(clip_next) - s : d_in;
    d_in = length(d_in) > 1e-6 ? normalize(d_in) : vec2(1.0, 0.0);
    d_out = length(d_out) > 1e-6 ? normalize(d_out) : d_in;
    vec2 tangent = normalize(d_in + d_out + vec2(1e-6));
    vec2 normal = vec2(-tangent.y, tangent.x);
    float miter = 1.0 / max(dot(normal, vec2(-d_in.y, d_in.x)), 0.35);

    float width_px = u_track_width;
    if (u_track_width_mode == 1) {
        width_px = max(u_track_width * u_focal_px / max(clip.w, 1e-3), u_track_min_px);
    }
    width_px += 2.0;  // room for anti-aliasing
    vec2 offset_px = normal * in_side * 0.5 * width_px * miter;
    clip.xy += offset_px / (0.5 * u_viewport) * clip.w;
    gl_Position = clip;
    v_side = in_side;
    v_dist = in_dist;
    v_width_px = width_px;
    v_world = p;
    v_log_z = 1.0 + clip.w;
}
