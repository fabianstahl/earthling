// Screen-aligned quad around a world position (4 vertices, triangle strip).
uniform mat4 u_view_proj;
uniform vec3 u_center;       // camera-relative
uniform vec2 u_viewport;
uniform float u_marker_size = 14.0;  // pixels (core diameter)
uniform float u_time;
uniform float u_marker_pulse = 0.8;

out vec2 v_local;   // -1..1 across the quad
out float v_log_z;
out float v_pulse;  // 0..1 phase of the pulse ring

void main() {
    vec2 corner = vec2((gl_VertexID & 1) * 2 - 1, (gl_VertexID >> 1) * 2 - 1);
    vec4 clip = u_view_proj * vec4(u_center, 1.0);
    float radius_px = u_marker_size * 2.5;  // room for the pulse ring and halo
    clip.xy += corner * radius_px / (0.5 * u_viewport) * clip.w;
    gl_Position = clip;
    v_local = corner;
    v_log_z = 1.0 + clip.w;
    v_pulse = fract(u_time * u_marker_pulse);
}
