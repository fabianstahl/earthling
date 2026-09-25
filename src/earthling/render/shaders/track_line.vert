in vec3 in_pos;         // relative to the drawable's origin
uniform mat4 u_view_proj;  // camera-relative (rotation only)
uniform vec3 u_offset;     // origin - camera position
out float v_log_z;

void main() {
    gl_Position = u_view_proj * vec4(in_pos + u_offset, 1.0);
    v_log_z = 1.0 + gl_Position.w;
}
