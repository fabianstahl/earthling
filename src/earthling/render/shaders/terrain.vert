#include "terrain_common.glsl"
in vec3 in_pos;  // ellipsoid position relative to the tile origin
in vec3 in_up;
uniform mat4 u_view_proj;
uniform vec3 u_offset;  // tile origin - camera position

out vec2 v_uv;
out vec2 v_tile_uv;
out float v_height;
out vec3 v_world;  // camera-relative position
out float v_log_z;

void main() {
    v_uv = vertex_uv(gl_VertexID);
    v_tile_uv = vertex_tile_uv(gl_VertexID);
    float h = height_at(v_uv);
    vec3 p = in_pos + in_up * h + u_offset;
    v_world = p;
    v_height = h / max(u_exaggeration, 1e-6);
    gl_Position = u_view_proj * vec4(p, 1.0);
    v_log_z = 1.0 + gl_Position.w;
}
