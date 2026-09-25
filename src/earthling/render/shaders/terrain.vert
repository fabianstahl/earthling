#include "terrain_common.glsl"
in vec3 in_pos;  // ellipsoid position relative to the node origin
in vec3 in_up;
in vec2 in_uv;    // node-local uv
in float in_skirt;
uniform mat4 u_view_proj;
uniform vec3 u_offset;  // node origin - camera position
uniform float u_skirt_depth;

out vec2 v_hm_uv;
out vec2 v_tile_uv;
out float v_height;
out vec3 v_world;  // camera-relative position
out float v_log_z;

void main() {
    v_tile_uv = in_uv;
    v_hm_uv = heightmap_uv(in_uv);
    float h = height_at(v_hm_uv) - in_skirt * u_skirt_depth;
    vec3 p = in_pos + in_up * h + u_offset;
    v_world = p;
    v_height = h / max(u_exaggeration, 1e-6);
    gl_Position = u_view_proj * vec4(p, 1.0);
    v_log_z = 1.0 + gl_Position.w;
}
