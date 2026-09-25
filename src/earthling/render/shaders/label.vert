// Screen-space label geometry (glyph quads and leader lines), laid out on the CPU in output
// pixels with y pointing down.
in vec2 in_pos;
in vec2 in_uv;
in vec3 in_params;  // alpha, screen px per atlas px, mode (0 glyph, 1 light shape, 2 dark shape)

uniform vec2 u_viewport;

out vec2 v_uv;
out float v_alpha;
out float v_scale;
out float v_mode;

void main() {
    vec2 ndc = vec2(in_pos.x / u_viewport.x * 2.0 - 1.0, 1.0 - in_pos.y / u_viewport.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_uv;
    v_alpha = in_params.x;
    v_scale = in_params.y;
    v_mode = in_params.z;
}
