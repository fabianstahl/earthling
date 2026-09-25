// Stats overlay geometry laid out on the CPU in output pixels (y down).
in vec2 in_pos;
in vec2 in_uv;
in vec4 in_color;   // display colour, straight alpha
in vec2 in_params;  // screen px per atlas px, mode (0 = SDF glyph, 1 = solid)

uniform vec2 u_viewport;

out vec2 v_uv;
out vec4 v_color;
out vec2 v_params;

void main() {
    vec2 ndc = vec2(in_pos.x / u_viewport.x * 2.0 - 1.0, 1.0 - in_pos.y / u_viewport.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_uv;
    v_color = in_color;
    v_params = in_params;
}
