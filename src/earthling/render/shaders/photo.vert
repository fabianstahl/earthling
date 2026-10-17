// One photo (with its frame and shadow) as a quad laid out in output pixels (y down).
in vec2 in_local;  // pixels from the photo centre, before rotation

uniform vec2 u_viewport;
uniform vec2 u_center;  // output pixels
uniform float u_angle;  // radians, clockwise on screen

out vec2 v_local;

void main() {
    float c = cos(u_angle), s = sin(u_angle);
    vec2 p = u_center + vec2(c * in_local.x - s * in_local.y, s * in_local.x + c * in_local.y);
    gl_Position = vec4(p.x / u_viewport.x * 2.0 - 1.0, 1.0 - p.y / u_viewport.y * 2.0, 0.0, 1.0);
    v_local = in_local;
}
