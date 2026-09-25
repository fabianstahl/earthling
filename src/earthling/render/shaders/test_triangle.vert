in vec2 in_pos;
in vec3 in_color;
uniform float u_time;
out vec3 v_color;

void main() {
    float c = cos(u_time * 0.5);
    float s = sin(u_time * 0.5);
    vec2 p = mat2(c, s, -s, c) * in_pos;
    gl_Position = vec4(p, 0.0, 1.0);
    v_color = in_color;
}
