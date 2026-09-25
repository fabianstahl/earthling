#include "logdepth.glsl"
in float v_log_z;
uniform vec4 u_color;
out vec4 f_color;

void main() {
    write_log_depth(v_log_z);
    f_color = u_color;
}
