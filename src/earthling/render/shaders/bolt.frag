#include "logdepth.glsl"
in float v_side;
in float v_strength;
in float v_log_z;
out vec4 f_color;

uniform vec3 u_color = vec3(0.6, 0.7, 1.0);  // linear

void main() {
    write_log_depth(v_log_z);
    float core = 1.0 - v_side * v_side;
#ifdef GLOW_PASS
    f_color = vec4(u_color * v_strength * core * 25.0, 1.0);
#else
    f_color = vec4(u_color * v_strength * core * 60.0, 1.0);
#endif
}
