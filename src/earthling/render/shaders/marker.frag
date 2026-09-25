#include "logdepth.glsl"
in vec2 v_local;
in float v_log_z;
in float v_pulse;
out vec4 f_color;

uniform vec3 u_color = vec3(1.0, 0.9, 0.6);  // linear
uniform float u_marker_glow = 3.0;
uniform float u_marker_pulse = 0.8;

void main() {
    write_log_depth(v_log_z);
    float r = length(v_local) * 2.5;  // in units of the marker size (the quad spans 2.5 sizes)
    float core = 1.0 - smoothstep(0.42, 0.5, r);          // solid dot (diameter = size)
    float rim = smoothstep(0.34, 0.42, r) * core;           // darker rim for contrast
    float ring_r = 0.5 + v_pulse * 1.9;
    float ring = (1.0 - smoothstep(0.0, 0.08, abs(r - ring_r))) * (1.0 - v_pulse);
    if (u_marker_pulse <= 0.0) ring = 0.0;
    float halo = exp(-r * 2.2) * 0.6;
#ifdef GLOW_PASS
    if (core + ring + halo < 0.01) discard;
    f_color = vec4(u_color * u_marker_glow * (core + ring * 0.6 + halo), 1.0);
#else
    vec3 color = mix(u_color * 1.6, u_color * 0.35, rim);
    float alpha = max(core, ring * 0.7);
    if (alpha <= 0.002) discard;
    f_color = vec4(mix(u_color * 1.2, color, core), alpha);
#endif
}
