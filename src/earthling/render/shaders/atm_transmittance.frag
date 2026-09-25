// Transmittance LUT: from (r, mu) to the top of the atmosphere (ground not considered).
#include "atmosphere_lut.glsl"
in vec2 v_uv;
out vec4 f_color;

void main() {
    float r, mu;
    transmittance_params(v_uv, r, mu);
    vec3 pos = vec3(0.0, 0.0, r);
    vec3 dir = vec3(sqrt(max(1.0 - mu * mu, 0.0)), 0.0, mu);
    float t_max = ray_sphere(pos, dir, R_TOP).y;
    const int STEPS = 40;
    vec3 depth = vec3(0.0);
    for (int i = 0; i < STEPS; ++i) {
        float t = t_max * (float(i) + 0.5) / float(STEPS);
        vec3 p = pos + dir * t;
        depth += medium_at(length(p) - R_BOTTOM).extinction;
    }
    f_color = vec4(exp(-min(depth * t_max / float(STEPS), vec3(80.0))), 1.0);
}
