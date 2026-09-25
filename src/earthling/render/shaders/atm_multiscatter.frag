// Multiple scattering LUT (Hillaire 2020, section 5.5): the second-order light arriving at a
// point from all directions assuming isotropic phase, scaled by the geometric series of all
// higher orders, 1 / (1 - f_ms). Depends on the height and the sun zenith only.
#include "atmosphere_lut.glsl"
in vec2 v_uv;
out vec4 f_color;

void main() {
    vec2 uv = vec2(to_unit(v_uv.x, MULTISCATTER_SIZE.x), to_unit(v_uv.y, MULTISCATTER_SIZE.y));
    float mu_s = uv.x * 2.0 - 1.0;
    float r = R_BOTTOM + clamp(uv.y, 0.0, 1.0) * (R_TOP - R_BOTTOM - 10.0) + 1.0;
    vec3 pos = vec3(0.0, 0.0, r);
    vec3 sun = vec3(0.0, sqrt(max(1.0 - mu_s * mu_s, 0.0)), mu_s);
    const int N = 8;       // 8 x 8 directions over the sphere
    const int STEPS = 20;
    const float ISOTROPIC = 1.0 / (4.0 * PI);
    vec3 second = vec3(0.0);
    vec3 f_ms = vec3(0.0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float theta = 2.0 * PI * (float(i) + 0.5) / float(N);
            float cos_phi = 1.0 - 2.0 * (float(j) + 0.5) / float(N);
            float sin_phi = sqrt(max(1.0 - cos_phi * cos_phi, 0.0));
            vec3 dir = vec3(cos(theta) * sin_phi, sin(theta) * sin_phi, cos_phi);
            vec2 atm = ray_sphere(pos, dir, R_TOP);
            float t1 = atm.y;
            vec2 ground = ray_sphere(pos, dir, R_BOTTOM);
            bool hits_ground = hits_ahead(ground);
            if (hits_ground) t1 = ground.x;
            vec3 radiance = vec3(0.0), as_one = vec3(0.0), throughput = vec3(1.0);
            float t_prev = 0.0;
            for (int k = 0; k < STEPS; ++k) {
                float t = t1 * (float(k) + 1.0) / float(STEPS);
                float dt = t - t_prev;
                vec3 p = pos + dir * (0.5 * (t + t_prev));
                t_prev = t;
                float rr = length(p);
                Medium m = medium_at(rr - R_BOTTOM);
                float lit = hits_ahead(ray_sphere(p, sun, R_BOTTOM)) ? 0.0 : 1.0;
                vec3 sun_t = transmittance_to_top(rr, dot(p / rr, sun)) * lit;
                vec3 step_t = exp(-m.extinction * dt);
                vec3 inv_ext = 1.0 / max(m.extinction, vec3(1e-12));
                radiance += throughput * (m.scattering * sun_t * ISOTROPIC) * (1.0 - step_t) * inv_ext;
                as_one += throughput * m.scattering * (1.0 - step_t) * inv_ext;
                throughput *= step_t;
            }
            if (hits_ground) {  // sunlight reflected by the ground (Lambertian)
                vec3 p = pos + dir * t1;
                vec3 n = normalize(p);
                float cos_sun = max(dot(n, sun), 0.0);
                radiance += throughput * transmittance_to_top(length(p), dot(n, sun))
                          * cos_sun * GROUND_ALBEDO / PI;
            }
            second += radiance;
            f_ms += as_one;
        }
    }
    float count = float(N * N);
    // isotropic phase times the full sphere: the mean over the directions
    second /= count;
    f_ms /= count;
    f_color = vec4(second / max(vec3(1.0) - f_ms, vec3(1e-3)), 1.0);
}
