// Precomputed atmosphere after Hillaire 2020, "A Scalable and Production Ready Sky and
// Atmosphere Rendering Technique": a transmittance LUT and a multiple-scattering LUT (both
// only depend on the medium), a per-frame sky-view LUT and a per-frame aerial perspective
// volume. Uses the medium constants of atmosphere.glsl. All radiance is per unit sun
// irradiance; the passes scale it with u_sun_illuminance.
#include "atmosphere.glsl"

#define PI 3.14159265358979
const float R_BOTTOM = PLANET_RADIUS;
const float R_TOP = ATMOSPHERE_RADIUS;
const vec3 GROUND_ALBEDO = vec3(0.1);  // average terrain (forest, rock); snow is brighter
const vec2 TRANSMITTANCE_SIZE = vec2(256.0, 64.0);
const vec2 MULTISCATTER_SIZE = vec2(32.0, 32.0);
const vec2 SKYVIEW_SIZE = vec2(192.0, 108.0);

uniform sampler2D u_transmittance_lut;
uniform sampler2D u_multiscatter_lut;

struct Medium {
    vec3 scat_r;      // Rayleigh scattering
    vec3 scat_m;      // Mie scattering
    vec3 scattering;
    vec3 extinction;  // incl. Mie and ozone absorption
};

Medium medium_at(float h) {
    Medium m;
    m.scat_r = RAYLEIGH_BETA * u_rayleigh_scale * exp(-h / RAYLEIGH_SCALE_HEIGHT);
    m.scat_m = vec3(MIE_BETA * u_mie_scale * exp(-h / MIE_SCALE_HEIGHT));
    m.scattering = m.scat_r + m.scat_m;
    m.extinction = m.scat_r + m.scat_m * MIE_EXTINCTION
                 + OZONE_BETA * u_rayleigh_scale * ozone_density(h);
    return m;
}

// Unit range <-> texel centres (values at the borders are stored exactly)
float from_unit(float x, float size) { return 0.5 / size + x * (1.0 - 1.0 / size); }
float to_unit(float u, float size) { return (u - 0.5 / size) / (1.0 - 1.0 / size); }

// --- transmittance LUT (Bruneton 2017 parameterisation) ------------------------------------
vec2 transmittance_uv(float r, float mu) {
    float H = sqrt(R_TOP * R_TOP - R_BOTTOM * R_BOTTOM);
    float rho = sqrt(max(r * r - R_BOTTOM * R_BOTTOM, 0.0));
    float disc = r * r * (mu * mu - 1.0) + R_TOP * R_TOP;
    float d = max(0.0, -r * mu + sqrt(max(disc, 0.0)));
    float d_min = R_TOP - r;
    float d_max = rho + H;
    float x_mu = (d - d_min) / max(d_max - d_min, 1e-3);
    float x_r = rho / H;
    return vec2(from_unit(clamp(x_mu, 0.0, 1.0), TRANSMITTANCE_SIZE.x),
                from_unit(clamp(x_r, 0.0, 1.0), TRANSMITTANCE_SIZE.y));
}

void transmittance_params(vec2 uv, out float r, out float mu) {
    float x_mu = to_unit(uv.x, TRANSMITTANCE_SIZE.x);
    float x_r = to_unit(uv.y, TRANSMITTANCE_SIZE.y);
    float H = sqrt(R_TOP * R_TOP - R_BOTTOM * R_BOTTOM);
    float rho = H * x_r;
    r = sqrt(rho * rho + R_BOTTOM * R_BOTTOM);
    float d_min = R_TOP - r;
    float d_max = rho + H;
    float d = d_min + x_mu * (d_max - d_min);
    mu = d == 0.0 ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * r * d);
    mu = clamp(mu, -1.0, 1.0);
}

vec3 transmittance_to_top(float r, float mu) {
    return texture(u_transmittance_lut, transmittance_uv(r, mu)).rgb;
}

// --- multiple scattering LUT: (cos sun zenith, height) -> isotropic multi-scattered light ----
vec3 multiscatter_at(float r, float mu_s) {
    vec2 uv = vec2(mu_s * 0.5 + 0.5, (r - R_BOTTOM) / (R_TOP - R_BOTTOM));
    uv = vec2(from_unit(clamp(uv.x, 0.0, 1.0), MULTISCATTER_SIZE.x),
              from_unit(clamp(uv.y, 0.0, 1.0), MULTISCATTER_SIZE.y));
    return texture(u_multiscatter_lut, uv).rgb;
}

// --- sky-view LUT: (azimuth relative to the sun, view zenith) around the camera -------------
// Latitude is concentrated at the horizon, azimuth towards the sun.
vec2 skyview_uv(float r, float view_zenith_cos, float light_view_cos, bool hits_ground) {
    float v_horizon = sqrt(max(r * r - R_BOTTOM * R_BOTTOM, 0.0));
    float beta = acos(clamp(v_horizon / r, -1.0, 1.0));
    float zenith_horizon = PI - beta;
    float zenith = acos(clamp(view_zenith_cos, -1.0, 1.0));
    float v;
    if (!hits_ground) {
        float c = 1.0 - sqrt(max(1.0 - zenith / zenith_horizon, 0.0));
        v = c * 0.5;
    } else {
        float c = sqrt(max((zenith - zenith_horizon) / beta, 0.0));
        v = c * 0.5 + 0.5;
    }
    float u = sqrt(clamp(-light_view_cos * 0.5 + 0.5, 0.0, 1.0));
    return vec2(from_unit(u, SKYVIEW_SIZE.x), from_unit(v, SKYVIEW_SIZE.y));
}

void skyview_params(vec2 uv, float r, out float view_zenith_cos, out float light_view_cos) {
    uv = vec2(to_unit(uv.x, SKYVIEW_SIZE.x), to_unit(uv.y, SKYVIEW_SIZE.y));
    float v_horizon = sqrt(max(r * r - R_BOTTOM * R_BOTTOM, 0.0));
    float beta = acos(clamp(v_horizon / r, -1.0, 1.0));
    float zenith_horizon = PI - beta;
    float zenith;
    if (uv.y < 0.5) {
        float c = 1.0 - 2.0 * uv.y;
        c = 1.0 - c * c;
        zenith = zenith_horizon * c;
    } else {
        float c = uv.y * 2.0 - 1.0;
        zenith = zenith_horizon + beta * c * c;
    }
    view_zenith_cos = cos(zenith);
    light_view_cos = -(uv.x * uv.x * 2.0 - 1.0);
}

// --- integration -----------------------------------------------------------------------------
// In-scattered radiance (single scattering with the real phase functions plus the multiple
// scattering term) along [0, t_max] from `pos`, and the transmittance of that segment.
vec3 integrate_scattering(vec3 pos, vec3 dir, float t_max, vec3 sun_dir, int steps,
                          out vec3 transmittance) {
    transmittance = vec3(1.0);
    vec2 atm = ray_sphere(pos, dir, R_TOP);
    if (atm.x > atm.y || atm.y < 0.0) return vec3(0.0);
    float t0 = max(atm.x, 0.0);
    float t1 = atm.y;
    vec2 ground = ray_sphere(pos, dir, R_BOTTOM);
    if (hits_ahead(ground)) t1 = min(t1, ground.x);
    t1 = min(t1, t_max);
    if (t1 <= t0) return vec3(0.0);
    float mu = dot(dir, sun_dir);
    float pr = phase_rayleigh(mu);
    float pm = phase_mie(mu);
    vec3 radiance = vec3(0.0);
    vec3 throughput = vec3(1.0);
    float t_prev = t0;
    for (int i = 0; i < steps; ++i) {
        float s = (float(i) + 1.0) / float(steps);
        float t = t0 + (t1 - t0) * s * s;  // dense near the start
        float dt = t - t_prev;
        float t_mid = 0.5 * (t + t_prev);
        t_prev = t;
        vec3 p = pos + dir * t_mid;
        float r = length(p);
        vec3 up = p / r;
        Medium m = medium_at(r - R_BOTTOM);
        float mu_s = dot(up, sun_dir);
        float lit = hits_ahead(ray_sphere(p, sun_dir, R_BOTTOM)) ? 0.0 : 1.0;  // earth shadow
        vec3 sun = transmittance_to_top(r, mu_s) * lit;
        vec3 source = sun * (m.scat_r * pr + m.scat_m * pm)
                    + multiscatter_at(r, mu_s) * m.scattering;
        vec3 step_t = exp(-m.extinction * dt);
        // energy-conserving analytic integration over the step (Hillaire 2015)
        radiance += throughput * (source - source * step_t) / max(m.extinction, vec3(1e-12));
        throughput *= step_t;
    }
    transmittance = throughput;
    return radiance;
}
