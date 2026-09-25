// Single-scattering atmosphere (Rayleigh + Mie), raymarched in real time.
// Coordinates: metres, planet centre at origin, local "up" = +z.
// Constants and the optical depth LUT layout must match render/atmosphere.py.
const float PLANET_RADIUS = 6371e3;
const float ATMOSPHERE_RADIUS = 6471e3;
const float ATMOSPHERE_HEIGHT = ATMOSPHERE_RADIUS - PLANET_RADIUS;
const float RAYLEIGH_SCALE_HEIGHT = 8e3;
const float MIE_SCALE_HEIGHT = 1.2e3;
const vec3 RAYLEIGH_BETA = vec3(5.5e-6, 13.0e-6, 22.4e-6);
const float MIE_BETA = 21e-6;
const float MIE_EXTINCTION = 1.1;
const float MIE_G = 0.758;
const vec3 OZONE_BETA = vec3(0.65e-6, 1.881e-6, 0.085e-6);
const float OZONE_CENTER = 25e3;
const float OZONE_HALF_WIDTH = 15e3;

uniform float u_rayleigh_scale = 1.0;  // air density multiplier
uniform float u_mie_scale = 1.0;       // haze / aerosol multiplier
uniform sampler2D u_optical_depth;     // RGB: Rayleigh / Mie / ozone optical depth to the sky

// (near, far) intersection distances of a ray with a sphere; far < near means miss.
vec2 ray_sphere(vec3 origin, vec3 dir, float radius) {
    float b = dot(origin, dir);
    float c = dot(origin, origin) - radius * radius;
    float d = b * b - c;
    if (d < 0.0) return vec2(1e30, -1e30);
    d = sqrt(d);
    return vec2(-b - d, -b + d);
}

float phase_rayleigh(float mu) {
    return 3.0 / (16.0 * 3.14159265) * (1.0 + mu * mu);
}

float phase_mie(float mu) {
    float g = MIE_G, gg = g * g;
    return 3.0 / (8.0 * 3.14159265) * ((1.0 - gg) * (1.0 + mu * mu))
           / ((2.0 + gg) * pow(1.0 + gg - 2.0 * mu * g, 1.5));
}

float ozone_density(float h) {
    return max(0.0, 1.0 - abs(h - OZONE_CENTER) / OZONE_HALF_WIDTH);
}

// Optical depth (Rayleigh, Mie, ozone) from a point at height h towards the top of the
// atmosphere along a ray whose zenith angle has cosine mu.
vec3 optical_depth_to_sky(float h, float mu) {
    float v = sqrt(clamp(h / ATMOSPHERE_HEIGHT, 0.0, 1.0));
    float s = clamp(mu, -1.0, 1.0);
    float u = 0.5 + 0.5 * sign(s) * sqrt(abs(s));
    return texture(u_optical_depth, vec2(u, v)).rgb;
}

vec3 extinction(vec3 od) {
    return exp(-min(RAYLEIGH_BETA * u_rayleigh_scale * od.x
                    + MIE_BETA * MIE_EXTINCTION * u_mie_scale * od.y
                    + OZONE_BETA * u_rayleigh_scale * od.z, vec3(80.0)));
}

// In-scattered radiance along the ray segment [0, max_dist] (unit sun irradiance) and the
// transmittance of that segment.
vec3 scatter(vec3 origin, vec3 dir, float max_dist, vec3 sun_dir, out vec3 transmittance) {
    const int SAMPLES = 32;
    vec3 beta_r = RAYLEIGH_BETA * u_rayleigh_scale;
    float beta_m = MIE_BETA * u_mie_scale;
    vec2 hit = ray_sphere(origin, dir, ATMOSPHERE_RADIUS);
    transmittance = vec3(1.0);
    if (hit.x > hit.y) return vec3(0.0);
    float t0 = max(hit.x, 0.0);
    float t1 = min(hit.y, max_dist);
    vec2 ground = ray_sphere(origin, dir, PLANET_RADIUS);
    if (ground.x > 0.0) t1 = min(t1, ground.x);
    if (t1 <= t0) return vec3(0.0);
    float span = t1 - t0;
    vec3 od_view = vec3(0.0);
    vec3 sum_r = vec3(0.0), sum_m = vec3(0.0);
    for (int i = 0; i < SAMPLES; ++i) {
        // quadratic spacing: dense near the camera where the air is densest
        float s = (float(i) + 0.5) / float(SAMPLES);
        float t = t0 + span * s * s;
        float dt = span * 2.0 * s / float(SAMPLES);
        vec3 p = origin + dir * t;
        float r = length(p);
        float h = r - PLANET_RADIUS;
        float dr = exp(-h / RAYLEIGH_SCALE_HEIGHT) * dt;
        float dm = exp(-h / MIE_SCALE_HEIGHT) * dt;
        vec3 d_od = vec3(dr, dm, ozone_density(h) * dt);
        od_view += d_od * 0.5;  // half step before, half after the sample
        vec3 od_sun = optical_depth_to_sky(h, dot(p / r, sun_dir));
        vec3 att = extinction(od_view + od_sun);
        sum_r += dr * att;
        sum_m += dm * att;
        od_view += d_od * 0.5;
    }
    transmittance = extinction(od_view);
    float mu = dot(dir, sun_dir);
    return sum_r * beta_r * phase_rayleigh(mu) + sum_m * beta_m * phase_mie(mu);
}
