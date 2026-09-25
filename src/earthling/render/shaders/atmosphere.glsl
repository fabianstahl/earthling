// Atmosphere medium (Rayleigh + Mie + ozone), shared by the LUT passes (atmosphere_lut.glsl).
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

// True if the ray really hits the sphere in front of its origin (not a miss, not behind).
bool hits_ahead(vec2 hit) {
    return hit.x <= hit.y && hit.x > 0.0;
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
