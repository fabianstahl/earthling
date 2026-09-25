#include "atmosphere.glsl"
in vec2 v_ndc;
out vec4 f_color;

uniform mat4 u_inv_view_proj;     // camera-relative, rotation only
uniform vec3 u_sun_dir;           // ENU, towards the sun
uniform float u_sun_illuminance;  // scale of the sky radiance
uniform float u_camera_height;    // metres above sea level
uniform float u_sun_disc = 1.0;
uniform float u_star_brightness = 1.0;
uniform float u_night = 0.0;      // 0 day .. 1 full night (stars fade in)
uniform mat3 u_to_celestial;      // ENU -> equatorial frame (for the star field)
uniform float u_pixel_angle;      // radians per pixel (star size)

float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

vec3 hash33(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}

vec3 stars(vec3 dir) {
    vec3 c = normalize(u_to_celestial * dir);
    vec3 result = vec3(0.0);
    for (int layer = 0; layer < 2; ++layer) {
        float scale = 70.0 * float(layer + 1);
        vec3 cell = floor(c * scale);
        for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dz = -1; dz <= 1; ++dz) {
            vec3 id = cell + vec3(dx, dy, dz) + float(layer) * 101.0;
            float h = hash13(id);
            if (h < 0.9) continue;
            vec3 pos = normalize((cell + vec3(dx, dy, dz) + hash33(id)) / scale);
            float d = acos(clamp(dot(c, pos), -1.0, 1.0));
            float size = max(u_pixel_angle * 0.7, 1e-5);
            float mag = pow((h - 0.9) / 0.1, 8.0) * (layer == 0 ? 4.0 : 1.0);
            vec3 tint = mix(vec3(0.75, 0.82, 1.0), vec3(1.0, 0.86, 0.7), hash13(id + 7.0));
            result += tint * mag * exp(-(d * d) / (size * size));
        }
    }
    return result;
}

void main() {
    vec4 far = u_inv_view_proj * vec4(v_ndc, 1.0, 1.0);
    vec3 dir = normalize(far.xyz / far.w);
    vec3 origin = vec3(0.0, 0.0, PLANET_RADIUS + max(u_camera_height, 1.0));
    vec3 sun = normalize(u_sun_dir);
    vec3 trans;
    vec3 radiance = scatter(origin, dir, 1e9, sun, trans) * u_sun_illuminance;
    bool below_horizon = ray_sphere(origin, dir, PLANET_RADIUS).x > 0.0;
    // sun disc (angular radius ~0.27 deg) with limb darkening
    float cos_d = dot(dir, sun);
    float disc_edge = cos(radians(0.27));
    if (cos_d > disc_edge && !below_horizon) {
        float r = sqrt(max(0.0, 1.0 - (1.0 - cos_d) / (1.0 - disc_edge)));
        radiance += trans * u_sun_illuminance * 60.0 * u_sun_disc * (0.4 + 0.6 * r);
    }
    // stars, dimmed by the atmosphere near the horizon
    if (u_night > 0.0 && !below_horizon) {
        radiance += stars(dir) * u_night * u_star_brightness * 0.02 * trans;
    }
    // faint airglow so the night sky is not pitch black
    radiance += vec3(0.0004, 0.0006, 0.0012) * u_night * (below_horizon ? 0.3 : 1.0);
    f_color = vec4(radiance, 1.0);
}
