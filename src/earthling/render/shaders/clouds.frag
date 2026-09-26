// Volumetric cloud and fog layers, raymarched against the scene depth. Output: in-scattered
// radiance (rgb) and transmittance (a), blended as  color = rgb + dst * a.
#include "atmosphere_lut.glsl"
#include "fog.glsl"
#include "weather.glsl"
in vec2 v_uv;
in vec2 v_ndc;
out vec4 f_color;

uniform mat4 u_inv_view_proj;        // camera-relative
uniform sampler2D u_depth;           // scene log depth
uniform float u_log_depth_coef;
uniform vec3 u_cam_forward;
uniform vec3 u_camera_enu;           // camera position (frame ENU)
uniform vec3 u_sun_radiance;
uniform vec3 u_sky_ambient;
uniform vec3 u_ground_ambient;
uniform int u_steps = 64;
const float FIRST_STEP_M = 40.0;     // step length at the cloud entry (see main)
uniform float u_max_distance = 150e3;
uniform float u_jitter = 0.0;        // sub-frame index (dither changes per anti-aliasing sample)
uniform float u_pixel_angle = 0.001; // radians per pixel (noise level of detail)
#define MAX_FLASHES 2
uniform int u_flash_count = 0;
uniform vec4 u_flash_pos[MAX_FLASHES];    // camera-relative position, radius (m)
uniform vec3 u_flash_color[MAX_FLASHES];  // linear radiance scale

float ign(vec2 p) {  // interleaved gradient noise (dither)
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

float hg(float mu, float g) {
    float gg = g * g;
    return (1.0 - gg) / (4.0 * PI * pow(1.0 + gg - 2.0 * g * mu, 1.5));
}

// [t0, t1] where the ray is inside the shell [inner, outer] (radii from the planet centre)
vec2 shell_segment(vec3 origin, vec3 dir, float inner, float outer) {
    float r0 = length(origin);
    vec2 a = ray_sphere(origin, dir, outer);
    vec2 b = ray_sphere(origin, dir, inner);
    bool hits_inner = b.x <= b.y;
    if (a.x > a.y) return vec2(1.0, 0.0);
    if (r0 < inner) {
        return vec2(max(b.y, 0.0), a.y);
    }
    if (r0 <= outer) {
        float end = a.y;
        if (hits_inner && b.x > 0.0) end = b.x;
        return vec2(0.0, end);
    }
    if (a.x < 0.0) return vec2(1.0, 0.0);
    float end = (hits_inner && b.x > 0.0) ? b.x : a.y;
    return vec2(a.x, end);
}

float total_density_lod(vec3 rel, float lod, float detail, out float hf_out) {
    float h = length(vec3(0.0, 0.0, R_BOTTOM + max(u_camera_height, 1.0)) + rel) - R_BOTTOM;
    vec2 xy = u_camera_enu.xy + rel.xy;
    float d = 0.0;
    hf_out = 0.5;
    for (int i = 0; i < u_layer_count; ++i) {
        float di = cloud_density_lod(i, xy, h, lod, detail);
        if (di > 0.0) {
            d += di;
            hf_out = clamp((h - u_layer_shape[i].x) / max(u_layer_shape[i].y, 1.0), 0.0, 1.0);
        }
    }
    return d;
}

vec3 layer_albedo(vec3 rel) {
    float h = length(vec3(0.0, 0.0, R_BOTTOM + max(u_camera_height, 1.0)) + rel) - R_BOTTOM;
    vec3 albedo = vec3(0.0);
    float w = 0.0;
    for (int i = 0; i < u_layer_count; ++i) {
        vec4 s = u_layer_shape[i];
        if (h >= s.x && h <= s.x + s.y) {
            albedo += u_layer_color[i].rgb * u_layer_color[i].a;
            w += 1.0;
        }
    }
    return w > 0.0 ? albedo / w : vec3(0.9);
}

float total_density(vec3 rel, out float hf_out) {
    return total_density_lod(rel, 0.0, 1.0, hf_out);
}

void main() {
    if (u_layer_count == 0) {
        f_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    vec4 far = u_inv_view_proj * vec4(v_ndc, 1.0, 1.0);
    vec3 dir = normalize(far.xyz / far.w);
    float depth = texture(u_depth, v_uv).r;
    float scene = u_max_distance;
    if (depth < 0.999999) {
        float view_z = exp2(depth / u_log_depth_coef) - 1.0;
        scene = min(view_z / max(dot(dir, u_cam_forward), 1e-3), u_max_distance);
    }
    vec3 origin = vec3(0.0, 0.0, R_BOTTOM + max(u_camera_height, 1.0));
    vec3 sun = normalize(u_sun_dir);
    float mu = dot(dir, sun);
    float phase = mix(hg(mu, 0.65), hg(mu, -0.25), 0.3) * 4.0 * PI;
    float dither = fract(ign(gl_FragCoord.xy) + u_jitter * 0.618034);
    vec3 radiance = vec3(0.0);
    float transmittance = 1.0;
    float weighted_t = 0.0, weight = 0.0;
    for (int i = 0; i < u_layer_count && transmittance > 0.01; ++i) {
        vec4 s = u_layer_shape[i];
        vec2 seg = shell_segment(origin, dir, R_BOTTOM + s.x, R_BOTTOM + s.x + s.y);
        seg.y = min(seg.y, scene);
        if (seg.y <= seg.x) continue;
        float len = seg.y - seg.x;
        int steps = clamp(int(float(u_steps) * clamp(len / 6000.0, 0.25, 1.0)), 8, u_steps);
        // Steps grow geometrically from the entry point: dense clouds are opaque within a few
        // hundred metres, so even steps along long (grazing) segments would be all-or-nothing
        // samples that flicker with the smallest camera move.
        float n = float(steps);
        float warp = clamp(log(len / (n * FIRST_STEP_M)), 0.0, 8.0);
        float norm = warp > 1e-3 ? 1.0 / (exp(warp) - 1.0) : 0.0;
        for (int k = 0; k < steps && transmittance > 0.01; ++k) {
            float u = (float(k) + dither) / n;
            float t, dt;
            if (warp > 1e-3) {
                t = seg.x + len * (exp(warp * u) - 1.0) * norm;
                dt = len * warp * exp(warp * u) * norm / n;
            } else {
                t = seg.x + len * u;
                dt = len / n;
            }
            vec3 rel = dir * t;
            float hf;
            // noise level from the sample footprint (pixel size / step) vs. the voxel size
            float footprint = max(t * u_pixel_angle, dt * 0.5);
            float lod = log2(footprint / (u_layer_noise[i].x / 128.0));
            float detail = 1.0 - smoothstep(8000.0, 20000.0, t);
            float d = total_density_lod(rel, lod, detail, hf);
            d *= 1.0 - smoothstep(0.7 * u_max_distance, u_max_distance, t);
            if (d <= 1e-6) continue;
            // light towards the sun: a few samples with growing steps
            float tau = 0.0;
            float ds = 60.0;
            float along = 0.0;
            for (int j = 0; j < 5; ++j) {
                along += ds;
                float hj;
                tau += total_density_lod(rel + sun * along, max(lod, 0.0) + 1.0, 0.0, hj) * ds;
                ds *= 1.9;
            }
            float sun_t = exp(-tau) * (1.0 - exp(-2.0 * tau) * 0.5);  // Beer + powder
            sun_t *= shadow_from_layers_above(rel, u_camera_enu, u_camera_height, sun);
            vec3 albedo = layer_albedo(rel);
            vec3 ambient = mix(u_ground_ambient, u_sky_ambient, 0.35 + 0.65 * hf);
            ambient = mix(ambient, vec3(dot(ambient, vec3(0.3333))), 0.6);  // grey bases
            vec3 light = u_sun_radiance * sun_t * phase + ambient * 0.8;
            for (int f = 0; f < u_flash_count; ++f) {
                float dist = length(rel - u_flash_pos[f].xyz);
                light += u_flash_color[f] * exp(-dist / max(u_flash_pos[f].w, 1.0));
            }
            vec3 source = light * albedo * d;
            float step_t = exp(-d * dt);
            radiance += transmittance * (source - source * step_t) / d;
            weighted_t += t * transmittance * (1.0 - step_t);
            weight += transmittance * (1.0 - step_t);
            transmittance *= step_t;
        }
    }
    if (weight > 0.0) {
        // aerial perspective between the camera and the cloud
        // (the scene behind already has it up to its own distance; it gets attenuated by the
        // cloud, so the in-scattering in front of the cloud is added for the covered part)
        float dist = weighted_t / weight;
        vec3 inscatter = apply_atmosphere(vec3(0.0), dir, dist);
        vec3 through = apply_atmosphere(vec3(1.0), dir, dist) - inscatter;
        radiance = radiance * through + inscatter * (1.0 - transmittance);
    }
    f_color = vec4(radiance, transmittance);
}
