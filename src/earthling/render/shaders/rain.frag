// Rain streaks: several world-anchored layers on cylinders around the camera (3 .. 32 m), with
// an angular constant streak width, falling with the timeline time and slanted by the wind.
// Additive onto the HDR target; occluded by the scene depth; only below the rain cloud base.
#include "atmosphere_lut.glsl"
in vec2 v_uv;
in vec2 v_ndc;
out vec4 f_color;

uniform mat4 u_inv_view_proj;
uniform sampler2D u_depth;
uniform float u_log_depth_coef;
uniform vec3 u_cam_forward;
uniform float u_camera_height;
uniform float u_time;
uniform float u_rain_intensity = 0.7;
uniform float u_rain_base = 2000.0;    // cloud base (m above sea level)
uniform vec2 u_rain_wind = vec2(3.0, 0.0);
uniform vec3 u_rain_light = vec3(0.3); // linear radiance lighting the drops
uniform float u_pixel_angle = 0.001;

float hash11(float n) { return fract(sin(n * 127.1) * 43758.5453); }

void main() {
    vec4 far = u_inv_view_proj * vec4(v_ndc, 1.0, 1.0);
    vec3 dir = normalize(far.xyz / far.w);
    float depth = texture(u_depth, v_uv).r;
    float scene = 1e9;
    if (depth < 0.999999) {
        float view_z = exp2(depth / u_log_depth_coef) - 1.0;
        scene = view_z / max(dot(dir, u_cam_forward), 1e-3);
    }
    float horiz = length(dir.xy);
    if (horiz < 0.03 || u_rain_intensity <= 0.0) {
        f_color = vec4(0.0);
        return;
    }
    float az = atan(dir.y, dir.x);
    vec2 tangent = vec2(-sin(az), cos(az));
    const float FALL = 8.0;                     // m/s
    const float EXPOSURE = 1.0 / 60.0;          // motion blur of one video frame
    float sum = 0.0;
    for (int layer = 0; layer < 6; ++layer) {
        float r = 2.5 * pow(1.8, float(layer));
        float t = r / horiz;
        if (t > scene) continue;
        vec3 p = dir * t;
        float h = u_camera_height + p.z;
        if (h > u_rain_base) continue;
        float cell = 0.02 * r;                      // lane spacing: constant on screen
        float u = az * r + p.z * dot(u_rain_wind, tangent) / FALL;
        float lane_id = floor(u / cell) + float(layer) * 1013.0;
        float h1 = hash11(lane_id);
        float h2 = hash11(lane_id * 1.37 + 5.1);
        if (h1 > u_rain_intensity * 0.9 + 0.1) continue;     // sparser lanes in light rain
        float x = fract(u / cell) - 0.5 - (h2 - 0.5) * 0.6;
        float width = max(0.045, u_pixel_angle * t / cell * 0.45);
        float across = exp(-x * x / (width * width));
        float period = 0.8 + h2 * 2.0;              // metres between the drops of a lane
        float streak_len = FALL * EXPOSURE * 1.2 + 0.03;
        float along = fract((p.z + u_time * FALL * (0.9 + 0.2 * h1) + h2 * 17.0) / period);
        float head = streak_len / period;
        float streak = smoothstep(0.0, head * 0.2, along) * (1.0 - smoothstep(head * 0.6, head, along));
        float fade = 0.55 / (1.0 + 0.35 * float(layer));
        sum += across * streak * fade;
    }
    f_color = vec4(u_rain_light * sum * 0.5 * (0.35 + 0.65 * u_rain_intensity), 0.0);
}
