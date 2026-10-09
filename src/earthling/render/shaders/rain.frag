// Rain streaks (or snowflakes): several world-anchored layers on cylinders around the camera
// (3 .. 32 m), with an angular constant streak width, falling with the timeline time and slanted
// by the wind. Additive onto the HDR target; occluded by the scene depth; only below the rain
// cloud base. Optionally a rainbow where sunlit rain hangs between the camera and the scene.
#include "atmosphere_lut.glsl"
#include "colormaps.glsl"
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
uniform int u_snow = 0;                // 1: snowflakes instead of rain streaks
uniform float u_rainbow = 0.0;         // rainbow strength (0 = none)
uniform vec3 u_rainbow_sun = vec3(0.0); // direction towards the sun (frame ENU)
uniform vec3 u_rainbow_light = vec3(0.0); // sun radiance reaching the rain

float hash11(float n) { return fract(sin(n * 127.1) * 43758.5453); }

// Rainbow colours over the angle from the antisolar point (degrees): the primary bow from violet
// (40.6) to red (42.4), a darker band outside it, the weaker secondary bow with reversed colours
// (50.2 .. 53.2) and the brighter sky inside the primary bow.
vec3 rainbow(float angle) {
    float t1 = (angle - 40.5) / 2.0;
    float t2 = (angle - 50.2) / 3.0;
    vec3 c = vec3(0.0);
    if (t1 > 0.0 && t1 < 1.0) {
        vec3 hue = hsv_to_rgb(vec3((1.0 - t1) * 0.78, 1.0, 1.0));
        c += hue * (1.0 - pow(abs(2.0 * t1 - 1.0), 3.0));
    }
    if (t2 > 0.0 && t2 < 1.0) {
        vec3 hue = hsv_to_rgb(vec3(t2 * 0.78, 1.0, 1.0));
        c += 0.4 * hue * (1.0 - pow(abs(2.0 * t2 - 1.0), 3.0));
    }
    c += vec3(0.12) * (1.0 - smoothstep(30.0, 40.5, angle)) * smoothstep(5.0, 30.0, angle);
    return c;
}

void main() {
    // a point on the near plane: the far plane is numerically unstable for huge far / near
    vec4 ray = u_inv_view_proj * vec4(v_ndc, -1.0, 1.0);
    vec3 dir = normalize(ray.xyz / ray.w);
    float depth = texture(u_depth, v_uv).r;
    float scene = 1e9;
    if (depth < 0.999999) {
        float view_z = exp2(depth / u_log_depth_coef) - 1.0;
        scene = view_z / max(dot(dir, u_cam_forward), 1e-3);
    }
    vec3 result = vec3(0.0);
    // rainbow: sunlit rain along the ray (below the cloud base, in front of the scene)
    if (u_rainbow > 0.0 && u_rain_intensity > 0.0 && u_snow == 0) {
        float angle = degrees(acos(clamp(dot(dir, -normalize(u_rainbow_sun)), -1.0, 1.0)));
        if (angle < 54.0) {
            float reach = min(scene, 30000.0);
            if (u_camera_height > u_rain_base) {
                reach = 0.0;  // above the rain: no bow
            } else if (dir.z > 1e-4) {
                reach = min(reach, (u_rain_base - u_camera_height) / dir.z);
            }
            float amount = 1.0 - exp(-reach / 2500.0);
            result += rainbow(angle) * u_rainbow_light * amount * u_rainbow * 0.05
                      * (0.4 + 0.6 * u_rain_intensity);
        }
    }
    float horiz = length(dir.xy);
    if (horiz < 0.03 || u_rain_intensity <= 0.0) {
        f_color = vec4(result, 0.0);
        return;
    }
    float az = atan(dir.y, dir.x);
    vec2 tangent = vec2(-sin(az), cos(az));
    bool snow = u_snow != 0;
    float FALL = snow ? 1.1 : 8.0;              // m/s
    const float EXPOSURE = 1.0 / 60.0;          // motion blur of one video frame
    float sum = 0.0;
    for (int layer = 0; layer < 6; ++layer) {
        float r = 2.5 * pow(1.8, float(layer));
        float t = r / horiz;
        if (t > scene) continue;
        vec3 p = dir * t;
        float h = u_camera_height + p.z;
        if (h > u_rain_base) continue;
        float cell = (snow ? 0.035 : 0.02) * r;  // lane spacing: constant on screen
        // the lanes slant with the wind (snow at most ~50 degrees: lanes running almost
        // horizontally break the flakes into streaks)
        float shear = dot(u_rain_wind, tangent) / FALL;
        if (snow) shear = clamp(shear, -1.2, 1.2);
        float u = az * r + p.z * shear;
        float lane_id = floor(u / cell) + float(layer) * 1013.0;
        float h1 = hash11(lane_id);
        float h2 = hash11(lane_id * 1.37 + 5.1);
        if (h1 > u_rain_intensity * 0.9 + 0.1) continue;     // sparser lanes in light rain
        float period = snow ? 0.5 + h2 * 1.2 : 0.8 + h2 * 2.0;  // metres between drops of a lane
        float fall = p.z + u_time * FALL * (0.9 + 0.2 * h1) + h2 * 17.0;
        float x = fract(u / cell) - 0.5 - (h2 - 0.5) * 0.6;
        if (snow) {
            // the lanes are sheared by the wind (u above): measure the distance to the flake
            // centre in real metres, the vertical part as seen on screen (looking up or down,
            // the cylinder's vertical metres shrink by horiz), so flakes stay round
            float drop = floor(fall / period);
            float phase = hash11(drop * 1.618 + lane_id * 0.71);
            float jitter = (phase - 0.5) * 0.85;              // flakes off the lane centre ...
            x -= jitter + 0.2 * sin(u_time * (1.3 + phase) + phase * 6.28);  // ... tumbling
            float dz = (fract(fall / period) - 0.5) * period;  // metres above the flake
            vec2 d = vec2(x * cell - dz * shear, dz * horiz);
            // (flakes smaller than a pixel are drawn at pixel size and fainter, and never
            // wider than their lane)
            float want = max(0.06 * cell, u_pixel_angle * t * 0.8);
            float size = min(want, 0.22 * cell);
            float flake = exp(-dot(d, d) / (size * size)) * (size / want);
            sum += flake * 1.6 / (1.0 + 0.35 * float(layer));
            continue;
        }
        float width = max(0.045, u_pixel_angle * t / cell * 0.45);
        float across = exp(-x * x / (width * width));
        float streak_len = FALL * EXPOSURE * 1.2 + 0.03;
        float along = fract(fall / period);
        float head = streak_len / period;
        float streak = smoothstep(0.0, head * 0.2, along) * (1.0 - smoothstep(head * 0.6, head, along));
        float fade = 0.55 / (1.0 + 0.35 * float(layer));
        sum += across * streak * fade;
    }
    result += u_rain_light * sum * 0.5 * (0.35 + 0.65 * u_rain_intensity) * (snow ? 1.8 : 1.0);
    f_color = vec4(result, 0.0);
}
