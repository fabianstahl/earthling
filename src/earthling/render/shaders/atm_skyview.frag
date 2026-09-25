// Sky-view LUT for the current camera height and sun (rendered every frame).
#include "atmosphere_lut.glsl"
in vec2 v_uv;
out vec4 f_color;

uniform float u_camera_height;
uniform vec3 u_sun_dir;

void main() {
    float r = R_BOTTOM + max(u_camera_height, 1.0);
    float view_zenith_cos, light_view_cos;
    skyview_params(v_uv, r, view_zenith_cos, light_view_cos);
    vec3 sun_enu = normalize(u_sun_dir);
    // local frame: +x towards the sun's azimuth
    vec3 sun = vec3(sqrt(max(1.0 - sun_enu.z * sun_enu.z, 0.0)), 0.0, sun_enu.z);
    float s = sqrt(max(1.0 - view_zenith_cos * view_zenith_cos, 0.0));
    vec3 dir = vec3(s * light_view_cos, s * sqrt(max(1.0 - light_view_cos * light_view_cos, 0.0)),
                    view_zenith_cos);
    vec3 transmittance;
    vec3 radiance = integrate_scattering(vec3(0.0, 0.0, r), dir, 1e9, sun, 30, transmittance);
    f_color = vec4(radiance, 1.0);
}
