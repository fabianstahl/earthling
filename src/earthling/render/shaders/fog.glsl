// Aerial perspective (precomputed volume, see atmosphere_lut.glsl) and exponential height
// fog. Requires atmosphere_lut.glsl.
uniform float u_camera_height;        // metres above sea level
uniform vec3 u_sun_dir = vec3(-0.5, 0.6, 0.6);  // ENU, towards the sun
uniform float u_sun_illuminance;      // sky radiance scale (shared with the sky pass)
uniform float u_aerial_strength = 0.0;  // set from the store (off without one)
uniform bool u_fog_enabled = false;
uniform float u_fog_density = 0.0;    // extinction per metre at the base height
uniform float u_fog_base = 1000.0;    // metres above sea level
uniform float u_fog_falloff = 300.0;  // height scale in metres
uniform vec3 u_fog_ambient;           // linear radiance of fog lit by the sky
uniform vec3 u_fog_sun;               // linear radiance of fog lit by the sun (before phase)
uniform float u_fog_glow = 0.6;       // forward scattering anisotropy (0 .. 0.95)
uniform sampler3D u_ap_inscatter;     // aerial perspective volume: (screen uv, sqrt(d / max))
uniform sampler3D u_ap_transmittance;
uniform vec2 u_ap_screen = vec2(1.0);  // render target size (froxels span the screen)
uniform float u_ap_max_dist = 300e3;
uniform float u_rain_haze = 0.0;      // rain: extinction per metre (reduced visibility)

float phase_hg(float mu, float g) {
    float gg = g * g;
    return (1.0 - gg) / (4.0 * 3.14159265 * pow(1.0 + gg - 2.0 * g * mu, 1.5));
}

// Optical depth of the height fog along a ray from the camera (dir normalised, ENU).
float fog_optical_depth(vec3 dir, float dist) {
    float b = 1.0 / max(u_fog_falloff, 1.0);
    float start = u_fog_density * exp(-b * (u_camera_height - u_fog_base));
    float k = b * dir.z * dist;
    // integral of exp(-k s) over s in [0, 1], stable for small k
    float shape = abs(k) > 1e-4 ? (1.0 - exp(-k)) / k : 1.0 - 0.5 * k;
    return start * dist * shape;
}

// Applies aerial perspective and height fog to radiance seen at distance `dist` along `dir`.
vec3 apply_atmosphere(vec3 color, vec3 dir, float dist) {
    vec3 origin = vec3(0.0, 0.0, PLANET_RADIUS + max(u_camera_height, 1.0));
    vec3 sun = normalize(u_sun_dir);
    if (u_aerial_strength > 0.0) {
        vec2 uv = gl_FragCoord.xy / u_ap_screen;
        float w = sqrt(clamp(dist / u_ap_max_dist, 0.0, 1.0));
        vec3 ap = vec3(uv, w);
        float fade = clamp(w * 64.0, 0.0, 1.0);  // nothing at the camera (first slice at w=1/64)
        vec3 inscatter = texture(u_ap_inscatter, ap).rgb * fade * u_sun_illuminance;
        vec3 trans = mix(vec3(1.0), texture(u_ap_transmittance, ap).rgb, fade);
        color = color * mix(vec3(1.0), trans, u_aerial_strength) + inscatter * u_aerial_strength;
    }
    if (u_fog_enabled && u_fog_density > 0.0) {
        float tau = min(fog_optical_depth(dir, dist), 60.0);
        float t = exp(-tau);
        vec3 light = u_fog_ambient + u_fog_sun * (4.0 * 3.14159265) * phase_hg(dot(dir, sun), u_fog_glow);
        color = color * t + light * (1.0 - t);
    }
    if (u_rain_haze > 0.0) {
        float t = exp(-min(dist * u_rain_haze, 60.0));
        color = color * t + u_fog_ambient * 1.1 * (1.0 - t);
    }
    return color;
}
