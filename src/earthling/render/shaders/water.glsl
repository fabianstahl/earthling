// Water surface (sea, lakes): Fresnel reflection of the sky, the sun's glint after Cox & Munk
// (1954) with the slope variance of the wind's waves, and moving wave normals close up. The
// satellite imagery stays the colour of the water body below the reflection. Needs
// atmosphere_lut.glsl and fog.glsl (sky radiance, sun light) before it.
uniform sampler2D u_water;         // r: water coverage, g: open water (0 at the shore)
uniform bool u_has_water = false;
uniform bool u_water_enabled = false;
uniform float u_water_reflection = 1.0;  // strength of reflection and glint
uniform float u_water_waves = 1.0;       // wave relief close up
uniform float u_water_wind = 5.0;        // m/s: rougher water spreads the glint
uniform float u_water_wind_dir = 60.0;   // degrees, the waves run towards it
uniform float u_water_time = 0.0;        // seconds (timeline)
uniform sampler2D u_skyview_lut;

// Sky radiance seen in direction `dir` (frame ENU), from the camera's sky-view LUT.
vec3 water_sky(vec3 dir) {
    vec3 d = u_atm_frame * dir;
    d.z = max(d.z, 0.01);  // reflections never look below the horizon
    d = normalize(d);
    vec3 sun = u_atm_frame * normalize(u_sun_dir);
    float r = PLANET_RADIUS + max(u_camera_height, 1.0);
    float light_view_cos = 1.0;
    if (length(d.xy) > 1e-5 && length(sun.xy) > 1e-5) {
        light_view_cos = dot(normalize(d.xy), normalize(sun.xy));
    }
    return texture(u_skyview_lut, skyview_uv(r, d.z, light_view_cos, false)).rgb * u_sun_illuminance;
}

// Wave slopes (dh/dx, dh/dy) at world position xy: a few deep-water waves around the wind
// direction. Waves shorter than a few pixels are left out (they would shimmer); their slope
// variance goes into the roughness instead (returned in .z).
vec3 water_waves(vec2 xy, float footprint) {
    const float G = 9.81;
    vec2 slope = vec2(0.0);
    float lost = 0.0;
    float base = radians(u_water_wind_dir);
    for (int i = 0; i < 22; ++i) {
        float fi = float(i);
        float h1 = fract(sin(fi * 12.9898 + 1.3) * 43758.5453);
        float h2 = fract(sin(fi * 78.233 + 4.1) * 43758.5453);
        // 300 m swell .. 1.3 m ripples: the longer waves carry the pattern into the distance
        float wavelength = 300.0 * pow(0.78, fi) * (0.8 + 0.4 * h2);
        float angle = base + (h1 - 0.5) * 2.6;                         // +-75 degrees
        vec2 dir = vec2(sin(angle), cos(angle));
        float k = 6.2831853 / wavelength;
        // slope amplitude (long swell is flatter)
        float s = 0.032 * u_water_waves * (0.5 + h2) * mix(0.45, 1.0, smoothstep(80.0, 20.0, wavelength));
        float omega = sqrt(G * k);
        float phase = k * dot(dir, xy) - omega * u_water_time + h1 * 6.2831853;
        // waves under ~20 pixels per wavelength alias (shimmer, and rings around the camera
        // where one component after the other fades): their slopes become roughness
        // (a wide fade: components leave one after the other, no visible edge)
        float keep = 1.0 - smoothstep(0.012, 0.07, footprint / wavelength);
        slope += dir * s * cos(phase) * keep;
        lost += 0.5 * s * s * (1.0 - keep);
    }
    return vec3(slope, lost);
}

// The water's look at a fragment: `base` is the lit imagery colour; returns the new colour.
vec3 water_shade(vec3 base, vec3 world, vec3 up, float open_water, vec3 sun_radiance) {
    vec3 view = normalize(-world);                    // towards the camera
    vec2 xy = (world + u_camera_enu).xy;
    float footprint = max(length(fwidth(xy)), 1e-3);  // metres per pixel
    vec3 w = water_waves(xy, footprint);
    // slope variance of the wind's capillary waves (Cox & Munk), plus the waves left out
    float sigma2 = max(0.003 + 0.00512 * u_water_wind, 0.002) + w.z;
    vec3 east = normalize(cross(vec3(0.0, 1.0, 0.0), up));
    vec3 north = cross(up, east);
    float waves = open_water * 0.85 + 0.15;           // calm along the shore
    vec3 n = normalize(up - (east * w.x + north * w.y) * waves);
    float cos_v = max(dot(n, view), 0.02);
    // (rough water seen at a grazing angle shows facets turned to the camera: the reflectance
    // levels off instead of reaching 1)
    float fresnel = min(0.02 + 0.98 * pow(1.0 - cos_v, 5.0), 0.55 - 8.0 * sigma2);
    vec3 refl = reflect(-view, n);
    vec3 sky = water_sky(refl);
    // sun glint: the share of facets tilted to mirror the sun to the camera
    vec3 sun = normalize(u_sun_dir);
    vec3 h = normalize(sun + view);
    float cos_b = max(dot(h, up), 1e-3);
    float tan2 = (1.0 - cos_b * cos_b) / (cos_b * cos_b);
    float p = exp(-tan2 / sigma2) / (3.14159265 * sigma2 * pow(cos_b, 4.0));
    float cos_l = dot(up, sun);
    float glint = 0.0;
    if (cos_l > 0.0) {
        float f_h = 0.02 + 0.98 * pow(1.0 - max(dot(h, view), 0.0), 5.0);
        glint = f_h * p / (4.0 * max(dot(up, view), 0.05));
        glint *= 0.5 + 0.5 * open_water;
    }
    vec3 surface = base * (1.0 - fresnel) + sky * fresnel + sun_radiance * min(glint, 8.0);
    return mix(base, surface, u_water_reflection);
}
