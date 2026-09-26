// Icons (premultiplied RGBA), SDF captions with an outline and pin lines. Display colours.
in vec2 v_uv;
in float v_alpha;
in float v_mode;
in float v_scale;
out vec4 f_color;

uniform sampler2D u_icon;
uniform sampler2D u_atlas;
uniform float u_spread;
uniform float u_outline_px = 1.8;

void main() {
    if (v_mode < 0.5) {
        f_color = texture(u_icon, v_uv) * v_alpha;
    } else if (v_mode < 1.5) {
        float d = texture(u_atlas, v_uv).r;
        float aa = max(fwidth(d), 1e-4) * 0.7;
        float outline = max(0.5 - u_outline_px / (2.0 * u_spread * max(v_scale, 1e-3)), 0.06);
        float text = smoothstep(0.5 - aa, 0.5 + aa, d);
        float halo = smoothstep(outline - aa, outline + aa, d);
        vec3 c = mix(vec3(0.05), vec3(1.0), text);
        float a = halo * v_alpha;
        f_color = vec4(c * a, a);
    } else {
        vec3 c = v_mode < 2.5 ? vec3(0.05) : vec3(1.0);
        f_color = vec4(c * v_alpha, v_alpha);
    }
}
