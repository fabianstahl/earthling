// Signed-distance-field text with an outline (halo) for readability on any background.
// Colours are display (sRGB) values: labels are drawn after tone mapping. Premultiplied output.
in vec2 v_uv;
in float v_alpha;
in float v_scale;
in float v_mode;
out vec4 f_color;

uniform sampler2D u_atlas;
uniform float u_spread;  // SDF spread in atlas pixels
uniform vec3 u_label_color = vec3(1.0);
uniform vec3 u_label_outline_color = vec3(0.05);
uniform float u_label_outline = 2.0;  // pixels

void main() {
    if (v_mode > 0.5) {
        vec3 c = v_mode > 1.5 ? u_label_outline_color : u_label_color;
        f_color = vec4(c * v_alpha, v_alpha);
        return;
    }
    float d = texture(u_atlas, v_uv).r;
    float aa = max(fwidth(d), 1e-4) * 0.7;
    // the SDF only reaches u_spread atlas pixels: thinner outlines on very small text
    float outline = max(0.5 - u_label_outline / (2.0 * u_spread * max(v_scale, 1e-3)), 0.06);
    float text = smoothstep(0.5 - aa, 0.5 + aa, d);
    float halo = smoothstep(outline - aa, outline + aa, d);
    vec3 c = mix(u_label_outline_color, u_label_color, text);
    float a = halo * v_alpha;
    f_color = vec4(c * a, a);
}
