// Solid shapes and SDF text with a dark outline. Premultiplied output, display colours.
in vec2 v_uv;
in vec4 v_color;
in vec2 v_params;
out vec4 f_color;

uniform sampler2D u_atlas;
uniform float u_spread;
uniform float u_outline_px = 1.5;
uniform float u_outline_alpha = 0.6;

void main() {
    if (v_params.y > 0.5) {
        f_color = vec4(v_color.rgb * v_color.a, v_color.a);
        return;
    }
    float d = texture(u_atlas, v_uv).r;
    float aa = max(fwidth(d), 1e-4) * 0.7;
    float outline = max(0.5 - u_outline_px / (2.0 * u_spread * max(v_params.x, 1e-3)), 0.06);
    float text = smoothstep(0.5 - aa, 0.5 + aa, d);
    float halo = smoothstep(outline - aa, outline + aa, d) * u_outline_alpha;
    float a = max(text, halo) * v_color.a;
    vec3 c = v_color.rgb * text / max(max(text, halo), 1e-4);  // dark halo behind the text
    f_color = vec4(c * a, a);
}
