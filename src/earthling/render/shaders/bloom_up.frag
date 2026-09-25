// 3x3 tent upsample, added onto the next larger level (additive blending on the target).
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_source;
uniform vec2 u_texel;     // 1 / source size
uniform float u_radius = 1.0;
uniform float u_weight = 1.0;

void main() {
    vec2 t = u_texel * u_radius;
    vec3 sum = texture(u_source, v_uv).rgb * 4.0;
    sum += (texture(u_source, v_uv + vec2(-t.x, 0)).rgb + texture(u_source, v_uv + vec2(t.x, 0)).rgb
          + texture(u_source, v_uv + vec2(0, -t.y)).rgb + texture(u_source, v_uv + vec2(0, t.y)).rgb) * 2.0;
    sum += texture(u_source, v_uv + vec2(-t.x, -t.y)).rgb + texture(u_source, v_uv + vec2(t.x, -t.y)).rgb
         + texture(u_source, v_uv + vec2(-t.x, t.y)).rgb + texture(u_source, v_uv + vec2(t.x, t.y)).rgb;
    f_color = vec4(sum / 16.0 * u_weight, 1.0);
}
