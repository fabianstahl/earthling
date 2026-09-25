// 13-tap downsample (Jimenez, "Next Generation Post Processing in Call of Duty").
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_source;
uniform vec2 u_texel;  // 1 / source size

void main() {
    vec2 t = u_texel;
    vec3 a = texture(u_source, v_uv + t * vec2(-2, 2)).rgb;
    vec3 b = texture(u_source, v_uv + t * vec2(0, 2)).rgb;
    vec3 c = texture(u_source, v_uv + t * vec2(2, 2)).rgb;
    vec3 d = texture(u_source, v_uv + t * vec2(-2, 0)).rgb;
    vec3 e = texture(u_source, v_uv).rgb;
    vec3 f = texture(u_source, v_uv + t * vec2(2, 0)).rgb;
    vec3 g = texture(u_source, v_uv + t * vec2(-2, -2)).rgb;
    vec3 h = texture(u_source, v_uv + t * vec2(0, -2)).rgb;
    vec3 i = texture(u_source, v_uv + t * vec2(2, -2)).rgb;
    vec3 j = texture(u_source, v_uv + t * vec2(-1, 1)).rgb;
    vec3 k = texture(u_source, v_uv + t * vec2(1, 1)).rgb;
    vec3 l = texture(u_source, v_uv + t * vec2(-1, -1)).rgb;
    vec3 m = texture(u_source, v_uv + t * vec2(1, -1)).rgb;
    vec3 color = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625
               + (j + k + l + m) * 0.125;
    f_color = vec4(color, 1.0);
}
