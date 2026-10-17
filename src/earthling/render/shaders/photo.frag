// The picture, a white frame around it and a soft drop shadow. Premultiplied output in display
// colours (drawn after tone mapping, like the HUD).
in vec2 v_local;
out vec4 f_color;

uniform sampler2D u_image;
uniform vec2 u_half;          // half size of the picture (px)
uniform float u_border;       // frame width (px)
uniform float u_shadow;       // shadow strength 0..1
uniform float u_shadow_blur;  // px
uniform vec2 u_shadow_offset; // px (in the rotated frame)
uniform float u_zoom;         // >= 1: zoom into the picture
uniform float u_alpha;

float box(vec2 p, vec2 half_size, float radius) {
    vec2 q = abs(p) - half_size + radius;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
}

void main() {
    float aa = max(fwidth(v_local.x), 1e-3);
    vec2 outer = u_half + u_border;
    float corner = min(u_border * 0.35, 6.0);
    float d_frame = box(v_local, outer, corner);
    float d_photo = box(v_local, u_half, 0.0);
    float frame_a = 1.0 - smoothstep(-aa, aa, d_frame);
    float photo_a = 1.0 - smoothstep(-aa, aa, d_photo);
    vec2 uv = (v_local / (2.0 * u_half)) / u_zoom + 0.5;
    vec3 picture = texture(u_image, uv).rgb;
    vec3 c = mix(vec3(0.97, 0.96, 0.94), picture, photo_a);
    // the shadow: a blurred box below the frame
    float d_shadow = box(v_local - u_shadow_offset, outer, corner);
    float shadow_a = u_shadow * 0.75 * (1.0 - smoothstep(-u_shadow_blur, u_shadow_blur, d_shadow));
    float a = frame_a + shadow_a * (1.0 - frame_a);
    vec3 premul = c * frame_a;  // (the shadow is black)
    f_color = vec4(premul, a) * u_alpha;
}
