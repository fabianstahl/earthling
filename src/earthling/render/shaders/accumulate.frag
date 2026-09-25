// Adds a rendered sub-frame into the accumulation buffer (additive blending), or resolves
// the accumulation into the output target (u_weight = 1, no blending).
out vec4 f_color;
uniform sampler2D u_source;
uniform float u_weight = 1.0;

void main() {
    f_color = texelFetch(u_source, ivec2(gl_FragCoord.xy), 0) * u_weight;
}
