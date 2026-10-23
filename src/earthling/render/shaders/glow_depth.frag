// The scene depth into the centre of the (larger) glow buffer; the margin around it is far
// away: glowing objects just outside the picture are not hidden, their glow reaches into it.
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_depth;
uniform ivec2 u_offset;  // the margin in pixels

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy) - u_offset;
    bool inside = all(greaterThanEqual(p, ivec2(0))) && all(lessThan(p, textureSize(u_depth, 0)));
    gl_FragDepth = inside ? texelFetch(u_depth, p, 0).r : 1.0;
    f_color = vec4(0.0);
}
