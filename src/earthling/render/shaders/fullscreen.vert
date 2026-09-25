// Fullscreen triangle without vertex buffers: draw 3 vertices.
out vec2 v_uv;
out vec2 v_ndc;

void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    v_ndc = p * 2.0 - 1.0;
    gl_Position = vec4(v_ndc, 0.0, 1.0);
}
