// Logarithmic depth (fragment shaders only). Vertex shaders pass `1.0 + gl_Position.w`
// as v_log_z; fragment shaders call write_log_depth(v_log_z). Gives usable depth precision
// from 0.5 m to thousands of kilometres.
uniform float u_log_depth_coef;  // 1 / log2(far + 1)

void write_log_depth(float log_z) {
    gl_FragDepth = log2(max(log_z, 1e-6)) * u_log_depth_coef;
}
