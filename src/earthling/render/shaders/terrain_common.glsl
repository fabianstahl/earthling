// Terrain tile helpers shared by terrain shaders.
#define MESH_GRID 64
#define HEIGHTMAP_SAMPLES 259.0
#define HEIGHTMAP_STRIDE 4.0

uniform sampler2D u_heightmap;
uniform float u_exaggeration;
uniform float u_sample_spacing;  // metres between heightmap samples
uniform mat3 u_tangent;          // columns: east, north, up (ENU) at tile centre

// Heightmap texture coordinate of mesh vertex (i, j); sample index 0 is texel 1.
vec2 vertex_uv(int vertex_id) {
    int i = vertex_id % (MESH_GRID + 1);
    int j = vertex_id / (MESH_GRID + 1);
    return (vec2(i, j) * HEIGHTMAP_STRIDE + 1.5) / HEIGHTMAP_SAMPLES;
}

float height_at(vec2 uv) {
    return texture(u_heightmap, uv).r * u_exaggeration;
}

// World-space (ENU) normal from central differences of the heightmap.
vec3 terrain_normal(vec2 uv) {
    float t = 1.0 / HEIGHTMAP_SAMPLES;
    float he = height_at(uv + vec2(t, 0.0));
    float hw = height_at(uv - vec2(t, 0.0));
    float hn = height_at(uv - vec2(0.0, t));  // rows run north -> south
    float hs = height_at(uv + vec2(0.0, t));
    vec3 n = normalize(vec3((hw - he) / (2.0 * u_sample_spacing),
                            (hs - hn) / (2.0 * u_sample_spacing), 1.0));
    return normalize(u_tangent * n);
}
