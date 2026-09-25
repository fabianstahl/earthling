// Terrain node helpers shared by terrain shaders.
#define HEIGHTMAP_SAMPLES 259.0

uniform sampler2D u_heightmap;  // r = height (m), g = valid (1/0)
uniform float u_exaggeration;
uniform float u_sample_spacing;  // metres between heightmap samples
uniform mat3 u_tangent;          // columns: east, north, up (ENU) at node centre
uniform float u_hm_scale;        // heightmap samples across this node
uniform vec2 u_hm_offset;        // heightmap sample index of the node's NW corner

// Heightmap texture coordinate for a node-local uv (0..1, v north -> south).
// Sample index s is stored at texel s + 1 (one border sample on each side).
vec2 heightmap_uv(vec2 tile_uv) {
    vec2 s = u_hm_offset + tile_uv * u_hm_scale;
    return (s + 1.5) / HEIGHTMAP_SAMPLES;
}

float height_at(vec2 hm_uv) {
    return texture(u_heightmap, hm_uv).r * u_exaggeration;
}

float valid_at(vec2 hm_uv) {
    return texture(u_heightmap, hm_uv).g;
}

// World-space (ENU) normal from central differences of the heightmap.
vec3 terrain_normal(vec2 hm_uv) {
    float t = 1.0 / HEIGHTMAP_SAMPLES;
    float he = height_at(hm_uv + vec2(t, 0.0));
    float hw = height_at(hm_uv - vec2(t, 0.0));
    float hn = height_at(hm_uv - vec2(0.0, t));  // rows run north -> south
    float hs = height_at(hm_uv + vec2(0.0, t));
    vec3 n = normalize(vec3((hw - he) / (2.0 * u_sample_spacing),
                            (hs - hn) / (2.0 * u_sample_spacing), 1.0));
    return normalize(u_tangent * n);
}
