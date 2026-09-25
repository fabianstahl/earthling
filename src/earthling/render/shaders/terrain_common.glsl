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

// --- geomorphing: blend from the parent node's surface to this node's heights ---------------
uniform sampler2D u_parent_heightmap;
uniform float u_morph = 1.0;             // 0 = parent surface, 1 = own heights
uniform vec2 u_parent_uv_offset;         // this node's NW corner in parent uv (0 or 0.5)
uniform float u_parent_hm_scale;
uniform vec2 u_parent_hm_offset;
uniform float u_parent_sample_spacing;

vec2 parent_heightmap_uv(vec2 tile_uv) {
    vec2 s = u_parent_hm_offset + (u_parent_uv_offset + tile_uv * 0.5) * u_parent_hm_scale;
    return (s + 1.5) / HEIGHTMAP_SAMPLES;
}

float parent_height_at(vec2 parent_hm_uv) {
    return texture(u_parent_heightmap, parent_hm_uv).r * u_exaggeration;
}

float morphed_height(vec2 hm_uv, vec2 tile_uv) {
    float h = height_at(hm_uv);
    if (u_morph >= 1.0) return h;
    return mix(parent_height_at(parent_heightmap_uv(tile_uv)), h, u_morph);
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

// The same for the parent heightmap (geomorphing the shading along with the heights).
vec3 parent_terrain_normal(vec2 parent_hm_uv) {
    float t = 1.0 / HEIGHTMAP_SAMPLES;
    float he = parent_height_at(parent_hm_uv + vec2(t, 0.0));
    float hw = parent_height_at(parent_hm_uv - vec2(t, 0.0));
    float hn = parent_height_at(parent_hm_uv - vec2(0.0, t));
    float hs = parent_height_at(parent_hm_uv + vec2(0.0, t));
    vec3 n = normalize(vec3((hw - he) / (2.0 * u_parent_sample_spacing),
                            (hs - hn) / (2.0 * u_parent_sample_spacing), 1.0));
    return normalize(u_tangent * n);
}

vec3 morphed_normal(vec2 hm_uv, vec2 tile_uv) {
    vec3 n = terrain_normal(hm_uv);
    if (u_morph >= 1.0) return n;
    return normalize(mix(parent_terrain_normal(parent_heightmap_uv(tile_uv)), n, u_morph));
}
