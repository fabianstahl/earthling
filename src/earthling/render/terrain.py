"""Terrain rendering with quadtree LOD.

Every LOD node (an XYZ tile) gets a (GRID+1)^2 vertex grid plus a skirt around its border.
Vertex positions on the ellipsoid (height 0) and local up vectors are computed on the CPU in
float64 via ECEF, relative to the node's own origin, so Earth curvature is exact and float32
precision stays high. The vertex shader displaces vertices along ``up`` by the height sampled
from the node's heightmap (its own or a sub-rectangle of an ancestor's). The fragment shader
derives normals from the heightmap and discards fragments without DEM data.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass

import moderngl
import numpy as np

from earthling.core.geo import (
    LocalFrame,
    geodetic_to_ecef,
    mercator_to_lonlat,
    tile_bounds_mercator,
)
from earthling.data.dem import GRID_INTERVALS, HEIGHTMAP_SAMPLES
from earthling.data.terrain_data import TerrainData
from earthling.render import lod
from earthling.render.camera import Camera
from earthling.render.shader_library import ShaderLibrary

log = logging.getLogger(__name__)

MESH_GRID = 64  # intervals per tile edge
DEFAULT_MIN_H, DEFAULT_MAX_H = -100.0, 4900.0

TileKey = tuple[int, int, int]


# --- geometry ----------------------------------------------------------------------------
def perimeter_indices(n: int = MESH_GRID) -> np.ndarray:
    """Grid vertex indices along the border as a closed loop."""
    top = list(range(n + 1))
    right = [j * (n + 1) + n for j in range(1, n + 1)]
    bottom = [n * (n + 1) + i for i in range(n - 1, -1, -1)]
    left = [j * (n + 1) for j in range(n - 1, 0, -1)]
    return np.array(top + right + bottom + left, dtype=np.int64)


def grid_indices(n: int = MESH_GRID, skirt: bool = True) -> np.ndarray:
    """Triangles for an (n+1)^2 grid (rows north->south) followed by skirt quads."""
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = (j * (n + 1) + i).ravel()
    b = a + 1
    c = a + (n + 1)
    d = c + 1
    tris = [np.column_stack([a, c, b, b, c, d]).ravel()]
    if skirt:
        top = perimeter_indices(n)
        bottom = (n + 1) ** 2 + np.arange(len(top))
        top_next = np.roll(top, -1)
        bottom_next = np.roll(bottom, -1)
        tris.append(np.column_stack([top, top_next, bottom, bottom, top_next, bottom_next]).ravel())
    return np.concatenate(tris).astype(np.uint32)


def shared_grid_attributes(n: int = MESH_GRID) -> np.ndarray:
    """(uv.x, uv.y, skirt) for grid + skirt vertices, shared by all tiles."""
    t = np.linspace(0.0, 1.0, n + 1)
    u, v = np.meshgrid(t, t)
    grid = np.column_stack([u.ravel(), v.ravel(), np.zeros(u.size)])
    skirt = grid[perimeter_indices(n)].copy()
    skirt[:, 2] = 1.0
    return np.vstack([grid, skirt]).astype("f4")


@dataclass
class TileGeometry:
    origin: np.ndarray  # ENU of the tile centre at height 0 (float64)
    positions: np.ndarray  # vertices (grid + skirt) relative to origin, height 0
    ups: np.ndarray  # unit vectors in ENU
    tangent: np.ndarray  # 3x3: rows east, north, up at the tile centre (ENU coords)
    width_m: float


def tile_geometry(frame: LocalFrame, z: int, x: int, y: int, n: int = MESH_GRID) -> TileGeometry:
    min_x, min_y, max_x, max_y = tile_bounds_mercator(z, x, y)
    t = np.linspace(0.0, 1.0, n + 1)
    mx = min_x + t * (max_x - min_x)
    my = max_y - t * (max_y - min_y)  # rows north -> south, like the heightmap
    gx, gy = np.meshgrid(mx, my)
    lon, lat = mercator_to_lonlat(gx.ravel(), gy.ravel())
    ground = frame.ecef_to_enu(geodetic_to_ecef(lat, lon, 0.0))
    above = frame.ecef_to_enu(geodetic_to_ecef(lat, lon, 1.0))
    ups = above - ground
    centre = n // 2 * (n + 1) + n // 2
    origin = ground[centre].copy()
    up_c = ups[centre]
    east_c = ground[centre + 1] - ground[centre - 1]
    east_c -= up_c * np.dot(east_c, up_c)
    east_c /= np.linalg.norm(east_c)
    north_c = np.cross(up_c, east_c)
    per = perimeter_indices(n)
    positions = np.vstack([ground, ground[per]]) - origin
    ups = np.vstack([ups, ups[per]])
    width = (max_x - min_x) * np.cos(np.radians(float(lat[centre])))
    return TileGeometry(origin, positions, ups, np.array([east_c, north_c, up_c]), float(width))


# --- GPU resources -----------------------------------------------------------------------
class _HeightmapTexture:
    def __init__(self, ctx: moderngl.Context, heights: np.ndarray) -> None:
        valid = np.isfinite(heights)
        fill = float(np.nanmin(heights)) if valid.any() else 0.0
        data = np.dstack([np.where(valid, heights, fill), valid.astype(np.float32)])
        self.texture = ctx.texture(
            (HEIGHTMAP_SAMPLES, HEIGHTMAP_SAMPLES), 2, data.astype("f4").tobytes(), dtype="f4"
        )
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.texture.repeat_x = self.texture.repeat_y = False
        self.refs = 0

    def release(self) -> None:
        self.texture.release()


@dataclass
class _Node:
    key: TileKey
    geometry: TileGeometry
    vbo: moderngl.Buffer
    heightmap_key: TileKey | None
    hm_scale: float
    hm_offset: tuple[float, float]
    sample_spacing_m: float
    min_h: float
    max_h: float
    imagery: moderngl.Texture | None
    vao: moderngl.VertexArray | None = None
    last_used: int = 0


class TerrainLayer:
    def __init__(self, ctx: moderngl.Context, shaders: ShaderLibrary) -> None:
        self.ctx = ctx
        self.shaders = shaders
        self.visible = True
        self.exaggeration = 1.0
        self.debug_lod = False
        self.params = lod.LodParams()
        self.max_resident_nodes = 700
        self.load_budget_s = 0.012  # synchronous loading time per frame
        self.frame: LocalFrame | None = None
        self.data: TerrainData | None = None
        self.nodes: lod.NodeSet | None = None
        self._resident: OrderedDict[TileKey, _Node] = OrderedDict()
        self._heightmaps: dict[TileKey, _HeightmapTexture] = {}
        self._bounds: dict[TileKey, lod.NodeBounds] = {}
        self._ibo = ctx.buffer(grid_indices().tobytes())
        self._shared = ctx.buffer(shared_grid_attributes().tobytes())
        self._program: moderngl.Program | None = None
        self._frame_no = 0
        self.last_selection: lod.Selection | None = None

    # --- setup ------------------------------------------------------------------------
    def set_source(
        self, frame: LocalFrame, data: TerrainData | None, nodes: lod.NodeSet | None
    ) -> None:
        self.clear()
        self.frame = frame
        self.data = data
        self.nodes = nodes

    def clear(self) -> None:
        for node in list(self._resident.values()):
            self._release_node(node)
        self._resident.clear()
        for hm in self._heightmaps.values():
            hm.release()
        self._heightmaps.clear()
        self._bounds.clear()
        self.last_selection = None

    @property
    def resident_count(self) -> int:
        return len(self._resident)

    # --- bounds -----------------------------------------------------------------------
    def node_bounds(self, key: TileKey) -> lod.NodeBounds:
        b = self._bounds.get(key)
        if b is None:
            node = self._resident.get(key)
            if node is not None:
                lo, hi = node.min_h * self.exaggeration, node.max_h * self.exaggeration
            else:
                lo, hi = DEFAULT_MIN_H, DEFAULT_MAX_H * max(1.0, self.exaggeration)
            assert self.frame is not None
            b = lod.node_bounds(self.frame, key, lo, hi)
            self._bounds[key] = b
        return b

    def invalidate_bounds(self) -> None:
        """Call after changing the exaggeration."""
        self._bounds.clear()

    def bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Bounding box of the loaded root nodes (for framing the view)."""
        if self.nodes is None or self.frame is None:
            return None
        roots = [self._resident[k] for k in self.nodes.roots() if k in self._resident]
        roots = [n for n in roots if n.heightmap_key is not None]
        if not roots:
            return None
        pts = np.vstack([n.geometry.origin + n.geometry.positions for n in roots])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        lo[2] = min(n.min_h for n in roots)
        hi[2] = max(n.max_h for n in roots)
        return lo, hi

    # --- loading ----------------------------------------------------------------------
    def _load_node(self, key: TileKey) -> _Node:
        assert self.frame is not None and self.data is not None
        ref = self.data.heightmap_for(key)
        geo = tile_geometry(self.frame, *key)
        verts = np.hstack([geo.positions, geo.ups]).astype("f4")
        vbo = self.ctx.buffer(verts.tobytes())
        hm_key = None
        min_h, max_h = 0.0, 0.0
        scale, offset = float(GRID_INTERVALS), (0.0, 0.0)
        spacing = geo.width_m / GRID_INTERVALS
        if ref is not None:
            hm_key = ref.source
            hm = self._heightmaps.get(hm_key)
            if hm is None:
                hm = _HeightmapTexture(self.ctx, ref.heights)
                self._heightmaps[hm_key] = hm
            hm.refs += 1
            scale, offset = ref.scale, ref.offset
            spacing = geo.width_m / scale
            ox, oy = int(offset[0]), int(offset[1])
            size = int(np.ceil(scale)) + 3
            window = ref.heights[oy : oy + size, ox : ox + size]
            if np.isfinite(window).any():
                min_h, max_h = float(np.nanmin(window)), float(np.nanmax(window))
        rgb = self.data.imagery_for(key)
        tex = None
        if rgb is not None:
            h, w, _ = rgb.shape
            tex = self.ctx.texture((w, h), 3, np.ascontiguousarray(rgb).tobytes())
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex.anisotropy = 16.0
            tex.repeat_x = tex.repeat_y = False
        node = _Node(key, geo, vbo, hm_key, scale, offset, spacing, min_h, max_h, tex)
        node.last_used = self._frame_no
        self._resident[key] = node
        self._bounds.pop(key, None)  # recompute with real heights
        return node

    def _release_node(self, node: _Node) -> None:
        if node.vao is not None:
            node.vao.release()
        node.vbo.release()
        if node.imagery is not None:
            node.imagery.release()
        if node.heightmap_key is not None:
            hm = self._heightmaps.get(node.heightmap_key)
            if hm is not None:
                hm.refs -= 1
                if hm.refs <= 0:
                    hm.release()
                    del self._heightmaps[node.heightmap_key]

    def _evict(self, protected: set[TileKey]) -> None:
        excess = len(self._resident) - self.max_resident_nodes
        if excess <= 0 or self.nodes is None:
            return
        min_zoom = self.nodes.min_zoom
        candidates = sorted(
            (n for k, n in self._resident.items() if k not in protected and k[0] > min_zoom),
            key=lambda n: n.last_used,
        )
        for node in candidates[:excess]:
            self._release_node(node)
            del self._resident[node.key]

    # --- per frame --------------------------------------------------------------------
    def update(
        self, camera: Camera, view_proj, viewport_height: int, budget_s: float | None = None
    ) -> lod.Selection | None:
        if self.nodes is None or self.data is None or self.frame is None:
            return None
        self._frame_no += 1
        planes = lod.frustum_planes(view_proj)
        ppr = viewport_height / (2.0 * np.tan(np.radians(camera.fov_y) / 2.0))
        sel = lod.select_nodes(
            self.nodes,
            self.node_bounds,
            lambda k: k in self._resident,
            camera.position,
            planes,
            ppr,
            self.params,
        )
        deadline = time.perf_counter() + (self.load_budget_s if budget_s is None else budget_s)
        for key in sel.request:
            if time.perf_counter() > deadline:
                break
            if key not in self._resident:
                self._load_node(key)
        for key in sel.draw:
            node = self._resident.get(key)
            if node is not None:
                node.last_used = self._frame_no
        self._evict(set(sel.draw))
        self.last_selection = sel
        return sel

    def fully_loaded(self) -> bool:
        return self.last_selection is not None and not self.last_selection.request

    def render(self, camera: Camera, view_proj, viewport_height: int) -> None:
        if not self.visible:
            return
        sel = self.update(camera, view_proj, viewport_height)
        if sel is None:
            return
        program = self.shaders.get("terrain")
        if program is not self._program:
            for node in self._resident.values():
                if node.vao is not None:
                    node.vao.release()
                node.vao = None
            self._program = program
        program["u_view_proj"].write(view_proj)
        program["u_log_depth_coef"] = camera.log_depth_coef
        program["u_exaggeration"] = self.exaggeration
        program["u_heightmap"] = 0
        program["u_imagery"] = 1
        program["u_debug_lod"] = self.debug_lod
        for key in sel.draw:
            node = self._resident.get(key)
            if node is None or node.heightmap_key is None:
                continue
            if node.vao is None:
                node.vao = self.ctx.vertex_array(
                    program,
                    [
                        (node.vbo, "3f 3f", "in_pos", "in_up"),
                        (self._shared, "2f 1f", "in_uv", "in_skirt"),
                    ],
                    self._ibo,
                    index_element_size=4,
                )
            geo = node.geometry
            program["u_offset"] = camera.relative(geo.origin)
            # GLSL mat3 is column-major: columns = east, north, up
            program["u_tangent"].write(geo.tangent.astype("f4").tobytes())
            program["u_sample_spacing"] = node.sample_spacing_m
            program["u_hm_scale"] = node.hm_scale
            program["u_hm_offset"] = node.hm_offset
            program["u_skirt_depth"] = max(20.0, geo.width_m / MESH_GRID * 2.0)
            program["u_zoom"] = key[0]
            self._heightmaps[node.heightmap_key].texture.use(0)
            if node.imagery is not None:
                node.imagery.use(1)
            program["u_has_imagery"] = node.imagery is not None
            node.vao.render(moderngl.TRIANGLES)
