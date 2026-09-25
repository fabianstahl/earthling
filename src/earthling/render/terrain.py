"""Terrain rendering with quadtree LOD.

Every LOD node (an XYZ tile) gets a (GRID+1)^2 vertex grid plus a skirt around its border.
Vertex positions on the ellipsoid (height 0) and local up vectors are computed on the CPU in
float64 via ECEF, relative to the node's own origin, so Earth curvature is exact and float32
precision stays high. The vertex shader displaces vertices along ``up`` by the height sampled
from the node's heightmap (its own or a sub-rectangle of an ancestor's). The fragment shader
derives normals from the heightmap and discards fragments without DEM data.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import moderngl
import numpy as np

from earthling.core.geo import (
    LocalFrame,
    geodetic_to_ecef,
    mercator_to_lonlat,
    tile_bounds_mercator,
)
from earthling.core.properties import bind_uniforms
from earthling.data.dem import GRID_INTERVALS, HEIGHTMAP_SAMPLES
from earthling.data.terrain_data import TerrainData
from earthling.render import lod
from earthling.render.camera import Camera
from earthling.render.shader_library import ShaderLibrary

log = logging.getLogger(__name__)

MESH_GRID = 64  # intervals per tile edge
# texture unit per tile source (0 = heightmap, 2 = optical depth LUT, 3 = shadow atlas)
TEXTURE_UNITS = {"imagery": 1, "topo": 4, "borders": 5}
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


def _set(program: moderngl.Program, name: str, value) -> None:
    """Set a uniform if the compiled program uses it (variants optimise some away)."""
    if name in program:
        program[name] = value


# --- CPU preparation (runs on worker threads) ---------------------------------------------
@dataclass
class PreparedNode:
    key: TileKey
    geometry: TileGeometry
    vertices: np.ndarray  # float32 (pos, up)
    heightmap_key: TileKey | None
    heights: np.ndarray | None  # the source heightmap (shared by descendants)
    hm_scale: float
    hm_offset: tuple[float, float]
    sample_spacing_m: float
    min_h: float
    max_h: float
    textures: dict[str, np.ndarray | None]  # per tile source ("imagery", "topo", ...)


def prepare_node(
    frame: LocalFrame,
    data: TerrainData,
    key: TileKey,
    sources: frozenset[str] = frozenset({"imagery"}),
) -> PreparedNode:
    ref = data.heightmap_for(key)
    geo = tile_geometry(frame, *key)
    vertices = np.hstack([geo.positions, geo.ups]).astype("f4")
    scale, offset = float(GRID_INTERVALS), (0.0, 0.0)
    spacing = geo.width_m / GRID_INTERVALS
    min_h = max_h = 0.0
    hm_key = heights = None
    if ref is not None:
        hm_key, heights = ref.source, ref.heights
        scale, offset = ref.scale, ref.offset
        spacing = geo.width_m / scale
        ox, oy = int(offset[0]), int(offset[1])
        size = int(np.ceil(scale)) + 3
        window = ref.heights[oy : oy + size, ox : ox + size]
        if np.isfinite(window).any():
            min_h, max_h = float(np.nanmin(window)), float(np.nanmax(window))
    textures: dict[str, np.ndarray | None] = {}
    ready = getattr(data, "source_ready", lambda source: True)
    for source in sorted(sources):
        if not ready(source):
            continue  # not initialised yet: the node is refreshed later
        rgb = data.texture_for(key, source)
        textures[source] = None if rgb is None else np.ascontiguousarray(rgb)
    return PreparedNode(
        key, geo, vertices, hm_key, heights, scale, offset, spacing, min_h, max_h, textures
    )


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


class TexturePool:
    """Recycles imagery textures of equal size instead of reallocating them."""

    def __init__(self, ctx: moderngl.Context, max_free_per_size: int = 32) -> None:
        self.ctx = ctx
        self.max_free = max_free_per_size
        self._free: dict[tuple[int, int], list[moderngl.Texture]] = {}

    def acquire(self, data: np.ndarray) -> moderngl.Texture:
        """uint8 RGB images or float fields (stored as half floats)."""
        h, w, components = data.shape
        is_float = data.dtype.kind == "f"
        dtype = "f2" if is_float else "f1"
        payload = data.astype(np.float16).tobytes() if is_float else data.tobytes()
        free = self._free.get((w, h, components, dtype))
        if free:
            tex = free.pop()
            tex.write(payload)
        else:
            tex = self.ctx.texture((w, h), components, payload, dtype=dtype)
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex.anisotropy = 16.0
            tex.repeat_x = tex.repeat_y = False
        tex.build_mipmaps()
        return tex

    def release(self, tex: moderngl.Texture) -> None:
        free = self._free.setdefault((*tex.size, tex.components, tex.dtype), [])
        if len(free) < self.max_free:
            free.append(tex)
        else:
            tex.release()

    def clear(self) -> None:
        for textures in self._free.values():
            for tex in textures:
                tex.release()
        self._free.clear()


def node_gpu_bytes(rgb_size: tuple[int, int] | None) -> int:
    """Rough GPU memory of one node (vertex buffer + imagery with mipmaps, RGBA-padded)."""
    vbo = ((MESH_GRID + 1) ** 2 + 4 * MESH_GRID) * 24
    if rgb_size is None:
        return vbo
    w, h = rgb_size
    return vbo + int(w * h * 4 * 4 / 3)


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
    textures: dict[str, moderngl.Texture | None]  # per tile source; None = unavailable
    vaos: dict[int, moderngl.VertexArray] = field(default_factory=dict)  # per program
    last_used: int = 0

    @property
    def gpu_bytes(self) -> int:
        total = node_gpu_bytes(None)
        for tex in self.textures.values():
            if tex is not None:
                total += node_gpu_bytes(tex.size) - node_gpu_bytes(None)
        return total

    @property
    def imagery(self) -> moderngl.Texture | None:
        return self.textures.get("imagery")


class TerrainLayer:
    def __init__(
        self, ctx: moderngl.Context, shaders: ShaderLibrary, workers: int | None = None
    ) -> None:
        self.ctx = ctx
        self.shaders = shaders
        self.visible = True
        self.exaggeration = 1.0
        self.debug_lod = False
        self.params = lod.LodParams()
        self.memory_budget_mb = 3000
        self.store = None  # PropertyStore for automatic uniform binding
        self.required_sources: frozenset[str] = frozenset({"imagery"})
        self.lighting_uniforms: dict[str, object] = {}
        self.upload_budget_s = 0.006  # GPU upload time per frame
        self.frame: LocalFrame | None = None
        self.data: TerrainData | None = None
        self.nodes: lod.NodeSet | None = None
        self._resident: OrderedDict[TileKey, _Node] = OrderedDict()
        self._heightmaps: dict[TileKey, _HeightmapTexture] = {}
        self._bounds: dict[TileKey, lod.NodeBounds] = {}
        self._ibo = ctx.buffer(grid_indices().tobytes())
        self._shared = ctx.buffer(shared_grid_attributes().tobytes())
        self._pool = TexturePool(ctx)
        self._workers = workers or max(2, min(8, (os.cpu_count() or 4) - 1))
        self._executor = ThreadPoolExecutor(self._workers, thread_name_prefix="terrain")
        self._in_flight: dict[TileKey, Future] = {}
        self._failed: set[TileKey] = set()
        self._programs: dict[str, moderngl.Program] = {}
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
        if self.data is not None and hasattr(self.data, "close"):
            self.data.close()
        for future in self._in_flight.values():
            future.cancel()
        self._in_flight.clear()
        for node in list(self._resident.values()):
            self._release_node(node)
        self._resident.clear()
        for hm in self._heightmaps.values():
            hm.release()
        self._heightmaps.clear()
        self._bounds.clear()
        self._failed.clear()
        self.last_selection = None

    def shutdown(self) -> None:
        self.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._pool.clear()

    @property
    def resident_count(self) -> int:
        return len(self._resident)

    @property
    def pending_count(self) -> int:
        return len(self._in_flight)

    @property
    def gpu_bytes(self) -> int:
        return sum(n.gpu_bytes for n in self._resident.values())

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
    def _submit(self, key: TileKey, refresh: bool = False) -> None:
        assert self.frame is not None and self.data is not None
        if key in self._in_flight or key in self._failed or (key in self._resident and not refresh):
            return
        sources = self.required_sources
        node = self._resident.get(key)
        if refresh and node is not None:
            sources = frozenset(sources - node.textures.keys())  # only what is missing
        self._in_flight[key] = self._executor.submit(
            prepare_node, self.frame, self.data, key, sources
        )

    def _needs_refresh(self, node: _Node) -> bool:
        missing = self.required_sources - node.textures.keys()
        if not missing:
            return False
        ready = getattr(self.data, "source_ready", lambda source: True)
        return any(ready(source) for source in missing)

    def _upload(self, prepared: PreparedNode) -> _Node:
        vbo = self.ctx.buffer(prepared.vertices.tobytes())
        if prepared.heightmap_key is not None:
            hm = self._heightmaps.get(prepared.heightmap_key)
            if hm is None:
                hm = _HeightmapTexture(self.ctx, prepared.heights)
                self._heightmaps[prepared.heightmap_key] = hm
            hm.refs += 1
        textures = {
            name: (self._pool.acquire(rgb) if rgb is not None else None)
            for name, rgb in prepared.textures.items()
        }
        node = _Node(
            prepared.key,
            prepared.geometry,
            vbo,
            prepared.heightmap_key,
            prepared.hm_scale,
            prepared.hm_offset,
            prepared.sample_spacing_m,
            prepared.min_h,
            prepared.max_h,
            textures,
            last_used=self._frame_no,
        )
        old = self._resident.pop(prepared.key, None)
        if old is not None:
            # a refresh only prepares missing sources: keep the textures the old node had
            for name, tex in list(old.textures.items()):
                if name not in node.textures:
                    node.textures[name] = tex
                    del old.textures[name]
            self._release_node(old)
        self._resident[prepared.key] = node
        self._bounds.pop(prepared.key, None)  # recompute with real heights
        return node

    def _collect(self, budget_s: float | None) -> None:
        """Upload finished preparations (all of them if ``budget_s`` is None)."""
        deadline = None if budget_s is None else time.perf_counter() + budget_s
        for key, future in list(self._in_flight.items()):
            if deadline is not None and time.perf_counter() > deadline:
                break
            if not future.done():
                continue
            del self._in_flight[key]
            if future.cancelled():
                continue
            try:
                prepared = future.result()
            except Exception:
                log.exception("preparing terrain node %s failed", key)
                self._failed.add(key)  # do not retry forever (and never block export)
                continue
            self._upload(prepared)

    def _release_node(self, node: _Node) -> None:
        for vao in node.vaos.values():
            vao.release()
        node.vaos.clear()
        node.vbo.release()
        for tex in node.textures.values():
            if tex is not None:
                self._pool.release(tex)
        node.textures = {}
        if node.heightmap_key is not None:
            hm = self._heightmaps.get(node.heightmap_key)
            if hm is not None:
                hm.refs -= 1
                if hm.refs <= 0:
                    hm.release()
                    del self._heightmaps[node.heightmap_key]

    def _evict(self, protected: set[TileKey]) -> None:
        if self.nodes is None:
            return
        budget = self.memory_budget_mb * 1024 * 1024
        used = self.gpu_bytes
        if used <= budget:
            return
        min_zoom = self.nodes.min_zoom
        candidates = sorted(
            (n for k, n in self._resident.items() if k not in protected and k[0] > min_zoom),
            key=lambda n: n.last_used,
        )
        for node in candidates:
            if used <= budget:
                break
            used -= node.gpu_bytes
            self._release_node(node)
            del self._resident[node.key]

    # --- per frame --------------------------------------------------------------------
    def _select(self, camera: Camera, view_proj, viewport_height: int) -> lod.Selection:
        assert self.nodes is not None
        planes = lod.frustum_planes(view_proj)
        ppr = viewport_height / (2.0 * np.tan(np.radians(camera.fov_y) / 2.0))
        return lod.select_nodes(
            self.nodes,
            self.node_bounds,
            lambda k: k in self._resident or k in self._failed,
            camera.position,
            planes,
            ppr,
            self.params,
        )

    def update(self, camera: Camera, view_proj, viewport_height: int) -> lod.Selection | None:
        """Select nodes, upload finished ones within the budget and request missing ones."""
        if self.nodes is None or self.data is None or self.frame is None:
            return None
        self._frame_no += 1
        self._collect(self.upload_budget_s)
        sel = self._select(camera, view_proj, viewport_height)
        max_in_flight = self._workers * 3
        for key in sel.request:
            if len(self._in_flight) >= max_in_flight:
                break
            self._submit(key)
        for key in sel.draw:
            node = self._resident.get(key)
            if node is not None:
                node.last_used = self._frame_no
                if self._needs_refresh(node) and len(self._in_flight) < max_in_flight:
                    self._submit(key, refresh=True)  # e.g. a layer needs another texture
        self._evict(set(sel.draw))
        self.last_selection = sel
        return sel

    def fully_loaded(self) -> bool:
        sel = self.last_selection
        if sel is None or sel.request:
            return False
        return not any(
            self._needs_refresh(self._resident[k]) for k in sel.draw if k in self._resident
        )

    def finish_loading(
        self, camera: Camera, view_proj, viewport_height: int, timeout_s: float = 120.0
    ) -> bool:
        """Block until the view is loaded at full LOD (used by export and tests)."""
        if self.nodes is None or self.data is None:
            return True
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            sel = self._select(camera, view_proj, viewport_height)
            self.last_selection = sel
            stale = [
                k
                for k in sel.draw
                if k in self._resident and self._needs_refresh(self._resident[k])
            ]
            if not sel.request and not stale:
                return True
            for key in sel.request:
                self._submit(key)
            for key in stale:
                self._submit(key, refresh=True)
            for future in list(self._in_flight.values()):
                with contextlib.suppress(Exception):
                    future.result(timeout=max(0.0, deadline - time.perf_counter()))
            self._collect(None)
        return False

    def render(self, camera: Camera, view_proj, viewport_height: int) -> None:
        if not self.visible:
            return
        sel = self.update(camera, view_proj, viewport_height)
        if sel is None:
            return
        self.draw(camera, view_proj, sel.draw)

    def _vao(self, node: _Node, program: moderngl.Program) -> moderngl.VertexArray:
        vao = node.vaos.get(program.glo)
        if vao is None:
            vao = self.ctx.vertex_array(
                program,
                [
                    (node.vbo, "3f 3f", "in_pos", "in_up"),
                    (self._shared, "2f 1f", "in_uv", "in_skirt"),
                ],
                self._ibo,
                index_element_size=4,
            )
            node.vaos[program.glo] = vao
        return vao

    def _check_program(self, program: moderngl.Program, key: str) -> None:
        """Drop cached VAOs of a program that was hot-reloaded."""
        old = self._programs.get(key)
        if old is not None and old is not program:
            for node in self._resident.values():
                vao = node.vaos.pop(old.glo, None)
                if vao is not None:
                    vao.release()
        self._programs[key] = program

    def draw(
        self,
        camera: Camera,
        view_proj,
        keys: list[TileKey],
        shadow_pass: bool = False,
        extra_uniforms: dict[str, object] | None = None,
    ) -> None:
        if shadow_pass:
            program = self.shaders.get("terrain", defines={"SHADOW_PASS": 1})
        else:
            program = self.shaders.get("terrain")
        self._check_program(program, "shadow" if shadow_pass else "main")
        program["u_view_proj"].write(view_proj)
        _set(program, "u_exaggeration", self.exaggeration)
        _set(program, "u_heightmap", 0)
        if not shadow_pass:
            _set(program, "u_log_depth_coef", camera.log_depth_coef)
            for name, unit in TEXTURE_UNITS.items():
                _set(program, f"u_{name}", unit)
            _set(program, "u_debug_lod", self.debug_lod)
            if self.store is not None:
                bind_uniforms(program, self.store)
            for name, value in {**self.lighting_uniforms, **(extra_uniforms or {})}.items():
                if name == "_matrices":
                    # to_list() yields columns -> column-major (bytes(mat) is row-major!)
                    data = np.array([m.to_list() for m in value], dtype="f4").tobytes()  # type: ignore[union-attr]
                    program["u_shadow_matrix"].write(data)
                else:
                    _set(program, name, value)
        for key in keys:
            node = self._resident.get(key)
            if node is None or node.heightmap_key is None:
                continue
            vao = self._vao(node, program)
            geo = node.geometry
            _set(program, "u_offset", camera.relative(geo.origin))
            _set(program, "u_hm_scale", node.hm_scale)
            _set(program, "u_hm_offset", node.hm_offset)
            _set(program, "u_skirt_depth", max(20.0, geo.width_m / MESH_GRID * 2.0))
            self._heightmaps[node.heightmap_key].texture.use(0)
            if not shadow_pass:
                _set(program, "u_sample_spacing", node.sample_spacing_m)
                # GLSL mat3 is column-major: columns = east, north, up
                if "u_tangent" in program:
                    program["u_tangent"].write(geo.tangent.astype("f4").tobytes())
                _set(program, "u_zoom", key[0])
                for name, unit in TEXTURE_UNITS.items():
                    tex = node.textures.get(name)
                    if tex is not None:
                        tex.use(unit)
                    _set(program, f"u_has_{name}", tex is not None)
            vao.render(moderngl.TRIANGLES)

    # --- shadows ------------------------------------------------------------------------
    def select_shadow_casters(
        self, camera: Camera, light_view_proj, viewport_height: int
    ) -> list[TileKey]:
        """Coarser node selection covering the light frustum (occluders outside the view)."""
        if self.nodes is None or self.data is None or self.frame is None:
            return []
        planes = lod.frustum_planes(light_view_proj)
        ppr = viewport_height / (2.0 * np.tan(np.radians(camera.fov_y) / 2.0))
        params = lod.LodParams(self.params.pixel_threshold * 4.0, self.params.max_nodes)
        sel = lod.select_nodes(
            self.nodes,
            self.node_bounds,
            lambda k: k in self._resident,
            camera.position,
            planes,
            ppr,
            params,
        )
        for key in sel.request:
            if len(self._in_flight) >= self._workers * 3:
                break
            self._submit(key)
        for key in sel.draw:
            node = self._resident.get(key)
            if node is not None:
                node.last_used = self._frame_no
        return sel.draw

    def draw_shadow_casters(self, camera: Camera, light_view_proj, keys: list[TileKey]) -> None:
        planes = lod.frustum_planes(light_view_proj)
        visible = [
            k
            for k in keys
            if lod.sphere_visible(
                planes, self.node_bounds(k).center - camera.position, self.node_bounds(k).radius
            )
        ]
        self.draw(camera, light_view_proj, visible, shadow_pass=True)
