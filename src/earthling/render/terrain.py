"""Terrain tiles: grid meshes displaced by heightmap textures.

Each tile gets a (GRID+1)^2 vertex grid. Vertex positions on the ellipsoid (height 0) and the
local up vectors are computed on the CPU in float64 via ECEF, relative to the tile's own origin,
so Earth curvature is exact and float32 precision stays high. The vertex shader displaces each
vertex along ``up`` by the height sampled from the tile's heightmap texture; the fragment shader
derives normals from the full-resolution heightmap.
"""

from __future__ import annotations

import logging
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
from earthling.render.camera import Camera
from earthling.render.shader_library import ShaderLibrary

log = logging.getLogger(__name__)

MESH_GRID = 64  # intervals per tile edge
HEIGHTMAP_STRIDE = GRID_INTERVALS // MESH_GRID


def grid_indices(n: int = MESH_GRID) -> np.ndarray:
    """Triangle indices for an (n+1) x (n+1) vertex grid (row-major, row = north->south)."""
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = (j * (n + 1) + i).ravel()
    b = a + 1
    c = a + (n + 1)
    d = c + 1
    return np.column_stack([a, c, b, b, c, d]).astype(np.uint32).ravel()


@dataclass
class TileGeometry:
    origin: np.ndarray  # ENU of the tile centre at height 0 (float64)
    positions: np.ndarray  # (n+1)^2 x 3, relative to origin, height 0
    ups: np.ndarray  # (n+1)^2 x 3 unit vectors in ENU
    tangent: np.ndarray  # 3x3: rows east, north, up at the tile centre (ENU coords)
    sample_spacing_m: float  # ground distance between heightmap samples


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
    lat_c = float(lat[centre])
    spacing = (max_x - min_x) / GRID_INTERVALS * np.cos(np.radians(lat_c))
    return TileGeometry(
        origin=origin,
        positions=ground - origin,
        ups=ups,
        tangent=np.array([east_c, north_c, up_c]),
        sample_spacing_m=float(spacing),
    )


@dataclass
class TerrainTile:
    z: int
    x: int
    y: int
    geometry: TileGeometry
    heights: np.ndarray
    vbo: moderngl.Buffer
    heightmap: moderngl.Texture
    vao: moderngl.VertexArray | None = None
    imagery: moderngl.Texture | None = None
    min_h: float = 0.0
    max_h: float = 0.0

    def release(self) -> None:
        if self.vao is not None:
            self.vao.release()
        self.vbo.release()
        self.heightmap.release()
        if self.imagery is not None:
            self.imagery.release()

    def set_imagery(self, ctx: moderngl.Context, rgb: np.ndarray | None) -> None:
        if self.imagery is not None:
            self.imagery.release()
            self.imagery = None
        if rgb is None:
            return
        h, w, _ = rgb.shape
        tex = ctx.texture((w, h), 3, np.ascontiguousarray(rgb).tobytes())
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.anisotropy = 16.0
        tex.repeat_x = tex.repeat_y = False
        self.imagery = tex


class TerrainLayer:
    def __init__(self, ctx: moderngl.Context, shaders: ShaderLibrary) -> None:
        self.ctx = ctx
        self.shaders = shaders
        self.tiles: list[TerrainTile] = []
        self.exaggeration = 1.0
        self.visible = True
        self._ibo = ctx.buffer(grid_indices().tobytes())
        self._program: moderngl.Program | None = None

    def add_tile(
        self,
        frame: LocalFrame,
        z: int,
        x: int,
        y: int,
        heights: np.ndarray,
        imagery: np.ndarray | None = None,
    ) -> TerrainTile:
        geo = tile_geometry(frame, z, x, y)
        verts = np.hstack([geo.positions, geo.ups]).astype("f4")
        vbo = self.ctx.buffer(verts.tobytes())
        filled = np.nan_to_num(
            heights, nan=float(np.nanmin(heights)) if np.isfinite(heights).any() else 0.0
        )
        tex = self.ctx.texture(
            (HEIGHTMAP_SAMPLES, HEIGHTMAP_SAMPLES), 1, filled.astype("f4").tobytes(), dtype="f4"
        )
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = tex.repeat_y = False
        tile = TerrainTile(
            z, x, y, geo, filled, vbo, tex, min_h=float(filled.min()), max_h=float(filled.max())
        )
        tile.set_imagery(self.ctx, imagery)
        self.tiles.append(tile)
        return tile

    def clear(self) -> None:
        for tile in self.tiles:
            tile.release()
        self.tiles = []

    def bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.tiles:
            return None
        lo = np.min(
            [t.geometry.origin + t.geometry.positions.min(axis=0) for t in self.tiles], axis=0
        )
        hi = np.max(
            [t.geometry.origin + t.geometry.positions.max(axis=0) for t in self.tiles], axis=0
        )
        lo[2] = min(t.min_h for t in self.tiles)
        hi[2] = max(t.max_h for t in self.tiles)
        return lo, hi

    def render(self, camera: Camera, view_proj) -> None:
        if not self.visible or not self.tiles:
            return
        program = self.shaders.get("terrain")
        if program is not self._program:
            for tile in self.tiles:
                if tile.vao is not None:
                    tile.vao.release()
                tile.vao = None
            self._program = program
        program["u_view_proj"].write(view_proj)
        program["u_log_depth_coef"] = camera.log_depth_coef
        program["u_exaggeration"] = self.exaggeration
        program["u_heightmap"] = 0
        program["u_imagery"] = 1
        for tile in self.tiles:
            if tile.vao is None:
                tile.vao = self.ctx.vertex_array(
                    program,
                    [(tile.vbo, "3f 3f", "in_pos", "in_up")],
                    self._ibo,
                    index_element_size=4,
                )
            geo = tile.geometry
            program["u_offset"] = camera.relative(geo.origin)
            # GLSL mat3 is column-major: columns = east, north, up
            program["u_tangent"].write(geo.tangent.astype("f4").tobytes())
            program["u_sample_spacing"] = geo.sample_spacing_m
            tile.heightmap.use(0)
            if tile.imagery is not None:
                tile.imagery.use(1)
            program["u_has_imagery"] = tile.imagery is not None
            tile.vao.render(moderngl.TRIANGLES)
