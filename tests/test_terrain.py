import numpy as np
import pytest

from earthling.core.geo import LocalFrame, lonlat_to_tile
from earthling.data.dem import HEIGHTMAP_SAMPLES
from earthling.data.terrain_data import HeightmapRef
from earthling.render import lod
from earthling.render.camera import Camera, OrbitController
from earthling.render.renderer import Renderer
from earthling.render.terrain import MESH_GRID, grid_indices, shared_grid_attributes, tile_geometry

FRAME = LocalFrame(46.0, 7.0, 0.0)
N = MESH_GRID + 1


def test_grid_and_skirt_layout():
    idx = grid_indices(2, skirt=False)
    assert idx.size == 2 * 2 * 6 and idx.max() == 8
    full = grid_indices()
    attrs = shared_grid_attributes()
    assert full.max() == len(attrs) - 1
    assert attrs[: N * N, 2].max() == 0 and attrs[N * N :, 2].min() == 1
    assert len(attrs) == N * N + 4 * MESH_GRID


def test_adjacent_tiles_share_edge_vertices():
    tx, ty = lonlat_to_tile(7.0, 46.0, 12)
    x, y = int(tx), int(ty)
    a = tile_geometry(FRAME, 12, x, y)
    b = tile_geometry(FRAME, 12, x + 1, y)
    pa = (a.positions[: N * N] + a.origin).reshape(N, N, 3)
    pb = (b.positions[: N * N] + b.origin).reshape(N, N, 3)
    assert np.allclose(pa[:, -1], pb[:, 0], atol=1e-6)


def test_tangent_frame_and_width():
    tx, ty = lonlat_to_tile(7.0, 46.0, 12)
    g = tile_geometry(FRAME, 12, int(tx), int(ty))
    east, north, up = g.tangent
    assert east == pytest.approx([1, 0, 0], abs=0.01)
    assert north == pytest.approx([0, 1, 0], abs=0.01)
    assert up == pytest.approx([0, 0, 1], abs=0.01)
    assert g.width_m == pytest.approx(9783.9 * np.cos(np.radians(46.0)), rel=0.01)


class FakeTerrainData:
    """Constant plateau with a bump; imagery only at z >= 11."""

    def __init__(self):
        self.heights = np.full((HEIGHTMAP_SAMPLES, HEIGHTMAP_SAMPLES), 1500.0, dtype=np.float32)
        self.heights[100:150, 100:150] = 2500.0

    def heightmap_for(self, key):
        return HeightmapRef(key, self.heights, 256.0, (0.0, 0.0))

    def imagery_for(self, key):
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[..., 1] = 200
        return rgb


def test_render_lod_terrain(gl_ctx):
    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [])
    tx, ty = lonlat_to_tile(7.0, 46.0, 10)
    nodes = lod.NodeSet({10: [(int(tx), int(ty))], 11: [(2 * int(tx), 2 * int(ty))]})
    renderer.set_terrain_source(FakeTerrainData(), nodes)
    camera = Camera()
    fbo = gl_ctx.simple_framebuffer((96, 96))
    renderer.render(fbo, 96, 96, camera)  # loads roots
    lo, hi = renderer.scene_bounds()
    assert lo[2] == pytest.approx(1500) and hi[2] == pytest.approx(2500)
    OrbitController(camera).frame_bounds(lo, hi)
    for _ in range(5):
        renderer.render(fbo, 96, 96, camera)
    assert renderer.terrain.fully_loaded()
    img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(96, 96, 3).astype(int)
    green = (img[..., 1] > 150) & (img[..., 0] < 80)
    assert green.mean() > 0.2
