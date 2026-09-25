import numpy as np
import pytest

from earthling.core.geo import LocalFrame, lonlat_to_tile
from earthling.data.dem import HEIGHTMAP_SAMPLES
from earthling.render.camera import Camera, OrbitController
from earthling.render.renderer import Renderer
from earthling.render.terrain import MESH_GRID, grid_indices, tile_geometry

FRAME = LocalFrame(46.0, 7.0, 0.0)
N = MESH_GRID + 1


def test_grid_indices():
    idx = grid_indices(2)
    assert idx.size == 2 * 2 * 6 and idx.max() == 8


def test_adjacent_tiles_share_edge_vertices():
    tx, ty = lonlat_to_tile(7.0, 46.0, 12)
    x, y = int(tx), int(ty)
    a = tile_geometry(FRAME, 12, x, y)
    b = tile_geometry(FRAME, 12, x + 1, y)
    pa = (a.positions + a.origin).reshape(N, N, 3)
    pb = (b.positions + b.origin).reshape(N, N, 3)
    assert np.allclose(pa[:, -1], pb[:, 0], atol=1e-6)


def test_tangent_frame_and_spacing():
    tx, ty = lonlat_to_tile(7.0, 46.0, 12)
    g = tile_geometry(FRAME, 12, int(tx), int(ty))
    east, north, up = g.tangent
    assert east == pytest.approx([1, 0, 0], abs=0.01)
    assert north == pytest.approx([0, 1, 0], abs=0.01)
    assert up == pytest.approx([0, 0, 1], abs=0.01)
    assert g.sample_spacing_m == pytest.approx(9783.9 / 256 * np.cos(np.radians(46.0)), rel=0.01)


def test_render_terrain(gl_ctx):
    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [])
    tx, ty = lonlat_to_tile(7.0, 46.0, 11)
    heights = np.full((HEIGHTMAP_SAMPLES, HEIGHTMAP_SAMPLES), 1500.0, dtype=np.float32)
    heights[100:150, 100:150] = 2500.0
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb[..., 1] = 200
    renderer.set_terrain([(11, int(tx), int(ty), heights, rgb)])
    lo, hi = renderer.scene_bounds()
    assert lo[2] == pytest.approx(1500) and hi[2] == pytest.approx(2500)
    camera = Camera()
    OrbitController(camera).frame_bounds(lo, hi)
    fbo = gl_ctx.simple_framebuffer((96, 96))
    renderer.render(fbo, 96, 96, camera)
    img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(96, 96, 3).astype(int)
    sky = np.array(renderer.clear_color[:3]) * 255
    assert (np.abs(img - sky).sum(axis=2) > 40).mean() > 0.2
