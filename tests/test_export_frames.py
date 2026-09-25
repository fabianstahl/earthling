import numpy as np

from earthling.core.geo import LocalFrame, lonlat_to_tile
from earthling.core.scene import Scene
from earthling.export.frames import FrameRenderer, parse_resolution, save_png
from earthling.render import lod
from earthling.render.renderer import Renderer
from test_terrain import SlopeTerrainData

FRAME = LocalFrame(46.0, 7.0, 0.0)


def make(gl_ctx):
    scene = Scene()
    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [])
    renderer.store = scene.store
    renderer.timezone = "UTC"
    tx, ty = lonlat_to_tile(7.0, 46.0, 10)
    renderer.set_terrain_source(SlopeTerrainData(), lod.NodeSet({10: [(int(tx), int(ty))]}))
    anim = scene.animation
    anim.set_key("camera.pose", 0.0, (0.0, -30000.0, 20000.0, 0.0, -30.0, 0.0))
    anim.set_key("camera.pose", 4.0, (5000.0, -30000.0, 20000.0, 10.0, -30.0, 0.0))
    return scene, renderer


def test_parse_resolution():
    assert parse_resolution("3840x2160") == (3840, 2160)


def test_frames_are_deterministic_and_complete(gl_ctx):
    scene, renderer = make(gl_ctx)
    frames = FrameRenderer(renderer, scene.animation)
    a = frames.render(2.0, 160, 90)
    b = frames.render(2.0, 160, 90)
    assert a.shape == (90, 160, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert renderer.terrain.fully_loaded()
    hi = frames.render(2.0, 160, 90, bits=16)
    assert hi.dtype == np.uint16
    assert np.abs(hi.astype(float) / 257.0 - a).max() <= 1.0
    # quality overrides are only active while exporting
    assert renderer.overrides == {}
    other = frames.render(0.0, 160, 90)
    assert not np.array_equal(a, other)  # the camera moved
    frames.release()


def test_preview_scale_renders_to_full_output(gl_ctx):
    scene, renderer = make(gl_ctx)
    from earthling.render.camera import Camera

    fbo = gl_ctx.simple_framebuffer((128, 64))
    camera = Camera(position=np.array([0.0, -30000.0, 20000.0]), heading=0.0, pitch=-30.0)
    renderer.prepare(camera, 128, 64)
    renderer.render(fbo, 128, 64, camera, scale=0.5)
    assert renderer.target.size == (64, 32)
    img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(64, 128, 3)
    assert img[:32].mean() > 5 and img[32:].mean() > 5  # whole output covered


def test_png16(tmp_path):
    img = (np.arange(4 * 3 * 3).reshape(4, 3, 3) * 1000).astype(np.uint16)
    path = tmp_path / "x.png"
    save_png(img, path)
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data[24] == 16  # bit depth in IHDR
    from PIL import Image

    with Image.open(path) as im:
        assert im.size == (3, 4)
