import numpy as np
import pytest

from earthling.core.scene import Scene
from earthling.render.hud import HudLayer
from earthling.render.labels import FontAtlas
from earthling.render.scale import pixel_scale
from earthling.render.shader_library import ShaderLibrary
from test_hud import fake_path


def test_pixel_scale_uses_the_shorter_side():
    assert pixel_scale(1920, 1080) == pytest.approx(1.0)
    assert pixel_scale(3840, 2160) == pytest.approx(2.0)
    assert pixel_scale(1080, 1920) == pytest.approx(1.0)  # portrait: same sizes as 1080p
    assert pixel_scale(1080, 1080) == pytest.approx(1.0)


def test_export_offers_portrait_and_square():
    options = [o for o, _ in Scene().registry["export.resolution"].options]
    assert "1080x1920" in options and "1080x1080" in options


def test_hud_fits_a_portrait_frame(gl_ctx):
    store = Scene().store
    store.set("stats.visible", True)
    atlas = FontAtlas(gl_ctx)
    hud = HudLayer(gl_ctx, ShaderLibrary(gl_ctx), lambda: atlas)
    sizes = {}
    for w, h in ((960, 540), (540, 960)):
        fbo = gl_ctx.simple_framebuffer((w, h))
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        hud.render(fbo, w, h, store, fake_path(), 20_000.0, "UTC")
        img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(h, w, 3)
        cols = np.flatnonzero(img.sum(axis=(0, 2)) > 0)
        sizes[(w, h)] = cols[-1] - cols[0]
        assert cols[-1] < w - 5  # the panel stays inside the narrow frame
    assert sizes[(540, 960)] == pytest.approx(sizes[(960, 540)], rel=0.05)  # same size
