import numpy as np
import pytest

from earthling.render.lightning import (
    DURATION_S,
    Strike,
    active_strikes,
    bolt_paths,
    flash_intensity,
    hash01,
    strikes_between,
)


def test_schedule_is_deterministic_and_follows_the_rate():
    a = strikes_between(lambda t: 30.0, 7, 0.0, 600.0)
    b = strikes_between(lambda t: 30.0, 7, 0.0, 600.0)
    assert [s.start for s in a] == [s.start for s in b]
    assert 240 < len(a) < 360  # ~30 per minute over ten minutes
    other = strikes_between(lambda t: 30.0, 8, 0.0, 600.0)
    assert [s.start for s in other] != [s.start for s in a]
    assert strikes_between(lambda t: 0.0, 7, 0.0, 600.0) == []
    ramp = strikes_between(lambda t: 0.0 if t < 300 else 60.0, 7, 0.0, 600.0)
    assert all(s.start >= 300.0 for s in ramp) and len(ramp) > 200
    # any frame on its own sees the same strikes as a sequential render
    s = a[3]
    assert s in active_strikes(lambda t: 30.0, 7, s.start + 0.1)
    assert s not in active_strikes(lambda t: 30.0, 7, s.start + DURATION_S + 0.01)
    assert 0.0 <= hash01(1, 2, 3) < 1.0


def test_flash_has_return_strokes_and_ends():
    s = Strike(10, 1.0, 1)
    assert flash_intensity(s, 0.9) == 0.0 and flash_intensity(s, 1.0 + DURATION_S + 0.01) == 0.0
    values = [flash_intensity(s, 1.0 + t) for t in np.linspace(0.0, DURATION_S, 200)]
    assert values[0] == pytest.approx(1.0)
    rises = sum(1 for a, b in zip(values, values[1:], strict=False) if b > a + 0.05)
    assert rises >= 2  # two return strokes


def test_bolt_connects_cloud_and_ground():
    top, bottom = np.array([0.0, 0.0, 3000.0]), np.array([200.0, 100.0, 1200.0])
    paths = bolt_paths(top, bottom, Strike(5, 0.0, 3))
    (main, strength), *branches = paths
    assert strength == 1.0 and np.allclose(main[0], top) and np.allclose(main[-1], bottom)
    assert len(main) == 2**7 + 1 and 2 <= len(branches) <= 4
    jag = np.linalg.norm(np.diff(main, axis=0), axis=1).sum()
    assert jag > 1.1 * np.linalg.norm(bottom - top)  # jagged, not straight
    again = bolt_paths(top, bottom, Strike(5, 0.0, 3))
    assert np.allclose(again[0][0], main)  # deterministic shape


def test_storm_flash_brightens_the_frame(gl_ctx):
    import datetime

    from test_weather import render

    night = {"sun__datetime": datetime.datetime(2026, 7, 1, 23, 30), "rain__enabled": True,
             "rain__cloud_base": 3500.0, "lightning__rate": 60.0}  # fmt: skip
    strikes = strikes_between(lambda t: 60.0, 1, 0.0, 30.0)
    from earthling.render.renderer import Renderer  # noqa: F401

    quiet_t = next(t for t in np.arange(0.0, 30.0, 0.05)
                   if not active_strikes(lambda _t: 60.0, 1, t))  # fmt: skip
    dark = _render_at(gl_ctx, render, quiet_t, night)
    lit = _render_at(gl_ctx, render, strikes[0].start + 0.005, night)
    assert lit.mean() > dark.mean() + 3.0


def _render_at(gl_ctx, render, t, props):
    import earthling.render.renderer as renderer_module

    original = renderer_module.Renderer.render

    def at_time(self, *args, **kwargs):
        self.timeline_time = t
        return original(self, *args, **kwargs)

    renderer_module.Renderer.render = at_time
    try:
        return render(gl_ctx, 0.2, **props)
    finally:
        renderer_module.Renderer.render = original


def test_bolts_do_not_write_depth(gl_ctx):
    """Clouds and rain read the scene depth: a bolt must not punch holes into them."""
    from earthling.core.scene import Scene
    from earthling.render.camera import Camera
    from earthling.render.lightning import LightningLayer
    from earthling.render.shader_library import ShaderLibrary

    camera = Camera()
    camera.position = np.array([0.0, -3000.0, 1000.0])
    camera.heading, camera.pitch = 0.0, 0.0
    strike = Strike(0, 0.0, 1)
    path = np.array([[0.0, 0.0, 3000.0], [30.0, 0.0, 1500.0], [0.0, 0.0, 0.0]])
    layer = LightningLayer(gl_ctx, ShaderLibrary(gl_ctx))
    layer.active = [(strike, 1.0, path[0], [(path, 1.0)])]
    color = gl_ctx.texture((64, 64), 4, dtype="f2")
    depth = gl_ctx.depth_texture((64, 64))
    fbo = gl_ctx.framebuffer([color], depth)
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
    layer.render(camera, camera.view_projection(1.0), 64, 64, Scene().store, glow=False)
    pixels = np.frombuffer(color.read(), dtype=np.float16).reshape(64, 64, 4)
    assert pixels[..., :3].max() > 1.0  # the bolt is drawn
    assert np.frombuffer(depth.read(), dtype=np.float32).min() == 1.0  # but leaves no depth


def test_branching_adds_forks_of_forks():
    top, bottom = np.array([0.0, 0.0, 4000.0]), np.array([300.0, 0.0, 1500.0])
    counts = {}
    for b in (0.0, 1.0):
        paths = [bolt_paths(top, bottom, Strike(slot, 0.0, 7), branching=b) for slot in range(20)]
        counts[b] = np.mean([len(p) for p in paths])
        again = bolt_paths(top, bottom, Strike(3, 0.0, 7), branching=b)
        assert [len(p[0]) for p in again] == [len(p[0]) for p in paths[3]]  # deterministic
    assert counts[1.0] > 2.5 * counts[0.0]
    rich = bolt_paths(top, bottom, Strike(5, 0.0, 7), branching=1.0)
    assert min(i for _, i in rich) < 0.35  # forks of forks are fainter than first branches
