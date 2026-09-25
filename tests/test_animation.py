from datetime import datetime

import pytest

from earthling.core.animation import (
    Animation,
    Curve,
    Interp,
    cubic_bezier_ease,
    oklab_to_srgb,
    snap_to_frame,
    srgb_to_oklab,
)
from earthling.core.properties import PropertyDef, PropertyRegistry, PropertyStore, PType
from earthling.core.scene import Scene

F = PropertyDef("a.f", "F", PType.FLOAT, 0.0, minimum=-100, maximum=100)
INT = PropertyDef("a.i", "I", PType.INT, 0)
B = PropertyDef("a.b", "B", PType.BOOL, False)
E = PropertyDef("a.e", "E", PType.ENUM, "x", options=(("x", "X"), ("y", "Y")))
C = PropertyDef("a.c", "C", PType.COLOR, (0.0, 0.0, 0.0))
V = PropertyDef("a.v", "V", PType.VEC3, (0.0, 0.0, 0.0))
T = PropertyDef("a.t", "T", PType.DATETIME, datetime(2026, 7, 1, 12, 0))
N = PropertyDef("a.n", "N", PType.FLOAT, 0.0, animatable=False)


def curve(defn, *points, interp=None):
    c = Curve(defn)
    for t, v in points:
        c.set_key(t, v, interp)
    return c


def test_cubic_bezier_presets():
    assert cubic_bezier_ease(0.5, 0, 0, 1, 1) == pytest.approx(0.5)
    ease_in = cubic_bezier_ease(0.25, 0.42, 0.0, 1.0, 1.0)
    ease_out = cubic_bezier_ease(0.25, 0.0, 0.0, 0.58, 1.0)
    assert ease_in < 0.25 < ease_out
    assert cubic_bezier_ease(0.5, 0.42, 0.0, 0.58, 1.0) == pytest.approx(0.5, abs=1e-6)
    # overshoot handles
    assert cubic_bezier_ease(0.8, 0.3, 1.5, 0.7, 1.5) > 1.0


def test_linear_step_and_clamping():
    c = curve(F, (0.0, 0.0), (2.0, 10.0))
    assert c.evaluate(-1) == 0.0 and c.evaluate(5) == 10.0
    assert c.evaluate(1.0) == pytest.approx(5.0)
    c.keys[0].interp = Interp.STEP
    assert c.evaluate(1.999) == 0.0 and c.evaluate(2.0) == 10.0


def test_ease_presets_shape():
    for interp, lower in ((Interp.EASE_IN, True), (Interp.EASE_OUT, False)):
        c = curve(F, (0.0, 0.0), (1.0, 10.0), interp=interp)
        v = c.evaluate(0.25)
        assert (v < 2.5) if lower else (v > 2.5)
    c = curve(F, (0.0, 0.0), (1.0, 10.0), interp=Interp.EASE_IN_OUT)
    assert c.evaluate(0.5) == pytest.approx(5.0, abs=1e-4)


def test_bezier_handles():
    c = curve(F, (0.0, 0.0), (1.0, 10.0), interp=Interp.BEZIER)
    c.keys[0].out_handle = (0.5, 0.0)
    c.keys[1].in_handle = (0.5, 0.0)
    assert c.evaluate(0.5) == pytest.approx(5.0, abs=1e-4)
    assert c.evaluate(0.2) < 2.0  # slow start


def test_type_rules():
    assert curve(INT, (0, 0), (1, 10)).evaluate(0.44) == 4
    assert curve(B, (0, False), (1, True)).evaluate(0.99) is False
    assert curve(E, (0, "x"), (1, "y"), interp=Interp.LINEAR).evaluate(0.9) == "x"
    assert curve(V, (0, (0, 0, 0)), (1, (2, 4, 6))).evaluate(0.5) == pytest.approx((1, 2, 3))
    t = curve(T, (0, datetime(2026, 7, 1, 12, 0)), (10, datetime(2026, 7, 1, 22, 0)))
    assert t.evaluate(5) == datetime(2026, 7, 1, 17, 0)


def test_oklab_roundtrip_and_colour_midpoint():
    for rgb in ((1.0, 0.0, 0.0), (0.2, 0.5, 0.9), (1.0, 1.0, 1.0)):
        assert oklab_to_srgb(srgb_to_oklab(rgb)) == pytest.approx(rgb, abs=1e-5)
    mid = curve(C, (0, (0.0, 0.0, 1.0)), (1, (1.0, 1.0, 0.0))).evaluate(0.5)
    # the sRGB average would be a dull grey (0.5, 0.5, 0.5); OKLab keeps it lighter
    assert sum(mid) / 3 > 0.55


def test_set_key_replaces_and_sorts():
    c = Curve(F)
    c.set_key(2.0, 1.0)
    c.set_key(1.0, 2.0)
    c.set_key(2.0, 3.0)
    assert [k.time for k in c.keys] == [1.0, 2.0] and c.keys[1].value == 3.0


def make_animation():
    registry = PropertyRegistry([F, E, T, N])
    store = PropertyStore(registry)
    return Animation(store, duration=10.0, fps=25.0), store


def test_animation_apply_and_snapping():
    anim, store = make_animation()
    anim.set_key("a.f", 0.01, 1.0)  # snapped to frame 0
    anim.set_key("a.f", 4.0, 9.0)
    anim.set_key("a.e", 2.0, "y")
    assert anim.has_key("a.f", 0.0)
    anim.apply(2.0)
    assert store["a.f"] == pytest.approx(5.0) and store["a.e"] == "y"
    assert anim.key_times() == [0.0, 2.0, 4.0]
    assert snap_to_frame(0.019, 25) == pytest.approx(0.0)
    assert anim.frame_count == 251
    with pytest.raises(ValueError):
        anim.set_key("a.n", 0.0, 1.0)
    assert anim.remove_key("a.e", 2.0) and not anim.is_animated("a.e")


def test_animation_json_roundtrip():
    anim, _ = make_animation()
    anim.set_key("a.f", 0.0, 1.0, Interp.BEZIER)
    anim.curves["a.f"].keys[0].out_handle = (0.2, 0.7)
    anim.set_key("a.f", 1.0, 2.0)
    anim.set_key("a.t", 3.0, datetime(2026, 7, 1, 20, 0))
    data = anim.to_json()
    other, _ = make_animation()
    assert other.load_json(data) == []
    assert other.curves["a.f"].keys[0].out_handle == (0.2, 0.7)
    assert other.curves["a.t"].keys[0].value == datetime(2026, 7, 1, 20, 0)
    assert other.load_json({"curves": {"zzz": []}}) == ["animation of unknown property 'zzz'"]


def test_scene_saves_animation(tmp_path):
    scene = Scene()
    scene.animation.set_key("sun.datetime", 0.0, datetime(2026, 7, 1, 12, 0))
    scene.animation.set_key("sun.datetime", 20.0, datetime(2026, 7, 1, 22, 0))
    path = scene.save(tmp_path / "s.json")
    other = Scene()
    other.load(path)
    other.animation.apply(10.0)
    assert other.store["sun.datetime"] == datetime(2026, 7, 1, 17, 0)
