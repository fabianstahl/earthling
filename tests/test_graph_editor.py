from datetime import UTC, datetime

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QUndoStack

from earthling.app.keyframing import KeyframeEditor
from earthling.app.timeline import TimelineController
from earthling.core.animation import Curve, HandleMode, Interp
from earthling.core.properties import PropertyDef, PType
from earthling.core.scene import Scene

F = PropertyDef("a.f", "F", PType.FLOAT, 0.0)
COL = PropertyDef("a.c", "C", PType.COLOR, (0.0, 0.0, 0.0))
DT = PropertyDef("a.t", "T", PType.DATETIME, datetime(2026, 7, 1, 12, tzinfo=UTC))


def bezier_curve(points, definition=F, mode=HandleMode.AUTO):
    c = Curve(definition)
    for t, v in points:
        k = c.set_key(t, v, Interp.BEZIER)
        k.handle_mode = mode
    return c


def slope(c, t, h=1e-5):
    return (c.evaluate(t + h) - c.evaluate(t - h)) / (2 * h)


def test_auto_handles_are_smooth_and_clamped():
    c = bezier_curve([(0.0, 0.0), (1.0, 10.0), (3.0, 12.0), (4.0, 0.0)])
    # smooth through the middle key: equal slope from both sides
    left = (c.evaluate(1.0) - c.evaluate(1.0 - 1e-4)) / 1e-4
    right = (c.evaluate(1.0 + 1e-4) - c.evaluate(1.0)) / 1e-4
    assert left == pytest.approx(right, rel=2e-3) and left > 0
    # the maximum key has a flat tangent and the curve does not overshoot it
    assert slope(c, 3.0) == pytest.approx(0.0, abs=1e-2)
    samples = [c.evaluate(t) for t in np.linspace(0, 4, 401)]
    assert max(samples) <= 12.0 + 1e-9 and min(samples) >= -1e-9


def test_vector_handles_are_linear_and_free_handles_are_kept():
    c = bezier_curve([(0.0, 0.0), (1.0, 10.0)], mode=HandleMode.VECTOR)
    assert c.evaluate(0.25) == pytest.approx(2.5, abs=1e-6)
    c.keys[0].handle_mode = HandleMode.FREE
    c.keys[0].out_handle = (0.5, 0.0)
    c.keys[1].handle_mode = HandleMode.FREE
    c.keys[1].in_handle = (0.5, 0.0)
    assert c.evaluate(0.2) < 2.0


def test_drag_handle_converts_linear_segment_without_jumps():
    c = Curve(F)
    c.set_key(0.0, 0.0)
    c.set_key(1.0, 10.0)
    before = [c.evaluate(t) for t in (0.25, 0.5, 0.75)]
    # grabbing the handle where it already is changes nothing
    c.drag_handle(1, "in", 1.0 - 1.0 / 3.0, 10.0 - 10.0 / 3.0)
    assert c.keys[0].interp is Interp.BEZIER
    assert [c.evaluate(t) for t in (0.25, 0.5, 0.75)] == pytest.approx(before, abs=1e-6)
    assert c.keys[1].handle_mode is HandleMode.ALIGNED
    c.drag_handle(0, "out", 0.25, 5.0)
    assert c.keys[0].out_handle == pytest.approx((0.25, 0.5))
    assert c.evaluate(0.1) > 1.0  # fast start
    assert c.handle_points(0)["out"] == pytest.approx((0.25, 5.0))


def test_aligned_handles_stay_collinear():
    c = bezier_curve([(0.0, 0.0), (1.0, 10.0), (2.0, 30.0)])
    c.drag_handle(1, "out", 1.3, 14.0)
    assert c.keys[1].handle_mode is HandleMode.ALIGNED
    pts = c.handle_points(1)
    (ti, vi), (to, vo) = pts["in"], pts["out"]
    assert (10.0 - vi) / (1.0 - ti) == pytest.approx((vo - 10.0) / (to - 1.0))
    assert slope(c, 1.0) == pytest.approx((vo - 10.0) / (to - 1.0), rel=1e-3)
    # free handles are independent
    c.keys[1].handle_mode = HandleMode.FREE
    in_before = c.keys[1].in_handle
    c.drag_handle(1, "out", 1.2, 18.0)
    assert c.keys[1].in_handle == in_before


def test_scalar_channels():
    color = bezier_curve([(0.0, (0, 0, 0)), (2.0, (1, 1, 1)), (3.0, (1, 0, 0))], definition=COL)
    assert color.graph_kind == "progress"
    assert color.scalar_at(2.0) == pytest.approx(1.0)
    assert 0.0 < color.scalar_at(1.0) < 1.0
    when = datetime(2026, 7, 1, 20, 0, tzinfo=UTC)
    dt = bezier_curve([(0.0, when), (1.0, when.replace(hour=22))], definition=DT)
    assert dt.graph_kind == "value"
    assert dt.scalar_at(1.0) - dt.scalar_at(0.0) == pytest.approx(7200.0)
    assert dt.from_scalar(dt.to_scalar(when) + 60.0, when) == when.replace(minute=1)


def test_handle_modes_json_roundtrip_and_legacy_files():
    c = bezier_curve([(0.0, 0.0), (1.0, 10.0)])
    c.keys[0].handle_mode = HandleMode.FREE
    c.keys[0].out_handle = (0.2, 0.7)
    data = c.to_json()
    other = Curve.from_json(F, data)
    assert [k.handle_mode for k in other.keys] == [HandleMode.FREE, HandleMode.AUTO]
    assert other.keys[0].out_handle == (0.2, 0.7)
    legacy = [
        {"t": 0.0, "v": 0.0, "interp": "bezier", "out": [0.2, 0.7], "in": [0.3, 0.0]},
        {"t": 1.0, "v": 1.0, "interp": "linear"},
    ]
    old = Curve.from_json(F, legacy)
    assert old.keys[0].handle_mode is HandleMode.FREE
    assert old.keys[1].handle_mode is HandleMode.AUTO


BOOL_PID = "camera.constant_speed"


def make_editor():
    scene = Scene()
    scene.animation.duration, scene.animation.fps = 10.0, 10.0
    timeline = TimelineController(scene.animation)
    stack = QUndoStack()
    return scene, timeline, stack, KeyframeEditor(scene.animation, timeline, stack)


def test_set_handle_mode_makes_segments_bezier_and_undoes(qapp):
    scene, tl, stack, ed = make_editor()
    anim = scene.animation
    for t, v in ((0.0, 30.0), (2.0, 60.0), (4.0, 50.0)):
        anim.set_key("view.fov", t, v)
    ed.set_handle_mode([("view.fov", 2.0)], HandleMode.AUTO)
    keys = anim.curves["view.fov"].keys
    assert keys[0].interp is Interp.BEZIER and keys[1].interp is Interp.BEZIER
    assert keys[2].interp is Interp.LINEAR
    stack.undo()
    assert anim.curves["view.fov"].keys[0].interp is Interp.LINEAR


def point_of(view, pid, t):
    curve = view.curve(pid)
    i = curve.keys.index(curve.key_at(t))
    p = view.point(pid, curve, t, curve.scalars()[i])
    return QPoint(round(p.x()), round(p.y()))


def test_graph_editor_drag_key_and_handle_with_undo(qtbot):
    from earthling.ui.graph_editor import GraphEditor

    scene, tl, stack, ed = make_editor()
    anim = scene.animation
    anim.set_key("view.fov", 0.0, 30.0)
    anim.set_key("view.fov", 4.0, 70.0)
    anim.set_key("view.fov", 8.0, 50.0)
    anim.set_key(BOOL_PID, 0.0, True)
    ed.changed.emit()
    widget = GraphEditor(ed)
    qtbot.addWidget(widget)
    widget.resize(900, 400)
    widget.show()
    qtbot.waitExposed(widget)
    view = widget.view
    assert BOOL_PID not in view.channels  # stepped: nothing to plot
    assert "view.fov" in view.channels
    view.frame_all()
    # drag the key at 4 s up by ~ 1/4 of the plot height and right by 1 s
    start = point_of(view, "view.fov", 4.0)
    r = view.plot_rect()
    target_t = 5.0
    end = QPoint(round(view.tx(target_t)), round(start.y() - r.height() / 4))
    qtbot.mousePress(view, Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(view, pos=QPoint((start.x() + end.x()) // 2, (start.y() + end.y()) // 2))
    qtbot.mouseMove(view, pos=end)
    qtbot.mouseRelease(view, Qt.MouseButton.LeftButton, pos=end)
    curve = anim.curves["view.fov"]
    moved = curve.key_at(5.0)
    assert moved is not None and moved.value > 70.0
    assert ("view.fov", 5.0) in view.selection
    assert stack.count() == 1  # one undo step for the whole drag
    # grab the (selected) key's in handle: the segment becomes Bézier
    ed.set_handle_mode([("view.fov", 5.0)], HandleMode.AUTO)
    handle = curve.handle_points(curve.keys.index(moved))["in"]
    hp = view.point("view.fov", curve, *handle)
    press = QPoint(round(hp.x()), round(hp.y()))
    to = QPoint(press.x() - 30, press.y() + 40)
    qtbot.mousePress(view, Qt.MouseButton.LeftButton, pos=press)
    qtbot.mouseMove(view, pos=to)
    qtbot.mouseRelease(view, Qt.MouseButton.LeftButton, pos=to)
    assert moved.handle_mode is HandleMode.ALIGNED
    assert curve.handle_points(curve.keys.index(moved))["in"] != pytest.approx(handle)
    assert stack.count() == 3
    stack.undo()
    stack.undo()
    stack.undo()
    curve = anim.curves["view.fov"]
    assert curve.key_at(4.0).value == pytest.approx(70.0) and curve.key_at(5.0) is None
    widget.close()  # qtbot only holds a weak reference: don't leave it painting after the test


def test_graph_editor_normalized_view_and_zoom(qtbot):
    from earthling.ui.graph_editor import GraphEditor

    scene, tl, stack, ed = make_editor()
    anim = scene.animation
    anim.set_key("view.fov", 0.0, 20.0)
    anim.set_key("view.fov", 4.0, 100.0)
    anim.set_key("follow.distance", 0.0, 500.0)
    anim.set_key("follow.distance", 4.0, 1000.0)
    ed.changed.emit()
    widget = GraphEditor(ed)
    qtbot.addWidget(widget)
    widget.resize(800, 300)
    widget.show()
    widget.buttons["normalize"].setChecked(True)
    view = widget.view
    for pid in ("view.fov", "follow.distance"):
        curve = view.curve(pid)
        offset, scale = view.transform(pid, curve)
        assert (curve.scalars()[0] - offset) * scale == pytest.approx(0.0)
        assert (curve.scalars()[-1] - offset) * scale == pytest.approx(1.0)
    span = view.t1 - view.t0
    qtbot.wait(1)
    from PyQt6.QtCore import QPointF
    from PyQt6.QtGui import QWheelEvent

    event = QWheelEvent(QPointF(400, 150), QPointF(400, 150), QPoint(0, 0), QPoint(0, 120),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.ShiftModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)  # fmt: skip
    view.wheelEvent(event)
    assert view.t1 - view.t0 < span
    # hiding a channel removes it from the view
    widget.channel_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert len(view.channels) == 1
    widget.close()
