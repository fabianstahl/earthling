"""The key views with mouse and keyboard: selecting (click, Shift/Ctrl, box), moving, deleting,
copying and pasting keys in the dope sheet; panning, zooming, scrubbing, constrained drags and
the context menus of the graph editor; the key button next to a setting."""

from datetime import UTC, datetime

import pytest
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QKeyEvent, QKeySequence, QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import QApplication

from earthling.core.animation import INTERP_LABELS, HandleMode, Interp
from helpers import menu_choices
from test_graph_editor import make_editor

L, R, M = Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton
NONE, SHIFT = Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ShiftModifier
CTRL, ALT = Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.AltModifier


def mouse(widget, kind, pos, button=L, mods=NONE, buttons=None):
    point = QPointF(pos)
    held = button if buttons is None else buttons
    QApplication.sendEvent(widget, QMouseEvent(kind, point, widget.mapToGlobal(point), button,
                                               held, mods))  # fmt: skip


def press(widget, pos, button=L, mods=NONE):
    mouse(widget, QEvent.Type.MouseButtonPress, pos, button, mods)


def move(widget, pos, button=L, mods=NONE):
    mouse(widget, QEvent.Type.MouseMove, pos, Qt.MouseButton.NoButton, mods, buttons=button)


def release(widget, pos, button=L, mods=NONE):
    mouse(widget, QEvent.Type.MouseButtonRelease, pos, button, mods)


def drag(widget, start, end, button=L, mods=NONE):
    press(widget, start, button, mods)
    move(widget, QPointF((start.x() + end.x()) / 2, (start.y() + end.y()) / 2), button, mods)
    move(widget, end, button, mods)
    release(widget, end, button, mods)


def double_click(widget, pos):
    mouse(widget, QEvent.Type.MouseButtonDblClick, pos)


def key(widget, k, mods=NONE):
    QApplication.sendEvent(widget, QKeyEvent(QEvent.Type.KeyPress, k, mods))


def shortcut(widget, standard):
    """Press a standard key sequence (Ctrl+A, Ctrl+C, ...)."""
    combo = QKeySequence(standard)[0]
    key(widget, combo.key(), combo.keyboardModifiers())


# --- dope sheet ------------------------------------------------------------------------------
@pytest.fixture
def dope(qtbot):
    from earthling.ui.dope_sheet import DopeSheet

    scene, timeline, stack, editor = make_editor()
    anim = scene.animation
    for t, v in ((0.0, 30.0), (2.0, 60.0), (4.0, 40.0)):
        anim.set_key("view.fov", t, v)
    for t, v in ((1.0, 500.0), (3.0, 900.0)):
        anim.set_key("follow.distance", t, v)
    editor.changed.emit()
    sheet = DopeSheet(editor, lambda t: 200.0 + t * 50.0, lambda x: (x - 200.0) / 50.0)
    qtbot.addWidget(sheet)
    sheet.resize(800, 200)
    sheet.stack, sheet.anim = stack, anim
    return sheet


def at(sheet, pid, t) -> QPointF:
    return QPointF(sheet.time_to_x(t), sheet.row_y(sheet.rows().index(pid)))


def test_dope_sheet_selects_with_clicks_shift_ctrl_and_a_box(dope):
    changed = []
    dope.selection_changed.connect(lambda: changed.append(1))
    press(dope, at(dope, "view.fov", 2.0))
    release(dope, at(dope, "view.fov", 2.0))
    assert dope.selection == {("view.fov", 2.0)} and changed
    press(dope, at(dope, "follow.distance", 3.0), mods=SHIFT)  # Shift adds
    release(dope, at(dope, "follow.distance", 3.0))
    press(dope, at(dope, "view.fov", 2.0), mods=CTRL)  # Ctrl toggles
    release(dope, at(dope, "view.fov", 2.0))
    assert dope.selection == {("follow.distance", 3.0)}
    press(dope, QPointF(700, 15))  # an empty spot clears
    release(dope, QPointF(700, 15))
    assert dope.selection == set()
    # a box over both rows from 0.5 s to 3.5 s
    drag(dope, QPointF(dope.time_to_x(0.5), 2), QPointF(dope.time_to_x(3.5), 38))
    assert dope.selection == {("view.fov", 2.0), ("follow.distance", 1.0),
                              ("follow.distance", 3.0)}  # fmt: skip
    dope.grab()  # (painted with the selection)
    double_click(dope, at(dope, "view.fov", 4.0))
    assert dope.editor.timeline.time == 4.0
    assert dope.key_at(QPointF(10, 500)) is None  # (below the rows)


def test_dope_sheet_moves_deletes_copies_and_pastes_keys(dope):
    fov = dope.anim.curves["view.fov"]
    start = at(dope, "view.fov", 2.0)
    press(dope, start)
    move(dope, QPointF(start.x() + 26, start.y()))  # half a second (snapped to frames)
    dope.grab()  # (the keys are drawn where they are dragged to)
    release(dope, QPointF(start.x() + 26, start.y()))
    assert [k.time for k in fov.keys] == [0.0, 2.5, 4.0] and dope.stack.count() == 1
    assert dope.selection == {("view.fov", 2.5)}
    dope.stack.undo()
    fov = dope.anim.curves["view.fov"]  # (undo restores a copy of the curve)
    assert [k.time for k in fov.keys] == [0.0, 2.0, 4.0]
    shortcut(dope, QKeySequence.StandardKey.SelectAll)
    assert len(dope.selection) == 5
    dope.selection = {("view.fov", 2.0), ("view.fov", 4.0)}
    shortcut(dope, QKeySequence.StandardKey.Copy)
    dope.editor.timeline.set_time(6.0)
    shortcut(dope, QKeySequence.StandardKey.Paste)  # at the playhead, the same spacing
    assert [k.time for k in fov.keys] == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert dope.selection == {("view.fov", 6.0), ("view.fov", 8.0)}
    key(dope, Qt.Key.Key_Delete)
    assert [k.time for k in fov.keys] == [0.0, 2.0, 4.0] and dope.selection == set()
    dope.selection = {("follow.distance", 1.0)}
    key(dope, Qt.Key.Key_Backspace)
    assert [k.time for k in dope.anim.curves["follow.distance"].keys] == [3.0]
    key(dope, Qt.Key.Key_X)  # (other keys: ignored)


def test_dope_sheet_context_menu_sets_interpolation_and_deletes(dope, monkeypatch):
    step = f"Interpolation › {INTERP_LABELS[Interp.STEP]}"
    shown = menu_choices(monkeypatch, None, step, "Paste at playhead", "Delete")
    press(dope, QPointF(700, 15), R)  # nothing selected: only the menu
    assert shown[-1]["Copy"] is False and shown[-1]["Delete"] is False
    assert shown[-1]["Paste at playhead"] is False  # (empty clipboard)
    press(dope, at(dope, "view.fov", 2.0), R)  # right-click selects the key under the mouse
    assert dope.anim.curves["view.fov"].key_at(2.0).interp is Interp.STEP
    dope.editor.copy(dope.selection)
    dope.editor.timeline.set_time(5.0)
    press(dope, at(dope, "view.fov", 2.0), R)
    assert dope.anim.curves["view.fov"].key_at(5.0) is not None
    assert dope.selection == {("view.fov", 5.0)}
    press(dope, at(dope, "view.fov", 5.0), R)
    assert dope.anim.curves["view.fov"].key_at(5.0) is None


def test_dope_sheet_paints_warnings_and_its_empty_state(qtbot):
    from earthling.ui.dope_sheet import DopeSheet

    scene, timeline, stack, editor = make_editor()
    sheet = DopeSheet(editor, lambda t: 200.0 + t * 50.0, lambda x: (x - 200.0) / 50.0)
    qtbot.addWidget(sheet)
    sheet.resize(600, 100)
    sheet.grab()  # (no animated settings: a hint)
    layers = [v for v, _ in scene.store.registry["layers.a"].options]
    scene.animation.set_key("layers.a", 0.0, layers[0])
    scene.animation.set_key("layers.a", 2.0, layers[1])  # a visible jump: drawn red
    editor.changed.emit()
    assert sheet.rows() == ["layers.a"] and sheet.minimumHeight() >= 60
    sheet.grab()
    assert [w[:2] for w in sheet._warnings] == [("layers.a", 2.0)]


# --- key button ------------------------------------------------------------------------------
def test_key_button_toggles_a_key_at_the_playhead(qtbot):
    from earthling.ui.dope_sheet import KeyButton

    scene, timeline, stack, editor = make_editor()
    button = KeyButton("view.fov", editor)
    qtbot.addWidget(button)
    button.grab()  # not animated: hollow grey
    timeline.set_time(1.0)
    button.click()
    assert scene.animation.has_key("view.fov", 1.0)
    button.grab()  # keyed: filled
    timeline.set_time(2.0)
    button.grab()  # animated, no key here: hollow orange
    button.click()
    button.click()  # (and off again)
    assert [k.time for k in scene.animation.curves["view.fov"].keys] == [1.0]
    stack.undo()
    stack.undo()
    stack.undo()
    assert not scene.animation.is_animated("view.fov")


# --- graph editor ----------------------------------------------------------------------------
@pytest.fixture
def graph(qtbot):
    from earthling.ui.graph_editor import GraphView

    scene, timeline, stack, editor = make_editor()
    anim = scene.animation
    for t, v in ((0.0, 30.0), (4.0, 70.0), (8.0, 50.0)):
        anim.set_key("view.fov", t, v)
    view = GraphView(editor)
    qtbot.addWidget(view)
    view.resize(900, 400)
    view.channels = ["view.fov"]
    view.frame_all()
    view.anim, view.stack = anim, stack
    return view


def gpoint(view, pid, t) -> QPointF:
    curve = view.curve(pid)
    k = curve.key_at(t)
    return view.point(pid, curve, t, curve.scalars()[curve.keys.index(k)])


def test_graph_view_box_selection_pan_zoom_and_scrub(graph):
    fov = graph.anim.curves["view.fov"]
    a, b = gpoint(graph, "view.fov", 0.0), gpoint(graph, "view.fov", 4.0)
    drag(graph, QPointF(a.x() - 10, min(a.y(), b.y()) - 10),
         QPointF(b.x() + 10, max(a.y(), b.y()) + 10))  # fmt: skip
    assert graph.selection == {("view.fov", 0.0), ("view.fov", 4.0)}
    press(graph, QPointF(gpoint(graph, "view.fov", 8.0)), mods=SHIFT)  # Shift adds
    release(graph, QPointF(gpoint(graph, "view.fov", 8.0)))
    assert len(graph.selection) == 3
    graph.grab()  # (handles of the selected keys are drawn)
    # middle drag (or Alt + left drag) pans the view
    t0, v0 = graph.t0, graph.v0
    drag(graph, QPointF(400, 200), QPointF(300, 250), M)
    assert graph.t0 > t0 and graph.v0 > v0
    t0 = graph.t0
    drag(graph, QPointF(300, 200), QPointF(400, 200), L, ALT)
    assert graph.t0 < t0 and len(graph.selection) == 3  # (panning keeps the selection)
    # the wheel zooms both axes, Ctrl the values only, Shift the time only
    for mods, time_zoomed, values_zoomed in ((NONE, True, True), (CTRL, False, True),
                                             (SHIFT, True, False)):  # fmt: skip
        span_t, span_v = graph.t1 - graph.t0, graph.v1 - graph.v0
        pos = QPointF(450, 200)
        graph.wheelEvent(QWheelEvent(pos, pos, QPoint(0, 0), QPoint(0, 120),
                                     Qt.MouseButton.NoButton, mods,
                                     Qt.ScrollPhase.NoScrollPhase, False))  # fmt: skip
        assert (graph.t1 - graph.t0 < span_t) == time_zoomed, mods
        assert (graph.v1 - graph.v0 < span_v) == values_zoomed, mods
    # Home frames all, F the selection
    key(graph, Qt.Key.Key_Home)
    full = graph.t1 - graph.t0
    graph.selection = {("view.fov", 4.0)}
    key(graph, Qt.Key.Key_F)
    assert graph.t1 - graph.t0 < full
    # the time axis below the plot scrubs the playhead
    axis_y = graph.plot_rect().bottom() + 8
    press(graph, QPointF(graph.tx(2.0), axis_y))
    move(graph, QPointF(graph.tx(3.0), axis_y))
    release(graph, QPointF(graph.tx(3.0), axis_y))
    assert graph.editor.timeline.time == pytest.approx(3.0, abs=0.06)
    double_click(graph, gpoint(graph, "view.fov", 8.0))
    assert graph.editor.timeline.time == 8.0
    assert [k.time for k in fov.keys] == [0.0, 4.0, 8.0]  # (nothing moved)


def test_graph_view_shift_constrains_drags_and_drops_replace_keys(graph):
    def fov():
        return graph.anim.curves["view.fov"]  # (undo replaces the curve)

    def constrained(start, end):
        """Press a key, then hold Shift while dragging (Shift on the press toggles it)."""
        press(graph, start)
        move(graph, QPointF((start.x() + end.x()) / 2, (start.y() + end.y()) / 2), mods=SHIFT)
        move(graph, end, mods=SHIFT)
        release(graph, end, mods=SHIFT)

    start = gpoint(graph, "view.fov", 4.0)
    # mostly sideways with Shift: only the time changes
    constrained(start, QPointF(graph.tx(5.0), start.y() - 30))
    assert [k.time for k in fov().keys] == [0.0, 5.0, 8.0] and fov().key_at(5.0).value == 70.0
    # mostly up with Shift: only the value changes
    start = gpoint(graph, "view.fov", 5.0)
    constrained(start, QPointF(start.x() + 5, start.y() - 60))
    assert fov().key_at(5.0) is not None and fov().key_at(5.0).value > 70.0
    assert graph.stack.count() == 2
    # dropped onto another key: it replaces that key
    drag(graph, gpoint(graph, "view.fov", 5.0), QPointF(graph.tx(8.0), start.y() - 60))
    assert [k.time for k in fov().keys] == [0.0, 8.0] and fov().keys[1].value > 70.0
    graph.stack.undo()
    assert [k.time for k in fov().keys] == [0.0, 5.0, 8.0]
    # Ctrl+A, then Delete
    shortcut(graph, QKeySequence.StandardKey.SelectAll)
    assert len(graph.selection) == 3
    key(graph, Qt.Key.Key_Delete)
    assert "view.fov" not in graph.anim.curves  # (the last key took the curve along)
    graph.grab()  # (no curves: a hint)


def test_graph_view_context_menu(graph, monkeypatch):
    shown = menu_choices(monkeypatch, None, "Handles › Vector", "Interpolation › Linear",
                         "Frame selected (F)", "Delete")  # fmt: skip
    press(graph, QPointF(450, 30), R)  # nothing selected
    assert shown[-1]["Frame all (Home)"] and not shown[-1]["Delete"]
    assert not shown[-1]["Handles › Vector"]
    press(graph, gpoint(graph, "view.fov", 4.0), R)  # selects the key under the mouse
    key4 = graph.anim.curves["view.fov"].key_at(4.0)
    assert graph.selection == {("view.fov", 4.0)} and key4.handle_mode is HandleMode.VECTOR
    press(graph, gpoint(graph, "view.fov", 4.0), R)
    assert graph.anim.curves["view.fov"].key_at(4.0).interp is Interp.LINEAR
    full = graph.t1 - graph.t0
    press(graph, gpoint(graph, "view.fov", 4.0), R)
    assert graph.t1 - graph.t0 < full  # framed around the selected key
    press(graph, gpoint(graph, "view.fov", 4.0), R)
    assert graph.anim.curves["view.fov"].key_at(4.0) is None


def test_graph_editor_channels_buttons_and_datetime_axis(qtbot):
    from earthling.ui.graph_editor import GraphEditor

    scene, timeline, stack, editor = make_editor()
    anim = scene.animation
    widget = GraphEditor(editor)
    qtbot.addWidget(widget)
    widget.resize(800, 300)
    assert widget.channel_list.count() == 0
    widget.view.grab()  # (nothing to show: a hint)
    when = datetime(2026, 7, 1, 8, 0, tzinfo=UTC)
    anim.set_key("sun.datetime", 0.0, when)
    anim.set_key("sun.datetime", 6.0, when.replace(hour=18))
    editor.changed.emit()  # a first curve: shown and framed
    assert widget.view.channels == ["sun.datetime"]
    assert widget.view._datetime_axis() is not None  # the value axis shows clock times
    widget.view.grab()
    anim.set_key("view.fov", 0.0, 30.0)
    anim.set_key("view.fov", 6.0, 60.0)
    editor.changed.emit()
    assert widget.view._datetime_axis() is None  # (mixed channels: plain numbers)
    widget.channel_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    hidden = widget.channel_list.item(0).data(Qt.ItemDataRole.UserRole)
    anim.set_key("view.fov", 3.0, 45.0)
    anim.set_key("follow.distance", 0.0, 500.0)
    editor.changed.emit()  # (rebuilt: the hidden channel stays hidden)
    assert hidden not in widget.view.channels and len(widget.view.channels) == 2
    editor.changed.emit()  # (no new channel: nothing rebuilt)
    # the handle buttons act on the selected keys
    widget.view.selection = {("view.fov", 3.0)}
    widget.buttons["handles_free"].click()
    assert anim.curves["view.fov"].key_at(3.0).handle_mode is HandleMode.FREE
    widget.buttons["frame_sel"].click()
    widget.buttons["normalize"].click()
    assert widget.view.normalized
    widget.view.grab()
