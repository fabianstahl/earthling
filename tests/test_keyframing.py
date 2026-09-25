import pytest
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QMouseEvent, QUndoStack

from earthling.app.keyframing import KeyframeEditor
from earthling.app.timeline import TimelineController
from earthling.core.animation import Interp
from earthling.core.scene import Scene


def make(qapp):
    scene = Scene()
    scene.animation.duration, scene.animation.fps = 10.0, 10.0
    timeline = TimelineController(scene.animation)
    stack = QUndoStack()
    return scene, timeline, stack, KeyframeEditor(scene.animation, timeline, stack)


def test_toggle_key_and_undo(qapp):
    scene, tl, stack, ed = make(qapp)
    tl.set_time(1.0)
    ed.toggle_key("view.fov")
    assert scene.animation.has_key("view.fov", 1.0) and ed.has_key_now("view.fov")
    ed.toggle_key("view.fov")
    assert not scene.animation.is_animated("view.fov")
    stack.undo()
    assert scene.animation.has_key("view.fov", 1.0)
    stack.undo()
    assert not scene.animation.is_animated("view.fov")
    stack.redo()
    assert scene.animation.has_key("view.fov", 1.0)


def test_keys_drive_values_and_auto_key(qapp):
    scene, tl, stack, ed = make(qapp)
    ed.set_key("view.fov", 30.0)
    tl.set_time(4.0)
    ed.set_key("view.fov", 70.0)
    tl.set_time(2.0)
    assert scene.store["view.fov"] == pytest.approx(50.0)
    # auto key: editing an animated property creates a key at the playhead
    assert ed.on_property_edited("view.fov", 20.0, interactive=False)
    assert scene.animation.curves["view.fov"].key_at(2.0).value == 20.0
    assert not ed.on_property_edited("terrain.exaggeration", 2.0, interactive=False)


def test_move_delete_interp_copy_paste(qapp):
    scene, tl, stack, ed = make(qapp)
    for t, v in ((0.0, 10.0), (1.0, 20.0), (2.0, 30.0)):
        tl.set_time(t)
        ed.set_key("view.fov", v)
    moved = ed.move_keys([("view.fov", 1.0)], 0.52)  # snapped to 0.5 s steps of 0.1
    assert moved == [("view.fov", 1.5)]
    # moving onto an existing key replaces it
    ed.move_keys([("view.fov", 1.5)], 0.5)
    times = [k.time for k in scene.animation.curves["view.fov"].keys]
    assert times == [0.0, 2.0]
    ed.set_interpolation([("view.fov", 0.0)], Interp.EASE_IN)
    assert scene.animation.curves["view.fov"].keys[0].interp is Interp.EASE_IN
    ed.copy([("view.fov", 0.0), ("view.fov", 2.0)])
    tl.set_time(5.0)
    pasted = ed.paste()
    assert pasted == [("view.fov", 5.0), ("view.fov", 7.0)]
    ed.delete_keys(pasted)
    assert [k.time for k in scene.animation.curves["view.fov"].keys] == [0.0, 2.0]
    for _ in range(5):  # delete, paste, interpolation, move, move
        stack.undo()
    assert [k.time for k in scene.animation.curves["view.fov"].keys] == [0.0, 1.0, 2.0]


def press(widget, x, y, button=Qt.MouseButton.LeftButton, mods=Qt.KeyboardModifier.NoModifier):
    pos = QPointF(x, y)
    widget.mousePressEvent(
        QMouseEvent(QMouseEvent.Type.MouseButtonPress, pos, pos, button, button, mods)
    )


def release(widget, x, y):
    pos = QPointF(x, y)
    b = Qt.MouseButton.LeftButton
    widget.mouseReleaseEvent(
        QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            pos,
            pos,
            b,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )  # fmt: skip
    )


def move(widget, x, y):
    pos = QPointF(x, y)
    b = Qt.MouseButton.LeftButton
    widget.mouseMoveEvent(
        QMouseEvent(
            QMouseEvent.Type.MouseMove,
            pos,
            pos,
            Qt.MouseButton.NoButton,
            b,
            Qt.KeyboardModifier.NoModifier,
        )  # fmt: skip
    )


def test_dope_sheet_select_drag_and_delete(qtbot):
    from earthling.ui.timeline_widget import TimelineWidget

    scene, tl, stack, ed = make(None)
    for t in (1.0, 3.0):
        tl.set_time(t)
        ed.set_key("view.fov")
        ed.set_key("sun.intensity")
    widget = TimelineWidget(tl, ed)
    qtbot.addWidget(widget)
    widget.resize(900, 300)
    widget.show()
    sheet = widget.dope_sheet
    rows = sheet.rows()
    assert rows == ["view.fov", "sun.intensity"]  # registry order
    x1 = sheet.time_to_x(1.0)
    y0 = sheet.row_y(0)
    press(sheet, x1, y0)
    assert sheet.selection == {("view.fov", 1.0)}
    move(sheet, sheet.time_to_x(2.0), y0)
    release(sheet, sheet.time_to_x(2.0), y0)
    assert [k.time for k in scene.animation.curves["view.fov"].keys] == [2.0, 3.0]
    assert sheet.selection == {("view.fov", 2.0)}
    # rubber band over the second row
    press(sheet, sheet.time_to_x(0.5), sheet.row_y(1) - 5)
    move(sheet, sheet.time_to_x(3.5), sheet.row_y(1) + 5)
    release(sheet, sheet.time_to_x(3.5), sheet.row_y(1) + 5)
    assert sheet.selection == {("sun.intensity", 1.0), ("sun.intensity", 3.0)}
    sheet.delete_selected()
    assert not scene.animation.is_animated("sun.intensity")
    stack.undo()
    assert scene.animation.is_animated("sun.intensity")


def test_window_key_buttons_and_auto_key(qtbot):
    from earthling.app.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    try:
        window.timeline.set_time(0.0)
        editor = window.property_panel.editors["view.fov"]
        grid = window.property_panel.sections["View"].grid
        row = next(
            r for r in range(grid.rowCount()) if grid.itemAtPosition(r, 1).widget() is editor
        )
        grid.itemAtPosition(row, 2).widget().click()
        assert window.scene.animation.is_animated("view.fov")
        window.timeline.set_time(2.0)
        editor.spin.setValue(80.0)  # auto-key
        assert window.scene.animation.curves["view.fov"].key_at(2.0).value == 80.0
        window.timeline.set_time(1.0)
        assert window.scene.store["view.fov"] == pytest.approx(65.0)
        assert window.scene.dirty
    finally:
        window.undo_stack.setClean()  # no "unsaved changes" dialog when the test closes it
        window.scene.mark_dirty(False)
