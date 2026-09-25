from PyQt6.QtGui import QUndoStack

from earthling.app.undo import set_property
from earthling.core.scene import Scene


def test_undo_redo_and_drag_merging(qapp):
    scene = Scene()
    store = scene.store
    stack = QUndoStack()
    set_property(stack, store, "terrain.exaggeration", 2.0)
    # a slider drag: many interactive updates followed by the final value
    for v in (2.1, 2.2, 2.3):
        set_property(stack, store, "terrain.exaggeration", v, interactive=True)
    set_property(stack, store, "terrain.exaggeration", 2.3, interactive=False)
    set_property(stack, store, "light.azimuth", 10.0)
    assert stack.count() == 3
    stack.undo()
    assert store["light.azimuth"] == 315.0
    stack.undo()
    assert store["terrain.exaggeration"] == 2.0
    stack.undo()
    assert store["terrain.exaggeration"] == 1.0
    stack.redo()
    stack.redo()
    assert store["terrain.exaggeration"] == 2.3
    # a new drag after the finished one is a separate command
    set_property(stack, store, "terrain.exaggeration", 3.0, interactive=True)
    assert stack.count() == 3  # redo tail (azimuth) dropped, new command added


def test_no_op_edits_are_not_recorded(qapp):
    scene = Scene()
    stack = QUndoStack()
    set_property(stack, scene.store, "view.fov", 50.0)
    assert stack.count() == 0


def test_window_undo_marks_clean_again(qtbot):
    from earthling.app.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.set_property("view.fov", 30.0)
    assert window.scene.dirty
    window.undo_stack.undo()
    assert window.scene.store["view.fov"] == 50.0
    assert not window.scene.dirty
    # panel edits are undoable too
    window.property_panel.editors["view.fov"].spin.setValue(40.0)
    assert window.undo_stack.count() == 1
    window.undo_stack.undo()
    assert window.property_panel.editors["view.fov"].spin.value() == 50.0
