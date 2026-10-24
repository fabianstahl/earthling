"""The inspector's editors as the user works them: drag fields (Shift fine, Ctrl coarse, typing
with Enter/F2, arrow keys), lists, corners, colours, dates, texts and vectors; resetting a row,
copying and pasting a section; values shown as text."""

from datetime import datetime

import pytest
from PyQt6.QtCore import QDateTime, QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QColor, QGuiApplication, QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import QApplication, QColorDialog, QWidget

from earthling.core.properties import (
    PropertyDef,
    PropertyRegistry,
    PropertyStore,
    PType,
    boolean,
    color,
    enum,
    flt,
    integer,
    section,
    shown_if,
)
from earthling.ui.property_panel import DragField, PropertyPanel, format_value, retire
from helpers import menu_choices

L = Qt.MouseButton.LeftButton
NONE, SHIFT, CTRL = (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ShiftModifier,
                     Qt.KeyboardModifier.ControlModifier)  # fmt: skip
WHEN = datetime(2026, 7, 1, 12, 0)
CORNERS = [("top_left", "Top left"), ("top_right", "Top right"), ("bottom_left", "Bottom left"),
           ("bottom_right", "Bottom right")]  # fmt: skip
STYLES = [("soft", "Soft and warm"), ("hard", "Hard and cold"), ("flat", "Flat"), ("neon", "Neon")]
DEFS = section(
    "Look",
    flt("look.gain", "Gain", 1.0, 0.0, 10.0, step=0.1, decimals=1),
    integer("look.count", "Count", 3, 0, 9),
    enum("look.style", "Style", "soft", STYLES),
    enum("look.corner", "Corner", "top_left", CORNERS),
    color("look.tint", "Tint", (1.0, 0.5, 0.0)),
    PropertyDef("look.when", "When", PType.DATETIME, WHEN, "Look"),
    PropertyDef("look.title", "Title", PType.TEXT, "", "Look"),
    PropertyDef("look.offset", "Offset", PType.VEC3, (0.0, 0.0, 0.0), "Look", decimals=1),
) + section(
    "Hidden",
    boolean("hidden.on", "On", False),
    flt("hidden.value", "Value", 0.5, 0.0, 1.0, when=shown_if("hidden.on")),
)


def drag_field(qtbot, field, dx, mods=NONE):
    """Press in the middle of ``field``, move ``dx`` pixels sideways, release."""
    start = QPointF(field.width() / 2, 12)
    end = QPointF(start.x() + dx, 12)
    for kind, pos, button in ((QEvent.Type.MouseButtonPress, start, L),
                              (QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton),
                              (QEvent.Type.MouseButtonRelease, end, L)):  # fmt: skip
        QApplication.sendEvent(field, QMouseEvent(kind, pos, pos, button, L, mods))


def test_drag_field_gains_typing_and_steps(qtbot):
    field = DragField(0.0, 10.0, step=0.1, decimals=2)
    qtbot.addWidget(field)
    field.resize(200, 24)  # (dragging spans at least 160 px: here 200)
    edits = []
    field.edited.connect(lambda v, interactive: edits.append((v, interactive)))
    field.set_value(5.0)
    drag_field(qtbot, field, 20)  # a tenth of the width: a tenth of the range
    assert field.value == pytest.approx(6.0)
    drag_field(qtbot, field, 20, SHIFT)  # Shift: ten times finer
    assert field.value == pytest.approx(6.1)
    drag_field(qtbot, field, -20, CTRL)  # Ctrl: five times coarser
    assert field.value == pytest.approx(1.1)
    assert edits[-1] == (pytest.approx(1.1), False)  # (the release reports a final value)
    drag_field(qtbot, field, 2)  # (a tiny wobble is a click: it starts typing)
    assert field._edit is not None and field.value == pytest.approx(1.1)
    qtbot.keyClicks(field._edit, "abc")
    qtbot.keyClick(field._edit, Qt.Key.Key_Return)  # not a number: unchanged
    assert field._edit is None and field.value == pytest.approx(1.1)
    for start_key in (Qt.Key.Key_Return, Qt.Key.Key_F2):  # Enter or F2 types a value
        qtbot.keyClick(field, start_key)
        field._edit.selectAll()
        qtbot.keyClicks(field._edit, "-2,5")  # (a decimal comma is fine)
        qtbot.keyClick(field._edit, Qt.Key.Key_Return)
        assert field.value == 0.0  # clamped to the range
    field.set_value(5.0)
    qtbot.keyClick(field, Qt.Key.Key_Right)
    qtbot.keyClick(field, Qt.Key.Key_Down)
    qtbot.keyClick(field, Qt.Key.Key_Down)
    assert field.value == pytest.approx(4.9)
    wheel = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, 120),
                        Qt.MouseButton.NoButton, NONE, Qt.ScrollPhase.NoScrollPhase,
                        False)  # fmt: skip
    QApplication.sendEvent(field, wheel)
    assert not wheel.isAccepted() and field.value == pytest.approx(4.9)  # (the panel scrolls)
    for state in ("animated", "keyed", "static"):  # the bar tints animated values
        field.set_state(state)
        field.grab()
    field.setEnabled(False)
    field.grab()


def test_enter_finishes_typing(qtbot):
    """Regression: Enter committed the typed value, then went on to the drag field as "type a
    value" and opened a new typing field at once (typing could not be left by keyboard)."""
    field = DragField(0.0, 10.0)
    qtbot.addWidget(field)
    field.show()
    qtbot.keyClick(field, Qt.Key.Key_Return)
    qtbot.keyClicks(field._edit, "3")
    qtbot.keyClick(field._edit, Qt.Key.Key_Return)
    assert field.value == 3.0 and field._edit is None


def test_drag_field_unbounded_logarithmic_and_integer(qtbot):
    free = DragField(None, None, step=0.5, decimals=1, unit="m")
    qtbot.addWidget(free)
    free.resize(200, 24)
    assert not free.bounded and free.fraction() == 0.0
    drag_field(qtbot, free, 10)  # (no range: a step per pixel)
    assert free.value == 5.0 and free.text() == "5.0 m"
    log = DragField(1.0, 1000.0, logarithmic=True)
    qtbot.addWidget(log)
    log.set_value(100.0)
    assert log.fraction() == pytest.approx(2 / 3)
    qtbot.keyClick(log, Qt.Key.Key_Up)
    assert log.value == pytest.approx(105.0)
    count = DragField(0, 9, integer=True)
    qtbot.addWidget(count)
    count.commit(3.7)
    assert count.value == 4.0 and count.text() == "4"


@pytest.fixture
def panel(qtbot):
    store = PropertyStore(PropertyRegistry(DEFS))
    edits = []

    def on_edit(pid, value, interactive):
        edits.append((pid, value))
        store.set(pid, value)

    batches = []

    def batch(label, fn):
        batches.append(label)
        fn()

    view = PropertyPanel(store, on_edit, batch=batch)
    qtbot.addWidget(view)
    view.resize(500, 900)
    view.show()
    view.edits, view.batches = edits, batches
    return view


def test_every_kind_of_editor_edits_its_value(panel, qtbot, monkeypatch):
    store, editors = panel.store, panel.editors
    editors["look.count"].commit(7.4)
    assert store["look.count"] == 7 and isinstance(store["look.count"], int)
    style = editors["look.style"]  # (long choices: a list, not segments)
    style.combo.activated.emit(2)
    assert store["look.style"] == "flat"
    store.set("look.style", "neon")
    assert style.combo.currentText() == "Neon"
    corner = editors["look.corner"]
    corner.group.button(3).click()
    assert store["look.corner"] == "bottom_right" and corner.name.text() == "Bottom right"
    tint = editors["look.tint"]
    assert tint.button.text() == "#FF8000"
    monkeypatch.setattr(QColorDialog, "getColor", lambda *a, **k: QColor())  # (cancelled)
    tint.button.click()
    assert store["look.tint"] == (1.0, 0.5, 0.0)
    monkeypatch.setattr(QColorDialog, "getColor", lambda *a, **k: QColor(0, 0, 255))
    tint.button.click()
    assert store["look.tint"] == (0.0, 0.0, 1.0) and tint.button.text() == "#0000FF"
    when = editors["look.when"]
    when.edit.setDateTime(QDateTime(datetime(2026, 7, 2, 18, 30)))
    assert store["look.when"] == datetime(2026, 7, 2, 18, 30)
    title = editors["look.title"]
    title.edit.setFocus()
    qtbot.keyClicks(title.edit, "Day one")
    qtbot.keyClick(title.edit, Qt.Key.Key_Return)
    assert store["look.title"] == "Day one"
    offset = editors["look.offset"]
    offset.spins[1].setValue(12.5)
    assert store["look.offset"] == (0.0, 12.5, 0.0)
    store.set("look.offset", (1.0, 2.0, 3.0))  # the editors follow the store
    assert [s.value() for s in offset.spins] == [1.0, 2.0, 3.0]
    assert panel.rows["look.offset"].changed and not panel.rows["look.gain"].changed


def test_rows_reset_and_sections_copy_and_paste(panel, monkeypatch):
    store = panel.store
    store.set("look.gain", 4.0)
    shown = menu_choices(monkeypatch, "Reset to the default", "Copy the setting's id", None)
    row = panel.rows["look.gain"]
    row.customContextMenuRequested.emit(QPoint(10, 10))
    assert store["look.gain"] == 1.0 and not row.changed
    row.customContextMenuRequested.emit(QPoint(10, 10))
    assert QGuiApplication.clipboard().text() == "look.gain"
    row.customContextMenuRequested.emit(QPoint(10, 10))
    assert shown[-1]["Reset to the default"] is False  # (already the default)
    row.flash()  # (a searched row lights up for a moment)
    assert row.property("flash") is True
    row._flash_off()
    # a section's values go to the clipboard as JSON and come back
    look = panel.sections["Look"]
    store.set("look.gain", 2.5)
    store.set("look.style", "hard")
    look.copy_values()
    store.set("look.gain", 9.0)
    store.set("look.style", "flat")
    look.paste_values()
    assert store["look.gain"] == 2.5 and store["look.style"] == "hard"
    assert panel.batches[-1] == "Paste Look"
    QGuiApplication.clipboard().setText('{"look.gain": "x", "look.count": 5, "other.id": 1}')
    look.paste_values()  # (bad values and unknown settings are skipped)
    assert store["look.gain"] == 2.5 and store["look.count"] == 5
    for text in ("not json", "[1, 2]"):
        QGuiApplication.clipboard().setText(text)
        look.paste_values()  # (ignored)
    look.reset_all()
    assert store["look.gain"] == 1.0 and store["look.count"] == 3 and store["look.style"] == "soft"
    # a section whose rows are all hidden by their conditions is hidden itself
    hidden = panel.sections["Hidden"]
    assert hidden.isVisible() and not panel.rows["hidden.value"].isVisible()
    assert hidden.depends_on("hidden.on") and not hidden.depends_on("look.gain")


def test_values_as_text_and_retiring_a_view(qtbot):
    d = {x.id: x for x in DEFS}
    assert format_value(d["look.gain"], 2.26) == "2.3"
    assert format_value(d["look.count"], 4) == "4"
    assert format_value(d["hidden.on"], True) == "On" and format_value(d["hidden.on"], 0) == "Off"
    assert format_value(d["look.style"], "hard") == "Hard and cold"
    assert format_value(d["look.tint"], (0.0, 1.0, 0.0)) == "#00FF00"
    assert format_value(d["look.when"], WHEN) == "2026-07-01 12:00"
    assert format_value(d["look.title"], "") == "–" and format_value(d["look.title"], "A") == "“A”"
    assert format_value(d["look.offset"], (1, 2, 3)) == ""
    # a view with the focus inside is deleted without reporting edits on the way out
    view = QWidget()
    qtbot.addWidget(view)
    field = DragField(0.0, 1.0, parent=view)
    view.show()
    field.setFocus()
    edits = []
    field.edited.connect(lambda *a: edits.append(a))
    retire(view)
    retire(None)
    assert not view.isVisible() and not field.hasFocus() and field.signalsBlocked()
    assert edits == []
