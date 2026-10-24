"""A monkey in the interface (slow suite): thousands of random steps a user could take -
menu actions, buttons, keys, the scene list, the inspector, the palette, value drags, the
timeline, clicks, drags and the wheel in the 3D view, undo and redo - on an open project.
The dialogs and popup menus it opens are filled in at random and accepted, or cancelled;
file dialogs offer GPX files (a broken one too), photos, scenes and folders. Nothing may
raise or freeze the window for long; the scene stays saveable. The steps come from a fixed
seed: a failure repeats, and its message lists the steps that led there."""

import json
import random
import re
import shutil
from pathlib import Path

import pytest
from PyQt6.QtCore import QPoint, QPointF, Qt, QTimer
from PyQt6.QtGui import QColor, QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QInputDialog,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
)

from earthling.data import net
from earthling.ui.inspector import CATEGORIES
from earthling.ui.property_panel import DragField
from helpers import gpx_text, longest_freeze, menu_choices, walk
from test_photo_editor import photo_file
from test_ui_actions import close_dialogs, menu_actions, window  # noqa: F401 (the fixture)

pytestmark = pytest.mark.slow
STEPS = 1500
NEVER = ("Quit", "Start Screen", "Open Recent")  # (ends the session under test)
KEYS = [Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Return,
        Qt.Key.Key_Escape, Qt.Key.Key_Tab, Qt.Key.Key_Delete, Qt.Key.Key_Space, Qt.Key.Key_F2,
        Qt.Key.Key_A, Qt.Key.Key_K, Qt.Key.Key_C, Qt.Key.Key_T, Qt.Key.Key_W, Qt.Key.Key_5,
        Qt.Key.Key_Home, Qt.Key.Key_Comma, Qt.Key.Key_Period, Qt.Key.Key_F6]  # fmt: skip
MODS = [Qt.KeyboardModifier.NoModifier] * 4 + [Qt.KeyboardModifier.ShiftModifier,
                                               Qt.KeyboardModifier.ControlModifier,
                                               Qt.KeyboardModifier.AltModifier]  # fmt: skip
TEXTS = ["", "Trient", "Zermatt", "46.05, 7.0", "1.5", "-3", "abc", "1e9"]
NO_BUTTON = Qt.MouseButton.NoButton


def visible(widgets):
    return [w for w in widgets if w.isVisible() and w.isEnabled()]


class Files:
    """Stands in for the file dialogs: GPX files (a broken one too), photos with and without a
    position, a broken image, scenes (a broken one too), a new or the project folder, a place
    to save to - or nothing, as if cancelled."""

    def __init__(self, folder: Path, project: Path, rng: random.Random, done: list) -> None:
        self.rng, self.project, self.folder = rng, project, folder
        self.done = done  # (what was chosen, for the failure message)
        files = folder / "files"
        files.mkdir()
        start = "2026-07-01T09:00:00+00:00"  # (during the demo's first day)
        (files / "near.gpx").write_text(gpx_text({"Near": walk(45.95, 6.95, start, 40, 60.0)}))
        (files / "untimed.gpx").write_text(gpx_text({"Untimed": walk(46.0, 7.0, None, 30,
                                                                     east_m=50.0)}))  # fmt: skip
        fixtures = Path(__file__).parent / "fixtures"
        shutil.copy(fixtures / "two_tracks.gpx", files / "two_tracks.gpx")
        shutil.copy(fixtures / "broken.gpx.txt", files / "broken.gpx")
        self.gpx = sorted(files.glob("*.gpx"))
        self.photos = [
            photo_file(files / "summit.jpg", (45.95, 6.95), "2026:07:01 11:00:00"),
            photo_file(files / "timed.jpg", taken="2026:07:02 10:00:00"),
            photo_file(files / "plain.png"),
        ]
        (files / "broken.png").write_bytes(b"not an image")
        self.images = [files / "plain.png", files / "broken.png"]
        (files / "broken.json").write_text("{")
        (files / "list.json").write_text("[]")
        self.scenes = [*project.glob("*.json"), files / "broken.json", files / "list.json"]
        self.count = 0

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (self.one(a), ""))
        monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *a, **k: (self.many(a), ""))
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (self.save(a), ""))
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: self.dir(a))
        monkeypatch.setattr(QColorDialog, "getColor", lambda *a, **k: self.colour())
        monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: self.number())
        answers = [QMessageBox.StandardButton.Save, QMessageBox.StandardButton.Discard,
                   QMessageBox.StandardButton.Cancel]  # fmt: skip
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: self.rng.choice(answers))

    def _pool(self, args) -> list[Path]:
        text = " ".join(str(a) for a in args[1:])
        if "GPX" in text:
            return self.gpx
        if "Scene" in text:
            return self.scenes
        return self.photos + self.images

    def _chose(self, args, chosen):
        names = [Path(c).name for c in chosen] if isinstance(chosen, list) else Path(chosen).name
        self.done.append(f"{args[1]!r}: {names or 'cancelled'}")
        return chosen

    def one(self, args) -> str:
        if self.rng.random() < 0.3:
            return self._chose(args, "")
        return self._chose(args, str(self.rng.choice(self._pool(args))))

    def many(self, args) -> list[str]:
        if self.rng.random() < 0.3:
            return self._chose(args, [])
        pool = self._pool(args)
        picks = self.rng.sample(pool, self.rng.randint(1, min(2, len(pool))))
        return self._chose(args, [str(p) for p in picks])

    def save(self, args) -> str:
        if self.rng.random() < 0.3:
            return self._chose(args, "")
        self.count += 1
        suffix = re.search(r"\*(\.\w+)", " ".join(str(a) for a in args[1:]))
        name = f"saved_{self.count}{suffix.group(1) if suffix else ''}"
        return self._chose(args, str(self.folder / name))

    def dir(self, args) -> str:
        if self.rng.random() < 0.3:
            return self._chose(args, "")
        if "new project" in str(args[1]):
            self.count += 1
            return self._chose(args, str(self.folder / f"project_{self.count}"))
        return self._chose(args, str(self.project))

    def colour(self) -> QColor:
        return QColor() if self.rng.random() < 0.3 else QColor(*self.rng.choices(range(256), k=3))

    def number(self):
        return (None, False) if self.rng.random() < 0.3 else (self.rng.uniform(0.0, 5.0), True)


class Dialogs:
    """What the monkey does with a dialog or popup menu that opened. They run their own event
    loop, so a timer looks at them. The first time: change some of the dialog's fields, then
    press what Enter would press, or cancel; pick an entry of a popup menu. Still open on the
    next look: closed (an export that started is cancelled). Clicks are posted: Qt does not run
    this timer again while it runs, and a click can open the next dialog."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.done: list[str] = []  # (for the failure message)
        self.timer = QTimer()
        self.timer.timeout.connect(self.look)

    def look(self) -> None:
        popup = QApplication.activePopupWidget()
        if popup is not None:
            popup.close()
            if isinstance(popup, QMenu) and not popup.property("monkey"):
                popup.setProperty("monkey", True)
                actions = [
                    a
                    for a in popup.actions()
                    if a.isEnabled()
                    and not a.isSeparator()
                    and a.menu() is None
                    and not any(n in a.text() for n in NEVER)
                ]
                if actions and self.rng.random() < 0.7:
                    action = self.rng.choice(actions)
                    self.done.append(f"popup entry {action.text()!r}")
                    QTimer.singleShot(0, action.trigger)
            return
        dialog = QApplication.activeModalWidget() or next(
            (w for w in QApplication.topLevelWidgets() if isinstance(w, QDialog) and w.isVisible()),
            None,
        )
        if dialog is None:
            return
        seen = dialog.property("monkey") or 0
        dialog.setProperty("monkey", seen + 1)
        name = type(dialog).__name__
        if seen == 0 and self.rng.random() < 0.65:
            self.fill(dialog)
            button = self.enter_button(dialog)
            if button is not None:
                self.done.append(f"{name}: {button.text()!r}")
                QTimer.singleShot(0, button.click)
                return
        if seen >= 3:
            dialog.hide()  # (refused to close)
        elif isinstance(dialog, QDialog):
            dialog.reject()
        else:
            dialog.close()

    def fill(self, dialog) -> None:
        fields = [w for w in visible(dialog.findChildren((QComboBox, QAbstractSpinBox,
                                                          QCheckBox, QLineEdit)))
                  if not isinstance(w.parent(), (QAbstractSpinBox, QComboBox))]  # fmt: skip
        for field in self.rng.sample(fields, min(len(fields), self.rng.randint(0, 3))):
            if isinstance(field, QComboBox) and field.count():
                field.setCurrentIndex(self.rng.randrange(field.count()))
            elif isinstance(field, QCheckBox):
                field.toggle()
            elif isinstance(field, (QSpinBox, QDoubleSpinBox)):
                low, high = field.minimum(), field.maximum()
                value = low + (high - low) * self.rng.random()
                field.setValue(int(value) if isinstance(field, QSpinBox) else value)
            elif isinstance(field, QAbstractSpinBox):  # (a time or a date)
                field.stepBy(self.rng.randint(-50, 50))
            elif isinstance(field, QLineEdit) and not re.search(r"[/\\:]", field.text()):
                field.setText(self.rng.choice(TEXTS))  # (never a file path: it stays put)

    @staticmethod
    def enter_button(dialog) -> QAbstractButton | None:
        """The default button, else the box's accepting one (OK, Save, Yes)."""
        for button in visible(dialog.findChildren(QPushButton)):
            if button.isDefault():
                return button
        accepting = (QDialogButtonBox.ButtonRole.AcceptRole, QDialogButtonBox.ButtonRole.YesRole)
        for box in dialog.findChildren(QDialogButtonBox):
            for button in visible(box.buttons()):
                if box.buttonRole(button) in accepting:
                    return button
        return None


def mouse(widget, kind, pos: QPointF, button, buttons) -> None:
    global_pos = QPointF(widget.mapToGlobal(pos.toPoint()))
    event = QMouseEvent(kind, pos, global_pos, button, buttons, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, event)


def view_step(viewport, rng: random.Random) -> str:
    """A click, drag, double-click or turn of the wheel in the 3D view."""
    width, height = viewport.width(), viewport.height()

    def anywhere() -> QPointF:
        return QPointF(rng.uniform(0, width), rng.uniform(0, height))

    gesture = rng.choice(["click", "drag", "drag", "double", "wheel"])
    at = anywhere()
    if gesture == "wheel":
        delta = QPoint(0, rng.choice([-360, -120, 120, 360]))
        event = QWheelEvent(at, QPointF(viewport.mapToGlobal(at.toPoint())), QPoint(), delta,
                            NO_BUTTON, Qt.KeyboardModifier.NoModifier,
                            Qt.ScrollPhase.NoScrollPhase, False)  # fmt: skip
        QApplication.sendEvent(viewport, event)
        return "view wheel"
    button = Qt.MouseButton.LeftButton
    if gesture == "drag":
        button = rng.choice([button, button, Qt.MouseButton.RightButton,
                             Qt.MouseButton.MiddleButton])  # fmt: skip
    kind = QMouseEvent.Type
    mouse(viewport, kind.MouseButtonPress, at, button, button)
    if gesture == "drag":
        end = anywhere()
        for k in range(1, 6):
            mouse(viewport, kind.MouseMove, at + (end - at) * (k / 5), NO_BUTTON, button)
        at = end
    mouse(viewport, kind.MouseButtonRelease, at, button, NO_BUTTON)
    if gesture == "double":
        mouse(viewport, kind.MouseButtonDblClick, at, button, button)
        mouse(viewport, kind.MouseButtonRelease, at, button, NO_BUTTON)
    return f"view {gesture} ({button.name})"


def step(window, qtbot, rng: random.Random) -> str:  # noqa: F811
    """One random step; returns what it did (for the failure message)."""
    kind = rng.choice(["action", "button", "key", "row", "category", "palette", "drag",
                       "time", "play", "undo", "redo", "view", "view"])  # fmt: skip
    if kind == "action":
        actions = [(p, a) for p, a in menu_actions(window.menuBar())
                   if not any(n in p for n in NEVER)]  # fmt: skip
        path, action = rng.choice(actions)
        action.trigger()
        return f"menu {path}"
    if kind == "button":
        buttons = [b for b in visible(window.findChildren(QAbstractButton))
                   if "Quit" not in b.toolTip()]  # fmt: skip
        button = rng.choice(buttons)
        button.click()
        return f"button {button.text() or button.toolTip()[:40]!r}"
    if kind == "key":
        target = window.focusWidget() or window.viewport
        key, mods = rng.choice(KEYS), rng.choice(MODS)
        if key == Qt.Key.Key_Q or (
            mods == Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_W
        ):
            return "skipped"
        qtbot.keyClick(target, key, mods)
        return f"key {key.name}+{mods.name} on {type(target).__name__}"
    if kind == "row":
        tree = window.scene_list.tree
        items = []
        for i in range(tree.topLevelItemCount()):
            top = tree.topLevelItem(i)
            items += [top] + [top.child(j) for j in range(top.childCount())]
        if items:
            tree.setCurrentItem(rng.choice(items))
        return "scene list row"
    if kind == "category":
        window.inspector.show_category(rng.choice(CATEGORIES).id)
        return "inspector category"
    if kind == "palette":
        window.show_palette(rng.choice(["fog", "sun", "key", "track", "label", "export", ""]))
        entries = [e for e in window.command_palette.shown if "Quit" not in e.title
                   and "Start Screen" not in e.title]  # fmt: skip
        window.command_palette.hide()
        if entries:
            rng.choice(entries).run()
        return "palette entry"
    if kind == "drag":
        fields = visible(window.findChildren(DragField))
        if fields:
            field = rng.choice(fields)
            field.commit(field.value + rng.uniform(-1.0, 1.0) * (field.step or 1.0) * 5)
        return "value drag"
    if kind == "time":
        window.timeline.set_time(rng.uniform(0.0, window.timeline.duration))
        return "timeline time"
    if kind == "play":
        window.timeline.toggle()
        return "play / pause"
    if kind == "view":
        return view_step(window.viewport, rng)
    stack = window.undo_stack
    (stack.undo if kind == "undo" else stack.redo)()
    return kind


@pytest.mark.parametrize("seed", [2026, 2027])
def test_a_monkey_in_the_interface(window, qtbot, monkeypatch, tmp_path, seed):  # noqa: F811
    # (a new project's tile cache is ~/earthling-cache: not the user's; nor is the folder of
    # anything saved under a relative name)
    for name in ("USERPROFILE", "HOME"):
        monkeypatch.setenv(name, str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    # (as in the application, a host that refused a request pauses: without that pause, every
    # tile of an export queued for its request slot to a refusing server - minutes)
    monkeypatch.setattr(net, "OFFLINE", net.OfflineHosts())
    menu_choices(monkeypatch)  # (context menus close at once)
    window.dialog_closer.stop()
    dialogs = Dialogs(random.Random(seed + 2))
    Files(tmp_path, window.project.folder, random.Random(seed + 1), dialogs.done).install(
        monkeypatch
    )
    dialogs.timer.start(40)
    rng = random.Random(seed)
    done = []
    worst = (0.0, "")
    with qtbot.capture_exceptions() as errors:  # (raised in Qt's event loop)
        try:
            for i in range(STEPS):
                what = []
                try:
                    frozen = longest_freeze(
                        qtbot, lambda w=what: w.append(step(window, qtbot, rng))
                    )
                    qtbot.waitUntil(lambda: QApplication.activeModalWidget() is None, timeout=20000)
                except Exception as exc:  # (which steps led there)
                    raise AssertionError(f"step {i}, after: {done[-15:]}") from exc
                done.append((what[0] if what else "?") + "".join(f" > {d}" for d in dialogs.done))
                dialogs.done.clear()
                if errors:
                    raise AssertionError(f"step {i}, after: {done[-15:]}") from errors[0][1]
                if frozen > worst[0]:
                    worst = (frozen, done[-1])
                assert frozen < 5.0, f"step {i} froze the window {frozen:.1f} s: {done[-15:]}"
                assert window.isVisible(), done[-20:]
                if i % 100 == 99:
                    json.dumps(window.scene.to_json(), default=str)  # (still saveable)
                    while window.undo_stack.canUndo():
                        window.undo_stack.undo()
        finally:
            dialogs.timer.stop()
            close_dialogs()
    window.timeline.pause()
    print(f"\n{STEPS} steps, the longest freeze {worst[0]:.2f} s: {worst[1]}")
