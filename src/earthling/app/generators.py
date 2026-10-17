"""Animation generators (Animation menu): keys written by the scene kit from the tracks.

Generate Camera (follow, map view, orbit, along a line), Draw Tracks (the head over a time
range), Sun from the Tracks (light from the recorded time) and Add Still Seconds. Each is one
undo step; the keys can be edited afterwards like any others.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
from PyQt6.QtCore import QTime
from PyQt6.QtGui import QUndoCommand
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QInputDialog,
    QTimeEdit,
    QVBoxLayout,
)

from earthling import scenekit
from earthling.scenekit import Journey
from earthling.ui.widgets import double_spin, row

CAMERA_STEP = 1.0 / 30.0  # camera samples (key_shots resamples them smoothly)


class AnimationCommand(QUndoCommand):
    """Undo by restoring the whole animation (keys, duration) as it was."""

    def __init__(self, label: str, animation, before: dict, after: dict, on_change) -> None:
        super().__init__(label)
        self.animation, self.before, self.after, self.on_change = (animation, before, after,
                                                                   on_change)  # fmt: skip
        self._first = True

    def redo(self) -> None:
        if self._first:  # applied when pushed
            self._first = False
            return
        self.animation.load_json(self.after)
        self.on_change()

    def undo(self) -> None:
        self.animation.load_json(self.before)
        self.on_change()


def window_journey(window) -> Journey | None:
    """The journey of the open project as the viewport shows it (None before a project)."""
    renderer = window.viewport.renderer
    if window.session is None or renderer is None or renderer.tracks.path is None:
        return None
    tz = ZoneInfo(window.session.config.project.timezone)
    return Journey(window.session, renderer.tracks.path, list(renderer.tracks._gpu),
                   renderer.terrain.data, tz)  # fmt: skip


def run(window, label: str, generate, settings: dict | None = None,
        needs_tracks: bool = True) -> bool:  # fmt: skip
    """Run ``generate(journey, scene)`` as one undo step (``settings``: plain property values
    set first). False without a project."""
    journey = window_journey(window)
    if journey is None and needs_tracks:
        window.statusBar().showMessage("Open a project with tracks first", 5000)
        return False
    scene, stack = window.scene, window.undo_stack
    stack.beginMacro(label)
    for pid, value in (settings or {}).items():
        if not scene.animation.is_animated(pid):
            window.set_property(pid, value)
    before = scene.animation.to_json()
    scenekit.set_current(journey)  # (level horizon, local time)
    try:
        generate(journey, scene)
    finally:
        scenekit.set_current(None)
    after = scene.animation.to_json()
    if after != before:
        stack.push(AnimationCommand(label, scene.animation, before, after,
                                    window.animation_changed))  # fmt: skip
    stack.endMacro()
    window.animation_changed()
    return True


# --- dialogs ---------------------------------------------------------------------------------
class GeneratorDialog(QDialog):
    """A form with a time range and OK / Cancel."""

    def __init__(self, title: str, duration: float, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.form = QFormLayout()
        self.t0 = double_spin(0.0, duration, decimals=2, suffix=" s", value=0.0)
        self.t1 = double_spin(0.0, duration, decimals=2, suffix=" s", value=duration)
        self.form.addRow("From / to", row((self.t0, 1), (self.t1, 1)))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)  # fmt: skip
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(self.form)
        layout.addWidget(buttons)

    def times(self, step: float) -> np.ndarray:
        t0, t1 = sorted((self.t0.value(), self.t1.value()))
        times = np.arange(t0, t1 + 1e-9, step)
        return times if len(times) > 1 else np.array([t0, t0 + step])


class CameraDialog(GeneratorDialog):
    TYPES = [("follow", "Follow"), ("map", "Map view"), ("orbit", "Orbit"),
             ("line", "Along a line")]  # fmt: skip
    TIPS = {
        "follow": "A soft chase camera behind or beside the head (angle 0: from behind)",
        "map": "A calm view from above, north up at angle 0; zooms out to keep the head in view",
        "orbit": "Circles around the head or the look-at target, starting at the angle",
        "line": "Flies along the line drawn in the time range, looking across it; zooms out to "
                "the far distance halfway (flights, ferries)",
    }  # fmt: skip

    def __init__(self, duration: float, parent=None) -> None:
        super().__init__("Generate Camera", duration, parent)
        self.kind = QComboBox()
        for key, label in self.TYPES:
            self.kind.addItem(label, key)
        self.distance = double_spin(20.0, 2e7, decimals=0, suffix=" m", value=2500.0)
        self.far = double_spin(20.0, 2e7, decimals=0, suffix=" m", value=20000.0)
        self.pitch = double_spin(-89.0, 0.0, decimals=1, suffix="°", value=-35.0)
        self.angle = double_spin(-360.0, 360.0, decimals=1, suffix="°", value=30.0)
        self.turn = double_spin(-1080.0, 1080.0, decimals=0, suffix="°", value=90.0)
        self.centre = QComboBox()
        self.centre.addItem("The head", "head")
        self.centre.addItem("The look-at target", "target")
        rows = [("Camera", self.kind), ("Distance", self.distance), ("Far distance", self.far),
                ("Pitch", self.pitch), ("Angle", self.angle), ("Turn by", self.turn),
                ("Around", self.centre)]  # fmt: skip
        for i, (label, widget) in enumerate(rows):
            self.form.insertRow(i, label, widget)
        self.kind.currentIndexChanged.connect(self._update)
        self._update()

    def _update(self, *_args) -> None:
        kind = self.kind.currentData()
        self.kind.setToolTip(self.TIPS[kind])
        self.far.setEnabled(kind == "line")
        self.turn.setEnabled(kind == "orbit")
        self.centre.setEnabled(kind == "orbit")

    def generate(self, journey: Journey, scene) -> None:
        kind = self.kind.currentData()
        times = self.times(CAMERA_STEP)
        head_of = scenekit.head_distance_of(scene, journey.path)
        distance, pitch, angle = self.distance.value(), self.pitch.value(), self.angle.value()
        if kind == "follow":
            shots = scenekit.follow_shots(journey, times, head_of, distance=distance,
                                          pitch=pitch, side=angle)  # fmt: skip
        elif kind == "map":
            shots = scenekit.map_shots(journey, times, head_of,
                                       distance_of=lambda t: distance,
                                       heading_of=lambda t: angle,
                                       pitch_of=lambda t: pitch)  # fmt: skip
        elif kind == "orbit":
            if self.centre.currentData() == "target":
                target = np.asarray(scene.store["camera.target"], dtype=np.float64)

                def centre_of(t):
                    return target
            else:
                heads = np.array([journey.path.position_at(head_of(float(t))) for t in times])
                heads = scenekit.gaussian(heads, 1.0 / CAMERA_STEP)

                def centre_of(t):
                    return heads[min(int(round((t - times[0]) / CAMERA_STEP)), len(heads) - 1)]

            shots = scenekit.orbit_shots(journey, times, centre_of, distance=distance,
                                         pitch=pitch, heading=angle,
                                         turn=self.turn.value())  # fmt: skip
        else:
            shots = scenekit.line_shots(journey, times, head_of, near=distance,
                                        far=self.far.value(), pitch=pitch, side=angle)  # fmt: skip
        scenekit.clear_keys(scene, "camera.pose", float(times[0]), float(times[-1]))
        scenekit.key_shots(scene, shots)


class DrawDialog(GeneratorDialog):
    def __init__(self, duration: float, tracks: list, parent=None) -> None:
        super().__init__("Draw Tracks", duration, parent)
        self.first, self.last = QComboBox(), QComboBox()
        for i, g in enumerate(tracks):
            for combo in (self.first, self.last):
                combo.addItem(f"{i + 1}  {g.track.name}", i)
        self.last.setCurrentIndex(len(tracks) - 1)
        self.timing = QComboBox()
        self.timing.addItem("Recorded time (breaks pause the line)", True)
        self.timing.addItem("Even speed", False)
        self.soft = QCheckBox("Start and stop softly")
        self.soft.setChecked(True)
        self.alone = QCheckBox("Hide the line drawn before")
        self.alone.setToolTip("Sets the start of the drawn part (progress tail) to the first track")
        for i, (label, widget) in enumerate([("From track", self.first), ("To track", self.last),
                                              ("Timing", self.timing), ("", self.soft),
                                              ("", self.alone)]):  # fmt: skip
            self.form.insertRow(i + 1, label, widget)

    def tracks(self) -> tuple[int, int]:
        return tuple(sorted((self.first.currentData(), self.last.currentData())))

    def generate(self, journey: Journey, scene) -> None:
        first, last = self.tracks()
        d0, d1 = journey.path.track_range(first)[0], journey.path.track_range(last)[1]
        t0, t1 = sorted((self.t0.value(), self.t1.value()))
        scenekit.draw_keys(scene, journey, d0, d1, t0, t1, realtime=self.timing.currentData(),
                           soft=self.soft.isChecked())  # fmt: skip


class SunDialog(GeneratorDialog):
    def __init__(self, duration: float, parent=None) -> None:
        super().__init__("Sun from the Tracks", duration, parent)
        self.use_earliest, self.use_latest = QCheckBox("Not before"), QCheckBox("Not after")
        self.earliest, self.latest = QTimeEdit(QTime(8, 0)), QTimeEdit(QTime(19, 0))
        self.smooth = QCheckBox("Smooth over nights (the sun moves back instead of setting)")
        self.form.addRow(self.use_earliest, self.earliest)
        self.form.addRow(self.use_latest, self.latest)
        self.form.addRow("", self.smooth)

    @staticmethod
    def _hours(edit: QTimeEdit) -> float:
        t = edit.time()
        return t.hour() + t.minute() / 60.0

    def generate(self, journey: Journey, scene) -> None:
        times = self.times(0.5)
        head_of = scenekit.head_distance_of(scene, journey.path)
        scenekit.sun_keys(scene, journey, times, head_of,
                          earliest=self._hours(self.earliest) if self.use_earliest.isChecked()
                          else None,
                          latest=self._hours(self.latest) if self.use_latest.isChecked() else None,
                          smooth_nights=self.smooth.isChecked())  # fmt: skip


# --- menu actions ----------------------------------------------------------------------------
def generate_camera(window) -> None:
    dialog = CameraDialog(window.scene.animation.duration, window)
    if dialog.exec():
        run(window, "Generate camera", dialog.generate, {"camera.mode": "keys"})


def draw_tracks(window) -> None:
    journey = window_journey(window)
    if journey is None:
        window.statusBar().showMessage("Open a project with tracks first", 5000)
        return
    dialog = DrawDialog(window.scene.animation.duration, journey.path.tracks, window)
    if not dialog.exec():
        return
    settings = {}
    if dialog.alone.isChecked():
        d0 = journey.path.track_range(dialog.tracks()[0])[0]
        mode = window.scene.store["progress.mode"]
        settings["progress.tail"] = journey.path.progress_for_distance(d0 + 0.01, mode)
    run(window, "Draw tracks", dialog.generate, settings)


def sun_from_tracks(window) -> None:
    dialog = SunDialog(window.scene.animation.duration, window)
    if dialog.exec():
        run(window, "Sun from the tracks", dialog.generate, {"sun.follow_track": False})


def add_still_seconds(window) -> None:
    seconds, ok = QInputDialog.getDouble(window, "Add Still Seconds",
                                         "Seconds without motion at the start and the end:",
                                         1.0, 0.1, 10.0, 1)  # fmt: skip
    if ok:
        run(
            window,
            "Add still seconds",
            lambda journey, scene: scenekit.add_holds(scene, seconds),
            needs_tracks=False,
        )
