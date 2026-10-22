"""Controls over the 3D view: orbit / fly, the animated camera, frame all, the look-at
target, preview quality, and the export frame (the part of the view a video shows).

The viewport keeps the vertical field of view, so an export of another aspect ratio shows
the full height and a narrower (or wider) width: the overlay darkens what is outside it.
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, QRectF, QSize, Qt
from PyQt6.QtGui import QAction, QColor, QPainter, QPen
from PyQt6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QLabel, QToolButton, QWidget

from earthling.ui import theme


def export_frame(view_w: float, view_h: float, export_w: int, export_h: int) -> QRectF | None:
    """The export's rectangle in the view (full height, centred); None when the export is
    wider than the view (it shows more than the view can)."""
    aspect = export_w / max(1, export_h)
    width = view_h * aspect
    if width > view_w + 0.5:
        return None
    return QRectF((view_w - width) / 2, 0.0, width, view_h)


class FrameShade(QWidget):
    """Darkens the view outside the export frame (does not take the mouse)."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.resolution = (3840, 2160)
        self.enabled = True

    def paintEvent(self, event) -> None:
        if not self.enabled:
            return
        frame = export_frame(self.width(), self.height(), *self.resolution)
        if frame is None or frame.width() >= self.width() - 1:
            return
        p = QPainter(self)
        shade = QColor(6, 8, 11, 150)
        p.fillRect(QRectF(0, 0, frame.left(), self.height()), shade)
        p.fillRect(QRectF(frame.right(), 0, self.width() - frame.right(), self.height()), shade)
        p.setPen(QPen(QColor(255, 255, 255, 70), 1))
        p.drawRect(frame.adjusted(0.5, 0.5, -0.5, -0.5))
        p.end()


def _pill(parent: QWidget) -> QFrame:
    frame = QFrame(parent)
    frame.setObjectName("overlayPill")
    layout = QHBoxLayout(frame)
    layout.setContentsMargins(3, 3, 3, 3)
    layout.setSpacing(2)
    return frame


class ViewportOverlay(QObject):
    """Places the controls in the corners of the viewport and keeps them up to date."""

    MARGIN = 12

    def __init__(self, window) -> None:
        super().__init__(window)
        self.window = window
        viewport = self.viewport = window.viewport
        self.shade = FrameShade(viewport)
        # top left: orbit / fly and the animated camera
        self.modes = _pill(viewport)
        self.mode_group = QButtonGroup(self.modes)
        self.mode_buttons = {}
        for mode, icon, text, tip in (
            (viewport.ORBIT, "globe-hemisphere-west", "Orbit", "Orbit around a point (Tab)"),
            (viewport.FLY, "airplane-tilt", "Fly", "Fly like a drone: W A S D, Q E (Tab)"),
        ):
            b = self._button(self.modes, icon, text, tip, checkable=True)
            b.clicked.connect(lambda _=False, m=mode: viewport.set_mode(m))
            self.mode_group.addButton(b)
            self.mode_buttons[mode] = b
        self.camera = _pill(viewport)
        self.camera_button = self._button(self.camera, "video-camera", "Free camera",
                                          "Look through the animated camera (C)",
                                          checkable=True)  # fmt: skip
        self.camera_button.toggled.connect(viewport.set_animated_camera)
        # top right: view tools
        self.tools = _pill(viewport)
        for action in (window.frame_all_action, window.pick_target_action):
            b = self._button(self.tools, "", "", action.toolTip() or action.text())
            b.setDefaultAction(action)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.preview = self._button(self.tools, "lightning", "", "Preview quality: a faster "
                                    "view (exports always use full quality)",
                                    checkable=True)  # fmt: skip
        self.preview.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.preview.toggled.connect(lambda v: window.set_property("view.preview", v))
        # bottom left: the export frame
        self.frame_label = QLabel(viewport)
        self.frame_label.setObjectName("overlayLabel")
        self.frame_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.frame_action = QAction("Show the &Export Frame", window, checkable=True)
        self.frame_action.setChecked(True)
        self.frame_action.setToolTip("Darken what a video does not show; photos and the stats "
                                     "sit in the frame as in the video")  # fmt: skip
        self.frame_action.toggled.connect(self._show_frame)
        window.view_menu.addAction(self.frame_action)
        viewport.installEventFilter(self)
        viewport.mode_changed.connect(lambda _: self.sync())
        viewport.animated_camera_changed.connect(lambda _: self.sync())
        window.scene.store.subscribe(self._on_store)
        self.sync()

    @staticmethod
    def _button(parent: QFrame, icon: str, text: str, tip: str,
                checkable: bool = False) -> QToolButton:  # fmt: skip
        b = QToolButton(parent)
        b.setObjectName("overlayButton")
        if icon:
            b.setIcon(theme.icon(icon, theme.TEXT_2, theme.TEXT, on="#FFFFFF"))
        b.setIconSize(QSize(16, 16))
        b.setText(text)
        b.setToolTip(tip)
        b.setCheckable(checkable)
        b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        parent.layout().addWidget(b)
        return b

    def _show_frame(self, on: bool) -> None:
        self.shade.enabled = on
        self.frame_label.setVisible(on)
        self.sync()

    def _on_store(self, pid: str, value) -> None:
        if pid in ("export.resolution", "view.preview"):
            self.sync()

    def sync(self) -> None:
        viewport, store = self.viewport, self.window.scene.store
        for mode, b in self.mode_buttons.items():
            b.setChecked(viewport.mode == mode)
        animated = viewport.animated_camera
        self.camera_button.blockSignals(True)
        self.camera_button.setChecked(animated)
        self.camera_button.setText("Animated camera" if animated else "Free camera")
        self.camera_button.blockSignals(False)
        self.preview.blockSignals(True)
        self.preview.setChecked(bool(store["view.preview"]))
        self.preview.blockSignals(False)
        from earthling.export.frames import parse_resolution

        width, height = parse_resolution(store["export.resolution"])
        self.shade.resolution = (width, height)
        viewport.export_size = (width, height) if self.shade.enabled else None
        viewport.request_render()
        frame = export_frame(viewport.width(), viewport.height(), width, height)
        wider = " · shows more at the sides" if frame is None else ""
        self.frame_label.setText(f"Export frame  {width} × {height}{wider}")
        self.place()

    def place(self) -> None:
        viewport, m = self.viewport, self.MARGIN
        self.shade.setGeometry(viewport.rect())
        self.shade.update()
        for widget in (self.modes, self.camera, self.tools, self.frame_label):
            widget.adjustSize()
        self.modes.move(m, m)
        self.camera.move(m + self.modes.width() + 8, m)
        self.tools.move(viewport.width() - self.tools.width() - m, m)
        self.frame_label.move(m, viewport.height() - self.frame_label.height() - m)
        for widget in (self.shade, self.modes, self.camera, self.tools, self.frame_label):
            widget.raise_()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is self.viewport and event.type() == QEvent.Type.Resize:
            self.place()
        return False
