"""OpenGL viewport widget hosting a moderngl context."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import moderngl
import numpy as np
from pyglm import glm
from PyQt6.QtCore import QFileSystemWatcher, QPointF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QMouseEvent, QSurfaceFormat, QWheelEvent
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from earthling.core.geo import LocalFrame
from earthling.core.gpx import Track
from earthling.render.camera import Camera, FlyController, OrbitController
from earthling.render.renderer import Renderer

log = logging.getLogger(__name__)

MIN_GROUND_CLEARANCE_M = 5.0

FLY_KEYS = {
    Qt.Key.Key_W: FlyController.FORWARD,
    Qt.Key.Key_S: FlyController.BACK,
    Qt.Key.Key_A: FlyController.LEFT,
    Qt.Key.Key_D: FlyController.RIGHT,
    Qt.Key.Key_Q: FlyController.DOWN,
    Qt.Key.Key_E: FlyController.UP,
    Qt.Key.Key_Up: FlyController.FORWARD,
    Qt.Key.Key_Down: FlyController.BACK,
    Qt.Key.Key_Left: FlyController.LEFT,
    Qt.Key.Key_Right: FlyController.RIGHT,
    Qt.Key.Key_PageDown: FlyController.DOWN,
    Qt.Key.Key_PageUp: FlyController.UP,
}


def configure_default_surface_format() -> None:
    """Must be called before the QApplication is created."""
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(fmt)


class Viewport(QOpenGLWidget):
    fps_changed = pyqtSignal(float)
    stats_changed = pyqtSignal(str)
    camera_changed = pyqtSignal(str)  # human readable camera position
    mode_changed = pyqtSignal(str)
    shader_error = pyqtSignal(str)

    ORBIT, FLY = "orbit", "fly"

    def __init__(self, dev_mode: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.dev_mode = dev_mode
        self.ctx: moderngl.Context | None = None
        self.renderer: Renderer | None = None
        self.camera = Camera()
        self.orbit = OrbitController(self.camera)
        self.fly = FlyController(self.camera)
        self.mode = self.ORBIT
        self.frame: LocalFrame | None = None
        self._pending_scene: tuple[LocalFrame, list[Track]] | None = None
        self._pending_outlines = None
        self._pending_terrain = None
        self.store = None  # PropertyStore of the scene
        self.timezone = "UTC"
        self._shut_down = False
        self._camera_restored = False  # a saved camera must not be overridden by framing
        self._last_mouse: QPointF | None = None
        self._last_paint = time.perf_counter()
        self._frames = 0
        self._fps_timer = time.perf_counter()
        self._watcher: QFileSystemWatcher | None = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # On-demand rendering: a 60 Hz tick repaints only when something changed, so idle
        # views cost no CPU and background loading threads get the GIL.
        self._needs_render = True
        self._last_camera_state: tuple | None = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

    # --- scene -------------------------------------------------------------------------
    def set_scene(
        self, frame: LocalFrame, tracks: list[Track], reframe: bool = True, timezone: str = "UTC"
    ) -> None:
        self.frame = frame
        self.timezone = timezone
        self._camera_restored = False
        if self.renderer is None:
            self._pending_scene = (frame, tracks)
            return
        self.renderer.timezone = timezone
        self.makeCurrent()
        self.renderer.set_scene(frame, tracks)
        self.doneCurrent()
        self.request_render()
        if reframe:
            self.frame_all()

    def set_terrain_source(self, data, nodes) -> None:
        if self.renderer is None:
            self._pending_terrain = (data, nodes)
            return
        self.makeCurrent()
        self.renderer.set_terrain_source(data, nodes)
        self.doneCurrent()
        self.request_render()

    def set_outlines(self, lines) -> None:
        if self.renderer is None:
            self._pending_outlines = lines
            return
        self.makeCurrent()
        self.renderer.outlines.set_lines(lines)
        self.doneCurrent()
        self.request_render()

    def set_on_demand(self, enabled: bool) -> None:
        if self.renderer is not None and self.renderer.terrain.data is not None:
            self.renderer.terrain.data.on_demand = enabled

    def set_store(self, store) -> None:
        self.store = store
        store.subscribe(lambda pid, value: self.request_render())
        if self.renderer is not None:
            self.renderer.store = store

    def shutdown(self) -> None:
        """Stop background terrain work (idempotent)."""
        if self.renderer is not None and not self._shut_down:
            self._shut_down = True
            self.makeCurrent()
            self.renderer.terrain.shutdown()
            self.doneCurrent()

    # --- camera ------------------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        if mode == self.mode:
            return
        if mode == self.ORBIT:
            target = self.pick_center()
            if target is None:
                target = self.camera.position + self.camera.forward * 2000.0
            self.orbit.sync_from_camera(target)
        self.mode = mode
        self.fly.pressed.clear()
        self.mode_changed.emit(mode)

    def toggle_mode(self) -> None:
        self.set_mode(self.FLY if self.mode == self.ORBIT else self.ORBIT)

    def frame_all(self) -> None:
        bounds = self.renderer.scene_bounds() if self.renderer is not None else None
        if bounds is not None:
            self.frame_box(*bounds)

    def frame_box(self, lo, hi) -> None:
        self.mode = self.ORBIT
        self.orbit.frame_bounds(lo, hi)
        self.mode_changed.emit(self.mode)

    def go_to(self, lat: float, lon: float, height_above_ground: float) -> None:
        if self.frame is None:
            return
        ground = (self.ground_height(lat, lon) or 0.0) * self._exaggeration()
        self.camera.position = self.frame.geodetic_to_enu(lat, lon, ground + height_above_ground)
        self.camera.heading, self.camera.pitch, self.camera.roll = 0.0, -30.0, 0.0
        self.mode = self.FLY
        self.mode_changed.emit(self.mode)

    def ground_height(self, lat: float, lon: float) -> float | None:
        data = self.renderer.terrain.data if self.renderer is not None else None
        return data.height_at(lon, lat) if data is not None else None

    def camera_geodetic(self) -> tuple[float, float, float] | None:
        if self.frame is None:
            return None
        lat, lon, h = self.frame.enu_to_geodetic(self.camera.position)
        return float(lat), float(lon), float(h)

    def height_above_ground(self) -> float | None:
        geo = self.camera_geodetic()
        if geo is None:
            return None
        ground = self.ground_height(geo[0], geo[1])
        return None if ground is None else geo[2] - ground * self._exaggeration()

    def _exaggeration(self) -> float:
        return self.renderer.terrain.exaggeration if self.renderer is not None else 1.0

    def _clamp_to_ground(self) -> None:
        hag = self.height_above_ground()
        if hag is not None and hag < MIN_GROUND_CLEARANCE_M:
            lift = np.array([0.0, 0.0, MIN_GROUND_CLEARANCE_M - hag])
            self.camera.position = self.camera.position + lift

    def camera_state(self) -> dict:
        c = self.camera
        return {
            "position": [float(v) for v in c.position],
            "heading": c.heading,
            "pitch": c.pitch,
            "roll": c.roll,
            "mode": self.mode,
            "orbit_target": [float(v) for v in self.orbit.target],
        }

    def restore_camera_state(self, state: dict) -> bool:
        try:
            position = np.array(state["position"], dtype=np.float64)
            heading, pitch = float(state["heading"]), float(state["pitch"])
        except (KeyError, TypeError, ValueError):
            return False
        self._camera_restored = True
        self.camera.position = position
        self.camera.heading, self.camera.pitch = heading, pitch
        self.camera.roll = float(state.get("roll", 0.0))
        if state.get("mode") == self.ORBIT and "orbit_target" in state:
            self.orbit.sync_from_camera(np.array(state["orbit_target"], dtype=np.float64))
            self.camera.roll = float(state.get("roll", 0.0))
            self.mode = self.ORBIT
        else:
            self.mode = self.FLY
        self.mode_changed.emit(self.mode)
        return True

    # --- picking -----------------------------------------------------------------------
    def pick(self, x: float, y: float) -> np.ndarray | None:
        """ENU position of the rendered surface under widget pixel (x, y), or None."""
        if self.ctx is None or self.renderer is None:
            return None
        ratio = self.devicePixelRatio()
        width, height = int(self.width() * ratio), int(self.height() * ratio)
        px, py = int(x * ratio), int(height - 1 - y * ratio)
        if not (0 <= px < width and 0 <= py < height):
            return None
        self.makeCurrent()
        depth = self.renderer.read_depth(px, py)
        self.doneCurrent()
        if depth is None or depth >= 0.999999:
            return None  # sky
        return unproject_log_depth(self.camera, px, py, width, height, depth)

    def pick_center(self) -> np.ndarray | None:
        return self.pick(self.width() / 2, self.height() / 2)

    # --- Qt GL hooks -------------------------------------------------------------------
    def initializeGL(self) -> None:
        self.ctx = moderngl.create_context()
        log.info("OpenGL %s on %s", self.ctx.info["GL_VERSION"], self.ctx.info["GL_RENDERER"])
        self.renderer = Renderer(self.ctx)
        if self.dev_mode:
            self._watcher = QFileSystemWatcher(self)
            self._watcher.fileChanged.connect(self._on_shader_changed)
        if self._pending_scene is not None:
            frame, tracks = self._pending_scene
            self._pending_scene = None
            self.renderer.set_scene(frame, tracks)
            if not self._camera_restored:
                self.frame_all()
        if self._pending_terrain is not None:
            self.renderer.set_terrain_source(*self._pending_terrain)
            self._pending_terrain = None
        if self._pending_outlines is not None:
            self.renderer.outlines.set_lines(self._pending_outlines)
            self._pending_outlines = None
        self.renderer.store = self.store
        self.renderer.timezone = self.timezone

    def paintGL(self) -> None:
        if self.ctx is None or self.renderer is None:
            return
        now = time.perf_counter()
        dt = now - self._last_paint
        self._last_paint = now
        if self.mode == self.FLY:
            hag = self.height_above_ground()
            self.fly.step(dt, hag if hag is not None else 1000.0)
        self._clamp_to_ground()
        fbo = self.ctx.detect_framebuffer(self.defaultFramebufferObject())
        ratio = self.devicePixelRatio()
        width, height = int(self.width() * ratio), int(self.height() * ratio)
        self.renderer.render(fbo, width, height, self.camera)
        self._sync_watcher()
        self._count_frame(now)

    # --- input -------------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last_mouse = event.position()
        self.setFocus()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._last_mouse = None

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        target = self.pick(event.position().x(), event.position().y())
        if target is not None:
            self.orbit.sync_from_camera(target)
            self.mode = self.ORBIT
            self.mode_changed.emit(self.mode)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_mouse is None:
            return
        delta = event.position() - self._last_mouse
        self._last_mouse = event.position()
        buttons = event.buttons()
        if self.mode == self.FLY:
            if buttons & (Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton):
                self.fly.look(delta.x(), delta.y())
            return
        if buttons & Qt.MouseButton.LeftButton:
            self.orbit.rotate(delta.x(), delta.y())
        elif buttons & (Qt.MouseButton.RightButton | Qt.MouseButton.MiddleButton):
            self.orbit.pan(delta.x(), delta.y(), self.height())

    def wheelEvent(self, event: QWheelEvent) -> None:
        steps = event.angleDelta().y() / 120.0
        if self.mode == self.FLY:
            multiplier = self.fly.speed_multiplier * 1.25**steps
            self.fly.speed_multiplier = float(np.clip(multiplier, 0.05, 50.0))
        else:
            self.orbit.zoom(steps)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_Tab:
            self.toggle_mode()
            return
        if key in FLY_KEYS and not event.isAutoRepeat():
            if self.mode != self.FLY:
                self.set_mode(self.FLY)
            self.fly.pressed.add(FLY_KEYS[key])
        self._update_boost(event)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() in FLY_KEYS and not event.isAutoRepeat():
            self.fly.pressed.discard(FLY_KEYS[event.key()])
        self._update_boost(event)
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event) -> None:
        self.fly.pressed.clear()
        super().focusOutEvent(event)

    def focusNextPrevChild(self, next: bool) -> bool:  # noqa: A002 - Qt signature
        return False  # keep Tab for mode switching

    def _update_boost(self, event: QKeyEvent) -> None:
        mods = event.modifiers()
        if mods & Qt.KeyboardModifier.ShiftModifier:
            self.fly.boost = 4.0
        elif mods & Qt.KeyboardModifier.ControlModifier:
            self.fly.boost = 0.25
        else:
            self.fly.boost = 1.0

    # --- on-demand rendering --------------------------------------------------------------
    def request_render(self) -> None:
        self._needs_render = True

    def _camera_key(self) -> tuple:
        c = self.camera
        return (*(round(float(v), 3) for v in c.position), c.heading, c.pitch, c.roll, c.fov_y)

    def _busy(self) -> bool:
        if self.fly.pressed:
            return True
        t = self.renderer.terrain if self.renderer is not None else None
        return (
            t is not None and t.nodes is not None and (t.pending_count > 0 or not t.fully_loaded())
        )

    def _tick(self) -> None:
        key = self._camera_key()
        if key != self._last_camera_state:
            self._last_camera_state = key
            self._needs_render = True
        if self._needs_render or self._busy():
            self._needs_render = False
            self.update()

    # --- helpers -----------------------------------------------------------------------
    def camera_text(self) -> str:
        geo = self.camera_geodetic()
        if geo is None:
            return ""
        lat, lon, h = geo
        hag = self.height_above_ground()
        agl = f" ({hag:.0f} m above ground)" if hag is not None else ""
        return (
            f"{self.mode}  {abs(lat):.5f}°{'N' if lat >= 0 else 'S'} "
            f"{abs(lon):.5f}°{'E' if lon >= 0 else 'W'}  {h:.0f} m{agl}  "
            f"heading {self.camera.heading:.0f}°  pitch {self.camera.pitch:.0f}°"
        )

    def _count_frame(self, now: float) -> None:
        self._frames += 1
        if now - self._fps_timer >= 0.5:
            fps = self._frames / (now - self._fps_timer)
            self.fps_changed.emit(fps)
            text = f"{fps:5.1f} fps"
            if self.renderer is not None and self.renderer.terrain.nodes is not None:
                t = self.renderer.terrain
                drawn = len(t.last_selection.draw) if t.last_selection else 0
                text = (
                    f"terrain: {drawn} drawn, {t.resident_count} resident, "
                    f"{t.pending_count} loading, {t.gpu_bytes / 2**20:.0f} MB  |  {text}"
                )
            self.stats_changed.emit(text)
            self.camera_changed.emit(self.camera_text())
            self._frames = 0
            self._fps_timer = now

    def _sync_watcher(self) -> None:
        if self._watcher is None or self.renderer is None:
            return
        wanted = {str(p) for p in self.renderer.shaders.watched_files()}
        missing = wanted - set(self._watcher.files())
        if missing:
            self._watcher.addPaths(sorted(missing))

    def _on_shader_changed(self, path: str) -> None:
        if self.renderer is None:
            return
        self.makeCurrent()
        errors = self.renderer.shaders.reload_changed(Path(path))
        self.doneCurrent()
        # Editors often replace files on save, which removes them from the watcher.
        if self._watcher is not None and path not in self._watcher.files():
            self._watcher.addPath(path)
        for err in errors:
            self.shader_error.emit(err)
        self.request_render()


def unproject_log_depth(
    camera: Camera, px: float, py: float, width: int, height: int, depth: float
) -> np.ndarray:
    """World (ENU) position from a pixel (GL convention, y up) and its logarithmic depth."""
    w = 2.0 ** (depth / camera.log_depth_coef) - 1.0  # view-space distance along forward
    ndc = glm.vec4(2.0 * (px + 0.5) / width - 1.0, 2.0 * (py + 0.5) / height - 1.0, 1.0, 1.0)
    inv = glm.inverse(camera.view_projection(width / max(1, height)))
    far = inv * ndc
    direction = np.array([far.x, far.y, far.z]) / far.w
    direction /= np.linalg.norm(direction)
    t = w / max(1e-6, float(np.dot(direction, camera.forward)))
    return camera.position + direction * t
