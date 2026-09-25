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
from earthling.render.camera_rig import clamp_above_ground
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
    animated_camera_changed = pyqtSignal(bool)

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
        self._pending_camera_path = None
        self.store = None  # PropertyStore of the scene
        self.suspended = False  # no viewport rendering (e.g. while exporting a video)
        self.animated_camera = False  # follow the "camera.pose" property (timeline playback)
        self.pose_provider = None  # () -> pose; the camera rig (modes, transitions)
        self._pick_callback = None  # set by request_pick(): the next click picks a point
        self.timezone = "UTC"
        self._shut_down = False
        self._camera_restored = False  # a saved camera must not be overridden by framing
        self._last_mouse: QPointF | None = None
        self._last_paint = time.perf_counter()
        self._start_time = self._last_paint
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

    def set_terrain_source(self, data, nodes, labels=None) -> None:
        """``labels``: LabelData of the area (optional)."""
        if self.renderer is None:
            self._pending_terrain = (data, nodes, labels)
            return
        self.makeCurrent()
        self.renderer.set_terrain_source(data, nodes)
        self.renderer.set_labels(labels)
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
        store.subscribe(self._on_store_changed)
        if self.renderer is not None:
            self.renderer.store = store

    def _on_store_changed(self, pid: str, value) -> None:
        if self.animated_camera and pid.startswith(("camera.", "follow.", "progress.")):
            pose = self.current_animated_pose()
            if pose is not None:
                self.apply_pose(pose)
        self.request_render()

    def request_pick(self, callback) -> None:
        """The next left click on the terrain calls ``callback(enu_position)``."""
        self._pick_callback = callback
        self.setCursor(Qt.CursorShape.CrossCursor)

    def current_animated_pose(self):
        if self.pose_provider is not None:
            return self.pose_provider()
        return self.store["camera.pose"] if self.store is not None else None

    # --- animated camera ----------------------------------------------------------------
    def camera_pose(self) -> tuple[float, float, float, float, float, float]:
        c = self.camera
        return (*(float(v) for v in c.position), c.heading, c.pitch, c.roll)

    def apply_pose(self, pose) -> None:
        x, y, z, heading, pitch, roll = pose
        self.camera.position = np.array([x, y, z], dtype=np.float64)
        self.camera.heading, self.camera.pitch, self.camera.roll = heading, pitch, roll
        self.request_render()

    def set_animated_camera(self, enabled: bool) -> None:
        if enabled == self.animated_camera:
            return
        self.animated_camera = enabled
        if enabled:
            pose = self.current_animated_pose()
            if pose is not None:
                self.apply_pose(pose)
        if not enabled:
            self.mode = self.FLY
            self.mode_changed.emit(self.mode)
        if self.renderer is not None:
            self.renderer.camera_path.visible = not enabled
        self.animated_camera_changed.emit(enabled)

    def _take_manual_control(self) -> None:
        """Manual navigation leaves the animated camera (like any 3D editor)."""
        if self.animated_camera:
            self.set_animated_camera(False)

    def set_camera_path(self, lines) -> None:
        if self.renderer is None:
            self._pending_camera_path = lines
            return
        self.makeCurrent()
        self.renderer.camera_path.set_lines(lines)
        self.doneCurrent()
        self.request_render()

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

    def clamp_pose(self, pose):
        """Keep an animated pose above the terrain (follow/look-at paths may cut ridges)."""
        data = self.renderer.terrain.data if self.renderer is not None else None
        return clamp_above_ground(pose, self.frame, data, self._exaggeration())

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
        # the scene target may be smaller than the widget (preview quality)
        tw, th = self.renderer.target.size
        tx = min(int(px * tw / max(width, 1)), tw - 1)
        ty = min(int(py * th / max(height, 1)), th - 1)
        self.makeCurrent()
        depth = self.renderer.read_depth(tx, ty)
        self.doneCurrent()
        if depth is None or depth >= 0.999999:
            return None  # sky
        return unproject_log_depth(self.camera, tx, ty, tw, th, depth)

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
            data, nodes, labels = self._pending_terrain
            self.renderer.set_terrain_source(data, nodes)
            self.renderer.set_labels(labels)
            self._pending_terrain = None
        if self._pending_outlines is not None:
            self.renderer.outlines.set_lines(self._pending_outlines)
            self._pending_outlines = None
        if self._pending_camera_path is not None:
            self.renderer.camera_path.set_lines(self._pending_camera_path)
            self._pending_camera_path = None
        self.renderer.camera_path.visible = not self.animated_camera
        self.renderer.store = self.store
        self.renderer.timezone = self.timezone

    def paintGL(self) -> None:
        if self.ctx is None or self.renderer is None:
            return
        now = time.perf_counter()
        dt = now - self._last_paint
        self._last_paint = now
        if self.animated_camera:
            pose = self.current_animated_pose()
            if pose is not None:
                x, y, z, heading, pitch, roll = pose
                self.camera.position = np.array([x, y, z], dtype=np.float64)
                self.camera.heading, self.camera.pitch, self.camera.roll = heading, pitch, roll
        elif self.mode == self.FLY:
            hag = self.height_above_ground()
            self.fly.step(dt, hag if hag is not None else 1000.0)
        if not self.animated_camera:
            self._clamp_to_ground()
        self.renderer.time = now - self._start_time
        fbo = self.ctx.detect_framebuffer(self.defaultFramebufferObject())
        ratio = self.devicePixelRatio()
        width, height = int(self.width() * ratio), int(self.height() * ratio)
        scale = 1.0
        self.renderer.overrides = {}
        if self.store is not None and self.store["view.preview"]:
            scale = 0.5
            self.renderer.overrides = {
                "terrain.detail": self.store["terrain.detail"] * 2.0,
                "shadows.resolution": "2048",
            }
        self.renderer.render(fbo, width, height, self.camera, scale=scale)
        self._sync_watcher()
        self._count_frame(now)

    # --- input -------------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.setFocus()
        if self._pick_callback is not None and event.button() == Qt.MouseButton.LeftButton:
            callback, self._pick_callback = self._pick_callback, None
            self.unsetCursor()
            point = self.pick(event.position().x(), event.position().y())
            if point is not None:
                callback(point)
            return
        self._last_mouse = event.position()
        self._take_manual_control()

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
        self._take_manual_control()
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
            self._take_manual_control()
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
        s = self.store
        if (
            s is not None
            and s["marker.visible"]
            and s["marker.pulse"] > 0
            and s["progress.head"] < 1
        ):
            return True  # the pulsing marker is animated
        if self.renderer is not None and self.renderer.labels.loading:
            return True  # labels appear when their download finishes
        t = self.renderer.terrain if self.renderer is not None else None
        return (
            t is not None and t.nodes is not None and (t.pending_count > 0 or not t.fully_loaded())
        )

    def _tick(self) -> None:
        if self.suspended:
            return
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
