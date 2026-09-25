"""OpenGL viewport widget hosting a moderngl context."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import moderngl
from PyQt6.QtCore import QFileSystemWatcher, QPointF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QSurfaceFormat, QWheelEvent
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from earthling.core.geo import LocalFrame
from earthling.core.gpx import Track
from earthling.render.camera import Camera, OrbitController
from earthling.render.renderer import Renderer

log = logging.getLogger(__name__)


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
    shader_error = pyqtSignal(str)

    def __init__(self, dev_mode: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.dev_mode = dev_mode
        self.ctx: moderngl.Context | None = None
        self.renderer: Renderer | None = None
        self.camera = Camera()
        self.orbit = OrbitController(self.camera)
        self._pending_scene: tuple[LocalFrame, list[Track]] | None = None
        self._pending_outlines = None
        self._pending_terrain = None
        self._outlines_visible = True
        self._debug_lod = False
        self._last_mouse: QPointF | None = None
        self._frames = 0
        self._fps_timer = time.perf_counter()
        self._watcher: QFileSystemWatcher | None = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Continuous redraw; later steps switch to on-demand rendering where possible.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(0)

    # --- scene -------------------------------------------------------------------------
    def set_scene(self, frame: LocalFrame, tracks: list[Track], reframe: bool = True) -> None:
        if self.renderer is None:
            self._pending_scene = (frame, tracks)
            return
        self.makeCurrent()
        self.renderer.set_scene(frame, tracks)
        self.doneCurrent()
        if reframe:
            self.frame_all()

    def set_terrain_source(self, data, nodes) -> None:
        if self.renderer is None:
            self._pending_terrain = (data, nodes)
            return
        self.makeCurrent()
        self.renderer.set_terrain_source(data, nodes)
        self.doneCurrent()

    def set_outlines(self, lines) -> None:
        if self.renderer is None:
            self._pending_outlines = lines
            return
        self.makeCurrent()
        self.renderer.outlines.set_lines(lines)
        self.doneCurrent()

    def set_outlines_visible(self, visible: bool) -> None:
        self._outlines_visible = visible
        if self.renderer is not None:
            self.renderer.outlines.visible = visible

    def set_on_demand(self, enabled: bool) -> None:
        if self.renderer is not None and self.renderer.terrain.data is not None:
            self.renderer.terrain.data.on_demand = enabled

    def shutdown(self) -> None:
        if self.renderer is not None:
            self.makeCurrent()
            self.renderer.terrain.shutdown()
            self.doneCurrent()

    def set_debug_lod(self, enabled: bool) -> None:
        self._debug_lod = enabled
        if self.renderer is not None:
            self.renderer.terrain.debug_lod = enabled

    def frame_all(self) -> None:
        bounds = self.renderer.scene_bounds() if self.renderer is not None else None
        if bounds is not None:
            self.orbit.frame_bounds(*bounds)

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
            self.frame_all()
        if self._pending_terrain is not None:
            self.renderer.set_terrain_source(*self._pending_terrain)
            self._pending_terrain = None
        if self._pending_outlines is not None:
            self.renderer.outlines.set_lines(self._pending_outlines)
            self._pending_outlines = None
        self._pending_terrain = None
        self.renderer.outlines.visible = self._outlines_visible
        self.renderer.terrain.debug_lod = self._debug_lod

    def paintGL(self) -> None:
        if self.ctx is None or self.renderer is None:
            return
        fbo = self.ctx.detect_framebuffer(self.defaultFramebufferObject())
        ratio = self.devicePixelRatio()
        width, height = int(self.width() * ratio), int(self.height() * ratio)
        self.renderer.render(fbo, width, height, self.camera)
        self._sync_watcher()
        self._count_frame()

    # --- input -------------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last_mouse = event.position()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._last_mouse = None

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_mouse is None:
            return
        delta = event.position() - self._last_mouse
        self._last_mouse = event.position()
        buttons = event.buttons()
        if buttons & Qt.MouseButton.LeftButton:
            self.orbit.rotate(delta.x(), delta.y())
        elif buttons & (Qt.MouseButton.RightButton | Qt.MouseButton.MiddleButton):
            self.orbit.pan(delta.x(), delta.y(), self.height())

    def wheelEvent(self, event: QWheelEvent) -> None:
        self.orbit.zoom(event.angleDelta().y() / 120.0)

    # --- helpers -----------------------------------------------------------------------
    def _count_frame(self) -> None:
        self._frames += 1
        now = time.perf_counter()
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
