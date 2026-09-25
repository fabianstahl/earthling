"""Deterministic offscreen rendering of timeline frames (stills and video frames)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import moderngl
import numpy as np

from earthling.core.animation import Animation
from earthling.render.camera import Camera
from earthling.render.camera_rig import CameraRig, clamp_above_ground
from earthling.render.renderer import Renderer


def parse_resolution(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


@dataclass
class ExportQuality:
    detail: float = 0.75  # terrain detail threshold (px per texel)
    shadow_resolution: str = "8192"

    @classmethod
    def from_store(cls, store) -> ExportQuality:
        return cls(store["export.detail"], store["export.shadow_resolution"])


class FrameRenderer:
    """Renders any timeline time at full quality into an offscreen framebuffer.

    Every frame: apply the animation, compute the camera from the rig, wait until all
    terrain for the view and its shadow casters is loaded, render, read back.
    """

    def __init__(
        self,
        renderer: Renderer,
        animation: Animation,
        rig: CameraRig | None = None,
        quality: ExportQuality | None = None,
        on_time: Callable[[float], None] | None = None,
    ) -> None:
        self.renderer = renderer
        self.ctx: moderngl.Context = renderer.ctx
        self.animation = animation
        self.store = animation.store
        self.rig = rig or CameraRig(animation, self.store)
        self.quality = quality or ExportQuality.from_store(self.store)
        self.on_time = on_time  # e.g. keep the GUI timeline in sync
        self._size = (0, 0)
        self._color: moderngl.Texture | None = None
        self._fbo: moderngl.Framebuffer | None = None

    def _ensure_target(self, width: int, height: int) -> moderngl.Framebuffer:
        if (width, height) != self._size or self._fbo is None:
            self.release()
            # 16-bit float output keeps full precision for 10-bit video codecs
            self._color = self.ctx.texture((width, height), 4, dtype="f2")
            self._fbo = self.ctx.framebuffer([self._color])
            self._size = (width, height)
        return self._fbo

    def camera_at(self, time: float) -> Camera:
        self.rig.path = self.renderer.tracks.path
        pose = self.rig.pose(time)
        pose = clamp_above_ground(
            pose,
            self.renderer.frame,
            self.renderer.terrain.data,
            self.renderer.terrain.exaggeration,
        )
        camera = Camera()
        x, y, z, heading, pitch, roll = pose
        camera.position = np.array([x, y, z], dtype=np.float64)
        camera.heading, camera.pitch, camera.roll = heading, pitch, roll
        return camera

    def render(self, time: float, width: int, height: int, bits: int = 8) -> np.ndarray:
        """RGB image (height, width, 3), uint8 for ``bits`` = 8, else uint16, top row first."""
        self.animation.apply(time)
        if self.on_time is not None:
            self.on_time(time)
        renderer = self.renderer
        previous_overrides = renderer.overrides
        renderer.overrides = {
            "terrain.detail": self.quality.detail,
            "shadows.resolution": self.quality.shadow_resolution,
        }
        previous_time = renderer.time
        try:
            renderer.time = time
            # the track geometry (and so the follow camera) needs one pass to be built
            if renderer.tracks.path is None and renderer.tracks.tracks:
                renderer.render(self._ensure_target(width, height), width, height, Camera())
            camera = self.camera_at(time)
            renderer.prepare(camera, width, height)
            fbo = self._ensure_target(width, height)
            renderer.render(fbo, width, height, camera)
            self.ctx.finish()
            raw = fbo.read(components=3, dtype="f2")
        finally:
            renderer.overrides = previous_overrides
            renderer.time = previous_time
        img = np.frombuffer(raw, dtype=np.float16).reshape(height, width, 3)[::-1]
        img = np.clip(img.astype(np.float32), 0.0, 1.0)
        if bits == 8:
            return (img * 255.0 + 0.5).astype(np.uint8)
        return (img * 65535.0 + 0.5).astype(np.uint16)

    def release(self) -> None:
        for obj in (self._fbo, self._color):
            if obj is not None:
                obj.release()
        self._fbo = self._color = None
        self._size = (0, 0)


def save_png(image: np.ndarray, path) -> None:
    """8-bit RGB or 16-bit RGB PNG."""
    from PIL import Image

    if image.dtype == np.uint16:
        # Pillow cannot write 16-bit RGB directly; go through a 48-bit raw buffer
        import zlib

        _write_png16(image, path, zlib)
        return
    Image.fromarray(image).save(path)


def _write_png16(image: np.ndarray, path, zlib) -> None:
    import struct

    h, w, _ = image.shape
    raw = b"".join(b"\x00" + image[y].astype(">u2").tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(png)
