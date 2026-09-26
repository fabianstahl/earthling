"""Deterministic offscreen rendering of timeline frames (stills and video frames)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
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


def halton(index: int, base: int) -> float:
    """Low-discrepancy sequence in [0, 1) (sub-pixel sample positions)."""
    f, result = 1.0, 0.0
    while index > 0:
        f /= base
        result += f * (index % base)
        index //= base
    return result


@dataclass
class ExportQuality:
    detail: float = 0.75  # terrain detail threshold (px per texel)
    shadow_resolution: str = "8192"
    samples: int = 1  # jittered sub-frames per frame (anti-aliasing)
    motion_blur: float = 0.0  # shutter as a fraction of the frame time

    @classmethod
    def from_store(cls, store) -> ExportQuality:
        return cls(
            store["export.detail"],
            store["export.shadow_resolution"],
            int(store["export.samples"]),
            float(store["export.motion_blur"]),
        )

    @property
    def subframes(self) -> int:
        if self.motion_blur > 0.0:
            return max(self.samples, 8)
        return max(self.samples, 1)


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
        renderer.animation = animation
        self.store = animation.store
        self.rig = rig or CameraRig(animation, self.store)
        self.quality = quality or ExportQuality.from_store(self.store)
        self.on_time = on_time  # e.g. keep the GUI timeline in sync
        self._size = (0, 0)
        self._color: moderngl.Texture | None = None
        self._fbo: moderngl.Framebuffer | None = None
        self._acc: moderngl.Texture | None = None
        self._acc_fbo: moderngl.Framebuffer | None = None

    def _ensure_target(self, width: int, height: int) -> moderngl.Framebuffer:
        if (width, height) != self._size or self._fbo is None:
            self.release()
            # 16-bit float output keeps full precision for 10-bit video codecs
            self._color = self.ctx.texture((width, height), 4, dtype="f2")
            self._fbo = self.ctx.framebuffer([self._color])
            self._size = (width, height)
        return self._fbo

    def _accumulator(self) -> moderngl.Framebuffer:
        if self._acc_fbo is None:
            self._acc = self.ctx.texture(self._size, 4, dtype="f4")
            self._acc_fbo = self.ctx.framebuffer([self._acc])
        return self._acc_fbo

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

    def _draw(self, time: float, width: int, height: int) -> moderngl.Framebuffer:
        """Render ``time`` at full quality into the offscreen target (no readback). With
        anti-aliasing / motion blur, the average of jittered sub-frames spread over the
        shutter interval."""
        n = self.quality.subframes
        if n <= 1:
            return self._draw_single(time, width, height)
        if self.on_time is not None:
            self.on_time(time)
        shutter = self.quality.motion_blur / max(self.animation.fps, 1e-6)
        jitter = self.quality.samples > 1
        renderer = self.renderer
        fbo = self._ensure_target(width, height)
        acc = self._accumulator()
        acc.use()
        acc.clear(0.0, 0.0, 0.0, 0.0)
        passes = renderer.fullscreen
        program = passes("accumulate")
        try:
            for i in range(n):
                renderer.jitter_px = (
                    (halton(i + 1, 2) - 0.5, halton(i + 1, 3) - 0.5) if jitter else (0.0, 0.0)
                )
                renderer.jitter_index = i
                t = time + ((i + 0.5) / n - 0.5) * shutter
                self._draw_single(t, width, height, notify=False)
                acc.use()
                self.ctx.viewport = (0, 0, width, height)
                self.ctx.enable(moderngl.BLEND)
                self.ctx.blend_func = moderngl.ONE, moderngl.ONE
                assert self._color is not None
                self._color.use(0)
                program["u_source"] = 0
                program["u_weight"] = 1.0 / n
                passes.draw("accumulate")
                self.ctx.disable(moderngl.BLEND)
        finally:
            renderer.jitter_px = (0.0, 0.0)
            renderer.jitter_index = 0
            renderer.reset_state()
        # resolve into the (half float) output target
        fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        assert self._acc is not None
        self._acc.use(0)
        program["u_weight"] = 1.0
        passes.draw("accumulate")
        if shutter > 0.0:
            self.animation.apply(time)  # leave the scene at the frame time
        return fbo

    def _draw_single(
        self, time: float, width: int, height: int, notify: bool = True
    ) -> moderngl.Framebuffer:
        self.animation.apply(time)
        if notify and self.on_time is not None:
            self.on_time(time)
        renderer = self.renderer
        previous_overrides = renderer.overrides
        renderer.overrides = {
            "terrain.detail": self.quality.detail,
            "shadows.resolution": self.quality.shadow_resolution,
        }
        previous_time = renderer.time
        previous_timeline = renderer.timeline_time
        try:
            renderer.time = time
            renderer.timeline_time = time
            # the track geometry (and so the follow camera) needs one pass to be built
            if renderer.tracks.path is None and renderer.tracks.tracks:
                renderer.render(self._ensure_target(width, height), width, height, Camera())
            camera = self.camera_at(time)
            renderer.prepare(camera, width, height)
            fbo = self._ensure_target(width, height)
            renderer.render(fbo, width, height, camera)
        finally:
            renderer.overrides = previous_overrides
            renderer.time = previous_time
            renderer.timeline_time = previous_timeline
        return fbo

    def render(self, time: float, width: int, height: int, bits: int = 8) -> np.ndarray:
        """RGB image (height, width, 3), uint8 for ``bits`` = 8, else uint16, top row first."""
        fbo = self._draw(time, width, height)
        self.ctx.finish()
        return to_image(fbo.read(components=3, dtype="f2"), width, height, bits)

    def iter_frames(
        self, times: Iterable[float], width: int, height: int, bits: int = 8
    ) -> Iterator[np.ndarray]:
        """Render a sequence of frames. The readback of each frame goes into a pixel buffer
        object and is only collected after the next frame has been submitted, so the GPU
        keeps rendering while the previous frame is copied and converted."""
        size = width * height * 3 * 2
        buffers = [self.ctx.buffer(reserve=size) for _ in range(2)]
        pending: int | None = None
        try:
            for i, t in enumerate(times):
                fbo = self._draw(t, width, height)
                fbo.read_into(buffers[i % 2], components=3, dtype="f2")
                if pending is not None:
                    yield to_image(buffers[pending].read(), width, height, bits)
                pending = i % 2
            if pending is not None:
                yield to_image(buffers[pending].read(), width, height, bits)
        finally:
            for buf in buffers:
                buf.release()

    def release(self) -> None:
        for obj in (self._fbo, self._color, self._acc_fbo, self._acc):
            if obj is not None:
                obj.release()
        self._fbo = self._color = self._acc_fbo = self._acc = None
        self._size = (0, 0)


def to_image(raw: bytes, width: int, height: int, bits: int) -> np.ndarray:
    """Bottom-up float16 RGB readback -> top-down uint8/uint16 image."""
    img = np.frombuffer(raw, dtype=np.float16).reshape(height, width, 3)[::-1]
    img = np.clip(img.astype(np.float32), 0.0, 1.0)
    if bits == 8:
        return (img * 255.0 + 0.5).astype(np.uint8)
    return (img * 65535.0 + 0.5).astype(np.uint16)


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
