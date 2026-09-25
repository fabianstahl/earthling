"""Video encoding through ffmpeg (raw frames piped to the bundled ffmpeg binary)."""

from __future__ import annotations

import contextlib
import queue
import subprocess
import threading
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

COLOR_TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]


@dataclass(frozen=True)
class VideoPreset:
    id: str
    label: str
    extension: str
    bits: int  # 8: rgb24 input, 16: rgb48 input (for 10-bit codecs)
    args: tuple[str, ...]
    encoder: str  # ffmpeg encoder that must be available


PRESETS: dict[str, VideoPreset] = {
    p.id: p
    for p in (
        VideoPreset("prores4444", "ProRes 4444 (.mov, 10-bit 4:4:4, for editing)", ".mov", 16,
                    ("-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuv444p10le",
                     "-vendor", "apl0", "-qscale:v", "6"), "prores_ks"),
        VideoPreset("prores422hq", "ProRes 422 HQ (.mov, 10-bit 4:2:2)", ".mov", 16,
                    ("-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le",
                     "-vendor", "apl0"), "prores_ks"),
        VideoPreset("dnxhr444", "DNxHR 444 (.mov, 10-bit 4:4:4, for editing)", ".mov", 16,
                    ("-c:v", "dnxhd", "-profile:v", "dnxhr_444", "-pix_fmt", "yuv444p10le"),
                    "dnxhd"),
        VideoPreset("h264", "H.264 (.mp4, 8-bit, high quality)", ".mp4", 8,
                    ("-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p",
                     "-movflags", "+faststart"), "libx264"),
        VideoPreset("h265", "H.265 / HEVC (.mp4, 10-bit)", ".mp4", 16,
                    ("-c:v", "libx265", "-preset", "medium", "-crf", "16",
                     "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1", "-movflags", "+faststart"),
                    "libx265"),
        VideoPreset("h264_nvenc", "H.264 NVENC (.mp4, fast, NVIDIA GPU)", ".mp4", 8,
                    ("-c:v", "h264_nvenc", "-preset", "p6", "-rc", "vbr", "-cq", "16",
                     "-b:v", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart"),
                    "h264_nvenc"),
        VideoPreset("hevc_nvenc", "H.265 NVENC (.mp4, 10-bit, fast, NVIDIA GPU)", ".mp4", 16,
                    ("-c:v", "hevc_nvenc", "-preset", "p6", "-rc", "vbr", "-cq", "18",
                     "-b:v", "0", "-pix_fmt", "p010le", "-tag:v", "hvc1",
                     "-movflags", "+faststart"), "hevc_nvenc"),
    )
}  # fmt: skip


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@cache
def available_encoders() -> frozenset[str]:
    try:
        out = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception:  # no ffmpeg at all
        return frozenset()
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            names.add(parts[1])
    return frozenset(names)


def available_presets() -> list[VideoPreset]:
    encoders = available_encoders()
    return [p for p in PRESETS.values() if p.encoder in encoders]


class VideoWriterError(RuntimeError):
    pass


class VideoWriter:
    """Streams frames to ffmpeg. Frames are converted and written on a background thread,
    so the GPU can render the next frame meanwhile (``queue_size`` frames in flight)."""

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        fps: float,
        preset: VideoPreset,
        queue_size: int = 3,
    ) -> None:
        self.path = Path(path)
        self.width, self.height, self.fps = width, height, fps
        self.preset = preset
        pix_fmt = "rgb48le" if preset.bits == 16 else "rgb24"
        cmd = [
            ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{width}x{height}",
            "-r", f"{fps:g}", "-i", "-",
            *preset.args, *COLOR_TAGS, "-r", f"{fps:g}", str(self.path),
        ]  # fmt: skip
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL
        )
        self._stderr: list[bytes] = []
        self._err_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._err_thread.start()
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._error: BaseException | None = None
        self._writer = threading.Thread(target=self._write_loop, daemon=True, name="ffmpeg-writer")
        self._writer.start()
        self.frames_written = 0

    def _read_stderr(self) -> None:
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            self._stderr.append(line)

    def _write_loop(self) -> None:
        assert self._proc.stdin is not None
        while True:
            frame = self._queue.get()
            if frame is None:
                break
            try:
                if self.preset.bits == 16:
                    data = np.ascontiguousarray(frame, dtype="<u2").tobytes()
                else:
                    data = np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
                self._proc.stdin.write(data)
                self.frames_written += 1
            except BaseException as exc:  # broken pipe: ffmpeg died
                self._error = exc
                break

    def write(self, frame: np.ndarray) -> None:
        if self._error is not None or self._proc.poll() is not None:
            raise VideoWriterError(self._message("ffmpeg stopped"))
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"frame shape {frame.shape} != {(self.height, self.width, 3)}")
        self._queue.put(frame)

    def _message(self, prefix: str) -> str:
        detail = b"".join(self._stderr[-20:]).decode("utf-8", "replace").strip()
        return f"{prefix}: {detail}" if detail else prefix

    def close(self) -> None:
        self._queue.put(None)
        self._writer.join()
        if self._proc.stdin is not None:
            with contextlib.suppress(OSError):
                self._proc.stdin.close()
        code = self._proc.wait()
        self._err_thread.join(timeout=5)
        if code != 0 or self._error is not None:
            raise VideoWriterError(self._message(f"ffmpeg failed (exit code {code})"))

    def abort(self) -> None:
        """Stop without finishing the file (cancel)."""
        self._error = self._error or InterruptedError()
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)
        self._proc.kill()
        self._proc.wait()
        self._writer.join(timeout=5)


def export_video(
    frames,
    path: Path,
    width: int,
    height: int,
    fps: float,
    preset: VideoPreset,
    start: float,
    end: float,
    on_progress=None,
    should_cancel=None,
) -> int:
    """Render and encode [start, end] (inclusive of the start frame). Returns frames written.

    ``frames``: an :class:`earthling.export.frames.FrameRenderer`.
    """
    first = int(round(start * fps))
    last = int(round(end * fps))
    count = max(0, last - first)
    times = [(first + i) / fps for i in range(count)]
    if hasattr(frames, "iter_frames"):
        images = frames.iter_frames(times, width, height, bits=preset.bits)
    else:
        images = (frames.render(t, width, height, bits=preset.bits) for t in times)
    writer = VideoWriter(path, width, height, fps, preset)
    done = 0
    try:
        while done < count:
            if should_cancel is not None and should_cancel():
                writer.abort()
                return done
            writer.write(next(images))
            done += 1
            if on_progress is not None:
                on_progress(done, count)
    except BaseException:
        writer.abort()
        raise
    finally:
        close = getattr(images, "close", None)
        if close is not None:
            close()
    writer.close()
    return count
