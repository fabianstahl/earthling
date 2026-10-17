"""Photo overlays: pictures on the screen with a white frame, a soft shadow and a slow zoom.

Drawn after tone mapping at output resolution (like the HUD), in display colours. Positions
and sizes are fractions of the frame (``photo.<id>.*``), borders are given for 1080p and scale
with the output height.
"""

from __future__ import annotations

import math
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image, ImageOps

from earthling.core.photos import Photo
from earthling.render.gl_util import DynamicBuffer, draw_overlay, set_uniforms
from earthling.render.scale import pixel_scale
from earthling.render.shader_library import ShaderLibrary

MAX_PHOTO_PX = 2048  # longest side on the GPU (a third of a 4K frame needs ~1300 px)


def resolve_photo(ref: str, base_dir: Path | None) -> Path | None:
    if not ref:
        return None
    path = Path(ref).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path if path.exists() else None


def load_photo(path: Path, max_px: int = MAX_PHOTO_PX) -> np.ndarray:
    """RGB uint8, upright (EXIF orientation applied), longest side at most ``max_px``."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        scale = min(1.0, max_px / max(img.width, img.height))
        if scale < 1.0:
            size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
            img = img.resize(size, Image.Resampling.LANCZOS)
        return np.asarray(img)


class PhotoLayer:
    def __init__(self, ctx: moderngl.Context, shaders: ShaderLibrary) -> None:
        self.ctx = ctx
        self.shaders = shaders
        self.photos: list[Photo] = []
        self.base_dir: Path | None = None
        self._textures: dict[tuple, tuple[moderngl.Texture, float] | None] = {}
        self._stream = DynamicBuffer(ctx, "2f", "in_local", min_bytes=0)
        self.last_drawn: list[str] = []  # ids drawn in the last frame (tests)

    def set_photos(self, photos: list[Photo], base_dir: Path | None = None) -> None:
        self.photos = list(photos)
        self.base_dir = base_dir

    def _texture(self, photo: Photo) -> tuple[moderngl.Texture, float] | None:
        """(texture, width / height) of the photo, or None when the file is missing."""
        path = resolve_photo(photo.path, self.base_dir)
        if path is None:
            return None
        key = (str(path), path.stat().st_mtime_ns)
        if key not in self._textures:
            try:
                rgb = load_photo(path)
            except Exception:  # unreadable image: nothing drawn
                self._textures[key] = None
                return None
            h, w = rgb.shape[:2]
            # (row 0 = the top of the picture: v grows downwards like the screen y)
            tex = self.ctx.texture((w, h), 3, np.ascontiguousarray(rgb).tobytes())
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex.anisotropy = 8.0
            tex.repeat_x = tex.repeat_y = False
            self._textures[key] = (tex, w / h)
        return self._textures[key]

    def render(self, fbo, width: int, height: int, store) -> None:
        self.last_drawn = []
        if not self.photos or store is None:
            return
        k = pixel_scale(width, height)
        program = None
        for photo in self.photos:
            p = f"photo.{photo.id}."
            if p + "opacity" not in store.registry:
                continue
            alpha = float(store[p + "opacity"])
            size = float(store[p + "height"]) * float(store[p + "scale"])
            if alpha <= 0.0 or size <= 0.0:
                continue
            entry = self._texture(photo)
            if entry is None:
                continue
            texture, aspect = entry
            if program is None:
                program = self.shaders.get("photo")
            half_h = 0.5 * size * height
            half = (half_h * aspect, half_h)
            border = float(store[p + "border"]) * k * float(store[p + "scale"])
            shadow = float(store[p + "shadow"])
            blur = 14.0 * k
            offset = (5.0 * k, 9.0 * k)
            reach = border + blur + max(offset) + 2.0
            x, y = half[0] + reach, half[1] + reach
            quad = np.array([[-x, -y], [x, -y], [x, y], [-x, -y], [x, y], [-x, y]], dtype="f4")
            self._draw(program, fbo, texture, quad, width, height, {
                "u_center": (float(store[p + "x"]) * width, float(store[p + "y"]) * height),
                "u_angle": math.radians(float(store[p + "rotation"])),
                "u_half": half, "u_border": border, "u_shadow": shadow,
                "u_shadow_blur": blur, "u_shadow_offset": offset,
                "u_zoom": max(1.0, float(store[p + "zoom"])), "u_alpha": min(alpha, 1.0),
            })  # fmt: skip
            self.last_drawn.append(photo.id)

    def _draw(self, program, fbo, texture, quad, width, height, uniforms) -> None:
        vao = self._stream.vertex_array(program, quad.tobytes())
        texture.use(0)
        set_uniforms(program, {"u_image": 0, "u_viewport": (float(width), float(height)),
                               **uniforms})  # fmt: skip
        draw_overlay(self.ctx, fbo, vao, len(quad), width, height)

    def release(self) -> None:
        for entry in self._textures.values():
            if entry is not None:
                entry[0].release()
        self._textures.clear()
        self._stream.release()
