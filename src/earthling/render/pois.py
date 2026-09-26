"""Points of interest: 3D-anchored icons (static or animated) with effects, pins and captions.

Drawn after tone mapping at output resolution, like the labels. Occlusion is tested in the
vertex shader against the scene depth (3 x 3 taps around the anchor), so no readback is needed.
Sizes are given for 1080p output and scale with the output height.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

from earthling.core.pois import Poi, visibility_state
from earthling.render.labels import BASE_PX, SPREAD, FontAtlas
from earthling.render.shader_library import ShaderLibrary

ICON_DIR = Path(__file__).resolve().parents[1] / "assets" / "icons"
REFERENCE_HEIGHT = 1080.0
MAX_ICON_PX = 256
LIFT_M = 3.0  # anchor above the ground (avoids z-fighting with the terrain in the depth test)


def builtin_icons() -> list[str]:
    return sorted(p.stem for p in ICON_DIR.glob("*.png"))


def resolve_icon(ref: str, base_dir: Path | None) -> Path | None:
    if not ref:
        return None
    if ref.startswith("builtin:"):
        path = ICON_DIR / f"{ref.split(':', 1)[1]}.png"
        return path if path.exists() else None
    path = Path(ref).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path if path.exists() else None


@dataclass
class IconFrames:
    frames: list[np.ndarray]  # RGBA uint8, all the same size
    durations: list[float] = field(default_factory=list)  # seconds per frame

    @property
    def total(self) -> float:
        return sum(self.durations) if self.durations else 0.0

    def index_at(self, t: float) -> int:
        if len(self.frames) <= 1 or self.total <= 0:
            return 0
        t = t % self.total
        acc = 0.0
        for i, d in enumerate(self.durations):
            acc += d
            if t < acc:
                return i
        return len(self.frames) - 1


def load_icon(path: Path, cols: int = 1, rows: int = 1, fps: float = 12.0) -> IconFrames:
    """PNG (optionally a cols x rows sprite sheet), animated GIF or APNG."""
    img = Image.open(path)
    frames: list[Image.Image] = []
    durations: list[float] = []
    n = getattr(img, "n_frames", 1)
    if n > 1:  # GIF / APNG: the reader composites the frames when seeking in order
        for i in range(n):
            img.seek(i)
            frames.append(img.convert("RGBA").copy())
            durations.append(max(float(img.info.get("duration", 100) or 100), 20.0) / 1000.0)
    else:
        sheet = img.convert("RGBA")
        cols, rows = max(1, int(cols)), max(1, int(rows))
        w, h = sheet.width // cols, sheet.height // rows
        for r in range(rows):
            for c in range(cols):
                frames.append(sheet.crop((c * w, r * h, (c + 1) * w, (r + 1) * h)))
        durations = [1.0 / max(fps, 0.1)] * len(frames)
    scale = min(1.0, MAX_ICON_PX / max(frames[0].width, frames[0].height))
    if scale < 1.0:
        size = (max(1, round(frames[0].width * scale)), max(1, round(frames[0].height * scale)))
        frames = [f.resize(size, Image.Resampling.LANCZOS) for f in frames]
    return IconFrames([np.asarray(f) for f in frames], durations if len(frames) > 1 else [])


@dataclass
class _GpuIcon:
    texture: moderngl.Texture
    frames: IconFrames
    cols: int
    size: tuple[int, int]  # one frame (px)


class PoiLayer:
    def __init__(self, ctx: moderngl.Context, shaders: ShaderLibrary, atlas_provider) -> None:
        self.ctx = ctx
        self.shaders = shaders
        self.atlas_provider = atlas_provider
        self.pois: list[Poi] = []
        self.base_dir: Path | None = None
        self._icons: dict[tuple, _GpuIcon | None] = {}
        self._anchors: dict[tuple, np.ndarray] = {}
        self._buffer: moderngl.Buffer | None = None
        self._vaos: dict[int, moderngl.VertexArray] = {}
        self.last_drawn: list[str] = []  # ids drawn in the last frame (tests)

    def set_pois(self, pois: list[Poi], base_dir: Path | None = None) -> None:
        self.pois = list(pois)
        self.base_dir = base_dir

    # --- resources ----------------------------------------------------------------------
    def _icon(self, poi: Poi) -> _GpuIcon | None:
        path = resolve_icon(poi.icon, self.base_dir) or resolve_icon("builtin:pin", None)
        if path is None:
            return None
        key = (str(path), path.stat().st_mtime_ns, poi.sprite_cols, poi.sprite_rows, poi.fps)
        if key in self._icons:
            return self._icons[key]
        try:
            frames = load_icon(path, poi.sprite_cols, poi.sprite_rows, poi.fps)
        except Exception:  # unreadable image: fall back to the pin
            self._icons[key] = (
                None
                if poi.icon == "builtin:pin"
                else self._icon(Poi(poi.id, poi.name, poi.lon, poi.lat))
            )
            return self._icons[key]
        h, w = frames.frames[0].shape[:2]
        cols = math.ceil(math.sqrt(len(frames.frames)))
        rows = math.ceil(len(frames.frames) / cols)
        sheet = np.zeros((rows * h, cols * w, 4), dtype=np.float32)
        for i, f in enumerate(frames.frames):
            r, c = divmod(i, cols)
            rgba = f.astype(np.float32) / 255.0
            rgba[..., :3] *= rgba[..., 3:4]  # premultiplied: no dark fringes when filtering
            sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = rgba
        tex = self.ctx.texture((cols * w, rows * h), 4, (sheet * 255.0 + 0.5).astype(np.uint8))
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.repeat_x = tex.repeat_y = False
        icon = _GpuIcon(tex, frames, cols, (w, h))
        self._icons[key] = icon
        return icon

    def _anchor(self, poi: Poi, frame, terrain_data, exaggeration: float) -> np.ndarray:
        key = (poi.id, poi.lon, poi.lat, poi.height_offset_m, exaggeration, id(terrain_data),
               id(frame))  # fmt: skip
        anchor = self._anchors.get(key)
        if anchor is None:
            ground = None
            if terrain_data is not None and hasattr(terrain_data, "height_at"):
                ground = terrain_data.height_at(poi.lon, poi.lat)
            h = (ground or 0.0) * exaggeration + poi.height_offset_m + LIFT_M
            anchor = np.asarray(frame.geodetic_to_enu(poi.lat, poi.lon, h), dtype=np.float64)
            anchor = anchor.reshape(3)
            if ground is not None:  # heights not loaded yet are not cached
                self._anchors[key] = anchor
        return anchor

    # --- drawing -------------------------------------------------------------------------
    def render(self, fbo, camera, view_proj, frame, terrain_data, exaggeration, width, height,
               depth, depth_size, store, time: float, animation) -> None:  # fmt: skip
        self.last_drawn = []
        if not self.pois or store is None or frame is None or depth is None:
            return
        k = height / REFERENCE_HEIGHT
        program = self.shaders.get("poi")
        atlas: FontAtlas = self.atlas_provider()
        for poi in self.pois:
            p = f"poi.{poi.id}."
            if p + "visible" not in store.registry:
                continue
            pop, since = visibility_state(animation, poi.id, time, store[p + "visible"])
            if not store[p + "pop"]:
                pop = 1.0 if store[p + "visible"] else 0.0
            opacity = store[p + "opacity"] * min(1.0, pop * 1.5)
            if opacity <= 0.0 or pop <= 0.0:
                continue
            icon = self._icon(poi)
            if icon is None:
                continue
            anchor = self._anchor(poi, frame, terrain_data, exaggeration)
            rel = anchor - np.asarray(camera.position, dtype=np.float64)
            vertices = self._geometry(poi, icon, rel, store, since, pop, opacity, k, atlas)
            if not len(vertices):
                continue
            self._draw(program, fbo, icon, atlas, vertices, camera, view_proj, width, height,
                       depth, depth_size, k)  # fmt: skip
            self.last_drawn.append(poi.id)

    def _geometry(self, poi, icon, rel, store, since, pop, opacity, k, atlas) -> np.ndarray:
        p = f"poi.{poi.id}."
        size = poi.size_px * k * store[p + "scale"] * pop
        effect = store[p + "effect"]
        strength = store[p + "effect_strength"]
        phase = 2.0 * math.pi * store[p + "effect_speed"] * since
        angle, dy = 0.0, 0.0
        if effect == "bounce":
            dy = -abs(math.sin(phase / 2.0)) * 0.35 * size * strength
        elif effect == "pulse":
            size *= 1.0 + 0.15 * strength * math.sin(phase)
        elif effect == "wobble":
            angle = math.radians(12.0 * strength * math.sin(phase))
        elif effect == "spin":
            angle = phase
        elif effect == "float":
            dy = -0.08 * size * strength * (0.5 + 0.5 * math.sin(phase))
        fw, fh = icon.size
        w = size * fw / max(fh, 1)
        iconless = poi.icon in ("", "none")  # a caption only (e.g. a country name)
        lift = poi.lift_px * k * pop if store[p + "pin"] and not iconless else 0.0
        cx, cy = 0.0, -lift - size / 2.0 + dy
        if iconless:
            cy = 0.0
        index = icon.frames.index_at(since * store[p + "anim_speed"])
        rows = math.ceil(len(icon.frames.frames) / icon.cols)
        r, c = divmod(index, icon.cols)
        u0, v0 = c / icon.cols, r / rows
        u1, v1 = (c + 1) / icon.cols, (r + 1) / rows
        parts = []
        if lift > 0.0:
            parts.append(_quad(rel, -1.4 * k, -lift, 1.4 * k, 0.0, opacity, 2.0))
            parts.append(_quad(rel, -0.6 * k, -lift, 0.6 * k, 0.0, opacity, 3.0))
            parts.append(_quad(rel, -2.8 * k, -2.8 * k, 2.8 * k, 2.8 * k, opacity, 2.0))
        hw, hh = w / 2, size / 2
        corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, -hh], [hw, hh], [-hw, hh]])
        uvs = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v0], [u1, v1], [u0, v1]])
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotated = corners @ np.array([[cos_a, sin_a], [-sin_a, cos_a]])
        icon_quad = np.zeros((6, 10))
        icon_quad[:, :3] = rel
        icon_quad[:, 3:5] = rotated + (cx, cy)
        icon_quad[:, 5:7] = uvs
        icon_quad[:, 7], icon_quad[:, 8], icon_quad[:, 9] = opacity, 0.0, 1.0
        if not iconless:
            parts.append(icon_quad)
        if store[p + "caption"] and poi.caption:
            lines = poi.caption.split("\n")
            atlas.ensure("".join(lines))  # all glyphs first: adding some repacks the atlas
            title_size = store[p + "caption_size"] * k * pop
            sizes = [title_size] + [title_size * 0.78] * (len(lines) - 1)
            # stacked upwards from above the icon (centred on the anchor without an icon)
            total = sum(s * 1.2 for s in sizes)
            base = cy - size / 2.0 - 8.0 * k if not iconless else total / 2.0 - sizes[-1] * 0.25
            for line, text_size in zip(reversed(lines), reversed(sizes), strict=True):
                scale = text_size / BASE_PX
                rects, glyph_uvs = atlas.quads(line)
                if len(rects):
                    width = atlas.width(line) * scale
                    q = rects * scale
                    q[:, [0, 2]] += -width / 2.0
                    q[:, [1, 3]] += base
                    parts.append(_glyphs(rel, q, glyph_uvs, opacity, scale))
                base -= text_size * 1.2
        if not parts:
            return np.zeros((0, 10), dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)

    def _draw(self, program, fbo, icon, atlas, vertices, camera, view_proj, width, height,
              depth, depth_size, k) -> None:  # fmt: skip
        data = vertices.tobytes()
        if self._buffer is None or self._buffer.size < len(data):
            if self._buffer is not None:
                self._buffer.release()
                for vao in self._vaos.values():
                    vao.release()
                self._vaos.clear()
            self._buffer = self.ctx.buffer(reserve=max(len(data), 1 << 14) * 2)
        self._buffer.write(data)
        vao = self._vaos.get(program.glo)
        if vao is None:
            vao = self.ctx.vertex_array(
                program,
                [(self._buffer, "3f 2f 2f 3f", "in_anchor", "in_offset", "in_uv", "in_params")],
            )
            self._vaos[program.glo] = vao
        icon.texture.use(0)
        if atlas.texture is not None:
            atlas.texture.use(1)
        previous = depth.compare_func
        depth.compare_func = ""
        depth.use(2)
        for name, value in (
            ("u_icon", 0), ("u_atlas", 1), ("u_depth", 2),
            ("u_viewport", (float(width), float(height))),
            ("u_depth_size", (float(depth_size[0]), float(depth_size[1]))),
            ("u_log_depth_coef", camera.log_depth_coef), ("u_spread", float(SPREAD)),
            ("u_outline_px", float(1.8 * k)),
        ):  # fmt: skip
            if name in program:
                program[name] = value
        program["u_view_proj"].write(view_proj)
        fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        vao.render(moderngl.TRIANGLES, vertices=len(vertices))
        self.ctx.disable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        depth.compare_func = previous

    def release(self) -> None:
        for icon in self._icons.values():
            if icon is not None:
                icon.texture.release()
        self._icons.clear()
        if self._buffer is not None:
            self._buffer.release()
        for vao in self._vaos.values():
            vao.release()


def _quad(rel, x0, y0, x1, y1, alpha, mode) -> np.ndarray:
    out = np.zeros((6, 10))
    out[:, :3] = rel
    out[:, 3:5] = [[x0, y0], [x1, y0], [x1, y1], [x0, y0], [x1, y1], [x0, y1]]
    out[:, 7], out[:, 8], out[:, 9] = alpha, mode, 1.0
    return out


def _glyphs(rel, q, uv, alpha, scale) -> np.ndarray:
    n = len(q)
    out = np.zeros((n, 6, 10))
    x0, y0, x1, y1 = q.T
    u0, v0, u1, v1 = uv.T
    for j, (xs, ys, us, vs) in enumerate(
        (
            (x0, y0, u0, v0),
            (x1, y0, u1, v0),
            (x1, y1, u1, v1),
            (x0, y0, u0, v0),
            (x1, y1, u1, v1),
            (x0, y1, u0, v1),
        )
    ):
        out[:, j, 3], out[:, j, 4], out[:, j, 5], out[:, j, 6] = xs, ys, us, vs  # fmt: skip
    out[:, :, :3] = rel
    out[:, :, 7], out[:, :, 8], out[:, :, 9] = alpha, 1.0, scale
    return out.reshape(-1, 10)
