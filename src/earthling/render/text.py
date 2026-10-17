"""Text rendering: a signed-distance-field glyph atlas (Noto Sans) shared by the labels, the
stats overlay and the POI captions. Glyphs are rasterised once at BASE_PX and drawn at any size
with a crisp outline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = Path(__file__).parent / "fonts" / "NotoSans-SemiBold.ttf"
BASE_PX = 64  # glyph raster size in the atlas
SPREAD = 16  # SDF range in atlas pixels (limits the outline width of small text)
ATLAS_WIDTH = 1024


def signed_distance(mask: np.ndarray, spread: float) -> np.ndarray:
    """SDF of a binary glyph mask, mapped to 0..1 (0.5 on the outline)."""
    h, w = mask.shape
    pad = np.pad(mask, 1, constant_values=False)
    neighbours_out = ~pad[:-2, 1:-1] | ~pad[2:, 1:-1] | ~pad[1:-1, :-2] | ~pad[1:-1, 2:]
    neighbours_in = pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
    edge_in = np.argwhere(mask & neighbours_out).astype(np.float32)
    edge_out = np.argwhere(~mask & neighbours_in).astype(np.float32)
    points = np.argwhere(np.ones_like(mask)).astype(np.float32)

    def nearest(edges: np.ndarray) -> np.ndarray:
        if len(edges) == 0:
            return np.full(len(points), spread * 2.0, dtype=np.float32)
        best = np.full(len(points), np.inf, dtype=np.float32)
        chunk = max(1, 2_000_000 // len(edges))
        for start in range(0, len(points), chunk):
            p = points[start : start + chunk]
            d = np.sqrt(((p[:, None, :] - edges[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
            best[start : start + chunk] = d
        return best

    d_out = nearest(edge_out).reshape(h, w)  # inside pixels: distance to the outside
    d_in = nearest(edge_in).reshape(h, w)
    signed = np.where(mask, d_out - 0.5, -(d_in - 0.5))
    return np.clip(0.5 + signed / (2.0 * spread), 0.0, 1.0)


@dataclass
class Glyph:
    advance: float  # base px
    x0: float = 0.0  # quad offset from the pen position / baseline (base px, incl. spread)
    y0: float = 0.0
    w: int = 0  # quad size (atlas px)
    h: int = 0
    uv: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    image: np.ndarray | None = None


class FontAtlas:
    """Glyphs are rasterised on demand, so any character the font has can be used."""

    def __init__(self, ctx: moderngl.Context, path: Path = FONT_PATH) -> None:
        self.ctx = ctx
        self.font = ImageFont.truetype(str(path), BASE_PX)
        self.ascent, self.descent = self.font.getmetrics()
        self.glyphs: dict[str, Glyph] = {}
        self.texture: moderngl.Texture | None = None
        self.generation = 0  # increases with every repack (glyph uvs change)
        self._dirty = False

    def _rasterise(self, ch: str) -> Glyph:
        glyph = Glyph(advance=float(self.font.getlength(ch)))
        x0, y0, x1, y1 = self.font.getbbox(ch, anchor="ls")
        if x1 <= x0 or y1 <= y0:
            return glyph  # whitespace
        w, h = x1 - x0 + 2 * SPREAD, y1 - y0 + 2 * SPREAD
        img = Image.new("L", (w, h), 0)
        ImageDraw.Draw(img).text(
            (SPREAD - x0, SPREAD - y0), ch, font=self.font, fill=255, anchor="ls"
        )
        mask = np.asarray(img) >= 128
        glyph.x0, glyph.y0 = float(x0 - SPREAD), float(y0 - SPREAD)
        glyph.w, glyph.h = w, h
        glyph.image = (signed_distance(mask, SPREAD) * 255.0 + 0.5).astype(np.uint8)
        return glyph

    def ensure(self, text: str) -> None:
        for ch in set(text) - self.glyphs.keys():
            self.glyphs[ch] = self._rasterise(ch)
            self._dirty = True
        if self._dirty:
            self._pack()

    def _pack(self) -> None:
        drawn = sorted((g for g in self.glyphs.values() if g.image is not None), key=lambda g: -g.h)
        x = y = row = 0
        places = []
        for g in drawn:
            if x + g.w > ATLAS_WIDTH:
                x, y, row = 0, y + row + 1, 0
            places.append((g, x, y))
            x += g.w + 1
            row = max(row, g.h)
        height = 1 << max(6, math.ceil(math.log2(max(y + row + 1, 1))))
        atlas = np.zeros((height, ATLAS_WIDTH), dtype=np.uint8)
        for g, gx, gy in places:
            atlas[gy : gy + g.h, gx : gx + g.w] = g.image
            g.uv = (gx / ATLAS_WIDTH, gy / height, (gx + g.w) / ATLAS_WIDTH, (gy + g.h) / height)
        if self.texture is not None:
            self.texture.release()
        self.texture = self.ctx.texture((ATLAS_WIDTH, height), 1, atlas.tobytes())
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.texture.repeat_x = self.texture.repeat_y = False
        self._dirty = False
        self.generation += 1

    def width(self, text: str) -> float:
        return sum(self.glyphs[ch].advance for ch in text)

    def quads(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Glyph quads of ``text`` at base size: (n, 4) x0 y0 x1 y1 relative to the pen start on
        the baseline, and (n, 4) uv rectangles."""
        pen = 0.0
        rects, uvs = [], []
        for ch in text:
            g = self.glyphs[ch]
            if g.image is not None:
                rects.append((pen + g.x0, g.y0, pen + g.x0 + g.w, g.y0 + g.h))
                uvs.append(g.uv)
            pen += g.advance
        return np.array(rects, dtype=np.float32).reshape(-1, 4), np.array(
            uvs, dtype=np.float32
        ).reshape(-1, 4)

    def release(self) -> None:
        if self.texture is not None:
            self.texture.release()
            self.texture = None


# --- layer ---------------------------------------------------------------------------------
