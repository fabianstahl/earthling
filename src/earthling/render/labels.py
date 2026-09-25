"""3D-anchored labels for peaks, passes, places and huts.

Per frame: filter the features (keyframeable properties), project the anchors, fade with
distance, test the anchors against the scene depth on the GPU (one texel per label, read
back), then place labels greedily by priority and fade out those that overlap an already
placed one. Text is drawn from a signed-distance-field glyph atlas with an outline, after tone
mapping and at output resolution, so labels are equally crisp in the viewport and in a 4K
export. Sizes are specified for 1080p output and scale with the output height.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from earthling.data.labels import LabelData, LabelFeature, enrich
from earthling.render.camera import Camera
from earthling.render.shader_library import ShaderLibrary

FONT_PATH = Path(__file__).parent / "fonts" / "NotoSans-SemiBold.ttf"
BASE_PX = 64  # glyph raster size in the atlas
SPREAD = 16  # SDF range in atlas pixels (limits the outline width of small text)
ATLAS_WIDTH = 1024
REFERENCE_HEIGHT = 1080.0
LIFT_M = 12.0  # anchors float a little above the ground
MAX_LABELS = 1024  # candidates per frame for the occlusion test
MAX_DECLUTTER = 400  # visible candidates considered for placement (by priority)
KIND_PRIORITY = {"place": 2.0, "peak": 2.0, "pass": 1.2, "hut": 0.8}
KIND_PROPERTIES = {
    "peak": "labels.peaks",
    "pass": "labels.passes",
    "place": "labels.places",
    "hut": "labels.huts",
}


# --- signed distance field font atlas --------------------------------------------------------
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
def smoothstep(e0: float, e1: float, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / max(e1 - e0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def label_texts(feature: LabelFeature, show_elevation: bool) -> tuple[str, str]:
    if show_elevation and feature.kind != "place" and feature.ele is not None:
        return feature.name, f"{feature.ele:.0f} m"
    return feature.name, ""


def static_priority(features: list[LabelFeature]) -> np.ndarray:
    out = np.zeros(len(features))
    for i, f in enumerate(features):
        p = KIND_PRIORITY.get(f.kind, 1.0)
        if f.kind == "place":
            p += f.rank * 0.8
        if f.kind == "peak":
            p += min((f.prominence or 0.0) / 400.0, 3.0) + (f.ele or 0.0) / 4000.0
        if f.kind in ("pass", "hut"):
            p += (f.ele or 0.0) / 5000.0
        if math.isfinite(f.track_distance_m):
            p += 1.5 * math.exp(-f.track_distance_m / 2000.0)
        out[i] = p
    return out


class LabelLayer:
    def __init__(self, ctx: moderngl.Context, shaders: ShaderLibrary) -> None:
        self.ctx = ctx
        self.shaders = shaders
        self.data: LabelData | None = None
        self.atlas: FontAtlas | None = None
        self.features: list[LabelFeature] = []
        self._version = -1
        self._ground = np.zeros(0)
        self._lonlat = np.zeros((0, 2))
        self._priority = np.zeros(0)
        self._anchor_key: tuple | None = None
        self._anchors = np.zeros((0, 3))
        self._layouts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._layout_generation = -1
        self._metrics_key: tuple | None = None
        self._vis_tex = ctx.texture((MAX_LABELS, 1), 1, dtype="f1")
        self._vis_fbo = ctx.framebuffer([self._vis_tex])
        self._vis_buffer = ctx.buffer(reserve=MAX_LABELS * 16)
        self._vis_vao: moderngl.VertexArray | None = None
        self._buffer: moderngl.Buffer | None = None
        self._vaos: dict[int, moderngl.VertexArray] = {}
        self.enricher = None  # callable(features) run once after loading (DEM, tracks)
        self.last_drawn = 0  # labels drawn in the last frame (tests, status)

    # --- data ---------------------------------------------------------------------------
    @property
    def loading(self) -> bool:
        return self.data is not None and not self.data.ready and self.data.allow_download

    def make_enricher(self, frame, terrain_data, tracks):
        """DEM heights, prominence and track distances for the features (background thread)."""

        def run(features: list[LabelFeature]) -> None:
            heights = getattr(terrain_data, "heights_at", None)
            lines = [(s.lon, s.lat) for t in tracks for s in t.segments]
            coarse = None
            if heights is not None:

                def coarse(lon, lat):  # prominence lines: z11 is plenty and reads few tiles
                    return heights(lon, lat, z=11)

            enrich(features, coarse, frame, lines)
            if heights is not None:
                ground = heights(
                    np.array([f.lon for f in features]), np.array([f.lat for f in features])
                )
                for f, g in zip(features, ground, strict=True):
                    f.ground = float(g) if np.isfinite(g) else f.ele

        return run

    def _sync(self) -> None:
        data = self.data
        if data is None or not data.ready or data.version == self._version:
            return
        features = [f for f in data.features or [] if f.anchor_height is not None]
        self.features = features
        self._version = data.version
        self._ground = np.array([f.anchor_height for f in features], dtype=np.float64)
        self._lonlat = np.array([(f.lon, f.lat) for f in features]).reshape(-1, 2)
        self._priority = static_priority(features)
        self._anchor_key = None

    def load_sync(self) -> bool:
        """For exports: make sure the features are loaded and enriched."""
        if self.data is None:
            return False
        if not self.data.ready:
            self.data.load(self.enricher)
        self._sync()
        return self.data.ready

    def _anchors_for(self, frame, exaggeration: float) -> np.ndarray:
        key = (id(frame), exaggeration, self._version)
        if key != self._anchor_key:
            h = self._ground * exaggeration + LIFT_M
            if len(h):
                self._anchors = frame.geodetic_to_enu(self._lonlat[:, 1], self._lonlat[:, 0], h)
            else:
                self._anchors = np.zeros((0, 3))
            self._anchor_key = key
        return self._anchors

    # --- rendering ----------------------------------------------------------------------
    def _filter(self, store) -> np.ndarray:
        n = len(self.features)
        keep = np.zeros(n, dtype=bool)
        enabled = {kind for kind, pid in KIND_PROPERTIES.items() if store[pid]}
        min_ele = store["labels.min_elevation"]
        min_prom = store["labels.min_prominence"]
        max_track = store["labels.max_track_distance"] * 1000.0
        for i, f in enumerate(self.features):
            if f.kind not in enabled:
                continue
            if f.kind != "place" and (f.ele or 0.0) < min_ele:
                continue
            if f.kind == "peak" and (f.prominence if f.prominence is not None else 0.0) < min_prom:
                continue
            if max_track > 0 and f.track_distance_m > max_track:
                continue
            keep[i] = True
        return keep

    def render(
        self,
        fbo: moderngl.Framebuffer,
        camera: Camera,
        view_proj,
        frame,
        exaggeration: float,
        width: int,
        height: int,
        depth: moderngl.Texture | None,
        depth_size: tuple[int, int],
        store,
    ) -> None:
        self.last_drawn = 0
        if store is None or not store["labels.visible"] or store["labels.opacity"] <= 0.0:
            return
        if self.data is not None and not self.data.ready and self.data.allow_download:
            self.data.start(self.enricher)
        self._sync()
        if not self.features or frame is None or depth is None:
            return
        anchors = self._anchors_for(frame, exaggeration)
        keep = self._filter(store)
        idx = np.flatnonzero(keep)
        if not len(idx):
            return
        rel = anchors[idx] - np.asarray(camera.position, dtype=np.float64)
        m = np.array(view_proj.to_list(), dtype=np.float64).T
        clip = np.c_[rel, np.ones(len(rel))] @ m.T
        w = clip[:, 3]
        front = w > 1.0
        ndc = clip[:, :2] / np.where(front, w, 1.0)[:, None]
        on_screen = front & (np.abs(ndc) < 1.15).all(axis=1)
        distance = np.linalg.norm(rel, axis=1)
        max_distance = store["labels.max_distance"] * 1000.0
        fade = 1.0 - smoothstep(0.7 * max_distance, max_distance, distance)
        sel = on_screen & (fade > 0.01)
        if not sel.any():
            return
        idx, rel, ndc, distance, fade = idx[sel], rel[sel], ndc[sel], distance[sel], fade[sel]
        priority = self._priority[idx] - 0.8 * np.log10(np.maximum(distance, 300.0) / 1000.0)
        order = np.argsort(-priority)[:MAX_LABELS]
        idx, rel, ndc, fade = idx[order], rel[order], ndc[order], fade[order]
        visible = self._visibility(rel, view_proj, camera, depth, depth_size)
        self._draw(fbo, idx, ndc, fade * visible, width, height, store)

    def _visibility(self, rel, view_proj, camera, depth, depth_size) -> np.ndarray:
        n = len(rel)
        program = self.shaders.get("label_vis")
        if self._vis_vao is None:
            self._vis_vao = self.ctx.vertex_array(program, [(self._vis_buffer, "4f", "in_anchor")])
        data = np.zeros((MAX_LABELS, 4), dtype=np.float32)
        data[:n, :3] = rel
        data[:n, 3] = 4.0 + 0.004 * np.linalg.norm(rel, axis=1)  # depth tolerance (m)
        self._vis_buffer.write(data.tobytes())
        program["u_view_proj"].write(view_proj)
        program["u_depth_size"] = (float(depth_size[0]), float(depth_size[1]))
        program["u_log_depth_coef"] = camera.log_depth_coef
        program["u_count"] = float(MAX_LABELS)
        previous = depth.compare_func
        depth.compare_func = ""  # plain depth values
        depth.use(0)
        program["u_depth"] = 0
        self._vis_fbo.use()
        self.ctx.viewport = (0, 0, MAX_LABELS, 1)
        self._vis_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._vis_vao.render(moderngl.POINTS, vertices=n)
        depth.compare_func = previous
        raw = self._vis_fbo.read(viewport=(0, 0, n, 1), components=1)
        return np.frombuffer(raw, dtype=np.uint8)[:n].astype(np.float64) / 255.0

    def _layout(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        assert self.atlas is not None
        if self._layout_generation != self.atlas.generation:
            self._layouts.clear()  # uvs of a previous packing
            self._layout_generation = self.atlas.generation
        layout = self._layouts.get(text)
        if layout is None:
            assert self.atlas is not None
            self.atlas.ensure(text)
            layout = self.atlas.quads(text)
            self._layouts[text] = layout
        return layout

    def _text_metrics(self, show_elevation: bool):
        """Per feature: texts and their widths at base size (cached)."""
        key = (self._version, show_elevation, self.atlas.generation if self.atlas else -1)
        if self._metrics_key != key:
            assert self.atlas is not None
            texts = [label_texts(f, show_elevation) for f in self.features]
            self.atlas.ensure("".join(n + e for n, e in texts))
            self._texts = texts
            self._name_w = np.array([self.atlas.width(n) for n, _ in texts])
            self._ele_w = np.array([self.atlas.width(e) for _, e in texts])
            self._is_place = np.array([f.kind == "place" for f in self.features])
            self._metrics_key = (self._version, show_elevation, self.atlas.generation)
        return self._texts, self._name_w, self._ele_w, self._is_place

    def _draw(self, fbo, idx, ndc, alpha0, width, height, store) -> None:
        if self.atlas is None:
            self.atlas = FontAtlas(self.ctx)
        k = height / REFERENCE_HEIGHT
        size = store["labels.size"] * k
        leader = store["labels.leader"] * k
        max_count = store["labels.max_count"]
        name_scale = size / BASE_PX
        ele_scale = 0.72 * name_scale
        texts, name_w_all, ele_w_all, is_place = self._text_metrics(store["labels.elevations"])
        ascent, descent = self.atlas.ascent, self.atlas.descent
        alpha0 = alpha0 * store["labels.opacity"]
        live = np.flatnonzero(alpha0 >= 0.02)[:MAX_DECLUTTER]
        if not len(live):
            return
        idx, alpha0 = idx[live], alpha0[live]
        # layout of every candidate (y down, pixels)
        x = (ndc[live, 0] * 0.5 + 0.5) * width
        y = (0.5 - ndc[live, 1] * 0.5) * height
        has_ele = ele_w_all[idx] > 0
        name_w, ele_w = name_w_all[idx] * name_scale, ele_w_all[idx] * ele_scale
        bottom = y - np.where(is_place[idx], 0.35, 1.0) * leader
        ele_base = bottom - descent * ele_scale
        name_base = np.where(has_ele, ele_base - ascent * ele_scale, bottom) - descent * name_scale
        top = name_base - ascent * name_scale
        half = np.maximum(name_w, ele_w) / 2.0
        pad = 3.0 * k
        rects = np.stack([x - half - pad, top - pad, x + half + pad, bottom + pad], axis=1)
        stems = np.stack([x - 3.0 * k, bottom, x + 3.0 * k, y + 3.0 * k], axis=1)
        # conflicts[i, j]: how much candidate i collides with candidate j once j is placed
        conflicts = np.maximum.reduce([
            _pairwise_fraction(rects, rects, of_first=True),  # text on text
            _pairwise_fraction(stems, rects, of_first=True),  # own leader through j's text
            _pairwise_fraction(rects, stems, of_first=False),  # own text over j's leader
        ])  # fmt: skip
        np.fill_diagonal(conflicts, 0.0)
        top_fade = smoothstep(0.0, 12.0 * k, top)
        blocked = np.zeros(len(idx))
        verts: list[np.ndarray] = []
        for i in range(len(idx)):
            a = alpha0[i] * (1.0 - _smooth(0.0, 0.06, blocked[i])) * top_fade[i]
            if a < 0.02:
                continue
            if a > 0.12:  # even faint (partly hidden) labels reserve their space
                blocked = np.maximum(blocked, conflicts[:, i])
            name, ele_text = texts[idx[i]]
            verts.append(self._label_vertices(
                x[i], y[i], bottom[i], name, name_base[i], name_scale, name_w[i],
                ele_text, ele_base[i], ele_scale, ele_w[i], a, k,
            ))  # fmt: skip
            if len(verts) >= max_count:
                break
        self.last_drawn = len(verts)
        if not verts:
            return
        # most important labels last, i.e. on top
        vertices = np.concatenate(verts[::-1]).astype(np.float32)
        self._submit(fbo, vertices, width, height, store)

    def _label_vertices(
        self,
        x,
        y,
        bottom,
        name,
        name_base,
        name_scale,
        name_w,
        ele_text,
        ele_base,
        ele_scale,
        ele_w,
        alpha,
        k,
    ) -> np.ndarray:
        parts = []
        # leader line and anchor dot: dark underlay, light core
        if bottom < y - 1.0:
            parts.append(_rect(x - 1.6 * k, bottom, x + 1.6 * k, y, alpha, 2.0))
            parts.append(_rect(x - 0.7 * k, bottom, x + 0.7 * k, y, alpha, 1.0))
        parts.append(_rect(x - 3.2 * k, y - 3.2 * k, x + 3.2 * k, y + 3.2 * k, alpha, 2.0))
        parts.append(_rect(x - 1.8 * k, y - 1.8 * k, x + 1.8 * k, y + 1.8 * k, alpha, 1.0))
        for text, base, scale, w in (
            (name, name_base, name_scale, name_w),
            (ele_text, ele_base, ele_scale, ele_w),
        ):
            if not text:
                continue
            rects, uvs = self._layout(text)
            if not len(rects):
                continue
            q = rects * scale
            q[:, [0, 2]] += x - w / 2.0
            q[:, [1, 3]] += base
            parts.append(_glyph_quads(q, uvs, alpha, scale))
        return np.concatenate(parts)

    def _submit(self, fbo, vertices: np.ndarray, width: int, height: int, store) -> None:
        program = self.shaders.get("label")
        data = vertices.tobytes()
        if self._buffer is None or self._buffer.size < len(data):
            if self._buffer is not None:
                self._buffer.release()
                for vao in self._vaos.values():
                    vao.release()
                self._vaos.clear()
            self._buffer = self.ctx.buffer(reserve=max(len(data), 1 << 16) * 2)
        self._buffer.write(data)
        vao = self._vaos.get(program.glo)
        if vao is None:
            vao = self.ctx.vertex_array(
                program, [(self._buffer, "2f 2f 3f", "in_pos", "in_uv", "in_params")]
            )
            self._vaos[program.glo] = vao
        assert self.atlas is not None and self.atlas.texture is not None
        self.atlas.texture.use(0)
        program["u_atlas"] = 0
        program["u_viewport"] = (float(width), float(height))
        if "u_spread" in program:
            program["u_spread"] = float(SPREAD)
        # display colours: labels are drawn after tone mapping (no linearisation)
        for name, value in (
            ("u_label_color", tuple(store["labels.color"])),
            ("u_label_outline_color", tuple(store["labels.outline_color"])),
            ("u_label_outline", float(store["labels.outline"] * height / REFERENCE_HEIGHT)),
        ):
            if name in program:
                program[name] = value
        fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        vao.render(moderngl.TRIANGLES, vertices=len(vertices))
        self.ctx.disable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def release(self) -> None:
        for obj in (self._vis_fbo, self._vis_tex, self._vis_buffer, self._buffer):
            if obj is not None:
                obj.release()
        for vao in self._vaos.values():
            vao.release()
        if self._vis_vao is not None:
            self._vis_vao.release()
        if self.atlas is not None:
            self.atlas.release()


def _smooth(e0: float, e1: float, x: float) -> float:
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _pairwise_fraction(a: np.ndarray, b: np.ndarray, of_first: bool) -> np.ndarray:
    """[i, j]: intersection of rectangles a[i] and b[j] as a fraction of a[i]'s area
    (``of_first``) or of b[j]'s area."""
    ix = np.clip(
        np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None
    )
    iy = np.clip(
        np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None
    )
    rects = a if of_first else b
    area = np.maximum((rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1]), 1.0)
    return ix * iy / (area[:, None] if of_first else area[None, :])


def _rect(x0, y0, x1, y1, alpha, mode) -> np.ndarray:
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y0], [x1, y1], [x0, y1]])
    out = np.zeros((6, 7))
    out[:, :2] = corners
    out[:, 2:4] = -1.0
    out[:, 4], out[:, 5], out[:, 6] = alpha, 1.0, mode
    return out


def _glyph_quads(q: np.ndarray, uv: np.ndarray, alpha: float, scale: float) -> np.ndarray:
    n = len(q)
    out = np.zeros((n, 6, 7))
    x0, y0, x1, y1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    u0, v0, u1, v1 = uv[:, 0], uv[:, 1], uv[:, 2], uv[:, 3]
    for j, (xs, ys, us, vs) in enumerate(
        ((x0, y0, u0, v0), (x1, y0, u1, v0), (x1, y1, u1, v1),
         (x0, y0, u0, v0), (x1, y1, u1, v1), (x0, y1, u0, v1))
    ):  # fmt: skip
        out[:, j, 0], out[:, j, 1], out[:, j, 2], out[:, j, 3] = xs, ys, us, vs
    out[:, :, 4], out[:, :, 5], out[:, :, 6] = alpha, scale, 0.0
    return out.reshape(-1, 7)
