"""3D-anchored labels for peaks, passes, places, huts, cities and countries.

Per frame: filter the features (keyframeable properties), project the anchors, fade with
distance, test the anchors against the scene depth on the GPU (one texel per label, read
back), then place labels greedily by priority and fade out those that overlap an already
placed one. Text is drawn from a signed-distance-field glyph atlas with an outline, after tone
mapping and at output resolution, so labels are equally crisp in the viewport and in a 4K
export. Sizes are specified for 1080p output and scale with the output height.
"""

from __future__ import annotations

import math

import moderngl
import numpy as np

from earthling.core.easing import smoothstep_array
from earthling.data.labels import LabelData, LabelFeature, enrich
from earthling.render.atmosphere import PLANET_RADIUS
from earthling.render.camera import Camera
from earthling.render.gl_util import DynamicBuffer, draw_overlay, set_uniforms
from earthling.render.scale import pixel_scale
from earthling.render.shader_library import ShaderLibrary
from earthling.render.text import (  # noqa: F401 (re-exported)
    ATLAS_WIDTH,
    BASE_PX,
    FONT_PATH,
    SPREAD,
    FontAtlas,
    Glyph,
    signed_distance,
)

LIFT_M = 12.0  # anchors float a little above the ground
MAX_LABELS = 1024  # candidates per frame for the occlusion test
MAX_DECLUTTER = 400  # visible candidates considered for placement (by priority)
FADE_S = 0.4  # a label placed or displaced fades in / out over this time
INCUMBENT_BONUS = 8  # a shown label goes before this many more important new ones
RESET_S = 0.5  # a jump in time (scrubbing, a single still) starts the placement afresh
KIND_PRIORITY = {"place": 2.0, "peak": 2.0, "pass": 1.2, "hut": 0.8, "city": 2.5,
                 "country": 7.0}  # fmt: skip
KIND_PROPERTIES = {
    "peak": "labels.peaks",
    "pass": "labels.passes",
    "place": "labels.places",
    "hut": "labels.huts",
    "city": "labels.cities",
    "country": "labels.countries",
}
WITH_ELEVATION = ("peak", "pass", "hut")
COUNTRY_SCALE = 1.3  # country names: larger, in capitals, centred on the point, no leader


def sight_in_clouds(rel: np.ndarray, camera_position, camera_height: float, bands,
                    samples: int = 12) -> np.ndarray:  # fmt: skip
    """True where the line of sight from the camera to an anchor (``rel``: camera-relative)
    passes through the height band (bottom, top) of a cloud layer: only there can clouds hide
    the label. (The cloud veil of a summit seen against the sky also holds the clouds far
    behind it.) Heights above a sphere under the camera, as in the cloud shader."""
    hit = np.zeros(len(rel), dtype=bool)
    if not bands or not len(rel):
        return hit
    position = np.asarray(camera_position, dtype=np.float64)
    up = position + (0.0, 0.0, PLANET_RADIUS)
    up /= np.linalg.norm(up)
    centre = -up * (PLANET_RADIUS + camera_height)  # relative to the camera
    t = np.linspace(0.0, 1.0, samples)
    heights = np.linalg.norm(rel[:, None, :] * t[None, :, None] - centre, axis=2) - PLANET_RADIUS
    low, high = heights.min(axis=1), heights.max(axis=1)
    for bottom, top in bands:
        hit |= (low <= top) & (high >= bottom)
    return hit


# --- signed distance field font atlas --------------------------------------------------------
def label_texts(feature: LabelFeature, show_elevation: bool, lang: str = "en") -> tuple[str, str]:
    name = feature.display(lang)
    if feature.kind == "country":
        return name.upper(), ""
    if show_elevation and feature.kind in WITH_ELEVATION and feature.ele is not None:
        return name, f"{feature.ele:.0f} m"
    return name, ""


def static_priority(features: list[LabelFeature]) -> np.ndarray:
    out = np.zeros(len(features))
    for i, f in enumerate(features):
        p = KIND_PRIORITY.get(f.kind, 1.0)
        if f.kind == "place":
            p += f.rank * 0.8
        if f.kind == "city":
            p += 0.8 * math.log10(max(f.population, 1e5) / 1e5)
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
        self._vis_buffer = ctx.buffer(reserve=MAX_LABELS * 20)
        self._vis_vao: moderngl.VertexArray | None = None
        self._stream = DynamicBuffer(ctx, "2f 2f 3f", "in_pos", "in_uv", "in_params")
        self.enricher = None  # callable(features) run once after loading (DEM, tracks)
        # the drawn track (labels.follow_progress): set by the renderer every frame
        self.path = None  # ProgressPath of the walked tracks
        self.head_m = math.inf  # global distance of the head
        self._passed_key: tuple | None = None
        self._passed = np.zeros(0)
        self.last_drawn = 0  # labels drawn in the last frame (tests, status)
        # the cloud layers' height bands (bottom, top) and the camera height: set by the
        # renderer every frame (clouds hide only the labels behind or in them)
        self.cloud_bands: list[tuple[float, float]] = []
        self.camera_height = 0.0
        # stable placement: how far each feature (index) is faded in, at which time
        self.time = 0.0  # set by the renderer every frame (animation or wall clock)
        self._shown: dict[int, float] = {}
        self._shown_time: float | None = None

    def get_atlas(self) -> FontAtlas:
        """The font atlas (shared with the stats overlay)."""
        if self.atlas is None:
            self.atlas = FontAtlas(self.ctx)
        return self.atlas

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
        self._shown = {}

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

    def _passed_at(self, anchors: np.ndarray) -> np.ndarray:
        """Per feature: the global track distance of the nearest point of the drawn path (the
        head passes the label there)."""
        path = self.path
        key = (id(path), getattr(path, "total_m", 0.0), self._anchor_key)
        if key != self._passed_key:
            self._passed = path.nearest_distances(anchors[:, :2])
            self._passed_key = key
        return self._passed

    # --- rendering ----------------------------------------------------------------------
    def _filter(self, store) -> np.ndarray:
        n = len(self.features)
        keep = np.zeros(n, dtype=bool)
        enabled = {kind for kind, pid in KIND_PROPERTIES.items() if store[pid]}
        min_ele = store["labels.min_elevation"]
        min_prom = store["labels.min_prominence"]
        max_track = store["labels.max_track_distance"] * 1000.0
        min_pop = store["labels.min_population"]
        for i, f in enumerate(self.features):
            if f.kind not in enabled:
                continue
            if f.kind in WITH_ELEVATION and (f.ele or 0.0) < min_ele:
                continue
            if f.kind == "peak" and (f.prominence if f.prominence is not None else 0.0) < min_prom:
                continue
            if f.kind == "city" and f.population < min_pop:
                continue
            if max_track > 0 and f.kind != "country" and f.track_distance_m > max_track:
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
        fade = 1.0 - smoothstep_array(0.7 * max_distance, max_distance, distance)
        if store["labels.follow_progress"] and self.path is not None:
            passed = self._passed_at(anchors)[idx]
            reveal = max(store["labels.progress_fade"], 1e-3)
            fade = fade * np.clip((self.head_m - passed) / reveal, 0.0, 1.0)
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
            self._vis_vao = self.ctx.vertex_array(
                program, [(self._vis_buffer, "4f 1f", "in_anchor", "in_clouds")]
            )
        data = np.zeros((MAX_LABELS, 5), dtype=np.float32)
        data[:n, :3] = rel
        data[:n, 3] = 4.0 + 0.004 * np.linalg.norm(rel, axis=1)  # depth tolerance (m)
        data[:n, 4] = sight_in_clouds(rel, camera.position, self.camera_height, self.cloud_bands)
        self._vis_buffer.write(data.tobytes())
        program["u_view_proj"].write(view_proj)
        program["u_depth_size"] = (float(depth_size[0]), float(depth_size[1]))
        program["u_log_depth_coef"] = camera.log_depth_coef
        program["u_count"] = float(MAX_LABELS)
        previous = depth.compare_func
        depth.compare_func = ""  # plain depth values
        depth.use(0)
        program["u_depth"] = 0
        if "u_veil" in program:  # labels in or behind clouds are hidden
            veil = getattr(self, "veil", None)
            program["u_has_veil"] = veil is not None
            if veil is not None:
                veil.use(1)
                program["u_veil"] = 1
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

    def _text_metrics(self, show_elevation: bool, lang: str = "en"):
        """Per feature: texts and their widths at base size (cached)."""
        generation = self.atlas.generation if self.atlas else -1
        key = (self._version, show_elevation, lang, generation)
        if self._metrics_key != key:
            assert self.atlas is not None
            texts = [label_texts(f, show_elevation, lang) for f in self.features]
            self.atlas.ensure("".join(n + e for n, e in texts))
            self._texts = texts
            self._name_w = np.array([self.atlas.width(n) for n, _ in texts])
            self._ele_w = np.array([self.atlas.width(e) for _, e in texts])
            self._is_place = np.array([f.kind in ("place", "city") for f in self.features])
            self._is_country = np.array([f.kind == "country" for f in self.features])
            self._metrics_key = (self._version, show_elevation, lang, self.atlas.generation)
        return self._texts, self._name_w, self._ele_w, self._is_place

    def _draw(self, fbo, idx, ndc, alpha0, width, height, store) -> None:
        if self.atlas is None:
            self.atlas = FontAtlas(self.ctx)
        k = pixel_scale(width, height)
        size = store["labels.size"] * k
        leader = store["labels.leader"] * k
        max_count = store["labels.max_count"]
        lang = store["stats.language"]
        texts, name_w_all, ele_w_all, is_place = self._text_metrics(store["labels.elevations"],
                                                                    lang)  # fmt: skip
        ascent, descent = self.atlas.ascent, self.atlas.descent
        alpha0 = alpha0 * store["labels.opacity"]
        live = np.flatnonzero(alpha0 >= 0.02)[:MAX_DECLUTTER]
        if not len(live):
            return
        idx, alpha0 = idx[live], alpha0[live]
        # layout of every candidate (y down, pixels)
        x = (ndc[live, 0] * 0.5 + 0.5) * width
        y = (0.5 - ndc[live, 1] * 0.5) * height
        country = self._is_country[idx]
        name_scale = size / BASE_PX * np.where(country, COUNTRY_SCALE, 1.0)
        ele_scale = 0.72 * name_scale
        has_ele = ele_w_all[idx] > 0
        name_w, ele_w = name_w_all[idx] * name_scale, ele_w_all[idx] * ele_scale
        bottom = y - np.where(is_place[idx], 0.35, 1.0) * leader
        # countries: the name centred on the point
        bottom = np.where(country, y + 0.5 * (ascent + descent) * name_scale, bottom)
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
        top_fade = smoothstep_array(0.0, 12.0 * k, top)
        shown = self._stable_placement(idx, alpha0 * top_fade, conflicts, max_count)
        verts: list[np.ndarray] = []
        for i in range(len(idx)):
            a = alpha0[i] * shown[i] * top_fade[i]
            if a < 0.02:
                continue
            name, ele_text = texts[idx[i]]
            verts.append(self._label_vertices(
                x[i], y[i], bottom[i], name, name_base[i], name_scale[i], name_w[i],
                ele_text, ele_base[i], ele_scale[i], ele_w[i], a, k, dot=not country[i],
            ))  # fmt: skip
        self.last_drawn = len(verts)
        if not verts:
            return
        # most important labels last, i.e. on top
        vertices = np.concatenate(verts[::-1]).astype(np.float32)
        self._submit(fbo, vertices, width, height, store)

    def _stable_placement(self, idx: np.ndarray, alpha: np.ndarray, conflicts: np.ndarray,
                          max_count: int) -> np.ndarray:  # fmt: skip
        """How far each candidate (in priority order) is shown, 0..1.

        Greedy by priority, except that labels already shown go first (by INCUMBENT_BONUS
        places): a label that briefly touches another one keeps its place instead of
        blinking. Placed labels fade in, displaced ones fade out over FADE_S, and a new label
        waits until a fading one has made room. A jump in time starts afresh (a still shows
        the plain greedy placement)."""
        dt = None if self._shown_time is None else self.time - self._shown_time
        # (anti-aliasing samples share a time, motion blur samples may step back a little:
        # neither changes the state)
        fresh = dt is None or abs(dt) > RESET_S
        self._shown_time = self.time
        before = np.array([0.0 if fresh else self._shown.get(int(j), 0.0) for j in idx])
        rank = np.arange(len(idx)) - INCUMBENT_BONUS * (before > 0.5)
        target = np.zeros(len(idx))
        blocked = np.zeros(len(idx))
        count = 0
        for i in np.argsort(rank, kind="stable"):
            if alpha[i] < 0.02 or blocked[i] > 0.06 or count >= max_count:
                continue
            target[i] = 1.0
            count += 1
            blocked = np.maximum(blocked, conflicts[:, i])
        fading = (before > 0.02) & (target == 0.0)
        if fading.any():  # newcomers wait for the place of a label that fades out
            waits = (before < 0.02) & (target > 0.0) & (conflicts[:, fading] > 0.06).any(axis=1)
            target[waits] = 0.0
        if fresh:
            shown = target
        else:
            step = max(dt, 0.0) / FADE_S
            shown = before + np.clip(target - before, -step, step)
        self._shown = {int(j): float(v) for j, v in zip(idx, shown, strict=True) if v > 1e-3}
        return shown

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
        dot: bool = True,
    ) -> np.ndarray:
        parts = []
        # leader line and anchor dot: dark underlay, light core
        if bottom < y - 1.0:
            parts.append(_rect(x - 1.6 * k, bottom, x + 1.6 * k, y, alpha, 2.0))
            parts.append(_rect(x - 0.7 * k, bottom, x + 0.7 * k, y, alpha, 1.0))
        if dot:
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
        vao = self._stream.vertex_array(program, vertices.tobytes())
        assert self.atlas is not None and self.atlas.texture is not None
        self.atlas.texture.use(0)
        program["u_atlas"] = 0
        program["u_viewport"] = (float(width), float(height))
        # display colours: labels are drawn after tone mapping (no linearisation)
        set_uniforms(program, (
            ("u_spread", float(SPREAD)),
            ("u_label_color", tuple(store["labels.color"])),
            ("u_label_outline_color", tuple(store["labels.outline_color"])),
            ("u_label_outline", float(store["labels.outline"] * pixel_scale(width, height))),
        ))  # fmt: skip
        draw_overlay(self.ctx, fbo, vao, len(vertices), width, height)

    def release(self) -> None:
        for obj in (self._vis_fbo, self._vis_tex, self._vis_buffer):
            if obj is not None:
                obj.release()
        self._stream.release()
        if self._vis_vao is not None:
            self._vis_vao.release()
        if self.atlas is not None:
            self.atlas.release()


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
