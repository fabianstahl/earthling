"""GPS tracks as screen-space extruded ribbons with tube shading.

The geometry is built on the CPU (:mod:`earthling.render.track_geometry`), the journey axis
lives in :mod:`earthling.render.progress`; both are re-exported here. The vertex shader
extrudes the ribbon perpendicular to the projected line, either with a constant screen width or
a width in metres, with miter joins. Every vertex also carries the cumulative distance along
its track, which drives the animated progress.
"""

from __future__ import annotations

import moderngl
import numpy as np

from earthling.core.geo import LocalFrame
from earthling.core.gpx import Track
from earthling.render.camera import Camera
from earthling.render.gl_util import set_uniform
from earthling.render.progress import ProgressPath, journey_path
from earthling.render.scale import pixel_scale
from earthling.render.shader_library import ShaderLibrary
from earthling.render.track_geometry import (  # noqa: F401 (re-exported)
    GRADIENT_WINDOW_M,
    PRIMITIVE_RESTART,
    SPEED_WINDOW_S,
    VALUE_KINDS,
    VERTEX_FLOATS,
    BuiltTrack,
    TrackGeometryOptions,
    TrackSegmentGeometry,
    build_track_positions,
    build_tracks,
    calibrate_ground,
    motion_values,
    resample,
    resample_indices,
    ribbon_vertices,
    smooth,
)

# Distinct, saturated colors for tracks / days (sRGB).
PALETTE = [
    (1.00, 0.35, 0.20),
    (0.20, 0.70, 1.00),
    (1.00, 0.80, 0.15),
    (0.55, 0.90, 0.30),
    (0.90, 0.40, 0.90),
    (0.20, 0.90, 0.80),
    (1.00, 0.55, 0.65),
    (0.70, 0.60, 1.00),
]


def track_color(index: int) -> tuple[float, float, float]:
    return PALETTE[index % len(PALETTE)]


class TrackLayer:
    def __init__(self, ctx: moderngl.Context, shaders: ShaderLibrary) -> None:
        self.ctx = ctx
        self.shaders = shaders
        self._gpu: list[BuiltTrack] = []
        self.bounds: tuple[np.ndarray, np.ndarray] | None = None
        self.visible = True
        self.tracks: list[Track] = []
        self.frame: LocalFrame | None = None
        self.heights_at = None  # vectorised DEM height function
        self.water_at = None  # vectorised water coverage (tracks split by water)
        self._built_key: tuple | None = None
        self.uniforms: dict[str, object] = {}  # lighting/atmosphere uniforms
        self.path: ProgressPath | None = None
        self.pixel_scale = 1.0  # image height / 1080 of the current pass
        self.head_m = float("inf")  # global distance of the visible end (progress)
        self.tail_m = 0.0  # global distance of the visible start
        self.track_heads: dict[int, float] | None = None  # clock mode: head per track (id)
        self._viewport = (1, 1)

    def set_tracks(self, tracks: list[Track], frame: LocalFrame) -> None:
        self.tracks = tracks
        self.frame = frame
        self._built_key = None
        self.release()
        self.bounds = self._raw_bounds()

    def _raw_bounds(self):
        if not self.tracks or self.frame is None:
            return None
        pts = []
        for t in self.tracks:
            lat, lon, ele = t.all_points()
            pts.append(self.frame.geodetic_to_enu(lat, lon, np.nan_to_num(ele)))
        allp = np.vstack(pts)
        return allp.min(axis=0), allp.max(axis=0)

    def walked(self) -> list[Track]:
        return [t for t in self.tracks if t.role == "walked"]

    def ensure_built(self, options: TrackGeometryOptions, dem_version: object = None) -> None:
        key = (
            options.elevation_source,
            options.height_offset_m,
            options.exaggeration,
            options.spacing_m,
            options.smoothing,
            dem_version,
        )
        if key == self._built_key or self.frame is None:
            return
        self.release()
        self._gpu = build_tracks(self.tracks, self.frame, options, self.heights_at, self.water_at)
        for g in self._gpu:
            g.vbo, g.ibo = self.ctx.buffer(g.verts.tobytes()), self.ctx.buffer(g.indices.tobytes())
            g.verts = g.indices = None
        self.path = journey_path(self._gpu)
        self._built_key = key

    def render(
        self,
        camera: Camera,
        view_proj,
        width: int,
        height: int,
        store=None,
        glow_pass=False,
        ref_size: tuple[int, int] | None = None,
    ) -> None:
        """``width`` x ``height``: the viewport; ``ref_size``: the picture that on-screen
        sizes refer to, when the viewport is larger (the glow buffer's margin)."""
        if not self._gpu or not self.visible:
            return
        if glow_pass:
            program = self.shaders.get("track", defines={"GLOW_PASS": 1})
        else:
            program = self.shaders.get("track")
        program["u_view_proj"].write(view_proj)
        set_uniform(program, "u_log_depth_coef", camera.log_depth_coef)
        set_uniform(program, "u_viewport", (float(width), float(height)))
        ref_w, ref_h = ref_size or (width, height)
        focal = ref_h / (2.0 * np.tan(np.radians(camera.fov_y) / 2.0))
        set_uniform(program, "u_focal_px", float(focal))
        if store is not None:
            from earthling.core.properties import bind_uniforms

            bind_uniforms(program, store)
        # pixel sizes scale with the image height: a 4K export looks like the 1080p preview
        self.pixel_scale = pixel_scale(ref_w, ref_h)
        if store is not None:
            for name, pid in (("u_track_min_px", "tracks.min_px"),
                              ("u_track_outline", "tracks.outline"),
                              ("u_track_casing", "tracks.casing")):  # fmt: skip
                set_uniform(program, name, float(store[pid]) * self.pixel_scale)
        for name, value in self.uniforms.items():
            set_uniform(program, name, value)
        self.ctx.enable(moderngl.BLEND)
        if glow_pass:
            # emission composited "over" in draw order (premultiplied): a track overlapping
            # itself does not add up its glow, and a track on top of another one replaces its
            # glow instead of mixing both colours into a third one
            self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        else:
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        # depth test only: tracks on top of each other blend (a faint line must not hide the
        # one below), and clouds / rain keep the terrain depth
        fbo = self.ctx.fbo
        fbo.depth_mask = False
        casing = store is not None and store["tracks.casing"] > 0.0
        parts = (0, 1) if casing and not glow_pass else (1,)
        self._viewport = (width, height)
        view_m = np.array(view_proj.to_list(), dtype=np.float64).T  # row-major
        for part in parts:
            set_uniform(program, "u_part", part)
            self._draw_tracks(program, camera, store, view_m)
        fbo.depth_mask = True
        self.ctx.blend_equation = moderngl.FUNC_ADD
        self.ctx.disable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def _draw_tracks(self, program, camera: Camera, store, view_m=None) -> None:
        single = store is not None and store["tracks.color_mode"] == "single"
        base = {
            "opacity": store["tracks.opacity"] if store is not None else 1.0,
            "width": store["tracks.width"] if store is not None else 5.0,
            "glow": store["tracks.glow"] if store is not None else 1.2,
        }
        # planned routes first: the walked tracks are drawn on top of them; by clock (several
        # heads on one course) the lead group on top of the others
        lead = self.path.lead_group if self.track_heads is not None and self.path else None
        for g in sorted(self._gpu, key=lambda g: (g.track.role == "walked",
                                                  g.track.group == lead)):  # fmt: skip
            if not g.track.visible:
                continue
            style = self._group_style(store, g, single)
            if style is None:
                continue
            vao = g.vaos.get(program.glo)
            if vao is None:
                vao = self.ctx.vertex_array(
                    program,
                    [
                        (
                            g.vbo,
                            "3f 3f 3f 1f 1f 4f 1f 1f",
                            "in_pos",
                            "in_prev",
                            "in_next",
                            "in_side",
                            "in_dist",
                            "in_values",
                            "in_time",
                            "in_water",
                        )
                    ],
                    g.ibo,
                    index_element_size=4,
                    skip_errors=True,  # attributes a shader variant does not use
                )
                g.vaos[program.glo] = vao
            set_uniform(program, "u_offset", camera.relative(g.origin))
            color, opacity, width, glow, dash, dots = style
            set_uniform(program, "u_color", tuple(c**2.2 for c in color))
            set_uniform(program, "u_track_opacity", base["opacity"] * opacity)
            px = store is None or store["tracks.width_mode"] == "pixels"
            scale = self.pixel_scale if px else 1.0
            set_uniform(program, "u_track_width", base["width"] * width * scale)
            set_uniform(program, "u_track_glow", base["glow"] * glow)
            set_uniform(program, "u_dash_px", dash * self.pixel_scale)
            set_uniform(program, "u_dots", int(dots and dash > 0.0))
            p = f"trackgroup.{g.track.group}."
            water_style = (
                store is not None and g.track.split_water and p + "water_color" in store.registry
            )
            set_uniform(program, "u_water_split", int(bool(water_style)))
            if water_style:
                set_uniform(
                    program, "u_water_color", tuple(c**2.2 for c in store[p + "water_color"])
                )
                set_uniform(program, "u_water_dash_px", store[p + "water_dash"] * self.pixel_scale)
                set_uniform(program, "u_water_dots", int(store[p + "water_dots"]))
            set_uniform(program, "u_track_offset", g.offset_m)
            walked = g.track.role == "walked"  # planned routes are always drawn in full
            color_by = store["tracks.color_by"] if store is not None and walked else "none"
            kind = 0 if color_by == "none" else VALUE_KINDS.index(color_by) + 1
            set_uniform(program, "u_color_by", kind)
            if color_by != "none":
                set_uniform(program, "u_color_min", float(store["tracks.color_min"]))
                set_uniform(program, "u_color_max", float(store["tracks.color_max"]))
                ramps = [v for v, _ in store.registry["tracks.color_ramp"].options]
                set_uniform(program, "u_color_ramp", ramps.index(store["tracks.color_ramp"]))
            head = self.head_m
            if walked and self.track_heads is not None:  # clock mode: every track its own
                head = self.track_heads.get(id(g), -1.0)
            set_uniform(program, "u_head_m", min(head, 1e12) if walked else 1e12)
            set_uniform(program, "u_tail_m", self.tail_m if walked else 0.0)
            spaced = dash > 0.0 or (water_style and store[p + "water_dash"] > 0.0)
            mpp = 0.0
            if spaced and view_m is not None:
                lo, hi = -np.inf, np.inf
                if walked:  # only the drawn part
                    lo, hi = self.tail_m - g.offset_m, head - g.offset_m
                mpp = line_m_per_px(g.enu, g.dist, camera.position, view_m, *self._viewport,
                                    lo, hi)  # fmt: skip
            set_uniform(program, "u_line_mpp", mpp)
            vao.render(moderngl.TRIANGLE_STRIP)

    @staticmethod
    def _group_style(store, g: _GpuTrack, single: bool):
        """(colour, opacity, width factor, glow factor, dash px, dots) or None when hidden."""
        default_color = store["tracks.color"] if single and store is not None else None
        if default_color is None:
            default_color = track_color(g.index)
        p = f"trackgroup.{g.track.group}."
        if store is None or p + "visible" not in store.registry:
            return default_color, 1.0, 1.0, 1.0, 0.0, False
        if not store[p + "visible"] or store[p + "opacity"] <= 0.0:
            return None
        mode = store[p + "color_mode"]
        if mode == "single":
            color = store[p + "color"]
        elif mode == "per_track":
            pid = p + f"track{g.index}.color"
            color = store[pid] if pid in store.registry else track_color(g.index)
        else:
            color = default_color
        opacity = store[p + "opacity"]
        highlight = store[p + "highlight"]
        if highlight != "none":
            if g.track.name == highlight:
                color = store[p + "highlight_color"]
            else:
                opacity *= store[p + "dim"]
        dots = bool(store[p + "dots"]) if p + "dots" in store.registry else False
        return color, opacity, store[p + "width"], store[p + "glow"], store[p + "dash"], dots

    def release(self) -> None:
        for g in self._gpu:
            for vao in g.vaos.values():
                vao.release()
            g.vbo.release()
            g.ibo.release()
        self._gpu = []


def line_m_per_px(enu: np.ndarray, dist: np.ndarray, camera_position, m: np.ndarray, width: int,
                  height: int, lo: float = -np.inf, hi: float = np.inf) -> float:  # fmt: skip
    """Metres per pixel along the on-screen part of a line (0 when none of it is visible).

    The dashes and dots of a line take their spacing from this one value per frame (see
    track.frag): chosen per pixel, the switch to the next spacing ran along a tilted line as a
    wave while the camera moved, as if the line was drawn again. ``m``: the view-projection
    matrix (row-major); ``lo`` / ``hi``: the drawn range of ``dist``."""
    n = len(enu)
    if n < 2:
        return 0.0
    idx = np.unique(np.r_[np.arange(0, n, max(1, n // 512)), n - 1])
    rel = enu[idx] - np.asarray(camera_position, dtype=np.float64)
    clip = np.c_[rel, np.ones(len(rel))] @ m.T
    w = clip[:, 3]
    front = w > 1e-3
    ndc = clip[:, :2] / np.where(front, w, 1.0)[:, None]
    px = ndc * (0.5 * width, 0.5 * height)
    d = dist[idx]
    ls = np.linalg.norm(np.diff(px, axis=0), axis=1)
    dd = np.diff(d)
    mid_d = 0.5 * (d[1:] + d[:-1])
    edge = np.abs(0.5 * (ndc[1:] + ndc[:-1])).max(axis=1)
    inside = 1.0 - np.clip((edge - 1.0) / 0.3, 0.0, 1.0)  # fades out beyond the screen edges
    ok = front[1:] & front[:-1] & (mid_d >= lo) & (mid_d <= hi) & (ls > 1e-3) & (dd > 0.0)
    weight = np.where(ok, ls * inside, 0.0)
    if weight.sum() < 1.0:
        return 0.0
    log_mpp = np.log2(np.where(ok, dd, 1.0) / np.where(ok, ls, 1.0))
    # x 4/pi: the per-pixel scale (fwidth) of a line in an average direction on screen
    return float(2.0 ** (np.sum(weight * log_mpp) / weight.sum()) * 4.0 / np.pi)


_GpuTrack = BuiltTrack  # (old name)
