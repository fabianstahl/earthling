"""Track geometry on the CPU (float64, no GL): the input of the track ribbons and of the
journey axis.

1. resample every segment to a minimum point spacing and smooth it,
2. take heights from the DEM (plus a height offset) or from the GPX file,
3. convert to ENU (curvature correct) and expand every point into two vertices (left/right),
4. per point: distance, time, walked distance and the colouring values.

:func:`build_tracks` does all of it for a project's tracks; the TrackLayer uploads the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from earthling.core.geo import LocalFrame
from earthling.core.gpx import Track, haversine_m

PRIMITIVE_RESTART = 0xFFFFFFFF


def resample(lat: np.ndarray, lon: np.ndarray, ele: np.ndarray, spacing_m: float):
    """Drop points closer than ``spacing_m`` to the previously kept one (keeps the ends)."""
    idx = resample_indices(lat, lon, spacing_m)
    return lat[idx], lon[idx], ele[idx]


def resample_indices(lat: np.ndarray, lon: np.ndarray, spacing_m: float) -> np.ndarray:
    if len(lat) < 3 or spacing_m <= 0:
        return np.arange(len(lat))
    steps = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    keep = [0]
    last = 0.0
    for i in range(1, len(lat) - 1):
        if cumulative[i] - last >= spacing_m:
            keep.append(i)
            last = cumulative[i]
    keep.append(len(lat) - 1)
    return np.array(keep)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with shrinking windows at the ends (NaN-free input): point i
    averages the min(half, i, n - 1 - i) points on either side, so the first and the last point
    stay where they are. (Padding with the end values pulled both ends inwards, by kilometres
    on sparse lines such as a flight with a point every 10 km.)"""
    n = len(values)
    if window <= 1 or n < 3:
        return values
    half = window // 2
    r = np.minimum(half, np.minimum(np.arange(n), n - 1 - np.arange(n)))
    cum = np.concatenate([[0.0], np.cumsum(np.asarray(values, dtype=np.float64))])
    i = np.arange(n)
    return (cum[i + r + 1] - cum[i - r]) / (2 * r + 1)


GRADIENT_WINDOW_M = 60.0  # gradient over +- this distance (DEM noise at hairpins)
SPEED_WINDOW_S = 15.0  # speed over +- this time
# per-vertex values the tracks can be coloured by (tracks.color_by), in this order
VALUE_KINDS = ("elevation", "gradient", "speed", "heart_rate")


def motion_values(ground: np.ndarray, ele: np.ndarray, time: np.ndarray,
                  hr: np.ndarray | None = None) -> np.ndarray:  # fmt: skip
    """(n, 4): elevation (m), gradient (%), speed (km/h) and heart rate (bpm) per point, NaN
    where unknown. ``ground``: metres walked (non-decreasing), ``time``: epoch seconds."""
    n = len(ground)
    out = np.full((n, 4), np.nan)
    if n == 0:
        return out
    ground = np.asarray(ground, dtype=np.float64)
    ele = np.asarray(ele, dtype=np.float64)
    ok = np.isfinite(ele)
    if ok.sum() >= 2:
        e = np.interp(ground, ground[ok], ele[ok])
        lo = np.clip(ground - GRADIENT_WINDOW_M, ground[0], ground[-1])
        hi = np.clip(ground + GRADIENT_WINDOW_M, ground[0], ground[-1])
        span = np.maximum(hi - lo, 1e-6)
        out[:, 0] = e
        rise = np.interp(hi, ground, e) - np.interp(lo, ground, e)
        out[:, 1] = np.where(hi - lo > 1.0, 100.0 * rise / span, 0.0)
    t = np.asarray(time, dtype=np.float64)
    ok = np.isfinite(t)
    if ok.sum() >= 2 and np.all(np.diff(t[ok]) >= 0):
        tt, gg = t[ok], ground[ok]
        lo = np.clip(t - SPEED_WINDOW_S, tt[0], tt[-1])
        hi = np.clip(t + SPEED_WINDOW_S, tt[0], tt[-1])
        span = hi - lo
        with np.errstate(invalid="ignore", divide="ignore"):
            speed = 3.6 * (np.interp(hi, tt, gg) - np.interp(lo, tt, gg)) / span
        out[:, 2] = np.where(ok & (span > 0.5), speed, np.nan)
    if hr is not None:
        out[:, 3] = np.asarray(hr, dtype=np.float64)
    return out


@dataclass
class TrackGeometryOptions:
    elevation_source: str = "dem"  # "dem" or "gpx"
    height_offset_m: float = 3.0
    exaggeration: float = 1.0
    spacing_m: float = 5.0
    smoothing: int = 5  # moving-average window in points


@dataclass
class TrackSegmentGeometry:
    enu: np.ndarray  # (n, 3) float64
    dist: np.ndarray  # (n,) metres along the drawn (exaggerated) track
    time: np.ndarray  # (n,) seconds since the epoch, NaN where unknown
    ele: np.ndarray | None = None  # (n,) true elevation (no exaggeration / offset)
    ground: np.ndarray | None = None  # (n,) metres actually walked (3D, true heights)
    hr: np.ndarray | None = None  # (n,) heart rate (bpm), NaN where unknown
    water: np.ndarray | None = None  # (n,) 1 on water, 0 on land (groups split by water)
    gpx_ele: np.ndarray | None = None  # (n,) the file's (smoothed) elevations, NaN if none


def build_track_positions(
    track: Track, frame: LocalFrame, options: TrackGeometryOptions, heights_at=None,
    water_at=None,
) -> list[TrackSegmentGeometry]:  # fmt: skip
    """Per segment: ENU positions, cumulative distance along the track and timestamps.

    ``heights_at(lon, lat) -> heights`` (vectorised, NaN where unknown) provides DEM heights,
    ``water_at(lon, lat) -> coverage`` the water mask (tracks split by water).
    """
    result = []
    distance_offset = 0.0
    ground_offset = 0.0
    for seg in track.segments:
        idx = resample_indices(seg.lat, seg.lon, options.spacing_m)
        lat, lon, ele = seg.lat[idx], seg.lon[idx], seg.ele[idx]
        times = seg.time[idx]
        hr = seg.hr[idx] if getattr(seg, "hr", None) is not None else np.full(len(idx), np.nan)
        seconds = np.where(
            np.isnat(times), np.nan, times.astype("datetime64[ms]").astype(np.float64) / 1000.0
        )
        if len(lat) < 2:
            continue
        gpx_h = np.nan_to_num(ele, nan=np.nanmean(ele) if np.isfinite(ele).any() else 0.0)
        if options.elevation_source == "dem" and heights_at is not None:
            dem_h = heights_at(lon, lat)
            h = np.where(np.isfinite(dem_h), dem_h, gpx_h)
        else:
            h = gpx_h
        lat = smooth(lat, options.smoothing)
        lon = smooth(lon, options.smoothing)
        if options.elevation_source != "dem":
            h = smooth(h, options.smoothing)
        true_h = h
        h = h * options.exaggeration + options.height_offset_m
        enu = frame.geodetic_to_enu(lat, lon, h)
        steps = np.linalg.norm(np.diff(enu, axis=0), axis=1)
        dist = distance_offset + np.concatenate([[0.0], np.cumsum(steps)])
        distance_offset = float(dist[-1])
        if options.exaggeration != 1.0:
            true_steps = np.linalg.norm(
                np.diff(frame.geodetic_to_enu(lat, lon, true_h), axis=0), axis=1
            )
        else:
            true_steps = steps
        ground = ground_offset + np.concatenate([[0.0], np.cumsum(true_steps)])
        ground_offset = float(ground[-1])
        water = None
        if water_at is not None and getattr(track, "split_water", False):
            water = (np.asarray(water_at(lon, lat), dtype=np.float64) > 0.5).astype(np.float64)
            water = (smooth(water, 3) > 0.5).astype(np.float64)  # no single-point flicker
        gpx_ele = np.full(len(ele), np.nan)
        if np.isfinite(ele).all():
            gpx_ele = smooth(ele, options.smoothing)
        result.append(
            TrackSegmentGeometry(
                enu, dist, seconds, np.asarray(true_h, float), ground, hr, water, gpx_ele
            )  # fmt: skip
        )
    return result


def calibrate_ground(ground: np.ndarray, enu: np.ndarray, time: np.ndarray, marks,
                     end_km: float | None, frame: LocalFrame,
                     tolerance_m: float = 30.0) -> np.ndarray:  # fmt: skip
    """Walked metres corrected to known distances (config.DistanceMark): piecewise linear
    between the marks, ``end_km`` at the last point (without it, the factor of the last
    stretch continues). Position marks take the first point within ``tolerance_m`` of the
    nearest track point after the previous mark (loops pass the same places twice); marks that
    do not lie further along the track are ignored."""
    ground = np.asarray(ground, dtype=np.float64)
    if len(ground) < 2 or (not marks and end_km is None):
        return ground
    src, dst = [float(ground[0])], [float(ground[0])]
    after = 0
    t_ok = np.isfinite(time)
    for mark in marks:
        if mark.lat is not None:
            p = np.asarray(frame.geodetic_to_enu(mark.lat, mark.lon, 0.0), dtype=np.float64)
            gap = np.linalg.norm(enu[after:, :2] - p.reshape(3)[:2], axis=1)
            near = gap <= gap.min() + tolerance_m
            first = int(np.argmax(near))  # the first pass near the mark ...
            last = first + (int(np.argmax(~near[first:])) if (~near[first:]).any() else len(near))
            i = after + first + int(np.argmin(gap[first:last]))  # ... its closest point
            g = float(ground[i])
        elif t_ok.sum() >= 2:
            g = float(np.interp(mark.time.timestamp(), time[t_ok], ground[t_ok]))
            i = int(np.searchsorted(ground, g))
        else:
            continue
        if g <= src[-1] or mark.km * 1000.0 <= dst[-1]:
            continue
        src.append(g)
        dst.append(mark.km * 1000.0)
        after = min(i, len(ground) - 1)
    if end_km is not None and ground[-1] > src[-1] and end_km * 1000.0 > dst[-1]:
        src.append(float(ground[-1]))
        dst.append(end_km * 1000.0)
    if len(src) < 2:
        return ground
    out = np.interp(ground, src, dst)
    beyond = ground > src[-1]  # (no end distance) the last factor continues
    if beyond.any():
        factor = (dst[-1] - dst[-2]) / (src[-1] - src[-2])
        out[beyond] = dst[-1] + (ground[beyond] - src[-1]) * factor
    return out


VERTEX_FLOATS = 17


def ribbon_vertices(segments: list[TrackSegmentGeometry], origin: np.ndarray,
                    values: np.ndarray | None = None, t0: float = 0.0):  # fmt: skip
    """Vertex array (pos, prev, next, side, dist, values[4], time) and strip indices with
    primitive restart. ``values``: per point (all segments, see motion_values), ``time``
    is relative to ``t0`` (float32 keeps it exact to milliseconds over days)."""
    verts, indices = [], []
    base = 0
    first = 0
    for geometry in segments:
        enu, dist = geometry.enu, geometry.dist
        n = len(enu)
        rel = enu - origin
        prev = np.vstack([2 * rel[0] - rel[1], rel[:-1]])  # mirror the ends
        nxt = np.vstack([rel[1:], 2 * rel[-1] - rel[-2]])
        block = np.empty((n, 2, VERTEX_FLOATS), dtype=np.float64)
        vals = values[first : first + n] if values is not None else np.zeros((n, 4))
        rel_t = geometry.time - t0 if geometry.time is not None else np.full(n, np.nan)
        for k, side in enumerate((-1.0, 1.0)):
            block[:, k, 0:3] = rel
            block[:, k, 3:6] = prev
            block[:, k, 6:9] = nxt
            block[:, k, 9] = side
            block[:, k, 10] = dist
            block[:, k, 11:15] = np.nan_to_num(vals, nan=0.0)
            block[:, k, 15] = np.nan_to_num(rel_t, nan=-1e9)
            block[:, k, 16] = geometry.water if geometry.water is not None else 0.0
        verts.append(block.reshape(-1, VERTEX_FLOATS))
        first += n
        indices.append(np.arange(base, base + 2 * n, dtype=np.uint32))
        indices.append(np.array([PRIMITIVE_RESTART], dtype=np.uint32))
        base += 2 * n
    if not verts:
        return np.zeros((0, VERTEX_FLOATS), dtype="f4"), np.zeros(0, dtype=np.uint32)
    return np.vstack(verts).astype("f4"), np.concatenate(indices)


@dataclass
class BuiltTrack:
    """One track as drawn: its points in the local frame plus the ribbon vertices. The GPU
    buffers are filled in by the TrackLayer (None for a headless build, e.g. scene scripts)."""

    track: Track
    index: int  # position within its group
    origin: np.ndarray  # ENU origin of the vertex positions (float32 precision)
    length_m: float
    offset_m: float = 0.0  # start of this track on the global (all tracks) distance axis
    enu: np.ndarray | None = None  # all points (float64), for head positions
    dist: np.ndarray | None = None
    time: np.ndarray | None = None
    ele: np.ndarray | None = None  # true elevations
    ground: np.ndarray | None = None  # walked distance within the track
    hr: np.ndarray | None = None  # heart rate (bpm, NaN where unknown)
    t0: float = 0.0  # epoch seconds the vertex times are relative to
    values: np.ndarray | None = None  # (n, 4) elevation, gradient, speed, heart rate
    verts: np.ndarray | None = None  # ribbon vertices (until uploaded)
    indices: np.ndarray | None = None
    vbo: object = None  # moderngl.Buffer
    ibo: object = None
    vaos: dict = field(default_factory=dict)  # program glo -> moderngl.VertexArray


def build_tracks(tracks: list[Track], frame: LocalFrame, options: TrackGeometryOptions,
                 heights_at=None, water_at=None) -> list[BuiltTrack]:  # fmt: skip
    """Geometry of every track (no GPU needed): positions on the terrain (``heights_at``:
    vectorised DEM height function), distances, times, colouring values, ribbon vertices."""
    built = []
    group_index: dict[str, int] = {}
    for track in tracks:
        index = group_index.get(track.group, 0)
        group_index[track.group] = index + 1
        spacing = track.spacing_m or (options.spacing_m if track.role == "walked" else 25.0)
        track_options = replace(options, spacing_m=spacing)
        if getattr(track, "elevation", None) == "gpx":  # e.g. a flight: heights as recorded
            track_options = replace(track_options, elevation_source="gpx", height_offset_m=0.0,
                                    exaggeration=1.0)  # fmt: skip
        segments = build_track_positions(track, frame, track_options, heights_at, water_at)
        if not segments:
            continue
        origin = segments[0].enu[0].copy()
        all_time = np.concatenate([s.time for s in segments])
        if getattr(track, "distance_marks", None) or getattr(track, "distance_end_km", None):
            ground = calibrate_ground(np.concatenate([s.ground for s in segments]),
                                      np.vstack([s.enu for s in segments]), all_time,
                                      track.distance_marks, track.distance_end_km,
                                      frame)  # fmt: skip
            first = 0
            for seg in segments:
                seg.ground = ground[first : first + len(seg.ground)]
                first += len(seg.ground)
        # gradients from the recorded elevations where the file has them (a draped DEM is
        # noisy where a road is cut into a slope, e.g. at hairpins)
        gpx_ele = np.concatenate([s.gpx_ele for s in segments])
        if not np.isfinite(gpx_ele).all():
            gpx_ele = np.concatenate([s.ele for s in segments])
        values = motion_values(np.concatenate([s.ground for s in segments]), gpx_ele, all_time,
                               np.concatenate([s.hr for s in segments]))  # fmt: skip
        t_ok = all_time[np.isfinite(all_time)]
        t0 = float(t_ok[0]) if len(t_ok) else 0.0
        verts, indices = ribbon_vertices(segments, origin, values, t0)
        built.append(BuiltTrack(
            track,
            index,
            origin,
            float(segments[-1].dist[-1]),
            enu=np.vstack([s.enu for s in segments]),
            dist=np.concatenate([s.dist for s in segments]),
            time=np.concatenate([s.time for s in segments]),
            ele=np.concatenate([s.ele for s in segments]),
            ground=np.concatenate([s.ground for s in segments]),
            hr=np.concatenate([s.hr for s in segments]),
            t0=t0,
            values=values,
            verts=verts,
            indices=indices,
        ))  # fmt: skip
    return built
