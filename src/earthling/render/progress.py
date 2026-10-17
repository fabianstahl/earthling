"""The journey axis: all walked tracks in time order as one path (global distance <->
position <-> time), the basis of the progress animation, the stats and the camera rig."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from earthling.render.track_geometry import BuiltTrack


class ProgressPath:
    """The whole hike as one path: global distance (m) <-> position <-> time."""

    def __init__(self, tracks: list[BuiltTrack]) -> None:
        self.tracks = tracks
        self.total_m = tracks[-1].offset_m + tracks[-1].length_m if tracks else 0.0
        dists, times, eles, grounds, owners, hrs = [], [], [], [], [], []
        ground_offset = 0.0
        for g in tracks:
            dists.append(g.dist + g.offset_m)
            times.append(g.time)
            n = len(g.dist)
            eles.append(g.ele if g.ele is not None else np.full(n, np.nan))
            ground = g.ground if g.ground is not None else g.dist
            grounds.append(ground + ground_offset)
            ground_offset += float(ground[-1]) if n else 0.0
            owners.append(np.full(n, g.index))
            hrs.append(g.hr if g.hr is not None else np.full(n, np.nan))
        self.dist = np.concatenate(dists) if dists else np.zeros(0)
        self.time = np.concatenate(times) if times else np.zeros(0)
        self.ele = np.concatenate(eles) if eles else np.zeros(0)  # true elevation per point
        self.ground = np.concatenate(grounds) if grounds else np.zeros(0)  # walked metres
        self.track_index = np.concatenate(owners) if owners else np.zeros(0, dtype=int)
        self.hr = np.concatenate(hrs) if hrs else np.zeros(0)  # heart rate per point
        # elevation, gradient, speed, heart rate per point (as the tracks are coloured)
        values = [g.values for g in tracks if g.values is not None]
        self.values = np.vstack(values) if len(values) == len(tracks) and values else None
        # clock mode (several heads at once): a time shift per group (seconds) and the group
        # whose head leads the camera, the sun and the stats (set by the renderer)
        self.time_offsets: dict[str, float] = {}
        self.lead_group: str | None = None
        valid = np.isfinite(self.time)
        # time mode needs monotonic timestamps; otherwise fall back to distance
        self.has_time = bool(valid.sum() >= 2) and bool(np.all(np.diff(self.time[valid]) >= 0))
        self._valid = valid

    # --- clock mode: every track drawn up to the same (shifted) clock time ----------------------
    def _times(self, g) -> np.ndarray | None:
        t = g.time
        if t is None or not np.isfinite(t).any():
            return None
        return t + self.time_offsets.get(g.track.group, 0.0)

    def clock_range(self) -> tuple[float, float]:
        """First and last (shifted) time of all tracks (epoch seconds)."""
        lo, hi = np.inf, -np.inf
        for g in self.tracks:
            t = self._times(g)
            if t is not None:
                lo, hi = min(lo, np.nanmin(t)), max(hi, np.nanmax(t))
        return (float(lo), float(hi)) if np.isfinite(lo) else (0.0, 1.0)

    def clock_at(self, progress: float) -> float:
        lo, hi = self.clock_range()
        return lo + min(max(progress, 0.0), 1.0) * (hi - lo)

    def head_on(self, g, clock: float) -> float | None:
        """Distance along track ``g`` at the (shifted) clock time, None before its start."""
        t = self._times(g)
        if t is None:
            return None
        ok = np.isfinite(t)
        if clock < t[ok][0]:
            return None
        return float(np.interp(clock, t[ok], g.dist[ok]))

    def only(self, g) -> ProgressPath:
        """A path of track ``g`` alone (cached): the stats of one runner in clock mode."""
        cache = self.__dict__.setdefault("_single", {})
        if id(g) not in cache:
            cache[id(g)] = ProgressPath([replace(g, offset_m=0.0)])
        return cache[id(g)]

    def gap_at(self, clock: float) -> float | None:
        """Clock mode: seconds the second track needs to reach the lead track's head (negative:
        the lead is behind), None without a second track."""
        lead = self.lead()
        other = next((g for g in self.tracks if g is not lead), None)
        if lead is None or other is None:
            return None
        d_lead, d_other = self.head_on(lead, clock), self.head_on(other, clock)
        if d_lead is None or d_other is None:
            return None
        if d_other <= d_lead:  # the other one still has to get here
            t = self._times(other)
            ok = np.isfinite(t)
            return float(np.interp(d_lead, other.dist[ok], t[ok]) - clock)
        t = self._times(lead)
        ok = np.isfinite(t)
        return -float(np.interp(d_other, lead.dist[ok], t[ok]) - clock)

    def lead(self):
        """The track that leads in clock mode: the first of the lead group (or of all)."""
        for g in self.tracks:
            if g.track.group == self.lead_group:
                return g
        return self.tracks[0] if self.tracks else None

    def distance_for(self, progress: float, mode: str = "distance") -> float:
        progress = min(max(progress, 0.0), 1.0)
        if mode == "clock" and self.tracks:
            g = self.lead()
            d = self.head_on(g, self.clock_at(progress))
            return g.offset_m + (d if d is not None else 0.0)
        if mode == "time" and self.has_time:
            t = self.time[self._valid]
            d = self.dist[self._valid]
            target = t[0] + progress * (t[-1] - t[0])
            return float(np.interp(target, t, d))
        return progress * self.total_m

    def position_at(self, distance: float) -> np.ndarray | None:
        for g in self.tracks:
            if g.enu is None or g.dist is None or len(g.dist) == 0:
                continue
            if distance <= g.offset_m + g.length_m or g is self.tracks[-1]:
                local = min(max(distance - g.offset_m, 0.0), g.length_m)
                return np.array([np.interp(local, g.dist, g.enu[:, k]) for k in range(3)])
        return None

    def distance_for_time(self, seconds: float) -> float:
        """Distance of the recorded point at ``seconds`` (UTC epoch), clamped to the hike."""
        valid = self._valid
        if not valid.any():
            return 0.0
        return float(np.interp(seconds, self.time[valid], self.dist[valid]))

    def progress_for_distance(self, distance: float, mode: str = "distance") -> float:
        """Inverse of :meth:`distance_for` (the progress value that puts the head there)."""
        if self.total_m <= 0:
            return 0.0
        if mode == "clock" and self.tracks:
            g = self.lead()
            t = self._times(g)
            if t is not None:
                ok = np.isfinite(t)
                seconds = float(np.interp(distance - g.offset_m, g.dist[ok], t[ok]))
                lo, hi = self.clock_range()
                return (seconds - lo) / max(hi - lo, 1e-9)
        if mode == "time" and self.has_time:
            t = self.time[self._valid]
            seconds = float(np.interp(distance, self.dist[self._valid], t))
            return (seconds - t[0]) / max(t[-1] - t[0], 1e-9)
        return min(max(distance / self.total_m, 0.0), 1.0)

    def track_at(self, distance: float):
        """The (built) track the global distance lies on."""
        for g in self.tracks:
            if distance <= g.offset_m + g.length_m:
                return g
        return self.tracks[-1] if self.tracks else None

    def track_range(self, index: int) -> tuple[float, float]:
        """(start, end) distance of walked track ``index`` on the global axis."""
        g = self.tracks[index]
        return g.offset_m, g.offset_m + g.length_m

    def time_at(self, distance: float) -> float | None:
        valid = self._valid
        if not valid.any():
            return None
        return float(np.interp(distance, self.dist[valid], self.time[valid]))


def journey_path(built: list[BuiltTrack]) -> ProgressPath:
    """Progress runs along the walked tracks only, in time order: several walked groups (e.g.
    hikes, bus rides, flights) form one journey. Sets each track's offset on the global axis."""
    walked = [g for g in built if g.track.role == "walked"]
    walked.sort(key=_start_time)
    offset = 0.0
    for g in walked:
        g.offset_m = offset
        offset += g.length_m
    return ProgressPath(walked)


def _start_time(g: BuiltTrack) -> float:
    """First timestamp of a built track (tracks without time keep their order, at the end)."""
    t = g.time[np.isfinite(g.time)] if g.time is not None else ()
    return float(t[0]) if len(t) else float("inf")
