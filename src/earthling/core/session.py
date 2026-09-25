"""Everything derived from a loaded project: tracks, local frame, area of interest, tile plan.

Independent of Qt so it can be used by the GUI, the CLI and tests alike.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

from earthling.core.aoi import AreaOfInterest, TilePlan, compute_aoi, plan_tiles
from earthling.core.config import ConfigError, Project
from earthling.core.geo import LocalFrame, frame_for_bbox
from earthling.core.gpx import Track, load_gpx_folder
from earthling.data.cache import TileCache
from earthling.data.providers import TileProvider, get_provider

DEFAULT_FRAME = LocalFrame(46.0, 7.0, 0.0)


class Session:
    def __init__(self, project: Project) -> None:
        self.project = project
        self.tracks: list[Track]
        self.tracks, self.load_errors = load_gpx_folder(project.gpx_dir)
        try:
            self.imagery_providers: list[TileProvider] = [
                get_provider(pid) for pid in self.config.sources.imagery
            ]
        except KeyError as exc:
            raise ConfigError(f"sources.imagery: {exc.args[0]}") from exc
        if not self.imagery_providers:
            raise ConfigError("sources.imagery must name at least one provider")
        self.cache = TileCache(project.cache_dir)

    @property
    def imagery_provider(self) -> TileProvider:
        return self.imagery_providers[0]

    @property
    def config(self):
        return self.project.config

    @cached_property
    def aoi(self) -> AreaOfInterest | None:
        return compute_aoi(self.tracks, self.config.area)

    @cached_property
    def frame(self) -> LocalFrame:
        if self.aoi is None:
            return DEFAULT_FRAME
        return frame_for_bbox(*self.aoi.bounds)

    @cached_property
    def plan(self) -> TilePlan | None:
        if self.aoi is None:
            return None
        return plan_tiles(self.aoi, self.config.area)

    @cached_property
    def mean_track_elevation(self) -> float:
        values = [
            np.nanmean(s.ele) for t in self.tracks for s in t.segments if np.isfinite(s.ele).any()
        ]
        return float(np.mean(values)) if values else 0.0

    def outline_lines(self) -> list[tuple[np.ndarray, tuple[float, ...], bool]]:
        """AOI and zone outlines as ENU polylines at mean track elevation."""
        if self.aoi is None:
            return []
        h = self.mean_track_elevation
        lines = []

        def rings(geom):
            polys = getattr(geom, "geoms", [geom])
            for poly in polys:
                yield np.asarray(poly.exterior.coords)
                for interior in poly.interiors:
                    yield np.asarray(interior.coords)

        zone_colors = [(0.95, 0.95, 0.4, 0.8), (0.9, 0.6, 0.3, 0.7), (0.8, 0.4, 0.3, 0.6)]
        for index, zone in enumerate(self.aoi.zones[:-1]):
            color = zone_colors[index % len(zone_colors)]
            for ring in rings(zone):
                lines.append((self.frame.geodetic_to_enu(ring[:, 1], ring[:, 0], h), color, True))
        for ring in rings(self.aoi.aoi):
            enu = self.frame.geodetic_to_enu(ring[:, 1], ring[:, 0], h)
            lines.append((enu, (1.0, 1.0, 1.0, 0.9), True))
        return lines
