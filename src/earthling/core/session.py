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
        from earthling.data.project_sources import register_project_sources

        self.project_sources = register_project_sources(project)
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
        try:
            self.topo_providers: list[TileProvider] = [
                get_provider(pid) for pid in self.config.sources.topo
            ]
        except KeyError as exc:
            raise ConfigError(f"sources.topo: {exc.args[0]}") from exc
        from earthling.data.dem import DemSource, get_dem_source

        try:
            self.dem_sources: list[DemSource] = [get_dem_source(d) for d in self.config.sources.dem]
        except KeyError as exc:
            raise ConfigError(f"sources.dem: {exc.args[0]}") from exc
        if not self.dem_sources:
            raise ConfigError("sources.dem must name at least one source")
        self.cache = TileCache(project.cache_dir)

    @property
    def dem_source(self):
        return self.dem_sources[0]

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

    def first_local_start(self):
        """Local (project timezone) start time of the earliest track, naive, or None."""
        from datetime import UTC
        from zoneinfo import ZoneInfo

        starts = [t.stats.start_time for t in self.tracks if t.stats.start_time is not None]
        if not starts:
            return None
        tz = ZoneInfo(self.config.project.timezone)
        return min(starts).replace(tzinfo=UTC).astimezone(tz).replace(tzinfo=None)

    # --- terrain ---------------------------------------------------------------------------
    def tile_sources(self) -> dict[str, list[TileProvider]]:
        """Texture sources by layer tile-source name (providers in priority order)."""
        sources = {"imagery": list(self.imagery_providers)}
        if self.topo_providers:
            sources["topo"] = list(self.topo_providers)
        return sources

    def provider_tiles(self, kind: str) -> list[tuple[object, dict[int, np.ndarray]]]:
        """(provider or DEM source, {zoom: tiles}) for the tiles each one is needed for:
        within its coverage, not fully covered by a higher-priority one, up to its maximum
        zoom. ``kind``: 'imagery', 'topo' or 'dem'."""
        from earthling.data.coverage import needed_tiles

        if self.plan is None:
            return []
        providers = {
            "imagery": self.imagery_providers,
            "topo": self.topo_providers,
            "dem": self.dem_sources,
        }[kind]
        levels = self.plan.levels.get("dem" if kind == "dem" else "imagery", {})
        needs = needed_tiles([p.coverage_area for p in providers], levels)
        out = []
        for provider, tiles in zip(providers, needs, strict=True):
            max_zoom = getattr(provider, "max_zoom", None)
            if max_zoom is not None:
                tiles = {z: t for z, t in tiles.items() if z <= max_zoom}
            out.append((provider, tiles))
        return out

    def attribution(self) -> str:
        """Credits of the data sources in use (for the stats overlay)."""
        parts = []
        for kind in ("imagery", "dem"):
            for provider, tiles in self.provider_tiles(kind):
                text = provider.license.attribution
                if tiles and text and text not in parts:
                    parts.append(text)
        if self.labels is not None:
            from earthling.data.labels import ATTRIBUTION

            parts.append(ATTRIBUTION)
        return " · ".join(parts)

    def provider_report(self) -> str:
        from earthling.core.aoi import AVG_TILE_BYTES, format_bytes

        lines = []
        for kind, title in (("imagery", "imagery"), ("dem", "elevation")):
            lines.append(f"{title} sources (priority order):")
            for provider, tiles in self.provider_tiles(kind):
                count = sum(len(t) for t in tiles.values())
                size = format_bytes(count * AVG_TILE_BYTES[kind])
                region = "worldwide" if provider.coverage is None else provider.coverage
                lines.append(f"  {provider.name:<36} {count:>8} tiles  ~{size:>9}  ({region})")
        return "\n".join(lines)

    @cached_property
    def labels(self):
        """OpenStreetMap label features of the area (loaded on demand)."""
        from earthling.data.labels import LabelData

        if self.aoi is None:
            return None
        return LabelData(self.cache, self.aoi.bounds)

    @cached_property
    def borders(self):
        from earthling.data.borders import BorderData

        return BorderData(self.cache)

    def terrain_data(self, on_demand: bool = True):
        from earthling.data.terrain_data import TerrainData

        if self.plan is None:
            return None
        return TerrainData(
            self.cache,
            [source.id for source in self.dem_sources],
            self.tile_sources(),
            self.plan,
            on_demand=on_demand,
            borders=self.borders,
        )

    def terrain_nodes(self):
        from earthling.data.terrain_data import TEXEL_ZOOM_OFFSET
        from earthling.render.lod import NodeSet

        if self.plan is None:
            return None
        return NodeSet.from_plan(
            self.plan.levels.get("dem", {}), self.plan.levels.get("imagery", {}), TEXEL_ZOOM_OFFSET
        )
