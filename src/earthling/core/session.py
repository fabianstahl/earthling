"""Everything derived from a loaded project: tracks, local frame, area of interest, tile plan.

Independent of Qt so it can be used by the GUI, the CLI and tests alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import shapely

from earthling.core.aoi import AreaOfInterest, TilePlan, compute_aoi, plan_tiles_multi
from earthling.core.config import AreaSection, ConfigError, Project, TrackGroupDef
from earthling.core.geo import LocalFrame, frame_for_bbox
from earthling.core.gpx import Track, load_gpx_file, load_gpx_folder
from earthling.data.cache import TileCache
from earthling.data.providers import TileProvider, get_provider

DEFAULT_FRAME = LocalFrame(46.0, 7.0, 0.0)


@dataclass
class TrackGroup:
    definition: TrackGroupDef
    tracks: list[Track] = field(default_factory=list)
    area: AreaSection | None = None
    aoi: AreaOfInterest | None = None

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def label(self) -> str:
        return self.definition.label or self.definition.name.replace("_", " ").title()

    @property
    def role(self) -> str:
        return self.definition.role

    @property
    def labels(self) -> bool:
        flag = self.definition.labels
        return self.role == "walked" if flag is None else flag


def load_group_tracks(path: Path) -> tuple[list[Track], list[str]]:
    """GPX tracks of a folder, a file or a glob pattern."""
    if path.is_dir():
        return load_gpx_folder(path)
    files = sorted(path.parent.glob(path.name)) if any(c in path.name for c in "*?[") else [path]
    tracks: list[Track] = []
    errors: list[str] = []
    for f in files:
        try:
            tracks.extend(load_gpx_file(f))
        except Exception as exc:  # gpxpy raises various exception types
            errors.append(f"{f.name}: {exc}")
    if not files or not any(f.exists() for f in files):
        errors.append(f"no GPX files found: {path}")
    return tracks, errors


class Session:
    def __init__(self, project: Project) -> None:
        self.project = project
        from earthling.data.project_sources import register_project_sources

        self.project_sources = register_project_sources(project)
        self.load_errors: list[str] = []
        self.groups: list[TrackGroup] = self._load_groups()
        self.all_tracks: list[Track] = [t for g in self.groups for t in g.tracks]
        # "the hike": the walked tracks (progress, marker, stats, labels follow them)
        self.tracks: list[Track] = [t for t in self.all_tracks if t.role == "walked"]
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

    def _load_groups(self) -> list[TrackGroup]:
        config = self.project.config
        definitions = config.tracks or (
            TrackGroupDef(name="tracks", gpx=str(config.project.gpx_dir), label="Tracks"),
        )
        groups = []
        for d in definitions:
            path = Path(d.gpx).expanduser()
            if not path.is_absolute():
                path = self.project.folder / path
            tracks, errors = load_group_tracks(path)
            self.load_errors += errors
            for track in tracks:
                track.group, track.role = d.name, d.role
                track.spacing_m = d.spacing_m
            area = d.area or config.area
            groups.append(TrackGroup(d, tracks, area, compute_aoi(tracks, area)))
        return groups

    def track_group_properties(self) -> list:
        """Keyframeable style properties of every track group ("trackgroup.<name>.*")."""
        from earthling.render.parameters import boolean, color, enum, flt, section

        defs = []
        for g in self.groups:
            planned = g.role == "planned"
            p = f"trackgroup.{g.name}."
            members = [("none", "None")] + [(t.name, t.name) for t in g.tracks]
            defs += section(
                f"Tracks: {g.label}",
                boolean(p + "visible", "Show", True),
                flt(p + "opacity", "Opacity", 0.9 if planned else 1.0, 0.0, 1.0),
                enum(p + "color_mode", "Colours", "per_track" if planned else "auto",
                     [("auto", "As in the Tracks section"), ("per_track", "One colour per track"),
                      ("single", "Group colour")]),
                color(p + "color", "Group colour",
                      (0.95, 0.95, 0.95) if planned else (1.0, 0.35, 0.15)),
                flt(p + "width", "Width factor", 0.6 if planned else 1.0, 0.05, 5.0, step=0.05),
                flt(p + "dash", "Dashes", 10.0 if planned else 0.0, 0.0, 100.0, step=1.0,
                    decimals=0, unit="px", tooltip="Dash length on screen (0 = solid line)"),
                flt(p + "glow", "Glow factor", 0.3 if planned else 1.0, 0.0, 5.0),
                enum(p + "highlight", "Highlight", "none", members,
                     tooltip="Show one track in the highlight colour and dim the others"),
                color(p + "highlight_color", "Highlight colour", (0.25, 0.95, 0.35)),
                flt(p + "dim", "Others while highlighting", 0.35, 0.0, 1.0),
            )  # fmt: skip
        return defs

    def group(self, name: str) -> TrackGroup | None:
        return next((g for g in self.groups if g.name == name), None)

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
        """All groups together (the union of their areas)."""
        aois = [g.aoi for g in self.groups if g.aoi is not None]
        if not aois:
            return None
        if len(aois) == 1:
            return aois[0]
        return AreaOfInterest(
            aoi=shapely.union_all([a.aoi for a in aois]),
            zones=[shapely.union_all([a.zones[0] for a in aois])],
            tracks=shapely.union_all([a.tracks for a in aois]),
        )

    @cached_property
    def detail_aoi(self) -> AreaOfInterest | None:
        """The groups with labels (by default the walked ones): labels, local frame."""
        aois = [g.aoi for g in self.groups if g.aoi is not None and g.labels]
        if len(aois) == 1:
            return aois[0]
        if not aois:
            return self.aoi
        return AreaOfInterest(
            aoi=shapely.union_all([a.aoi for a in aois]),
            zones=[shapely.union_all([a.zones[0] for a in aois])],
            tracks=shapely.union_all([a.tracks for a in aois]),
        )

    @cached_property
    def frame(self) -> LocalFrame:
        aoi = self.detail_aoi
        if aoi is None:
            return DEFAULT_FRAME
        return frame_for_bbox(*aoi.bounds)

    @cached_property
    def plan(self) -> TilePlan | None:
        areas = [(g.aoi, g.area) for g in self.groups if g.aoi is not None and g.area is not None]
        if not areas:
            return None
        return plan_tiles_multi(areas)

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
        for group in self.groups:
            if group.aoi is None:
                continue
            for index, zone in enumerate(group.aoi.zones[:-1]):
                color = zone_colors[index % len(zone_colors)]
                for ring in rings(zone):
                    enu = self.frame.geodetic_to_enu(ring[:, 1], ring[:, 0], h)
                    lines.append((enu, color, True))
            for ring in rings(group.aoi.aoi):
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
        used_at = None
        if kind == "dem":
            from earthling.core.geo import ground_resolution_m
            from earthling.data.dem import SOURCE_DETAIL_FACTOR

            west, south, east, north = self.aoi.bounds if self.aoi is not None else (0, 0, 0, 0)
            lat = (south + north) / 2.0

            def used_at(i: int, z: int) -> bool:
                # fine sources only where the heightmaps are fine enough to benefit
                limit = providers[i].native_resolution_m * SOURCE_DETAIL_FACTOR
                return ground_resolution_m(lat, z) <= limit or i == len(providers) - 1

        needs = needed_tiles([p.coverage_area for p in providers], levels, used_at)
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

        if self.detail_aoi is None:
            return None
        return LabelData(self.cache, self.detail_aoi.bounds)

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
