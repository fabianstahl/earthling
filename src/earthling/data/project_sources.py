"""Data sources defined in a project's ``earthling.toml`` (``[[providers]]`` tables)."""

from __future__ import annotations

from pathlib import Path

from earthling.core.config import Project, ProviderDef
from earthling.data.dem import DEM_SOURCES, DemSource, StacDemSource, WmsDemSource
from earthling.data.providers import License, TileProvider, register


def _coverage_ref(value: str | None, folder: Path) -> str | None:
    """Shipped coverage names stay names; GeoJSON paths are resolved against the project."""
    if value is None or not Path(value).suffix:
        return value
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (folder / path).resolve())


def build_source(definition: ProviderDef, folder: Path) -> TileProvider | DemSource:
    d = definition
    license = License(d.license, d.attribution, d.license_url, d.commercial_use)
    coverage = _coverage_ref(d.coverage, folder)
    name = d.name or d.id
    if d.kind != "dem":
        return TileProvider(
            id=d.id,
            name=name,
            kind=d.kind,
            url_template=d.url,
            ext=d.ext,
            min_zoom=d.min_zoom,
            max_zoom=d.max_zoom,
            tile_size=d.tile_size,
            license=license,
            max_requests_per_second=d.requests_per_second,
            concurrency=d.concurrency,
            headers=dict(d.headers),
            placeholder_md5=frozenset(d.placeholder_md5),
            coverage=coverage,
            feather_m=d.feather_m,
        )
    source: DemSource
    if d.type == "wms":
        source = WmsDemSource()
        source.wms_url = d.url
        source.layer = d.layer
        source.oversample = d.oversample
        source.nodata_below = d.nodata_below
    else:
        source = StacDemSource()
        source.items_url = d.url
        source.asset_suffix = d.asset_suffix
    source.id = d.id
    source.name = name
    source.license = license
    source.coverage = coverage
    source.feather_m = d.feather_m
    source.native_resolution_m = d.native_resolution_m
    source.concurrency = d.concurrency
    source.max_requests_per_second = d.requests_per_second
    return source


def register_project_sources(project: Project) -> list[TileProvider | DemSource]:
    """Make the project's own providers available by id (they may replace built-in ones)."""
    sources = []
    for definition in project.config.providers:
        source = build_source(definition, project.folder)
        if isinstance(source, TileProvider):
            register(source)
        else:
            DEM_SOURCES[source.id] = source
        sources.append(source)
    return sources
