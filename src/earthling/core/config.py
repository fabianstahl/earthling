"""Project configuration (``earthling.toml``)."""

from __future__ import annotations

import tomllib
import zoneinfo
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

CONFIG_FILENAME = "earthling.toml"

SAMPLE_CONFIG = """\
# Earthling project configuration

[project]
gpx_dir   = "gpx/"                 # folder containing one or more .gpx files
cache_dir = "~/earthling-cache"    # tile cache, shared between projects
timezone  = "Europe/Paris"         # used for sun position / time of day

[area]
border_km = 15.0                   # area around the union of all tracks
# Resolution zones by distance to the track (first matching zone wins).
# The outermost zone must reach at least border_km.
zones = [
  { within_km = 2.0,  imagery_zoom = 17, dem_zoom = 13 },
  { within_km = 6.0,  imagery_zoom = 15, dem_zoom = 12 },
  { within_km = 15.0, imagery_zoom = 13, dem_zoom = 11 },
]

[sources]
# Priority order: regional sources are used within their coverage (blended at the edge),
# the worldwide ones everywhere else. Regional: ign_bdortho / ign_rgealti (France),
# swisstopo_swissimage / swisstopo_alti3d (Switzerland).
imagery = ["esri_world_imagery"]
dem     = ["copernicus_glo30"]
topo    = ["opentopomap"]          # topographic map layer (downloaded on demand)
"""


class ConfigError(Exception):
    pass


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectSection(_Model):
    gpx_dir: Path = Path("gpx")
    cache_dir: Path = Path("~/earthling-cache")
    timezone: str = "UTC"

    @field_validator("timezone")
    @classmethod
    def _valid_tz(cls, value: str) -> str:
        try:
            zoneinfo.ZoneInfo(value)
        except (zoneinfo.ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone '{value}'") from exc
        return value


class Zone(_Model):
    within_km: float = Field(gt=0)
    imagery_zoom: int = Field(ge=0, le=22)
    dem_zoom: int = Field(ge=0, le=18)


class AreaSection(_Model):
    border_km: float = Field(default=10.0, ge=0)
    zones: tuple[Zone, ...] = (Zone(within_km=10.0, imagery_zoom=14, dem_zoom=12),)

    @model_validator(mode="after")
    def _check_zones(self) -> Self:
        if not self.zones:
            raise ValueError("at least one zone is required")
        distances = [z.within_km for z in self.zones]
        if distances != sorted(distances) or len(set(distances)) != len(distances):
            raise ValueError("zones must be sorted by strictly increasing within_km")
        if distances[-1] < self.border_km:
            raise ValueError(
                f"outermost zone reaches {distances[-1]} km but border_km is {self.border_km} km"
            )
        return self

    def zone_for_distance(self, km: float) -> Zone | None:
        for zone in self.zones:
            if km <= zone.within_km:
                return zone
        return None

    @property
    def max_imagery_zoom(self) -> int:
        return max(z.imagery_zoom for z in self.zones)

    @property
    def max_dem_zoom(self) -> int:
        return max(z.dem_zoom for z in self.zones)


class SourcesSection(_Model):
    imagery: tuple[str, ...] = ("esri_world_imagery",)
    dem: tuple[str, ...] = ("copernicus_glo30",)
    topo: tuple[str, ...] = ("opentopomap",)


class ProviderDef(_Model):
    """A data source defined in the project (``[[providers]]``), see docs/adding-a-region.md.

    kind ``imagery``/``topo`` with type ``xyz`` (XYZ or WMTS tile URL with {z} {x} {y}), or kind
    ``dem`` with type ``wms`` (float32 BIL GetMap in EPSG:3857) or ``stac`` (GeoTIFF sheets
    listed by a STAC collection)."""

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    kind: Literal["imagery", "topo", "dem"]
    type: Literal["xyz", "wms", "stac"] = "xyz"
    name: str = ""
    url: str
    # regional coverage: a shipped name ("france", "switzerland", "austria") or a GeoJSON path
    coverage: str | None = None
    feather_m: float = Field(default=300.0, gt=0)
    attribution: str = ""
    license: str = "unknown"
    license_url: str = ""
    commercial_use: bool = False
    requests_per_second: float = Field(default=5.0, ge=0)
    concurrency: int = Field(default=4, ge=1, le=32)
    headers: dict[str, str] = {}
    # xyz
    ext: str = "jpg"
    min_zoom: int = Field(default=0, ge=0, le=22)
    max_zoom: int = Field(default=19, ge=0, le=22)
    tile_size: int = 256
    placeholder_md5: tuple[str, ...] = ()
    # dem
    layer: str = ""  # wms
    oversample: int = Field(default=2, ge=1, le=4)  # wms
    nodata_below: float = -1000.0  # wms
    asset_suffix: str = ""  # stac: which asset of an item, e.g. "_2_2056_5728.tif"
    native_resolution_m: float = Field(default=5.0, gt=0)

    @model_validator(mode="after")
    def _check_type(self) -> Self:
        if self.kind == "dem" and self.type == "xyz":
            raise ValueError("DEM providers need type = 'wms' or 'stac'")
        if self.kind != "dem" and self.type != "xyz":
            raise ValueError("imagery/topo providers must be type = 'xyz'")
        if self.type == "xyz" and not all(f"{{{k}}}" in self.url for k in "zxy"):
            raise ValueError("xyz url needs {z}, {x} and {y} placeholders")
        if self.type == "wms" and not self.layer:
            raise ValueError("wms providers need a layer")
        if self.type == "stac" and not self.asset_suffix:
            raise ValueError("stac providers need an asset_suffix")
        return self


class Config(_Model):
    project: ProjectSection = ProjectSection()
    area: AreaSection = AreaSection()
    sources: SourcesSection = SourcesSection()
    providers: tuple[ProviderDef, ...] = ()

    @classmethod
    def from_toml(cls, text: str) -> Config:
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"invalid TOML: {exc}") from exc
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            lines = []
            for err in exc.errors():
                loc = ".".join(str(p) for p in err["loc"]) or "<root>"
                lines.append(f"{loc}: {err['msg']}")
            raise ConfigError("invalid configuration:\n" + "\n".join(lines)) from exc


class Project:
    """A project folder containing ``earthling.toml`` and the GPX files."""

    def __init__(self, folder: Path, config: Config) -> None:
        self.folder = folder
        self.config = config

    @property
    def name(self) -> str:
        return self.folder.name

    @property
    def config_path(self) -> Path:
        return self.folder / CONFIG_FILENAME

    @property
    def gpx_dir(self) -> Path:
        return self._resolve(self.config.project.gpx_dir)

    @property
    def cache_dir(self) -> Path:
        return self._resolve(self.config.project.cache_dir)

    def _resolve(self, path: Path) -> Path:
        path = path.expanduser()
        return path if path.is_absolute() else (self.folder / path).resolve()

    @classmethod
    def load(cls, folder: str | Path) -> Project:
        folder = Path(folder).resolve()
        path = folder / CONFIG_FILENAME
        if not path.exists():
            raise ConfigError(f"{folder} contains no {CONFIG_FILENAME}")
        return cls(folder, Config.from_toml(path.read_text(encoding="utf-8")))

    @classmethod
    def create(cls, folder: str | Path) -> Project:
        folder = Path(folder).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / CONFIG_FILENAME
        if not path.exists():
            path.write_text(SAMPLE_CONFIG, encoding="utf-8")
        (folder / "gpx").mkdir(exist_ok=True)
        return cls.load(folder)
