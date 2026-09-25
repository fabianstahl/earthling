"""Project configuration (``earthling.toml``)."""

from __future__ import annotations

import tomllib
import zoneinfo
from pathlib import Path
from typing import Self

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
imagery = ["esri_world_imagery"]   # priority order
dem     = ["copernicus_glo30"]
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


class Config(_Model):
    project: ProjectSection = ProjectSection()
    area: AreaSection = AreaSection()
    sources: SourcesSection = SourcesSection()

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
