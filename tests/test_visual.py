"""Visual regression: small offline renders compared with reference images.

A synthetic project (two walked days over analytic hills, a terrain stub instead of tiles, label
features from a cached Overpass file) is rendered with the real export path
(``scene_renderer`` + ``FrameRenderer``, one sample) at 320x180, one reference per feature:
tracks, an elevation ramp, labels, POIs, clouds, rain with a lightning flash at night, snow and
fog, the stats overlay, the mini-map, the glow at dusk, a sunset. The references are PNGs in
``tests/fixtures/visual/``; all of them run in the normal suite (about 5 s together).

A failing test means the picture changed. If the change is intended, regenerate the references
deliberately (``-k name`` for single ones) and look at them before committing:

    .venv/Scripts/python.exe -m pytest tests/test_visual.py --update-visual

On a mismatch the actual image and a difference image (amplified so that the largest difference
is white) are written to the test's temporary folder; the failure message names them and the
measured difference. The limits (see "comparison" below) are above the run-to-run noise, which
is zero on the reference machine, and leave room for small driver differences, but catch a
changed colour, shader constant or overlay. A very different GPU may need its own references.
Skipped without OpenGL 4.3.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import moderngl
import numpy as np
import pytest
from PIL import Image

from earthling.core.config import Project
from earthling.core.geo import lonlat_to_tile, tile_bounds_lonlat, tile_to_lonlat
from earthling.core.pois import Poi
from earthling.data.dem import HEIGHTMAP_SAMPLES
from earthling.data.labels import cache_path
from earthling.data.terrain_data import HeightmapRef
from earthling.export.frames import ExportQuality, FrameRenderer, scene_renderer
from earthling.render import lod
from earthling.render.camera_rig import look_orientation
from earthling.render.lightning import active_strikes, strikes_between
from earthling.session import Session
from helpers import gpx_text, make_project, walk

REFERENCES = Path(__file__).parent / "fixtures" / "visual"
SIZE = (320, 180)

# one z10 tile holds the whole synthetic world (lon 6.68..7.03, lat 45.83..46.07)
Z = 10
TILE = tuple(int(v) for v in lonlat_to_tile(6.85, 45.95, Z))
WEST, SOUTH, EAST, NORTH = tile_bounds_lonlat(Z, *TILE)
LON0, LAT0 = 6.84, 45.93  # the hills' centre
M_PER_DEG_LON, M_PER_DEG_LAT = 77_300.0, 111_200.0


def relief(lon, lat):
    """Height (m): rolling hills, a summit north-east of the start and a valley."""
    x = (np.asarray(lon, dtype=np.float64) - LON0) * M_PER_DEG_LON
    y = (np.asarray(lat, dtype=np.float64) - LAT0) * M_PER_DEG_LAT
    hills = 260.0 * np.sin(x / 1900.0 + 0.4) * np.cos(y / 2600.0)
    summit = 1500.0 * np.exp(-((x - 4500.0) ** 2 + (y - 2500.0) ** 2) / (2.0 * 2300.0**2))
    valley = -350.0 * np.exp(-((x + 2500.0) ** 2) / (2.0 * 1500.0**2))
    return 1300.0 + hills + summit + valley + 0.02 * y


class Hills:
    """Terrain stub: the analytic relief as heightmaps (any tile), imagery coloured by height
    and slope with a field pattern; no tiles, no downloads."""

    on_demand = False  # (labels: no download either)

    def heightmap_for(self, key):
        z, x, y = key
        s = (np.arange(HEIGHTMAP_SAMPLES, dtype=np.float64) - 1.0) / (HEIGHTMAP_SAMPLES - 3)
        lon, lat = tile_to_lonlat(x + s[None, :], y + s[:, None], z)
        return HeightmapRef(key, relief(lon, lat).astype(np.float32), 256.0, (0.0, 0.0))

    def texture_for(self, key, source="imagery"):
        return self.imagery_for(key) if source == "imagery" else None

    def imagery_for(self, key):
        z, x, y = key
        s = (np.arange(256, dtype=np.float64) + 0.5) / 256.0
        lon, lat = tile_to_lonlat(x + s[None, :], y + s[:, None], z)
        h = relief(lon, lat)
        gy, gx = np.gradient(h)
        slope = np.clip(np.hypot(gx, gy) / 40.0, 0.0, 1.0)[..., None]
        t = np.clip((h - 1000.0) / 1800.0, 0.0, 1.0)[..., None]
        meadow, forest = np.array([110.0, 140.0, 60.0]), np.array([40.0, 70.0, 35.0])
        rock, snow = np.array([135.0, 125.0, 115.0]), np.array([235.0, 238.0, 242.0])
        fields = 0.5 + 0.5 * np.sign(np.sin(lon * 900.0) * np.sin(lat * 1300.0))[..., None]
        low = meadow * fields + forest * (1.0 - fields)
        rgb = low * (1.0 - t) + rock * t
        rgb = rgb * (1.0 - slope * 0.5) + rock * slope * 0.5
        rgb = np.where(h[..., None] > 2500.0, snow, rgb)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def height_at(self, lon, lat, z=None):
        return float(relief(lon, lat))

    def heights_at(self, lon, lat, z=None):
        return relief(lon, lat)


OVERPASS = {"elements": [
    {"type": "node", "id": 1, "lat": LAT0 + 2500.0 / M_PER_DEG_LAT,
     "lon": LON0 + 4500.0 / M_PER_DEG_LON, "tags": {"natural": "peak", "name": "Testhorn"}},
    {"type": "node", "id": 2, "lat": 45.955, "lon": 6.80,
     "tags": {"natural": "peak", "name": "Pointe Grise", "ele": "1650"}},
    {"type": "node", "id": 3, "lat": 45.94, "lon": 6.87,
     "tags": {"natural": "saddle", "name": "Col du Test", "ele": "2050"}},
    {"type": "node", "id": 4, "lat": 45.915, "lon": 6.835,
     "tags": {"tourism": "alpine_hut", "name": "Cabane"}},
    {"type": "node", "id": 5, "lat": 45.925, "lon": 6.78, "tags": {"place": "village",
                                                                   "name": "Testdorf"}},
]}  # fmt: skip


def on_ground(name: str, points: list[tuple]) -> str:
    """GPX of a walk whose elevations follow the relief (the stats and the ramp see hills)."""
    return gpx_text({name: [(lat, lon, round(float(relief(lon, lat)), 1), when)
                            for lat, lon, _, when in points]})  # fmt: skip


@dataclass
class World:
    ctx: moderngl.Context
    session: Session
    folder: Path

    def enu(self, lat: float, lon: float, h: float | None = None) -> np.ndarray:
        if h is None:
            h = float(relief(lon, lat))
        return np.asarray(self.session.frame.geodetic_to_enu(lat, lon, h), dtype=np.float64)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The synthetic project, loaded once per module (one GL context for all renders)."""
    import earthling.core.gpx as gpx

    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:  # pragma: no cover - depends on machine
        pytest.skip(f"no OpenGL 4.3 context available: {exc}")
    base = tmp_path_factory.mktemp("visual")
    days = {"day1.gpx": on_ground("Day 1", walk(45.905, 6.815, "2026-07-01T06:00:00", 120,
                                                north_m=45.0, east_m=12.0, pause=(50, 70))),
            "day2.gpx": on_ground("Day 2", walk(45.952, 6.825, "2026-07-02T06:30:00", 100,
                                                north_m=-8.0, east_m=55.0))}  # fmt: skip
    project = make_project(base / "trip", days)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gpx, "PARSE_CACHE_DIR", base / "gpx_cache")
        mp.setattr(Session, "terrain_data", lambda self, on_demand=True: Hills())
        mp.setattr(Session, "terrain_nodes", lambda self: lod.NodeSet({Z: [TILE[:2]]}))
        session = Session(Project.load(project))
        labels = session.labels
        path = cache_path(labels.cache, labels.bounds)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(OVERPASS), encoding="utf-8")
        yield World(ctx, session, project)
    ctx.release()


@dataclass
class Case:
    """One reference: property values, the camera (eye/target as lat, lon, height above the
    ground) and the time; ``pois`` are placed as in a scene."""

    settings: dict = field(default_factory=dict)
    eye: tuple = (45.86, 6.79, 2600.0)
    target: tuple = (45.94, 6.84, 0.0)
    time: float = 2.0
    pois: list = field(default_factory=list)


BASE = {
    "labels.visible": False,
    "stats.visible": False,
    "minimap.visible": False,
    "camera.mode": "keys",
    "progress.head": 0.7,
    "marker.size": 10.0,
}


def render(world: World, case: Case) -> np.ndarray:
    scene = world.session.new_scene()
    scene.path = world.folder / "scenes" / "visual.json"
    scene.set_pois(case.pois)
    for pid, value in {**BASE, **case.settings}.items():
        scene.store.set(pid, value)
    eye = world.enu(*case.eye[:2], float(relief(case.eye[1], case.eye[0])) + case.eye[2])
    target = world.enu(*case.target[:2], float(relief(case.target[1], case.target[0]))
                       + case.target[2])  # fmt: skip
    heading, pitch, roll = look_orientation(eye, target, 0.0)
    scene.animation.set_key("camera.pose", 0.0, (*eye, heading, pitch, roll))
    scene.animation.duration = 10.0
    renderer = scene_renderer(world.ctx, world.session, scene)
    quality = ExportQuality(detail=0.75, shadow_resolution="2048")  # (one sample)
    frames = FrameRenderer(renderer, scene.animation, quality=quality)
    try:
        return frames.render(case.time, *SIZE)
    finally:
        frames.release()
        renderer.terrain.shutdown()


# --- comparison --------------------------------------------------------------------------------
# Measured on the reference machine: repeated renders (in one process, in new processes, each
# test alone) are bit-identical, so the noise is 0. The limits leave room for a driver that
# rounds differently (a level here and there, a few flipped edge pixels) and were checked
# against small deliberate changes: a track 1 px narrower, the casing 1 px thinner, the marker
# 2 px smaller, glow -12 %, exposure +0.05 EV, the sun 30 min later, fog -8 %, a POI 10 %
# smaller, a mini-map colour +0.02 all fail.
MAX_MEAN = 0.2  # colour levels, mean over the picture: light, colour, exposure
BLOCK = 20  # px: the mean difference of any BLOCK x BLOCK square ...
MAX_BLOCK = 0.75  # ... stays below this (levels): local changes (glow, a line, an overlay)
OUTLIER_LEVELS = 8  # a pixel off by more than this in any channel is an outlier ...
MAX_OUTLIERS = 0.0003  # ... and at most this share of the pixels may be (17 px at 320x180)


@dataclass
class Difference:
    mean: float  # mean absolute difference (levels, over all pixels and channels)
    block: float  # largest mean absolute difference of a BLOCK x BLOCK square
    outliers: float  # share of the pixels off by more than OUTLIER_LEVELS
    worst: int  # largest channel difference

    @property
    def ok(self) -> bool:
        return (self.mean <= MAX_MEAN and self.block <= MAX_BLOCK
                and self.outliers <= MAX_OUTLIERS)  # fmt: skip

    def __str__(self) -> str:
        return (f"mean {self.mean:.3f} levels (max {MAX_MEAN}), worst {BLOCK}x{BLOCK} block"
                f" {self.block:.2f} (max {MAX_BLOCK}), {100 * self.outliers:.3f} % of the pixels"
                f" off by more than {OUTLIER_LEVELS} (max {100 * MAX_OUTLIERS:.3f} %),"
                f" worst pixel {self.worst}")  # fmt: skip


def difference(a: np.ndarray, b: np.ndarray) -> Difference:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    h, w = d.shape[0] // BLOCK * BLOCK, d.shape[1] // BLOCK * BLOCK
    blocks = d[:h, :w].mean(axis=2).reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK)
    return Difference(float(d.mean()), float(blocks.mean(axis=(1, 3)).max()),
                      float((d.max(axis=2) > OUTLIER_LEVELS).mean()), int(d.max()))  # fmt: skip


def check(request, name: str, image: np.ndarray, tmp_path: Path) -> None:
    reference = REFERENCES / f"{name}.png"
    if request.config.getoption("--update-visual"):
        reference.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(reference, optimize=True)
        return
    assert reference.exists(), f"no reference {reference}: run with --update-visual"
    expected = np.asarray(Image.open(reference).convert("RGB"))
    assert expected.shape == image.shape, f"{name}: size {image.shape} != {expected.shape}"
    diff = difference(image, expected)
    if diff.ok:
        return
    actual_path = tmp_path / f"{name}_actual.png"
    diff_path = tmp_path / f"{name}_diff.png"
    Image.fromarray(image).save(actual_path)
    gain = max(8, 255 // max(diff.worst, 1))  # (the largest difference shows white)
    amplified = np.clip(np.abs(image.astype(np.int16) - expected) * gain, 0, 255)
    Image.fromarray(amplified.astype(np.uint8)).save(diff_path)
    pytest.fail(f"{name} looks different from {reference.name}: {diff}\n"
                f"  actual:     {actual_path}\n  difference: {diff_path} (x{gain})\n"
                f"  intended? regenerate with --update-visual", pytrace=False)  # fmt: skip


# --- the references ----------------------------------------------------------------------------
def storm_time(rate: float, seed: int = 1) -> float:
    """A moment just after the first strike's flash began."""
    strike = strikes_between(lambda t: rate, seed, 0.0, 60.0)[0]
    assert active_strikes(lambda t: rate, seed, strike.start + 0.02)
    return strike.start + 0.02


CASES = {
    # the line with casing, glow, a drawn head and the marker; midday light with shadows
    # (close up and wide: sizes are given for 1080p, 30 px are 5 px here)
    "tracks": Case(
        {"tracks.casing": 10.0, "tracks.width": 30.0, "marker.size": 30.0},
        eye=(45.938, 6.836, 1800.0),
        target=(45.950, 6.855, 0.0),
    ),  # fmt: skip
    # coloured by elevation (ramp), seen from closer
    "track_ramp": Case(
        {
            "tracks.color_by": "elevation",
            "tracks.color_ramp": "turbo",
            "tracks.color_min": 800.0,
            "tracks.color_max": 2900.0,
            "progress.head": 1.0,
        },
        eye=(45.90, 6.80, 1200.0),
        target=(45.935, 6.83, 0.0),
    ),  # fmt: skip
    "labels": Case(
        {
            "labels.visible": True,
            "labels.size": 40.0,
            "labels.huts": True,
            "labels.places": True,
            "labels.max_distance": 60.0,
        }
    ),
    "pois": Case(
        pois=[
            Poi(
                id="hut",
                name="Hut",
                lon=6.835,
                lat=45.915,
                icon="builtin:hut",
                size_px=70.0,
                caption="Cabane",
            ),
            Poi(
                id="peak",
                name="Peak",
                lon=LON0 + 4500.0 / M_PER_DEG_LON,
                lat=LAT0 + 2500.0 / M_PER_DEG_LAT,
                icon="builtin:peak",
                size_px=70.0,
                caption="Testhorn",
            ),
            Poi(id="camp", name="Camp", lon=6.825, lat=45.952, icon="builtin:tent", size_px=60.0),
        ]
    ),  # fmt: skip
    "clouds": Case(
        {
            "weather.enabled": True,
            "weather.layer1.enabled": True,
            "weather.layer1.coverage": 0.55,
            "weather.cloud_shadows": 0.9,
        }
    ),
    "storm": Case(
        {
            "sun.datetime": dt.datetime(2026, 7, 1, 23, 30),
            "rain.enabled": True,
            "rain.cloud_base": 3500.0,
            "lightning.rate": 30.0,
        },
        time=storm_time(30.0),
    ),  # fmt: skip
    # snow above a line, ground fog in the valleys (terrain shader, fog)
    "snow_fog": Case(
        {
            "snow.cover": 1.0,
            "snow.line": 1500.0,
            "fog.enabled": True,
            "fog.density": 0.6,
            "fog.base": 1000.0,
        }
    ),
    "stats": Case({"stats.visible": True, "stats.show_speed": True}),
    "minimap": Case({"minimap.visible": True, "minimap.size": 600.0}),
    "glow": Case(
        {
            "sun.datetime": dt.datetime(2026, 7, 1, 21, 50),
            "tracks.glow": 3.0,
            "glow.intensity": 1.6,
            "tracks.emissive": 1.0,
            "marker.glow": 4.0,
        }
    ),
    "sunset": Case(
        {"sun.datetime": dt.datetime(2026, 7, 1, 21, 5)},
        eye=(45.87, 6.93, 500.0),
        target=(45.93, 6.83, 0.0),
    ),  # fmt: skip
}


@pytest.mark.parametrize("name", list(CASES))
def test_reference(world, name, request, tmp_path):
    check(request, name, render(world, CASES[name]), tmp_path)
