"""Shared test helpers: the demo project, a local frame, PNG bytes, terrain fakes, mock HTTP,
small synthetic projects and a meter for how long the user interface freezes."""

import io
import time
from pathlib import Path

import numpy as np
from PIL import Image

from earthling.core.geo import LocalFrame
from earthling.data.dem import HEIGHTMAP_SAMPLES
from earthling.data.terrain_data import HeightmapRef

DEMO = Path(__file__).parents[1] / "examples" / "alps_demo"  # two days around Trient
FRAME = LocalFrame(46.0, 7.0, 0.0)


def png(color, size: int = 256) -> bytes:
    """PNG bytes of a plain ``color`` tile, or of an RGB array."""
    if isinstance(color, np.ndarray):
        image = Image.fromarray(color.astype(np.uint8))
    else:
        image = Image.new("RGB", (size, size), color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class FakeTerrainData:
    """Constant plateau with a bump; imagery only at z >= 11."""

    def __init__(self):
        self.heights = np.full((HEIGHTMAP_SAMPLES, HEIGHTMAP_SAMPLES), 1500.0, dtype=np.float32)
        self.heights[100:150, 100:150] = 2500.0

    def heightmap_for(self, key):
        return HeightmapRef(key, self.heights, 256.0, (0.0, 0.0))

    def texture_for(self, key, source="imagery"):
        return self.imagery_for(key) if source == "imagery" else None

    def imagery_for(self, key):
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[..., 1] = 200
        return rgb


class SlopeTerrainData(FakeTerrainData):
    """Height rises towards the east -> the slope faces west."""

    def __init__(self):
        super().__init__()
        cols = np.arange(HEIGHTMAP_SAMPLES, dtype=np.float32)
        self.heights = np.tile(1000.0 + cols * 40.0, (HEIGHTMAP_SAMPLES, 1))

    def imagery_for(self, key):
        return np.full((64, 64, 3), 128, dtype=np.uint8)


def longest_freeze(qtbot, action=None, until=None, timeout: float = 10.0) -> float:
    """Seconds the Qt event loop could not run (the longest gap between ticks of a 5 ms timer)
    while ``action()`` runs and, afterwards, until ``until()`` is true: what the user sees as
    a frozen window."""
    from PyQt6.QtCore import Qt, QTimer

    ticks = [time.perf_counter()]
    gaps = [0.0]

    def tick() -> None:
        now = time.perf_counter()
        gaps.append(now - ticks[-1])
        ticks.append(now)

    timer = QTimer()
    timer.setTimerType(Qt.TimerType.PreciseTimer)
    timer.timeout.connect(tick)
    timer.start(5)
    qtbot.wait(20)
    try:
        if action is not None:
            action()
        if until is not None:
            qtbot.waitUntil(until, timeout=int(timeout * 1000))
        qtbot.wait(30)
    finally:
        timer.stop()
    return max(gaps)


def offline_demo(folder: Path) -> Path:
    """A copy of the demo project in ``folder`` with its own (empty) tile cache: tests that open
    it neither read the user's cache nor download (the network is refused anyway)."""
    import shutil

    target = folder / "alps_demo"
    shutil.copytree(DEMO, target)
    config = target / "earthling.toml"
    cache = (folder / "cache").as_posix()
    text = config.read_text(encoding="utf-8").replace('"~/earthling-cache"', f'"{cache}"')
    config.write_text(text, encoding="utf-8", newline="\n")
    return target


class FlatTerrain:
    """Terrain stub for scene scripts: everything at ``height`` m (no downloads in tests)."""

    def __init__(self, height: float = 1000.0):
        self.height = height

    def height_at(self, lon, lat, z=None):
        return self.height

    def heights_at(self, lon, lat, z=None):
        return np.full(np.shape(lon), self.height)


def gpx_text(tracks: dict[str, list[tuple]]) -> str:
    """GPX 1.1 with one <trk> per entry: name -> [(lat, lon, ele, iso time or None), ...]."""
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">',
    ]
    for name, points in tracks.items():
        out.append(f"<trk><name>{name}</name><trkseg>")
        for lat, lon, ele, when in points:
            stamp = f"<time>{when}</time>" if when else ""
            out.append(f'<trkpt lat="{lat:.6f}" lon="{lon:.6f}"><ele>{ele}</ele>{stamp}</trkpt>')
        out.append("</trkseg></trk>")
    out.append("</gpx>\n")
    return "\n".join(out)


def walk(lat0: float, lon0: float, start: str | None, minutes: int, north_m: float = 0.0,
         east_m: float = 0.0, pause: tuple[int, int] | None = None) -> list[tuple]:  # fmt: skip
    """A straight walk sampled every minute from ``start`` (ISO UTC, None: no timestamps),
    ``north_m``/``east_m`` per minute; ``pause`` = (from, to) minutes standing still."""
    import datetime as dt

    t0 = dt.datetime.fromisoformat(start) if start else None
    points, moved = [], 0
    for i in range(minutes + 1):
        if i and not (pause and pause[0] < i <= pause[1]):
            moved += 1
        lat = lat0 + moved * north_m / 111_320.0
        lon = lon0 + moved * east_m / (111_320.0 * np.cos(np.radians(lat0)))
        time = (t0 + dt.timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%SZ") if t0 else None
        points.append((lat, lon, 1000.0 + moved, time))
    return points


def make_project(folder: Path, gpx: dict[str, str], *, timezone: str = "Europe/Zurich",
                 extra: str = "") -> Path:  # fmt: skip
    """A tiny project (coarse zooms, the tile cache inside ``folder``): GPX files by name."""
    (folder / "gpx").mkdir(parents=True)
    for name, text in gpx.items():  # (names may contain a sub folder: "walked/day1.gpx")
        (folder / "gpx" / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / "gpx" / name).write_bytes(text.encode("utf-8"))
    toml = f"""[project]
gpx_dir   = "gpx/"
cache_dir = "cache"
timezone  = "{timezone}"

[area]
border_km = 1.0
zones = [{{ within_km = 1.0, imagery_zoom = 12, dem_zoom = 10 }}]
{extra}"""
    (folder / "earthling.toml").write_bytes(toml.encode("utf-8"))
    return folder


def mock_http(monkeypatch, handler) -> list:
    """Route the data layer's default HTTP clients to ``handler`` (request -> httpx.Response).
    Returns the list the requests are recorded in."""
    import httpx

    from earthling.data import check, dem, downloader, labels

    real = downloader.http_client
    requests = []

    def record(request):
        requests.append(request)
        return handler(request)

    def client(transport=None, **kwargs):
        return real(httpx.MockTransport(record), **kwargs)

    for module in (downloader, dem, check, labels):
        monkeypatch.setattr(module, "http_client", client)
    return requests
