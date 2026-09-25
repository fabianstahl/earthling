"""Test harness for data sources: ``earthling check-source <project> <id> [--at lon,lat]``.

Imagery: fetches the tile at the location on several zoom levels and checks the HTTP answer,
the image format, placeholders and the coverage. DEM: bakes heightmap tiles at the location and
checks the data, the alignment against Copernicus GLO-30 (slope correlation, in samples) and the
consistency between neighbouring LOD levels (a tile vs. its four children).
"""

from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from earthling.core.geo import lonlat_to_tile, tile_bounds_lonlat
from earthling.data.cache import TileCache
from earthling.data.dem import (
    DemBaker,
    DemSource,
    download_sources,
    sample_spacing_m,
)
from earthling.data.downloader import http_client
from earthling.data.providers import TileProvider


@dataclass
class CheckReport:
    lines: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def info(self, text: str) -> None:
        self.lines.append(text)

    def problem(self, text: str) -> None:
        self.problems.append(text)
        self.lines.append(f"PROBLEM: {text}")

    def warn(self, text: str) -> None:
        self.warnings.append(text)
        self.lines.append(f"warning: {text}")

    def text(self) -> str:
        verdict = "OK" if self.ok else f"{len(self.problems)} problem(s)"
        extra = f", {len(self.warnings)} warning(s)" if self.warnings else ""
        return "\n".join([*self.lines, f"result: {verdict}{extra}"])


def _coverage_line(report: CheckReport, source, lon: float, lat: float) -> None:
    coverage = source.coverage_area
    if coverage is None:
        report.info(f"coverage: worldwide; location {lon:.5f}, {lat:.5f}")
    elif coverage.contains_lonlat(lon, lat):
        report.info(f"coverage: {source.coverage}; location {lon:.5f}, {lat:.5f} is inside")
    else:
        report.problem(f"location {lon:.5f}, {lat:.5f} is outside the coverage {source.coverage}")


# --- imagery -------------------------------------------------------------------------------
def check_imagery(
    provider: TileProvider,
    lon: float,
    lat: float,
    client: httpx.Client | None = None,
    zooms: list[int] | None = None,
    out_dir: Path | None = None,
) -> CheckReport:
    report = CheckReport()
    report.info(
        f"imagery provider {provider.id} ({provider.name}), zoom {provider.min_zoom}"
        f"-{provider.max_zoom}, {provider.max_requests_per_second:g} requests/s"
    )
    report.info(f"license: {provider.license.name}; attribution: {provider.license.attribution}")
    _coverage_line(report, provider, lon, lat)
    client = client or http_client()
    if zooms is None:
        zooms = sorted(
            {
                min(max(z, provider.min_zoom), provider.max_zoom)
                for z in (10, 13, 16, provider.max_zoom)
            }
        )
    images = []
    for z in zooms:
        tx, ty = lonlat_to_tile(lon, lat, z)
        x, y = int(tx), int(ty)
        started = time.monotonic()
        try:
            response = client.get(provider.tile_url(z, x, y), headers=provider.headers)
        except httpx.HTTPError as exc:
            report.problem(f"z{z} {x}/{y}: {type(exc).__name__}: {exc}")
            continue
        ms = (time.monotonic() - started) * 1000
        kind = response.headers.get("content-type", "?")
        size = len(response.content)
        head = f"z{z:<2} {x}/{y}: HTTP {response.status_code} {kind} {size} B {ms:.0f} ms"
        if response.status_code != 200:
            report.problem(f"{head} (no tile at the location)")
            continue
        data = response.content
        if provider.is_placeholder(data):
            report.problem(f"{head}: known placeholder (no data)")
            continue
        try:
            with Image.open(io.BytesIO(data)) as img:
                rgb = np.asarray(img.convert("RGB"))
        except Exception as exc:
            report.problem(f"{head}: not an image ({exc}): {data[:80]!r}")
            continue
        size_ok = rgb.shape[:2] == (provider.tile_size, provider.tile_size)
        report.info(
            f"{head}, {rgb.shape[1]}x{rgb.shape[0]} px, mean colour "
            f"{tuple(int(c) for c in rgb.reshape(-1, 3).mean(axis=0))}"
        )
        if not size_ok:
            report.problem(f"z{z}: tile size {rgb.shape[1]} px, expected {provider.tile_size}")
        if rgb.std() < 1.0:
            md5 = hashlib.md5(data).hexdigest()
            report.warn(
                f"z{z}: uniform tile – a 'no data' placeholder? Add "
                f'placeholder_md5 = ["{md5}"] if so'
            )
        images.append((z, rgb))
    if out_dir is not None and images:
        out_dir.mkdir(parents=True, exist_ok=True)
        sheet = Image.new("RGB", (256 * len(images), 256))
        for i, (_, rgb) in enumerate(images):
            sheet.paste(Image.fromarray(rgb).resize((256, 256)), (256 * i, 0))
        path = out_dir / f"check_{provider.id}.png"
        sheet.save(path)
        report.info(f"tiles saved to {path}")
    return report


# --- elevation -----------------------------------------------------------------------------
def _slope(a: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(np.nan_to_num(a.astype(np.float64), nan=float(np.nanmean(a))))
    return np.hypot(gx, gy)


def _shifted(a: np.ndarray, s: float, axis: int) -> np.ndarray:
    k = int(np.floor(s))
    f = s - k
    return (1 - f) * np.roll(a, -k, axis) + f * np.roll(a, -(k + 1), axis)


def alignment(test: np.ndarray, reference: np.ndarray) -> tuple[float, float, float]:
    """(south, east, correlation): shift in samples of ``test`` relative to ``reference``."""
    a, b = _slope(test), _slope(reference)
    m = slice(10, -10)
    best = []
    for axis in (0, 1):
        best.append(
            max(
                (np.corrcoef(a[m, m].ravel(), _shifted(b, s, axis)[m, m].ravel())[0, 1], s)
                for s in np.arange(-2.5, 2.51, 0.0625)
            )
        )
    return best[0][1], best[1][1], min(best[0][0], best[1][0])


def lod_consistency(baker: DemBaker, z: int, x: int, y: int) -> tuple[float, float] | None:
    """(mean, rms) difference between tile z/x/y and its children at the shared samples."""
    parent = baker.bake_tile(z, x, y)
    if parent is None:
        return None
    mosaic = np.full((513, 513), np.nan)
    for j in range(2):
        for i in range(2):
            child = baker.bake_tile(z + 1, 2 * x + i, 2 * y + j)
            if child is not None:
                mosaic[j * 256 : j * 256 + 257, i * 256 : i * 256 + 257] = child[1:-1, 1:-1]
    d = parent[1:-1, 1:-1] - mosaic[::2, ::2]
    if not np.isfinite(d).any():
        return None
    return float(np.nanmean(d)), float(np.sqrt(np.nanmean(d**2)))


def check_dem(
    source: DemSource,
    lon: float,
    lat: float,
    cache: TileCache,
    client: httpx.Client | None = None,
    zooms: tuple[int, ...] | None = None,
    reference: DemSource | None = None,
) -> CheckReport:
    """``reference``: the DEM to compare with (default Copernicus GLO-30, downloaded if needed);
    pass ``False`` to skip the comparison. File sources are only downloaded for the z13 tile at
    the location, so they are checked at z13 and finer by default."""
    from earthling.data.dem import DEM_SOURCES

    report = CheckReport()
    kind = "direct (per tile)" if source.direct else "raster files"
    report.info(
        f"DEM source {source.id} ({source.name}), {kind}, native ~{source.native_resolution_m:g} m"
    )
    report.info(f"license: {source.license.name}; attribution: {source.license.attribution}")
    _coverage_line(report, source, lon, lat)
    client = client or http_client()

    def prepare(src: DemSource, z: int) -> DemBaker:
        if not src.direct:  # download the files around the location
            tx, ty = lonlat_to_tile(lon, lat, z)
            bounds = tile_bounds_lonlat(z, int(tx), int(ty))
            files = src.files_for_bounds(bounds, client=client)
            progress = download_sources(src, files, cache, client=client)
            report.info(
                f"{src.id}: {len(files)} source file(s) for the z{z} tile, "
                f"{progress.downloaded} downloaded, {progress.failed} failed"
            )
        return DemBaker(src, cache, client=client)

    if zooms is None:
        zooms = (12, 13, 14) if source.direct else (13, 14)
    baker = prepare(source, 13 if not source.direct else min(zooms))
    if reference is None:
        reference = DEM_SOURCES["copernicus_glo30"]
    ref_baker = prepare(reference, min(zooms)) if reference else None
    for z in zooms:
        tx, ty = lonlat_to_tile(lon, lat, z)
        key = (z, int(tx), int(ty))
        started = time.monotonic()
        try:
            heights = baker.bake_tile(*key)
        except Exception as exc:
            report.problem(f"z{z}: {type(exc).__name__}: {exc}")
            continue
        ms = (time.monotonic() - started) * 1000
        if heights is None:
            report.problem(f"z{z}: no data at the location")
            continue
        finite = np.isfinite(heights)
        spacing = sample_spacing_m(z, lat)
        line = (
            f"z{z:<2} ({spacing:5.1f} m samples): {finite.mean() * 100:5.1f} % data, "
            f"{np.nanmin(heights):7.1f} .. {np.nanmax(heights):7.1f} m, {ms:.0f} ms"
        )
        if ref_baker is not None:
            ref = ref_baker.bake_tile(*key)
            if ref is not None and finite.mean() > 0.5:
                south, east, r = alignment(heights, ref)
                bias = float(np.nanmean(heights - ref))
                line += (
                    f"; vs {reference.id}: shift south {south:+.2f} east {east:+.2f} samples "
                    f"(r={r:.2f}), mean difference {bias:+.1f} m"
                )
                if max(abs(south), abs(east)) >= 1.0 and r > 0.5:
                    worst = max(abs(south), abs(east))
                    report.warn(
                        f"z{z}: misaligned by ~{worst:.1f} samples ({worst * spacing:.0f} m)"
                    )
        report.info(line)
    z = max(zooms)
    tx, ty = lonlat_to_tile(lon, lat, z)
    consistency = lod_consistency(baker, z, int(tx), int(ty))
    if consistency is not None:
        mean, rms = consistency
        report.info(f"LOD z{z} vs z{z + 1}: mean {mean:+.2f} m, rms {rms:.2f} m")
        if abs(mean) > 1.0:
            report.warn(f"LOD levels are biased by {mean:+.1f} m (visible popping)")
    return report
