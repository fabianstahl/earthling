"""Download jobs for a session, shared by the GUI and the CLI."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

import shapely

from earthling.core.session import Session
from earthling.data.dem import DemBaker, download_sources
from earthling.data.downloader import DownloadProgress, TileDownloader

ProgressCallback = Callable[[DownloadProgress], None]


@dataclass
class DownloadJob:
    """A named unit of work that reports DownloadProgress and can be cancelled."""

    name: str
    run: Callable[[ProgressCallback | None], DownloadProgress]
    cancel: Callable[[], None]


def download_jobs(
    session: Session, kinds: set[str], max_zoom: int | None = None
) -> list[DownloadJob]:
    jobs: list[DownloadJob] = []
    plan = session.plan
    if plan is None:
        return jobs
    for kind in ("imagery", "topo"):
        if kind not in kinds:
            continue
        for provider, tiles in session.provider_tiles(kind):
            if tiles:
                jobs.append(_tile_job(provider, session, _flatten(tiles, max_zoom)))
            if kind == "topo":
                break  # further topo providers are fallbacks only (downloaded on demand)
    if "dem" in kinds:
        for source, tiles in session.provider_tiles("dem"):
            if tiles:
                jobs.extend(_dem_jobs(session, source, _flatten(tiles, max_zoom)))
    return jobs


def _flatten(tiles: dict, max_zoom: int | None) -> list[tuple[int, int, int]]:
    return [
        (z, int(x), int(y))
        for z in sorted(tiles)
        if max_zoom is None or z <= max_zoom
        for x, y in tiles[z]
    ]


def _tile_job(provider, session: Session, tiles: list[tuple[int, int, int]]) -> DownloadJob:
    """Download ``tiles`` of ``provider`` (the downloader skips zooms it does not serve)."""
    downloader = TileDownloader(provider, session.cache)
    return DownloadJob(
        f"{provider.kind.capitalize()} ({provider.name})",
        lambda cb: downloader.run(tiles, cb),
        downloader.cancel,
    )


def _dem_jobs(session: Session, source, tiles: list[tuple[int, int, int]]) -> list[DownloadJob]:
    area = session.aoi.aoi
    coverage = source.coverage_area
    if coverage is not None:
        area = area.intersection(coverage.lonlat)
    cancel = threading.Event()
    baker = DemBaker(source, session.cache)
    jobs = []
    if not source.direct:  # direct sources (WMS) are fetched per heightmap tile

        def download(cb):
            # listing may need the network (STAC catalogues): done when the job runs
            files = [
                f
                for f in source.files_for_bounds(area.bounds)
                if shapely.box(*f.bounds).intersects(area)
            ]
            return download_sources(source, files, session.cache, on_progress=cb, cancel=cancel)

        jobs.append(DownloadJob(f"DEM sources ({source.name})", download, cancel.set))
    jobs.append(
        DownloadJob(
            f"DEM heightmap tiles ({source.name})",
            lambda cb: baker.bake(tiles, cb, cancel),
            cancel.set,
        )
    )
    return jobs
