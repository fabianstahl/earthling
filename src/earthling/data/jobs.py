"""Download jobs for a session, shared by the GUI and the CLI."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

import shapely

from earthling.core.aoi import TilePlan
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
    if "imagery" in kinds:
        jobs.append(_imagery_job(session, plan, max_zoom))
    if "dem" in kinds:
        jobs.extend(_dem_jobs(session, plan, max_zoom))
    return jobs


def _imagery_job(session: Session, plan: TilePlan, max_zoom: int | None) -> DownloadJob:
    provider = session.imagery_provider
    tiles = [t for t in plan.tiles("imagery") if max_zoom is None or t[0] <= max_zoom]
    downloader = TileDownloader(provider, session.cache)
    return DownloadJob(
        f"Imagery ({provider.name})", lambda cb: downloader.run(tiles, cb), downloader.cancel
    )


def _dem_jobs(session: Session, plan: TilePlan, max_zoom: int | None) -> list[DownloadJob]:
    source = session.dem_source
    aoi = session.aoi.aoi
    files = [
        f for f in source.files_for_bounds(aoi.bounds) if shapely.box(*f.bounds).intersects(aoi)
    ]
    tiles = [t for t in plan.tiles("dem") if max_zoom is None or t[0] <= max_zoom]
    cancel = threading.Event()
    baker = DemBaker(source, session.cache)
    return [
        DownloadJob(
            f"DEM sources ({source.name})",
            lambda cb: download_sources(
                source, files, session.cache, on_progress=cb, cancel=cancel
            ),
            cancel.set,
        ),
        DownloadJob("DEM heightmap tiles", lambda cb: baker.bake(tiles, cb, cancel), cancel.set),
    ]
