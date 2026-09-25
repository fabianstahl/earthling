"""Download jobs for a session, shared by the GUI and the CLI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from earthling.core.session import Session
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
        provider = session.imagery_provider
        tiles = [t for t in plan.tiles("imagery") if max_zoom is None or t[0] <= max_zoom]
        downloader = TileDownloader(provider, session.cache)
        jobs.append(
            DownloadJob(
                f"Imagery ({provider.name})",
                lambda cb, d=downloader, t=tiles: d.run(t, cb),
                downloader.cancel,
            )
        )
    return jobs
