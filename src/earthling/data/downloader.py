"""Concurrent, rate-limited, resumable tile downloader."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from earthling.data import net
from earthling.data.cache import TileCache, write_atomic
from earthling.data.net import RateLimiter
from earthling.data.providers import TileProvider

log = logging.getLogger(__name__)


@dataclass
class DownloadProgress:
    total: int = 0
    done: int = 0  # finished (downloaded + skipped + missing + failed)
    downloaded: int = 0
    skipped: int = 0
    missing: int = 0
    failed: int = 0
    bytes: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return self.done >= self.total


ProgressCallback = Callable[[DownloadProgress], None]


http_client = net.client  # (the old name)


class TileDownloader:
    def __init__(
        self,
        provider: TileProvider,
        cache: TileCache,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        """``sleep``: replaces the back-off wait between retries (tests)."""
        self.provider = provider
        self.cache = cache
        self.client = client or net.client()
        self._sleep = sleep
        self._limiter = RateLimiter(provider.max_requests_per_second)
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def run(
        self,
        tiles: Iterable[tuple[int, int, int]],
        on_progress: Callable[[DownloadProgress], None] | None = None,
    ) -> DownloadProgress:
        p = self.provider
        todo: list[tuple[int, int, int]] = []
        progress = DownloadProgress()
        for z, x, y in tiles:
            if z < p.min_zoom or z > p.max_zoom:
                continue
            progress.total += 1
            if self.cache.is_known(p.id, z, x, y, p.ext):
                progress.skipped += 1
                progress.done += 1
            else:
                todo.append((z, x, y))
        if on_progress:
            on_progress(progress)

        def work(tile: tuple[int, int, int]) -> None:
            if self._cancel.is_set():
                return
            outcome, size, error = self._fetch(*tile)
            with self._lock:
                progress.done += 1
                if outcome == "ok":
                    progress.downloaded += 1
                    progress.bytes += size
                elif outcome == "missing":
                    progress.missing += 1
                else:
                    progress.failed += 1
                    if len(progress.errors) < 50:
                        progress.errors.append(error)
                if on_progress:
                    on_progress(progress)

        with ThreadPoolExecutor(max_workers=max(1, p.concurrency)) as pool:
            list(pool.map(work, todo))
        return progress

    def fetch_one(self, z: int, x: int, y: int) -> bool:
        """Download a single tile unless it is already known. True if it is cached afterwards."""
        p = self.provider
        if self.cache.has(p.id, z, x, y, p.ext):
            return True
        if self.cache.is_missing(p.id, z, x, y) or z > p.max_zoom:
            return False
        return self._fetch(z, x, y)[0] == "ok"

    def _fetch(self, z: int, x: int, y: int) -> tuple[str, int, str]:
        p = self.provider
        tile = f"{z}/{x}/{y}"
        try:
            response = net.get(
                self.client,
                p.tile_url(z, x, y),
                headers=p.headers,
                cancel=self._cancel,
                limiter=self._limiter,
                sleep=self._sleep,
            )
        except net.CancelledError:
            return "failed", 0, "cancelled"
        except httpx.HTTPError as exc:
            error = f"{tile}: {type(exc).__name__}: {exc}"
            if not isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):  # (net logs)
                log.warning("giving up on tile %s", error)
            return "failed", 0, error
        status = response.status_code
        if status == 200:
            data = response.content
            if not data or p.is_placeholder(data):
                self.cache.mark_missing(p.id, z, x, y)
                return "missing", 0, ""
            self.cache.write(p.id, z, x, y, p.ext, data)
            return "ok", len(data), ""
        if status in (204, 404):
            self.cache.mark_missing(p.id, z, x, y)
            return "missing", 0, ""
        error = f"{tile}: HTTP {status}"
        if net.retryable(status):
            log.warning("giving up on tile %s", error)
        return "failed", 0, error  # (other client errors are not retried)


# --- Natural Earth (borders, countries, cities): one GeoJSON file per dataset ------------------
NATURAL_EARTH_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson"


def natural_earth_path(cache: TileCache, dataset: str) -> Path:
    """Cache file of a Natural Earth dataset (e.g. ``ne_50m_admin_0_countries``)."""
    return cache.root / "_vector" / "naturalearth" / f"{dataset}.geojson"


def ensure_natural_earth(
    cache: TileCache, dataset: str, allow_download: bool = True
) -> Path | None:
    """The cached dataset file, downloaded first when missing; None when offline or when
    downloads are not allowed (the data is optional)."""
    path = natural_earth_path(cache, dataset)
    if path.exists():
        return path
    if not allow_download:
        return None
    url = f"{NATURAL_EARTH_URL}/{dataset}.geojson"
    log.info("downloading %s", url)
    try:
        with net.client() as client:
            response = net.get(client, url, timeout=120.0)
            response.raise_for_status()
        write_atomic(path, response.content)
    except Exception as exc:  # offline etc.
        log.warning("cannot download %s: %s", url, exc)
        return None
    return path
