"""Concurrent, rate-limited, resumable tile downloader."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import httpx

from earthling.data.cache import TileCache
from earthling.data.providers import USER_AGENT, TileProvider

log = logging.getLogger(__name__)

MAX_RETRIES = 4


class RateLimiter:
    """Thread-safe limiter spacing requests at least ``1 / rate`` seconds apart."""

    def __init__(self, rate: float) -> None:
        self.interval = 1.0 / rate if rate > 0 else 0.0
        self._lock = threading.Lock()
        self._next = time.monotonic()

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        delay = slot - now
        if delay > 0:
            time.sleep(delay)


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


def http_client(transport: httpx.BaseTransport | None = None, **kwargs) -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(30.0, connect=15.0),
        follow_redirects=True,
        transport=transport,
        limits=httpx.Limits(max_connections=16, max_keepalive_connections=16),
        **kwargs,
    )


class TileDownloader:
    def __init__(
        self,
        provider: TileProvider,
        cache: TileCache,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provider = provider
        self.cache = cache
        self.client = client or http_client()
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
        url = p.tile_url(z, x, y)
        delay = 1.0
        last_error = ""
        for _attempt in range(MAX_RETRIES + 1):
            if self._cancel.is_set():
                return "failed", 0, "cancelled"
            self._limiter.wait()
            try:
                response = self.client.get(url, headers=p.headers)
            except httpx.HTTPError as exc:
                last_error = f"{z}/{x}/{y}: {type(exc).__name__}: {exc}"
            else:
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
                last_error = f"{z}/{x}/{y}: HTTP {status}"
                if status == 429:
                    retry_after = response.headers.get("Retry-After", "")
                    delay = max(delay, float(retry_after)) if retry_after.isdigit() else delay * 2
                elif status < 500:
                    return "failed", 0, last_error  # other client errors are not retried
            self._sleep(delay)
            delay = min(delay * 2, 30.0)
        log.warning("giving up on tile %s", last_error)
        return "failed", 0, last_error
