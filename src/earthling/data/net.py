"""HTTP for every download: the shared client, retries with back-off and the offline pause.

``get`` / ``post`` retry timeouts, dropped connections, 5xx and 429 (honouring Retry-After)
with exponential back-off; other answers (200, 204, 4xx) are returned to the caller at once.
A host that cannot be reached at all (connection refused, no DNS, connect timeout) is not
retried: requests to it fail at once with ``HostOffline`` for ``OFFLINE_PAUSE_S`` seconds -
shared by every downloader in the process - and are then tried again. Retrying every tile
for half a minute would otherwise stall an offline export for many minutes.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import httpx

from earthling.data.providers import USER_AGENT

log = logging.getLogger(__name__)

MAX_RETRIES = 4
MAX_DELAY_S = 30.0  # longest back-off between two attempts
MAX_RETRY_AFTER_S = 60.0  # a server's Retry-After is honoured up to this
OFFLINE_PAUSE_S = 30.0  # after a connection error, requests to that host fail at once


class HostOffline(httpx.ConnectError):
    """The host could not be reached a moment ago; not tried again before the pause ends."""


class CancelledError(Exception):
    """The request was cancelled (``cancel`` event set) before or between attempts."""


class OfflineHosts:
    """Thread-safe registry of hosts that could not be reached. Requests to a host are refused
    until ``pause`` seconds after its last connection error; it stays listed (``hosts``) until
    it answers again."""

    def __init__(self, pause: float = OFFLINE_PAUSE_S, clock=time.monotonic) -> None:
        self.pause = pause
        self.clock = clock
        self._until: dict[str, float] = {}
        self._lock = threading.Lock()

    def paused(self, host: str) -> bool:
        with self._lock:
            until = self._until.get(host)
        return until is not None and self.clock() < until

    def failed(self, host: str) -> bool:
        """Start (or extend) the pause of ``host``; True if it was online before."""
        with self._lock:
            new = host not in self._until
            self._until[host] = self.clock() + self.pause
        return new

    def reached(self, host: str) -> None:
        with self._lock:
            self._until.pop(host, None)

    def hosts(self) -> list[str]:
        with self._lock:
            return sorted(self._until)


OFFLINE = OfflineHosts()


def offline_hosts() -> list[str]:
    """Hosts not reached since their last connection error (e.g. for an "offline" status)."""
    return OFFLINE.hosts()


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


def client(transport: httpx.BaseTransport | None = None, **kwargs) -> httpx.Client:
    """The HTTP client of the data layer (user agent, timeouts, redirects, pool size)."""
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(30.0, connect=15.0),
        follow_redirects=True,
        transport=transport,
        limits=httpx.Limits(max_connections=16, max_keepalive_connections=16),
        **kwargs,
    )


def backoff_wait(seconds: float, cancel: threading.Event | None = None) -> None:
    """The wait between two attempts; a cancel ends it early."""
    if cancel is None:
        time.sleep(seconds)
    else:
        cancel.wait(seconds)


def retryable(status: int) -> bool:
    return status == 429 or status >= 500


def _retry_after(response: httpx.Response) -> float:
    try:
        return min(float(response.headers.get("Retry-After", "")), MAX_RETRY_AFTER_S)
    except ValueError:  # missing, or an HTTP date
        return 0.0


def request(
    http: httpx.Client,
    method: str,
    url: str,
    *,
    stream: bool = False,
    cancel: threading.Event | None = None,
    limiter: RateLimiter | None = None,
    sleep: Callable[[float], None] | None = None,
    retries: int = MAX_RETRIES,
    **kwargs,
) -> httpx.Response:
    """Send a request with retries (see the module docstring) and return the response, also
    a 4xx or - after the last attempt - a 5xx one. Raises ``httpx.TransportError`` when the
    last attempt failed, ``HostOffline`` during a host's pause and ``CancelledError``.
    ``kwargs`` go to ``httpx.Client.build_request`` (params, headers, data, timeout ...);
    a ``stream`` response must be closed by the caller. ``limiter`` is waited for before
    every attempt; ``sleep`` replaces the back-off wait (tests)."""
    req = http.build_request(method, url, **kwargs)
    host = req.url.host
    delay = 1.0
    for attempt in range(retries + 1):
        if cancel is not None and cancel.is_set():
            raise CancelledError(str(req.url))
        if OFFLINE.paused(host):
            raise HostOffline(f"{host} is offline (tried again in a while)", request=req)
        if limiter is not None:
            limiter.wait()
        try:
            response = http.send(req, stream=stream)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            if OFFLINE.failed(host):
                log.warning("cannot reach %s, pausing its downloads: %s", host, exc)
            raise
        except httpx.TransportError:  # timeouts, dropped connections
            if attempt == retries:
                raise
            wait = delay
        else:
            OFFLINE.reached(host)
            if attempt == retries or not retryable(response.status_code):
                return response
            wait = max(delay, _retry_after(response))
            response.close()
        if sleep is not None:
            sleep(wait)
        else:
            backoff_wait(wait, cancel)
        delay = min(delay * 2, MAX_DELAY_S)
    raise AssertionError("unreachable")


def get(http: httpx.Client, url: str, **kwargs) -> httpx.Response:
    return request(http, "GET", url, **kwargs)


def post(http: httpx.Client, url: str, **kwargs) -> httpx.Response:
    return request(http, "POST", url, **kwargs)
