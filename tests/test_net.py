"""The shared HTTP helper: retries with back-off, Retry-After, cancel and the offline pause."""

import threading
import time

import httpx
import numpy as np
import pytest

from earthling.data import net
from earthling.data.cache import TileCache
from earthling.data.dem import HEIGHTMAP_SAMPLES, DemBaker, WmsDemSource
from earthling.data.downloader import TileDownloader
from earthling.data.providers import TileProvider

BACKOFF_WAIT = net.backoff_wait  # (the conftest replaces it with a no-op per test)


def server(answers, calls):
    """A mock transport answering with ``answers`` in turn (statuses, responses or exceptions
    to raise), 200 when they are used up."""

    def handler(request):
        calls.append(str(request.url))
        answer = answers.pop(0) if answers else 200
        if isinstance(answer, type) and issubclass(answer, Exception):
            raise answer("trouble", request=request)
        return answer if isinstance(answer, httpx.Response) else httpx.Response(answer)

    return net.client(httpx.MockTransport(handler))


@pytest.fixture
def clock(monkeypatch):
    """The offline registry with a clock the test moves (``clock[0]``) and the real pause."""
    now = [1000.0]
    monkeypatch.setattr(net, "OFFLINE", net.OfflineHosts(clock=lambda: now[0]))
    return now


def test_retries_back_off_exponentially_and_give_up():
    calls, sleeps = [], []
    client = server([503, 500, httpx.ReadTimeout, 502], calls)
    response = net.get(client, "https://a.example/x", sleep=sleeps.append)
    assert response.status_code == 200 and len(calls) == 5 and sleeps == [1.0, 2.0, 4.0, 8.0]
    # still failing after the last attempt: the 5xx answer is returned (no wait after it)
    calls.clear()
    sleeps.clear()
    client = server([503] * 10, calls)
    assert net.get(client, "https://a.example/x", sleep=sleeps.append).status_code == 503
    assert len(calls) == net.MAX_RETRIES + 1 and sleeps == [1.0, 2.0, 4.0, 8.0]
    # ... and a transport error is raised; the back-off is capped
    sleeps.clear()
    client = server([httpx.ReadTimeout] * 10, calls)
    with pytest.raises(httpx.ReadTimeout):
        net.get(client, "https://a.example/x", sleep=sleeps.append, retries=7)
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


def test_retry_after_is_honoured_up_to_a_limit():
    calls, sleeps = [], []
    too_many = [httpx.Response(429, headers={"Retry-After": "7"}),
                httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
                httpx.Response(429, headers={"Retry-After": "3600"}),
                httpx.Response(429)]  # fmt: skip
    client = server(too_many, calls)
    assert net.get(client, "https://a.example/x", sleep=sleeps.append).status_code == 200
    assert sleeps == [7.0, 2.0, net.MAX_RETRY_AFTER_S, 8.0]


def test_client_errors_are_not_retried():
    for status in (403, 404, 204):
        calls, sleeps = [], []
        client = server([status], calls)
        assert net.get(client, "https://a.example/x", sleep=sleeps.append).status_code == status
        assert len(calls) == 1 and sleeps == []


def test_cancel_before_and_during_the_back_off():
    calls = []
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(net.CancelledError):
        net.get(server([], calls), "https://a.example/x", cancel=cancel)
    assert calls == []
    cancel.clear()
    with pytest.raises(net.CancelledError):
        net.get(server([503] * 5, calls), "https://a.example/x", cancel=cancel,
                sleep=lambda s: cancel.set())  # fmt: skip
    assert len(calls) == 1
    # the real back-off wait ends at a cancel
    started = time.monotonic()
    BACKOFF_WAIT(30.0, cancel)
    assert time.monotonic() - started < 5.0


def test_an_unreachable_host_is_never_retried(clock):
    """Connection refused, no DNS or no connection within the connect timeout: failing at once
    (an export must not wait through the retries of every tile)."""
    for error in (httpx.ConnectError, httpx.ConnectTimeout):
        calls, sleeps = [], []
        client = server([error] * 10, calls)
        with pytest.raises(error):
            net.get(client, f"https://{error.__name__}.example/x", sleep=sleeps.append)
        assert len(calls) == 1 and sleeps == []


def test_the_offline_pause_is_shared_per_host_and_released(clock):
    calls, other_calls, waits = [], [], []

    class Limiter:
        def wait(self):
            waits.append(1)

    tiles = server([httpx.ConnectError], calls)
    dem = server([], calls)
    with pytest.raises(httpx.ConnectError):
        net.get(tiles, "https://data.example/tiles/1.png")
    assert net.offline_hosts() == ["data.example"]
    # another user of the same host fails at once: no request, no rate-limit wait
    with pytest.raises(net.HostOffline, match="data.example"):
        net.get(dem, "https://data.example/wms", params={"BBOX": "1"}, limiter=Limiter())
    assert len(calls) == 1 and waits == []
    # other hosts are not affected
    assert net.get(server([], other_calls), "https://other.example/x").status_code == 200
    # after the pause the host is tried again; once it answers it is online
    clock[0] += net.OFFLINE_PAUSE_S - 1.0
    with pytest.raises(net.HostOffline):
        net.get(dem, "https://data.example/wms")
    assert net.offline_hosts() == ["data.example"]
    clock[0] += 2.0
    assert net.get(dem, "https://data.example/wms", limiter=Limiter()).status_code == 200
    assert len(calls) == 2 and waits == [1] and net.offline_hosts() == []


def test_tiles_and_dem_of_one_server_stop_waiting_together(tmp_path, clock):
    """The tile downloader and the DEM baker share the pause of their server."""
    requests, online = [], [False]
    values = np.full((HEIGHTMAP_SAMPLES, HEIGHTMAP_SAMPLES), 5.0, "<f4").tobytes()

    def handler(request):
        requests.append(request.url.path)
        if not online[0]:
            raise httpx.ConnectError("no route to host", request=request)
        return httpx.Response(200, content=values, headers={"content-type": "image/x-bil"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = TileProvider(id="shared", name="Shared", kind="imagery", ext="png",
                            url_template="https://geo.example/tiles/{z}/{x}/{y}.png",
                            max_requests_per_second=0, concurrency=1)  # fmt: skip

    class Wms(WmsDemSource):
        id = name = "test_shared_host"
        wms_url = "https://geo.example/wms"
        layer = "DEM"
        max_requests_per_second = 0.0

    cache = TileCache(tmp_path)
    tiles = TileDownloader(provider, cache, client, sleep=lambda s: pytest.fail("retried"))
    assert tiles.run([(5, x, 0) for x in range(10)]).failed == 10
    baked = DemBaker(Wms(), cache, client=client).bake([(12, x, 1450) for x in range(10)])
    assert baked.failed == 10 and "offline" in baked.errors[0]
    assert requests == ["/tiles/5/0/0.png"]  # one attempt in all
    # the pause is over and the server is back
    clock[0] += net.OFFLINE_PAUSE_S
    online[0] = True
    assert DemBaker(Wms(), cache, client=client).bake([(12, 0, 1450)]).downloaded == 1
    assert net.offline_hosts() == []
