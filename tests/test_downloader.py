import threading

import httpx
import pytest

from earthling.data.cache import TileCache
from earthling.data.downloader import RateLimiter, TileDownloader, http_client
from earthling.data.providers import ESRI_WORLD_IMAGERY, TileProvider, get_provider

PROVIDER = TileProvider(
    id="test",
    name="Test",
    kind="imagery",
    url_template="https://tiles.example/{z}/{x}/{y}.png",
    ext="png",
    max_zoom=10,
    max_requests_per_second=0,
    placeholder_md5=frozenset({"5d41402abc4b2a76b9719d911017c592"}),  # md5("hello")
)


def make_transport(responses, calls):
    lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.url.path
        with lock:
            calls.append(key)
            queue = responses.get(key)
            status, body = queue.pop(0) if queue else (200, b"tile:" + key.encode())
        return httpx.Response(status, content=body)

    return httpx.MockTransport(handler)


def test_download_resume_missing_and_retries(tmp_path):
    calls = []
    responses = {
        "/3/1/2.png": [(404, b"")],
        "/3/2/2.png": [(503, b""), (200, b"late")],
        "/3/3/2.png": [(200, b"hello")],  # placeholder
        "/3/4/2.png": [(403, b"")],  # not retried
    }
    cache = TileCache(tmp_path)
    cache.write("test", 3, 0, 2, "png", b"cached")
    dl = TileDownloader(
        PROVIDER, cache, http_client(make_transport(responses, calls)), sleep=lambda s: None
    )
    tiles = [(3, x, 2) for x in range(6)] + [(12, 0, 0)]  # z12 is above max_zoom -> ignored
    progress = dl.run(tiles)
    assert progress.total == 6
    assert progress.skipped == 1
    assert progress.downloaded == 2  # x=2 (after a retry) and x=5
    assert progress.missing == 2
    assert progress.failed == 1
    assert cache.read("test", 3, 2, 2, "png") == b"late"
    assert cache.is_missing("test", 3, 1, 2) and cache.is_missing("test", 3, 3, 2)
    assert "/3/0/2.png" not in calls
    assert calls.count("/3/4/2.png") == 1

    # second run only retries the failed tile
    calls.clear()
    progress = dl.run(tiles)
    assert calls == ["/3/4/2.png"]
    assert progress.skipped == 5


def test_cancel_stops_requests(tmp_path):
    calls = []
    dl = TileDownloader(
        PROVIDER, TileCache(tmp_path), http_client(make_transport({}, calls)), sleep=lambda s: None
    )
    dl.cancel()
    progress = dl.run([(5, x, 0) for x in range(20)])
    assert calls == []
    assert progress.downloaded == 0


def test_rate_limiter_spacing():
    import time

    limiter = RateLimiter(50.0)
    start = time.monotonic()
    for _ in range(6):
        limiter.wait()
    assert time.monotonic() - start >= 5 / 50.0 * 0.9


def test_esri_placeholder_and_registry():
    assert get_provider("esri_world_imagery") is ESRI_WORLD_IMAGERY
    assert ESRI_WORLD_IMAGERY.tile_url(12, 2126, 1458).endswith("/tile/12/1458/2126")
    with pytest.raises(KeyError, match="unknown provider"):
        get_provider("nope")


def test_download_dialog_runs_jobs(qtbot, tmp_path):
    from earthling.app.download_dialog import DownloadDialog
    from earthling.core.config import Project
    from earthling.core.session import Session
    from earthling.data.downloader import DownloadProgress
    from earthling.data.jobs import DownloadJob

    session = Session(
        Project.load(__import__("pathlib").Path(__file__).parents[1] / "examples" / "alps_demo")
    )
    ran = []

    def run(cb):
        p = DownloadProgress(total=3, done=3, downloaded=3)
        cb(p)
        ran.append(True)
        return p

    dialog = DownloadDialog(session, lambda kinds: [DownloadJob("fake", run, lambda: None)])
    qtbot.addWidget(dialog)
    with qtbot.waitSignal(dialog.data_changed, timeout=5000):
        dialog.start()
    qtbot.waitUntil(lambda: dialog.start_btn.isEnabled(), timeout=5000)
    assert ran and "fake: done" in dialog.log.toPlainText()
