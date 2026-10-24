import moderngl
import pytest


@pytest.fixture
def gl_ctx():
    """Standalone OpenGL context for renderer tests; skipped where no GPU is available.

    Function-scoped: Qt widget tests make their own contexts current, which would leave a
    shared standalone context unusable.
    """
    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:  # pragma: no cover - depends on machine
        pytest.skip(f"no OpenGL 4.3 context available: {exc}")
    yield ctx
    ctx.release()


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path_factory):
    """Keep tests from touching the user's real settings (recent projects etc.), and from each
    other's: every test starts with empty settings. They live in an INI file of this session,
    not in the registry, which test runs in other checkouts share (and clear meanwhile)."""
    from PyQt6.QtCore import QCoreApplication, QSettings

    QCoreApplication.setOrganizationName("EarthlingTests")
    QCoreApplication.setApplicationName("EarthlingTests")
    ini = QSettings.Format.IniFormat
    QSettings.setDefaultFormat(ini)
    folder = tmp_path_factory.getbasetemp() / "settings"
    QSettings.setPath(ini, QSettings.Scope.UserScope, str(folder))
    QSettings().clear()
    yield


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Tests must never block on a message box (e.g. 'unsaved changes' on close)."""
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Discard)
    for name in ("warning", "critical", "information"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "about", lambda *a, **k: None)
    yield


@pytest.fixture(autouse=True)
def _isolated_gpx_cache(tmp_path_factory, monkeypatch):
    """Parsed-GPX cache in a temporary folder (not the user's ~/.cache)."""
    import earthling.core.gpx as gpx

    monkeypatch.setattr(gpx, "PARSE_CACHE_DIR", tmp_path_factory.getbasetemp() / "gpx_cache")
    yield


@pytest.fixture(autouse=True, scope="session")
def _no_network():
    """Real HTTP requests fail (tests use httpx.MockTransport, see ``helpers.mock_http``).

    Session-scoped, so background threads outliving a test (borders, labels) are offline too;
    code that treats the data as optional then simply goes without it."""
    import httpx

    def refuse(self, request):  # (as if offline: the code handles that)
        raise httpx.ConnectError(f"tests must not touch the network: {request.url}")

    # Never restored: daemon threads (a label download retrying) may still run while the
    # interpreter shuts down after the session, and reached the real Overpass server.
    httpx.HTTPTransport.handle_request = refuse
    yield


@pytest.fixture(autouse=True)
def _network_state(monkeypatch):
    """No real back-off waits between retries, and no offline pause unless a test sets one:
    daemon threads of earlier tests reach the refusing transport above and would otherwise
    pause the hosts (Overpass, Natural Earth) that later tests mock."""
    from earthling.data import net

    monkeypatch.setattr(net, "backoff_wait", lambda seconds, cancel=None: None)
    monkeypatch.setattr(net, "OFFLINE", net.OfflineHosts(pause=0.0))
    yield
