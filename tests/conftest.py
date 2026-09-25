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
def _isolated_qsettings():
    """Keep tests from touching the user's real settings (recent projects etc.)."""
    from PyQt6.QtCore import QCoreApplication

    QCoreApplication.setOrganizationName("EarthlingTests")
    QCoreApplication.setApplicationName("EarthlingTests")
    yield


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Tests must never block on a message box (e.g. 'unsaved changes' on close)."""
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Discard)
    for name in ("warning", "critical", "information"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: QMessageBox.StandardButton.Ok)
    yield
