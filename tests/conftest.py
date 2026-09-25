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
