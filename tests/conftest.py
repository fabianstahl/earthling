import moderngl
import pytest


@pytest.fixture(scope="session")
def gl_ctx():
    """Standalone OpenGL context for renderer tests; skipped where no GPU is available."""
    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:  # pragma: no cover - depends on machine
        pytest.skip(f"no OpenGL 4.3 context available: {exc}")
    yield ctx
    ctx.release()
