import numpy as np
import pytest

from earthling.render.renderer import Renderer
from earthling.render.shader_library import ShaderError, ShaderLibrary, preprocess


def test_preprocess_resolves_includes(tmp_path):
    (tmp_path / "a.glsl").write_text("float a() { return 1.0; }\n")
    (tmp_path / "b.glsl").write_text('#include "a.glsl"\nfloat b() { return a(); }\n')
    deps = set()
    out = preprocess('#include "b.glsl"\nvoid main() {}', tmp_path, {"FOO": 2}, deps)
    assert out.startswith("#version 430 core\n#define FOO 2\n")
    assert "float a()" in out and "float b()" in out
    assert {p.name for p in deps} == {"a.glsl", "b.glsl"}


def test_preprocess_detects_recursion(tmp_path):
    (tmp_path / "a.glsl").write_text('#include "a.glsl"\n')
    with pytest.raises(ShaderError):
        preprocess('#include "a.glsl"', tmp_path)


def test_hot_reload_keeps_old_program_on_error(gl_ctx, tmp_path):
    (tmp_path / "p.vert").write_text("void main() { gl_Position = vec4(0.0); }")
    frag = tmp_path / "p.frag"
    frag.write_text("out vec4 c; void main() { c = vec4(1.0); }")
    lib = ShaderLibrary(gl_ctx, tmp_path)
    first = lib.get("p")
    frag.write_text("this is not glsl")
    errors = lib.reload_changed(frag)
    assert errors and lib.get("p") is first
    frag.write_text("out vec4 c; void main() { c = vec4(0.5); }")
    assert lib.reload_changed(frag) == []
    assert lib.get("p") is not first


def test_renderer_draws_tracks(gl_ctx):
    from pathlib import Path

    from earthling.core.geo import LocalFrame
    from earthling.core.gpx import load_gpx_folder
    from earthling.render.camera import Camera, OrbitController

    tracks, _ = load_gpx_folder(Path(__file__).parents[1] / "examples" / "alps_demo" / "gpx")
    renderer = Renderer(gl_ctx)
    renderer.set_scene(LocalFrame(46.0, 6.99, 0.0), tracks)
    camera = Camera()
    OrbitController(camera).frame_bounds(*renderer.tracks.bounds)
    fbo = gl_ctx.simple_framebuffer((128, 128))
    renderer.render(fbo, 128, 128, camera)
    img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(128, 128, 3)
    background = np.array([20, 23, 28])
    lit = np.abs(img.astype(int) - background).sum(axis=2) > 60
    assert lit.sum() > 50
