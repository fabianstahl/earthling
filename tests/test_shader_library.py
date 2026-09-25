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


def test_renderer_draws_triangle(gl_ctx):
    renderer = Renderer(gl_ctx)
    fbo = gl_ctx.simple_framebuffer((64, 64))
    renderer.render(fbo, 64, 64, 0.0)
    img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(64, 64, 3)
    center = img[32, 32].astype(int)
    corner = img[1, 1].astype(int)
    assert center.sum() > corner.sum() + 100
