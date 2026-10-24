"""Area outlines (render/overlays.py): plain coloured lines in the 3D view."""

import numpy as np

from earthling.render.camera import Camera
from earthling.render.overlays import OutlineLayer
from earthling.render.shader_library import ShaderLibrary

S = 64  # framebuffer size


def square(half, north=1000.0):
    """A square of ``half`` metres around a point ``north`` metres ahead, upright."""
    return np.array([[-half, north, -half], [half, north, -half], [half, north, half],
                     [-half, north, half]], dtype=np.float64)  # fmt: skip


def draw(gl_ctx, layer, camera):
    fbo = gl_ctx.simple_framebuffer((S, S))
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 1.0)
    layer.render(camera, camera.view_projection(1.0))
    img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(S, S, 3)
    return img[::-1].astype(int)  # rows top -> bottom


def looking_north():
    return Camera(position=np.zeros(3), heading=0.0, pitch=0.0)


def test_lines_are_drawn_where_they_are(gl_ctx):
    layer = OutlineLayer(gl_ctx, ShaderLibrary(gl_ctx))
    red = (1.0, 0.0, 0.0, 1.0)
    layer.set_lines([(square(200.0), red, True)])
    img = draw(gl_ctx, layer, looking_north())
    closed = img[..., 0] > 200
    assert (img[closed][:, 1:] < 30).all()  # pure red (sRGB -> linear -> sRGB)
    ys, xs = np.nonzero(closed)
    assert abs(xs.mean() - (S - 1) / 2) < 1 and abs(ys.mean() - (S - 1) / 2) < 1  # ahead
    assert xs.min() > 10 and xs.max() < S - 10  # 400 m wide at 1 km
    left, mid = xs.min(), S // 2
    assert closed[mid, left]
    # open: the same points without the closing (left) edge
    layer.set_lines([(square(200.0), red, False)])
    open_line = draw(gl_ctx, layer, looking_north())[..., 0] > 200
    assert not open_line[mid, left] and open_line[mid, xs.max()]
    # camera-relative: a camera far away east sees the same picture of a line far away east
    far = 3_000_000.0
    layer.set_lines([(square(200.0) + [far, 0, 0], red, True)])
    camera = looking_north()
    camera.position = np.array([far, 0.0, 0.0])
    assert ((draw(gl_ctx, layer, camera)[..., 0] > 200) == closed).mean() > 0.99


def test_hidden_empty_and_short_lines(gl_ctx):
    layer = OutlineLayer(gl_ctx, ShaderLibrary(gl_ctx))
    green = (0.0, 1.0, 0.0, 1.0)
    layer.set_lines([(square(200.0)[:1], green, False), (np.zeros((0, 3)), green, True)])
    assert layer._lines == []  # fewer than two points: nothing to draw
    assert draw(gl_ctx, layer, looking_north()).max() == 0
    layer.set_lines([(square(200.0), green, True)])
    layer.visible = False
    assert draw(gl_ctx, layer, looking_north()).max() == 0
    layer.visible = True
    behind = looking_north()
    behind.heading = 180.0
    assert draw(gl_ctx, layer, behind).max() == 0
    assert draw(gl_ctx, layer, looking_north())[..., 1].max() > 200
    layer.release()
    assert layer._lines == [] and draw(gl_ctx, layer, looking_north()).max() == 0


def test_shader_reload_rebuilds_the_vertex_arrays(gl_ctx):
    """After a shader hot-reload the lines are drawn with the new program."""
    library = ShaderLibrary(gl_ctx)

    class Reloading:
        def __init__(self):
            self.program = library.get("track_line")

        def get(self, name):
            return self.program

    shaders = Reloading()
    layer = OutlineLayer(gl_ctx, shaders)
    layer.set_lines([(square(200.0), (0.0, 0.0, 1.0, 1.0), True)])
    first = draw(gl_ctx, layer, looking_north())
    old_vao = layer._lines[0].vao
    shaders.program = ShaderLibrary(gl_ctx).get("track_line")  # a recompiled program
    second = draw(gl_ctx, layer, looking_north())
    assert layer._lines[0].vao is not old_vao
    assert np.array_equal(first, second) and first[..., 2].max() > 200
