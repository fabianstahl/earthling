"""Step 19.8: vehicle markers at the head of the track."""

import math

import numpy as np
import pytest

from earthling.render.vehicles import VEHICLES, icon_rotation


def test_icons_exist():
    from earthling.render.vehicles import VEHICLE_DIR

    for name in VEHICLES:
        assert (VEHICLE_DIR / f"{name}.png").exists()


def test_icon_rotation():
    assert icon_rotation((0.0, -1.0), "rotate") == (pytest.approx(0.0), False)  # up the screen
    assert icon_rotation((1.0, 0.0), "rotate")[0] == pytest.approx(math.pi / 2)  # to the right
    assert icon_rotation((-1.0, 0.2), "flip") == (0.0, True)  # side views are mirrored
    assert icon_rotation((1.0, 0.2), "flip") == (0.0, False)
    assert icon_rotation(None, "rotate") == (0.0, False)


def test_vehicle_replaces_the_dot(gl_ctx):
    from earthling.core.scene import Scene
    from earthling.render.camera import Camera
    from earthling.render.renderer import Renderer
    from test_tracks import FRAME, straight_track

    renderer = Renderer(gl_ctx)
    track = straight_track()
    renderer.set_scene(FRAME, [track])
    scene = Scene()
    from earthling.core.session import Session  # noqa: F401  (properties below)
    from earthling.render.parameters import enum, flt
    from earthling.render.vehicles import MARKER_OPTIONS

    scene.set_dynamic("groups", [enum("trackgroup.tracks.marker", "Marker", "dot", MARKER_OPTIONS),
                                 flt("trackgroup.tracks.marker_size", "Size", 46.0, 8.0, 200.0)])
    store = scene.store
    renderer.store = store
    for pid, value in (("tracks.elevation", "gpx"), ("progress.head", 0.5), ("haze.aerial", 0.0),
                       ("tracks.visible", False)):  # fmt: skip
        store.set(pid, value)
    mid = FRAME.geodetic_to_enu(46.005, 7.0, 1500.0)
    camera = Camera(position=mid + np.array([0.0, 0.0, 1500.0]), heading=0.0, pitch=-89.9)
    fbo = gl_ctx.simple_framebuffer((200, 200))

    def draw():
        renderer.render(fbo, 200, 200, camera)
        return np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(200, 200, 3)

    store.set("tracks.visible", True)
    dot = draw().astype(int)
    assert renderer.vehicles == [] and renderer.marker.position is not None
    store.set("trackgroup.tracks.marker", "ship")
    ship = draw().astype(int)
    assert renderer.marker.position is None and renderer.vehicles[0].name == "ship"
    centre = (slice(80, 120), slice(80, 120))
    assert np.abs(ship[centre] - dot[centre]).sum() > 2000  # an icon instead of the dot
