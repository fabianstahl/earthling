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
    from earthling.core.properties import enum, flt
    from earthling.render.vehicles import MARKER_OPTIONS
    from earthling.session import Session  # noqa: F401  (properties below)

    scene.set_dynamic(
        "groups",
        [
            enum("trackgroup.tracks.marker", "Marker", "dot", MARKER_OPTIONS),
            flt("trackgroup.tracks.marker_size", "Size", 46.0, 8.0, 200.0),
        ],
    )
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


def test_vehicle_tint_and_label(gl_ctx):
    """Step 19.18: a tinted vehicle with a caption moving with it."""
    from earthling.core.properties import PropertyDef, PType, color, enum, flt
    from earthling.core.scene import Scene
    from earthling.render.camera import Camera
    from earthling.render.renderer import Renderer
    from earthling.render.vehicles import MARKER_OPTIONS
    from test_tracks import FRAME, straight_track

    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [straight_track()])
    scene = Scene()
    p = "trackgroup.tracks."
    scene.set_dynamic("groups", [
        enum(p + "marker", "Marker", "runner", MARKER_OPTIONS),
        flt(p + "marker_size", "Size", 160.0, 8.0, 200.0),
        color(p + "marker_tint", "Tint", (1.0, 1.0, 1.0)),
        PropertyDef(p + "marker_label", "Label", PType.TEXT, "")])  # fmt: skip
    store = scene.store
    renderer.store = store
    for pid, value in (("tracks.elevation", "gpx"), ("progress.head", 0.5), ("haze.aerial", 0.0),
                       ("glow.enabled", False)):  # fmt: skip
        store.set(pid, value)
    mid = FRAME.geodetic_to_enu(46.005, 7.0, 1500.0)
    camera = Camera(position=mid + np.array([0.0, 0.0, 1500.0]), heading=0.0, pitch=-89.9)
    fbo = gl_ctx.simple_framebuffer((200, 200))

    def draw():
        renderer.render(fbo, 200, 200, camera)
        return np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(200, 200, 3)[::-1]

    plain = draw().astype(int)
    store.set(p + "marker_tint", (1.0, 0.8, 0.2))  # gold
    gold = draw().astype(int)
    changed = np.abs(gold - plain).sum(axis=2) > 30
    assert changed.sum() > 20  # the icon got its colour
    assert (gold[changed][:, 2] < plain[changed][:, 2]).mean() > 0.8  # less blue: gold
    store.set(p + "marker_label", "2025")
    labelled = draw().astype(int)
    rows = np.flatnonzero((np.abs(labelled - gold).sum(axis=2) > 30).any(axis=1))
    icon_rows = np.flatnonzero(changed.any(axis=1))
    assert len(rows) and rows.max() <= icon_rows.min() + 2  # the caption above the icon


def test_travel_direction_is_horizontal_and_smooth():
    import numpy as np

    from earthling.render.vehicles import travel_direction

    # a plane landing steeply towards the north-east: the icon still points north-east
    here = np.array([0.0, 0.0, 500.0])
    d = travel_direction(here - np.array([700.0, 700.0, -400.0]), here,
                         here + np.array([5.0, 5.0, -480.0]))  # fmt: skip
    assert d[2] == 0.0 or abs(d[2]) < 1e-6
    assert np.allclose(d[:2], [np.sqrt(0.5), np.sqrt(0.5)], atol=0.01)
    assert travel_direction(here, here, here) is None
