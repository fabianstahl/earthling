from datetime import datetime

import numpy as np
import pytest

from earthling.core.geo import LocalFrame, lonlat_to_tile
from earthling.core.sun import local_to_aware, solar_position
from earthling.render.camera import Camera
from earthling.render.shadows import (
    cascade_splits,
    compute_cascades,
    frustum_slice_sphere,
    light_rotation,
)

SUN = (-0.8, 0.3, 0.5)


def test_cascade_splits_are_increasing_and_end_at_far():
    splits = cascade_splits(10.0, 40_000.0)
    assert splits == sorted(splits) and splits[-1] == pytest.approx(40_000.0)
    assert splits[0] < 40_000.0 / 4  # logarithmic: more resolution near the camera


def test_slice_sphere_contains_the_slice():
    cam = Camera(position=np.zeros(3), heading=30.0, pitch=-10.0)
    centre, radius = frustum_slice_sphere(cam, 16 / 9, 100.0, 1000.0)
    t = np.tan(np.radians(cam.fov_y) / 2)
    # far corner distance from the camera
    far_corner = 1000.0 * np.sqrt(1 + t * t + (t * 16 / 9) ** 2)
    assert np.linalg.norm(centre) + radius >= far_corner - 1e-6


def test_cascade_centres_snap_to_texels_in_world_space():
    rot = np.array(light_rotation(SUN).to_list()).T[:3, :3]
    for x in (0.0, 3.7, 11.2):  # camera moving sideways
        cam = Camera(position=np.array([x, 0.0, 2000.0]), heading=0.0, pitch=-20.0)
        setup = compute_cascades(cam, 1.5, SUN, 30_000.0, 4096)
        for c in setup.cascades:
            ls = rot @ (cam.position + c.centre_rel)
            assert ls[0] / c.texel_world == pytest.approx(round(ls[0] / c.texel_world), abs=1e-3)
            assert ls[1] / c.texel_world == pytest.approx(round(ls[1] / c.texel_world), abs=1e-3)


def test_ridge_casts_shadow(gl_ctx):
    from earthling.core.scene import Scene
    from earthling.data.dem import HEIGHTMAP_SAMPLES
    from earthling.data.terrain_data import HeightmapRef
    from earthling.render import lod
    from earthling.render.camera import OrbitController
    from earthling.render.renderer import Renderer

    frame = LocalFrame(46.0, 7.0, 0.0)

    class Ridge:
        def __init__(self):
            self.heights = np.full((HEIGHTMAP_SAMPLES, HEIGHTMAP_SAMPLES), 1000.0, np.float32)
            self.heights[:, 120:125] = 4000.0  # north-south wall

        def heightmap_for(self, key):
            return HeightmapRef(key, self.heights, 256.0, (0.0, 0.0))

        def texture_for(self, key, source="imagery"):
            return np.full((64, 64, 3), 150, dtype=np.uint8) if source == "imagery" else None

    renderer = Renderer(gl_ctx)
    renderer.set_scene(frame, [])
    store = Scene().store
    renderer.store = store
    renderer.timezone = "UTC"
    when = datetime(2026, 7, 1, 12, 0)
    real = solar_position(local_to_aware(when, "UTC"), 46.0, 7.0)
    store.set("sun.datetime", when)
    store.set("sun.azimuth_offset", 270.0 - real.azimuth)  # sun in the west
    store.set("sun.elevation_offset", 20.0 - real.elevation)
    store.set("haze.aerial", 0.0)
    store.set("shadows.distance", 200.0)
    tx, ty = lonlat_to_tile(7.0, 46.0, 10)
    renderer.set_terrain_source(Ridge(), lod.NodeSet({10: [(int(tx), int(ty))]}))
    camera = Camera()
    renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 128)
    orbit = OrbitController(camera)
    orbit.frame_bounds(*renderer.scene_bounds())
    orbit.heading, orbit.pitch = 0.0, -89.0
    orbit.apply()
    renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 128)
    fbo = gl_ctx.simple_framebuffer((128, 128))

    def row(shadows):
        store.set("shadows.enabled", shadows)
        renderer.render(fbo, 128, 128, camera)
        img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(128, 128, 3)
        return img[60:68].mean(axis=(0, 2))  # brightness across x (west -> east)

    lit, shadowed = row(False), row(True)
    # the wall sits at ~47 % of the tile width; east of it the ground must darken
    east = slice(70, 90)
    west = slice(20, 45)
    assert shadowed[east].mean() < 0.8 * lit[east].mean()
    assert shadowed[west].mean() == pytest.approx(lit[west].mean(), rel=0.05)
