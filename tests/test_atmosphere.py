from datetime import UTC, datetime

import numpy as np
import pytest

from earthling.render.atmosphere import (
    BLOCKED_DEPTH,
    MIE_SCALE_HEIGHT,
    RAYLEIGH_SCALE_HEIGHT,
    coord_from_mu,
    lookup_optical_depth,
    mu_from_coord,
    optical_depth_lut,
    sun_transmittance,
)
from earthling.render.lighting import enu_to_celestial


def test_zenith_optical_depth_equals_scale_heights():
    od_r, od_m, od_o = lookup_optical_depth(0.0, 1.0)
    assert od_r == pytest.approx(RAYLEIGH_SCALE_HEIGHT, rel=0.02)
    assert od_m == pytest.approx(MIE_SCALE_HEIGHT, rel=0.02)
    assert od_o == pytest.approx(15e3, rel=0.05)  # tent area = half width


def test_coord_mapping_roundtrip_and_blocked_rays():
    mu = np.linspace(-1, 1, 11)
    assert mu_from_coord(coord_from_mu(mu)) == pytest.approx(mu)
    lut = optical_depth_lut()
    assert lut[0, 0, 0] == BLOCKED_DEPTH  # straight down from the ground


def test_sun_reddens_towards_the_horizon():
    high = sun_transmittance(60.0, 1500.0)
    low = sun_transmittance(2.0, 1500.0)
    assert (low < high).all()
    assert low[0] > low[1] > low[2]  # red survives, blue is scattered away
    assert sun_transmittance(-5.0, 1500.0).max() < 1e-3


def test_celestial_matrix_maps_north_pole():
    m = enu_to_celestial(datetime(2026, 7, 1, 22, 0, tzinfo=UTC), 46.0, 7.0)
    # the celestial pole is due north at an elevation equal to the latitude
    pole = np.array([0.0, np.cos(np.radians(46.0)), np.sin(np.radians(46.0))])
    assert (m @ pole) == pytest.approx([0, 0, 1], abs=1e-9)
    assert np.allclose(m.T @ m, np.eye(3))


def test_sky_pass_colors(gl_ctx):
    from earthling.core.geo import LocalFrame
    from earthling.core.scene import Scene
    from earthling.render.camera import Camera
    from earthling.render.renderer import Renderer

    renderer = Renderer(gl_ctx)
    renderer.set_scene(LocalFrame(46.0, 7.0, 0.0), [])
    renderer.store = Scene().store
    renderer.timezone = "Europe/Paris"
    camera = Camera(position=np.array([0.0, 0.0, 2000.0]), heading=0.0, pitch=20.0)
    fbo = gl_ctx.simple_framebuffer((32, 32))

    def mean_color(hour):
        renderer.store.set("sun.datetime", datetime(2026, 7, 1, hour, 0))
        renderer.render(fbo, 32, 32, camera)
        return np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(-1, 3).mean(axis=0)

    noon = mean_color(13)
    night = mean_color(1)
    assert noon[2] > noon[0] + 30  # blue sky
    assert noon.sum() > 3 * night.sum()
