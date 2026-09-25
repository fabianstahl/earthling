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


def make_luts(gl_ctx):
    from earthling.render.renderer import FullscreenPasses
    from earthling.render.shader_library import ShaderLibrary
    from earthling.render.sky_luts import AtmosphereLuts

    shaders = ShaderLibrary(gl_ctx)
    return AtmosphereLuts(gl_ctx, shaders, FullscreenPasses(gl_ctx, shaders))


def read(tex, shape):
    return np.frombuffer(tex.read(), dtype=np.float16).astype(np.float64).reshape(shape)[..., :3]


def skyview_for(luts, elevation_deg, camera_height=2000.0, haze=1.0):
    import math

    from pyglm import glm

    e = math.radians(elevation_deg)
    luts.update(1.0, haze, camera_height, (0.0, math.cos(e), math.sin(e)), glm.mat4(1.0))
    return read(luts.skyview, (108, 192, 4))


def test_transmittance_lut_matches_the_optical_depth(gl_ctx):
    from earthling.render.atmosphere import lookup_optical_depth

    luts = make_luts(gl_ctx)
    skyview_for(luts, 30.0)
    t = read(luts.transmittance, (64, 256, 4))
    zenith_ground = t[0, 0]  # x_r = 0 (ground), x_mu = 0 (straight up)
    od = np.array(lookup_optical_depth(0.0, 1.0))
    rayleigh = np.array([5.5e-6, 13.0e-6, 22.4e-6])
    ozone = np.array([0.65e-6, 1.881e-6, 0.085e-6])
    expected = np.exp(-(rayleigh * od[0] + 21e-6 * 1.1 * od[1] + ozone * od[2]))
    assert zenith_ground == pytest.approx(expected, rel=0.02)
    assert (t[0, -1] < 0.05).all()  # grazing rays through the whole atmosphere


def test_sky_colours_from_noon_to_twilight(gl_ctx):
    luts = make_luts(gl_ctx)
    noon = skyview_for(luts, 60.0)
    zenith = noon[2].mean(axis=0)  # top rows: near the zenith
    assert zenith[2] > 1.5 * zenith[0]  # blue sky
    low = skyview_for(luts, 3.0)
    toward, away = low[50, 1], low[50, -2]  # just above the horizon
    assert toward[0] > 2.0 * toward[2]  # red/orange towards the setting sun
    assert away.sum() > 0.02 * toward.sum()  # the opposite sky stays lit (earth shadow test)
    twilight = skyview_for(luts, -3.0)
    assert 0.0 < twilight[50, 1].sum() < 0.2 * toward.sum()  # afterglow, much dimmer
    assert skyview_for(luts, -18.0).max() < 1e-3  # astronomical night


def test_medium_luts_are_cached_and_the_volume_is_monotonic(gl_ctx):
    import math

    from pyglm import glm

    from earthling.render.camera import Camera

    luts = make_luts(gl_ctx)
    camera = Camera(position=np.array([0.0, 0.0, 2000.0]), heading=0.0, pitch=0.0)
    inverse = glm.inverse(camera.view_projection(16 / 9))
    sun = (0.0, math.cos(0.5), math.sin(0.5))
    luts.update(1.0, 1.0, 2000.0, sun, inverse)
    luts.update(1.0, 1.0, 2000.0, sun, inverse)
    assert luts.static_updates == 1
    luts.update(1.5, 1.0, 2000.0, sun, inverse)
    assert luts.static_updates == 2
    inscatter = read(luts.ap_inscatter, (32, 32, 32, 4))[:, 16, 16]  # along the view centre
    trans = read(luts.ap_transmittance, (32, 32, 32, 4))[:, 16, 16]
    assert (np.diff(inscatter.sum(axis=1)) >= -1e-4).all() and inscatter[-1].sum() > 0.01
    assert (np.diff(trans.sum(axis=1)) <= 1e-4).all() and trans[0].min() > 0.99
