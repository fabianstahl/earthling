"""Step 18.3: rendering views from space (culling, depth, the camera's local atmosphere)."""

import numpy as np

from earthling.render.camera import far_for_height


def test_far_plane_does_not_cull_the_globe_from_space():
    from earthling.core.geo import LocalFrame
    from earthling.render import lod
    from earthling.render.camera import Camera

    frame = LocalFrame(35.0, 75.0, 0.0)
    eye = frame.geodetic_to_enu(35.0, 70.0, 10_000e3)
    cam = Camera(np.asarray(eye, dtype=np.float64), 100.0, -85.0, 0.0)
    cam.far = far_for_height(10_000e3)
    planes = lod.frustum_planes(cam.view_projection(16 / 9))
    planes = lod.with_depth_planes(planes, cam.forward, cam.near, cam.far)
    world = lod.node_bounds(frame, (0, 0, 0), 0.0, 8000.0)
    assert lod.sphere_visible(planes, world.center - cam.position, world.radius)
    # the zoom 0 tile's bounds contain the whole globe (dense samples)
    assert world.radius > 6_371e3


def test_lighting_and_atmosphere_follow_the_camera(gl_ctx):
    from earthling.core.geo import LocalFrame
    from earthling.core.scene import Scene
    from earthling.render.camera import Camera
    from earthling.render.renderer import Renderer

    renderer = Renderer(gl_ctx)
    frame = LocalFrame(35.0, 75.0, 0.0)
    renderer.set_scene(frame, [])
    renderer.store = Scene().store
    renderer.timezone = "UTC"
    near = Camera(np.array(frame.geodetic_to_enu(35.0, 75.0, 2000.0)))
    renderer.apply_properties(near)
    sun_near = np.array(renderer.lighting.sun_direction)
    assert np.allclose(renderer.lighting.atm_frame, np.eye(3).ravel(), atol=1e-9)
    far = Camera(np.array(frame.geodetic_to_enu(50.0, 8.6, 2000.0)))  # Frankfurt
    renderer.apply_properties(far)
    m = np.array(renderer.lighting.atm_frame).reshape(3, 3).T  # column-major
    up = frame.geodetic_to_enu(50.0, 8.6, 1000.0) - frame.geodetic_to_enu(50.0, 8.6, 0.0)
    assert np.allclose(m @ (up / np.linalg.norm(up)), (0.0, 0.0, 1.0), atol=1e-6)
    # the sun is the same direction in space, only the local elevation differs
    assert np.allclose(renderer.lighting.sun_direction, sun_near, atol=2e-3)


def test_bloom_keeps_its_size_relative_to_the_image(gl_ctx):
    from earthling.render.bloom import Bloom
    from earthling.render.renderer import Renderer

    renderer = Renderer(gl_ctx)
    bloom = Bloom(gl_ctx)
    bloom.ensure(512, 288)
    source = gl_ctx.texture((512, 288), 4, dtype="f2")
    full = bloom.run(source, renderer.fullscreen, 1.0, 4)
    coarse = bloom.run(source, renderer.fullscreen, 1.0, 4, skip=1)
    assert coarse.size[1] * 2 == full.size[1]  # one level coarser, same number of levels
    assert bloom.run(source, renderer.fullscreen, 1.0, 99, skip=99) is not None


def test_height_fog_from_space_is_finite(gl_ctx):
    """Height fog seen from hundreds of kilometres up: a thin column, no NaN (step 18.7)."""
    from earthling.core.geo import LocalFrame
    from earthling.core.scene import Scene
    from earthling.render.camera import Camera
    from earthling.render.renderer import Renderer

    renderer = Renderer(gl_ctx)
    frame = LocalFrame(9.75, 100.0, 0.0)
    renderer.set_scene(frame, [])
    store = Scene().store
    store.set("fog.enabled", True)
    store.set("fog.density", 0.25)
    store.set("fog.base", 150.0)
    store.set("fog.falloff", 350.0)
    renderer.store = store
    renderer.timezone = "UTC"
    fbo = gl_ctx.simple_framebuffer((64, 64))
    cam = Camera(np.array(frame.geodetic_to_enu(9.75, 100.0, 400e3)), 0.0, -80.0, 0.0)
    renderer.render(fbo, 64, 64, cam)
    img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(64, 64, 3)
    # (no terrain in this test: the sky and the empty ground below must not be fog-filled)
    assert img.std() > 0.0 or img.mean() < 250
