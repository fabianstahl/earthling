import numpy as np

from earthling.core.geo import LocalFrame, lonlat_to_tile
from earthling.render.layers import LAYER_IDS, LAYERS, generate_glsl, layer_properties
from earthling.render.parameters import build_registry
from earthling.render.shader_library import preprocess


def test_generated_glsl_has_all_layers_and_defaults():
    src = generate_glsl()
    for index, layer in enumerate(LAYERS):
        assert f"vec3 layer_{layer.id}(LayerInput li)" in src
        assert f"if (id == {index}) return layer_{layer.id}(li);" in src
    assert "uniform float u_sat_contrast = 1;" in src
    assert "uniform int u_elev_ramp = 0;" in src


def test_layer_properties_in_scene_registry():
    registry = build_registry()
    assert registry["layers.a"].options[0][0] == LAYER_IDS[0]
    for d in layer_properties():
        assert d.id in registry


def test_virtual_includes(tmp_path):
    out = preprocess(
        '#include "gen.glsl"\nvoid main() {}', tmp_path, virtual={"gen.glsl": "int x;"}
    )
    assert "int x;" in out


def test_every_layer_renders_differently(gl_ctx):
    from earthling.core.scene import Scene
    from earthling.render import lod
    from earthling.render.camera import Camera, OrbitController
    from earthling.render.renderer import Renderer
    from test_terrain import SlopeTerrainData

    renderer = Renderer(gl_ctx)
    renderer.set_scene(LocalFrame(46.0, 7.0, 0.0), [])
    renderer.store = Scene().store
    renderer.timezone = "Europe/Paris"
    tx, ty = lonlat_to_tile(7.0, 46.0, 10)
    renderer.set_terrain_source(SlopeTerrainData(), lod.NodeSet({10: [(int(tx), int(ty))]}))
    camera = Camera()
    renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 48)
    orbit = OrbitController(camera)
    orbit.frame_bounds(*renderer.scene_bounds())
    orbit.pitch = -80.0
    orbit.apply()
    renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 48)
    fbo = gl_ctx.simple_framebuffer((48, 48))
    means = {}
    for layer_id in LAYER_IDS:
        renderer.store.set("layers.a", layer_id)
        renderer.render(fbo, 48, 48, camera)
        img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(48, 48, 3)
        means[layer_id] = img[16:32, 16:32].reshape(-1, 3).mean(axis=0)
    ids = list(means)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            assert np.abs(means[ids[i]] - means[ids[j]]).sum() > 3, (ids[i], ids[j])
    # crossfade half way lands between the two layers
    renderer.store.set("layers.a", "satellite")
    renderer.store.set("layers.b", "hillshade")
    renderer.store.set("layers.mix", 0.5)
    renderer.render(fbo, 48, 48, camera)
    img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(48, 48, 3)
    mixed = img[16:32, 16:32].reshape(-1, 3).mean(axis=0).sum()
    lo, hi = sorted([means["satellite"].sum(), means["hillshade"].sum()])
    assert lo - 5 <= mixed <= hi + 5
