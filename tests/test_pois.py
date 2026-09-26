import numpy as np
import pytest
from PIL import Image

from earthling.core.pois import POP_IN_S, Poi, pop_in, slug, visibility_state
from earthling.core.scene import Scene
from earthling.render.pois import builtin_icons, load_icon, resolve_icon


def test_builtin_icons_and_resolution(tmp_path):
    names = builtin_icons()
    assert {"pin", "coffee", "turnaround", "tent", "camera", "flag"} <= set(names)
    assert resolve_icon("builtin:coffee", None).name == "coffee.png"
    assert resolve_icon("builtin:nope", None) is None
    (tmp_path / "mine.png").write_bytes(b"x")
    assert resolve_icon("mine.png", tmp_path) == tmp_path / "mine.png"


def test_icon_formats(tmp_path):
    coffee = load_icon(resolve_icon("builtin:coffee", None))
    assert len(coffee.frames) == 24 and coffee.total == pytest.approx(24 * 0.07)
    assert coffee.index_at(0.0) == 0 and coffee.index_at(0.071) == 1
    assert coffee.index_at(coffee.total + 0.01) == 0  # loops
    # all APNG frames are complete (composited), not just the changed regions
    opaque = [(f[..., 3] > 0).mean() for f in coffee.frames]
    assert min(opaque) > 0.6 * max(opaque)
    sheet = np.zeros((64, 192, 4), dtype=np.uint8)
    for i in range(3):
        sheet[:, i * 64 : (i + 1) * 64] = (80 * i, 0, 0, 255)
    Image.fromarray(sheet).save(tmp_path / "sheet.png")
    frames = load_icon(tmp_path / "sheet.png", cols=3, rows=1, fps=10)
    assert len(frames.frames) == 3 and frames.frames[2][0, 0, 0] == 160
    assert frames.index_at(0.25) == 2
    Image.new("RGBA", (1024, 512), (0, 255, 0, 255)).save(tmp_path / "big.png")
    big = load_icon(tmp_path / "big.png")
    assert big.frames[0].shape[:2] == (128, 256) and big.durations == []


def test_pop_timing_from_the_visibility_keys():
    scene = Scene()
    poi = Poi("cafe", "Café", 7.0, 46.0)
    scene.set_pois([poi])
    anim = scene.animation
    assert visibility_state(anim, "cafe", 3.0, True) == (1.0, 3.0)  # no keys: always there
    anim.set_key("poi.cafe.visible", 0.0, False)
    anim.set_key("poi.cafe.visible", 2.0, True)
    anim.set_key("poi.cafe.visible", 5.0, False)
    assert visibility_state(anim, "cafe", 1.0, False) == (0.0, 0.0)
    assert visibility_state(anim, "cafe", 0.0, False) == (0.0, 0.0)  # hidden start: no pop-out
    scale, since = visibility_state(anim, "cafe", 2.1, True)
    assert since == pytest.approx(0.1) and 0.0 < scale < 1.0
    assert visibility_state(anim, "cafe", 2.0 + POP_IN_S + 0.1, True)[0] == 1.0
    scale, _ = visibility_state(anim, "cafe", 5.1, False)  # shrinking
    assert 0.0 < scale < 1.0
    assert visibility_state(anim, "cafe", 6.0, False)[0] == 0.0
    assert pop_in(0.0) == 0.0 and max(pop_in(t / 100) for t in range(50)) > 1.05  # overshoot


def test_scene_roundtrip_with_pois():
    scene = Scene()
    pois = [Poi(slug("Coffee stop", set()), "Coffee stop", 6.9, 45.9, "builtin:coffee",
                caption="Cappuccino!")]  # fmt: skip
    pois.append(Poi(slug("Coffee stop", {pois[0].id}), "Coffee stop", 6.95, 45.92))
    assert [p.id for p in pois] == ["coffee_stop", "coffee_stop_2"]
    scene.set_pois(pois)
    assert "poi.coffee_stop.effect" in scene.registry
    assert scene.registry["poi.coffee_stop.visible"].section == "POI: Coffee stop"
    scene.store.set("poi.coffee_stop.effect", "bounce")
    scene.animation.set_key("poi.coffee_stop_2.opacity", 1.0, 0.5)
    doc = scene.to_json()
    other = Scene()
    assert other.load_json(doc) == []
    assert [p.caption for p in other.pois] == ["Cappuccino!", ""]
    assert other.store["poi.coffee_stop.effect"] == "bounce"
    assert other.animation.is_animated("poi.coffee_stop_2.opacity")
    other.set_pois(other.pois[:1])
    assert "poi.coffee_stop_2.opacity" not in other.registry


def render_poi(gl_ctx, poi, time=1.0, **props):
    from earthling.core.geo import lonlat_to_tile
    from earthling.render import lod
    from earthling.render.camera import Camera, OrbitController
    from earthling.render.renderer import Renderer
    from test_terrain import FRAME, FakeTerrainData

    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [])
    tx, ty = lonlat_to_tile(7.0, 46.0, 10)
    renderer.set_terrain_source(FakeTerrainData(), lod.NodeSet({10: [(int(tx), int(ty))]}))
    scene = Scene()
    scene.set_pois([poi])
    for pid, value in props.items():
        scene.store.set(pid.replace("__", "."), value)
    renderer.store = scene.store
    renderer.animation = scene.animation
    renderer.time = time
    renderer.pois.set_pois(scene.pois)
    camera = Camera()
    renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 240)
    orbit = OrbitController(camera)
    orbit.frame_bounds(*renderer.scene_bounds())
    orbit.distance *= 0.3
    orbit.pitch = -45.0
    orbit.apply()
    renderer.terrain.finish_loading(camera, camera.view_projection(320 / 240), 240)
    fbo = gl_ctx.simple_framebuffer((320, 240))
    renderer.render(fbo, 320, 240, camera)
    img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(240, 320, 3)[::-1]
    return renderer, img.astype(int)


def test_poi_rendering(gl_ctx):
    # (the fake terrain has no height lookup: the anchor height is the offset, above the bump)
    poi = Poi("camp", "Camp", 7.0, 46.0, "builtin:turnaround", size_px=300.0, caption="Closed",
              height_offset_m=2600.0)  # fmt: skip
    renderer, img = render_poi(gl_ctx, poi)
    assert renderer.pois.last_drawn == ["camp"]
    red = (img[..., 0] > 170) & (img[..., 1] < 70) & (img[..., 2] < 70)
    assert red.sum() > 200  # the red sign ring
    _, hidden = render_poi(gl_ctx, poi, poi__camp__visible=False)
    assert ((hidden[..., 0] > 170) & (hidden[..., 1] < 70)).sum() == 0
    buried = Poi("camp", "Camp", 7.0, 46.0, "builtin:turnaround", size_px=300.0,
                 height_offset_m=-3000.0)  # fmt: skip
    renderer, img_b = render_poi(gl_ctx, buried)
    assert ((img_b[..., 0] > 170) & (img_b[..., 1] < 70) & (img_b[..., 2] < 70)).sum() == 0
    _, wobbled = render_poi(gl_ctx, poi, time=1.1, poi__camp__effect="wobble",
                            poi__camp__effect_strength=3.0)  # fmt: skip
    assert np.abs(wobbled - img).sum() > 1000


def test_captions_without_icon_and_with_several_lines(gl_ctx):
    def white_rows(img):
        white = (img[..., 0] > 200) & (img[..., 1] > 200) & (img[..., 2] > 200)
        rows = np.flatnonzero(white.any(axis=1))
        return (rows.max() - rows.min()) if len(rows) else 0, white.sum()

    label = Poi("fr", "France", 7.0, 46.0, "none", caption="Frankreich", height_offset_m=2600.0)
    renderer, img = render_poi(gl_ctx, label, poi__fr__caption_size=72.0)
    assert renderer.pois.last_drawn == ["fr"]
    one_line, text_px = white_rows(img)
    assert text_px > 30
    sign = Poi(
        "fr", "France", 7.0, 46.0, "builtin:turnaround", size_px=300.0, height_offset_m=2600.0
    )
    _, with_icon = render_poi(gl_ctx, sign)
    red = lambda im: ((im[..., 0] > 170) & (im[..., 1] < 70) & (im[..., 2] < 70)).sum()  # noqa: E731
    assert red(with_icon) > 20 and red(img) == 0  # "none": no icon
    stacked = Poi("fr", "France", 7.0, 46.0, "none", caption="Etappe 2\n382 km",
                  height_offset_m=2600.0)  # fmt: skip
    _, img2 = render_poi(gl_ctx, stacked, poi__fr__caption_size=72.0)
    assert white_rows(img2)[0] > one_line * 1.6
