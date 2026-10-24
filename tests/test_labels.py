import json

import httpx
import numpy as np
import pytest

from earthling.core.geo import LocalFrame, lonlat_to_tile
from earthling.core.scene import Scene
from earthling.data.cache import TileCache
from earthling.data.labels import (
    LabelData,
    LabelFeature,
    cache_path,
    download_labels,
    enrich,
    estimate_prominence,
    parse_elevation,
    parse_overpass,
)
from earthling.render import lod
from earthling.render.camera import Camera, OrbitController
from earthling.render.labels import FontAtlas, sight_in_clouds, signed_distance
from earthling.render.renderer import Renderer
from helpers import FRAME, FakeTerrainData

OVERPASS = {
    "elements": [
        {"type": "node", "id": 1, "lat": 45.83, "lon": 6.86,
         "tags": {"natural": "peak", "name": "Mont Blanc", "ele": "4808"}},
        {"type": "node", "id": 2, "lat": 45.95, "lon": 7.03,
         "tags": {"natural": "saddle", "name": "Col de Balme", "ele": "2,191 m"}},
        {"type": "node", "id": 3, "lat": 45.92, "lon": 6.87,
         "tags": {"place": "town", "name": "Chamonix"}},
        {"type": "way", "id": 4, "center": {"lat": 45.9, "lon": 6.9},
         "tags": {"tourism": "alpine_hut", "name": "Refuge", "ele": "9000 ft"}},
        {"type": "node", "id": 5, "lat": 45.9, "lon": 6.9, "tags": {"natural": "peak"}},
        {"type": "node", "id": 6, "lat": 45.9, "lon": 6.9,
         "tags": {"amenity": "bench", "name": "x"}},
    ]
}  # fmt: skip


def test_parse_overpass_and_elevations():
    features = parse_overpass(OVERPASS)
    assert [(f.kind, f.name) for f in features] == [
        ("peak", "Mont Blanc"), ("pass", "Col de Balme"), ("place", "Chamonix"), ("hut", "Refuge")
    ]  # fmt: skip
    assert features[0].ele == 4808 and features[1].ele == pytest.approx(2.191)
    assert features[2].rank == 3 and features[2].ele is None
    assert features[3].ele == pytest.approx(9000 * 0.3048)
    assert features[0].osm_id == "node/1" and features[3].osm_id == "way/4"
    assert parse_elevation("3842 m") == 3842 and parse_elevation("ca. 3000") is None


def two_peaks(lon, lat):
    """Peak A (3000 m) at lon 7.0, peak B (2500 m) at 7.1, valley floor 1500 m between."""
    lon = np.asarray(lon)
    a = 3000.0 - 40000.0 * np.abs(lon - 7.0)
    b = 2500.0 - 40000.0 * np.abs(lon - 7.1)
    return np.maximum(np.maximum(a, b), 1500.0)


def test_prominence_estimate():
    a = LabelFeature("a", "peak", "A", 7.0, 46.0, ele=3000.0)
    b = LabelFeature("b", "peak", "B", 7.1, 46.0, ele=2500.0)
    estimate_prominence([a, b], two_peaks)
    assert b.prominence == pytest.approx(1000.0, abs=30.0)  # down to the valley floor
    assert a.prominence == pytest.approx(1500.0, abs=30.0)  # highest: drop to the lowest point


def test_enrich_fills_elevation_and_track_distance():
    frame = LocalFrame(46.0, 7.0, 0.0)
    near = LabelFeature("n", "hut", "Near", 7.0, 46.009)  # ~1 km north of the track
    far = LabelFeature("f", "place", "Far", 7.0, 46.09)
    track = (np.array([6.95, 7.05]), np.array([46.0, 46.0]))
    enrich([near, far], lambda lon, lat: np.full(np.shape(lon), 1234.0), frame, [track])
    assert near.ele == 1234.0 and near.track_distance_m == pytest.approx(1000.0, rel=0.02)
    assert far.track_distance_m == pytest.approx(10000.0, rel=0.02)


def test_label_data_from_cache_and_download(tmp_path):
    cache = TileCache(tmp_path)
    bounds = (6.8, 45.8, 7.2, 46.1)
    seen = []

    def handler(request):
        seen.append(request.content.decode())
        return httpx.Response(200, json=OVERPASS)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    path = download_labels(cache, bounds, client=client)
    assert path == cache_path(cache, bounds) and "natural" in seen[0] and "45.80000" in seen[0]
    data = LabelData(cache, bounds)
    data.allow_download = False
    enriched = []
    features = data.load(enricher=lambda fs: enriched.extend(fs))
    assert len(features) == 4 and data.ready and data.version == 1 and len(enriched) == 4
    missing = LabelData(cache, (0.0, 0.0, 1.0, 1.0))
    missing.allow_download = False
    assert missing.load() is None and not missing.ready
    path.write_text("not json", encoding="utf-8")
    broken = LabelData(cache, bounds)
    broken.allow_download = False
    assert broken.load() is None and broken.error


def test_signed_distance_of_a_disk():
    yy, xx = np.mgrid[0:40, 0:40]
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 <= 10**2
    sdf = signed_distance(mask, 8.0)
    assert sdf[20, 20] == 1.0 and sdf[0, 0] == 0.0
    assert sdf[20, 30] == pytest.approx(0.5, abs=0.05)  # on the outline
    assert sdf[20, 25] > sdf[20, 29] > sdf[20, 31] > sdf[20, 34]  # monotonic across the edge


def test_font_atlas(gl_ctx):
    atlas = FontAtlas(gl_ctx)
    atlas.ensure("Aiguille Éö")
    generation = atlas.generation
    rects, uvs = atlas.quads("Aö")
    assert len(rects) == 2 and (uvs[:, 2] > uvs[:, 0]).all()
    assert atlas.width("Aö") > atlas.width("A") > 0
    atlas.ensure("A")  # nothing new: no repack
    assert atlas.generation == generation
    atlas.ensure("ß")
    assert atlas.generation == generation + 1
    atlas.release()


def test_font_atlas_layout_does_not_depend_on_the_hash_seed():
    """The glyphs were packed in set order, which changes with Python's string hash seed: the
    same frame differed between runs (text by up to 30 colour levels at 180p)."""
    import os
    import subprocess
    import sys

    script = "; ".join([
        "from types import SimpleNamespace as N",
        "from earthling.render.text import FontAtlas",
        "atlas = FontAtlas(N(texture=lambda *a, **k: N(release=lambda: None)))",
        "atlas.ensure('Testhorn 2834 m - Day 1: 12.4 km, 5 h 10 min')",
        "print(sorted((c, g.uv) for c, g in atlas.glyphs.items()))",
    ])  # fmt: skip
    runs = [subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True,
                             env={**os.environ, "PYTHONHASHSEED": seed})
            for seed in ("1", "2")]  # fmt: skip
    layouts = {run.communicate(timeout=60)[0] for run in runs}
    assert all(run.returncode == 0 for run in runs) and len(layouts) == 1


def render_labels(gl_ctx, tmp_path, features, **props):
    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [])
    tx, ty = lonlat_to_tile(7.0, 46.0, 10)
    renderer.set_terrain_source(FakeTerrainData(), lod.NodeSet({10: [(int(tx), int(ty))]}))
    scene = Scene()
    scene.store.set("labels.max_distance", 200.0)  # the test camera is ~55 km away
    scene.store.set("labels.size", 72.0)  # 16 px at the 240 px test output
    for pid, value in props.items():
        scene.store.set(pid.replace("__", "."), value)
    renderer.store = scene.store
    data = LabelData(TileCache(tmp_path), (6.9, 45.9, 7.1, 46.1))
    data.allow_download = False
    data.features, data.version = features, 1
    renderer.labels.data = data
    camera = Camera()
    renderer.terrain.finish_loading(camera, camera.view_projection(1.0), 240)
    orbit = OrbitController(camera)
    orbit.frame_bounds(*renderer.scene_bounds())
    orbit.pitch = -35.0
    orbit.apply()
    renderer.terrain.finish_loading(camera, camera.view_projection(320 / 240), 240)
    fbo = gl_ctx.simple_framebuffer((320, 240))
    renderer.render(fbo, 320, 240, camera)
    img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(240, 320, 3)
    return renderer, img[::-1]


def peak(name, lon=7.0, lat=46.0, ground=3500.0, prominence=500.0):
    return LabelFeature(name, "peak", name, lon, lat, ele=ground, prominence=prominence,
                        ground=ground)  # fmt: skip


def white_pixels(img):
    return int(((img > 245).all(axis=2)).sum())


def test_labels_are_drawn_occluded_and_decluttered(gl_ctx, tmp_path):
    renderer, img = render_labels(gl_ctx, tmp_path, [peak("Visible")])
    assert renderer.labels.last_drawn == 1 and white_pixels(img) > 30
    _, plain = render_labels(gl_ctx, tmp_path, [peak("Visible")], labels__visible=False)
    assert white_pixels(plain) < white_pixels(img)
    # below the terrain surface: hidden by the depth test
    renderer, _ = render_labels(gl_ctx, tmp_path, [peak("Buried", ground=200.0)])
    assert renderer.labels.last_drawn == 0
    # two labels at the same spot: only the more important one
    renderer, _ = render_labels(gl_ctx, tmp_path, [peak("Big", prominence=900.0),
                                                   peak("Small", prominence=300.0)])  # fmt: skip
    assert renderer.labels.last_drawn == 1


def test_label_filters(gl_ctx, tmp_path):
    minor = peak("Minor", prominence=50.0)
    renderer, _ = render_labels(gl_ctx, tmp_path, [minor])
    assert renderer.labels.last_drawn == 0  # default minimum prominence 150 m
    renderer, _ = render_labels(gl_ctx, tmp_path, [minor], labels__min_prominence=0.0)
    assert renderer.labels.last_drawn == 1
    renderer, _ = render_labels(gl_ctx, tmp_path, [peak("P")], labels__peaks=False)
    assert renderer.labels.last_drawn == 0
    renderer, _ = render_labels(gl_ctx, tmp_path, [peak("P")], labels__min_elevation=4000.0)
    assert renderer.labels.last_drawn == 0


def test_labels_json_cache_roundtrip(tmp_path):
    cache = TileCache(tmp_path)
    bounds = (1.0, 2.0, 3.0, 4.0)
    path = cache_path(cache, bounds)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(OVERPASS), encoding="utf-8")
    data = LabelData(cache, bounds)
    data.start()
    assert data.wait(10.0) and len(data.features) == 4


def test_names_the_font_can_draw():
    from earthling.data.labels import display_name

    assert display_name({"name": "Mont Blanc"}) == "Mont Blanc"
    assert display_name({"name": "Эльбрус"}) == "Эльбрус"  # Cyrillic: the font has it
    assert display_name({"name": "सगरमाथा", "name:en": "Mount Everest"}) == "Mount Everest"
    assert display_name({"name": "珠穆朗玛峰", "int_name": "Qomolangma"}) == "Qomolangma"
    assert display_name({"name": "ดอยสุเทพ"}) is None  # nothing drawable: no label


def test_labels_hidden_in_clouds(gl_ctx, tmp_path):
    """Step 19.19: a peak behind a closed cloud deck keeps no label."""
    renderer, _ = render_labels(gl_ctx, tmp_path, [peak("Visible")])
    assert renderer.labels.last_drawn == 1
    deck = {"weather__enabled": True, "weather__layer1__base": 6000.0,
            "weather__layer1__thickness": 1500.0, "weather__layer1__coverage": 1.0,
            "weather__layer1__density": 3.0, "weather__layer1__variation": 0.0}  # fmt: skip
    renderer, _ = render_labels(gl_ctx, tmp_path, [peak("Hidden")], **deck)
    assert renderer.labels.last_drawn == 0


NE_COUNTRIES = {"features": [
    {"properties": {"NAME": "Mongolia", "NAME_EN": "Mongolia", "NAME_DE": "Mongolei",
                    "LABEL_X": 104.1, "LABEL_Y": 45.9, "POP_EST": 3.2e6, "ADM0_A3": "MNG"}},
    {"properties": {"NAME": "Chile", "NAME_EN": "Chile", "LABEL_X": -72.3, "LABEL_Y": -38.2}},
]}  # fmt: skip
NE_CITIES = {"features": [
    {"properties": {"name": "Ulaanbaatar", "nameascii": "Ulaanbaatar", "longitude": 106.9,
                    "latitude": 47.9, "pop_max": 1.5e6, "ne_id": 1}},
    {"properties": {"name": "Darkhan", "longitude": 105.9, "latitude": 49.5, "pop_max": 7.5e4}},
]}  # fmt: skip


def test_natural_earth_countries_and_cities(tmp_path):
    """Step 19.21: country names and large cities at any scale, no Overpass for huge areas."""
    from earthling.data.labels import is_large, ne_path

    cache = TileCache(tmp_path)
    for name, data in (("countries", NE_COUNTRIES), ("cities", NE_CITIES)):
        ne_path(cache, name).parent.mkdir(parents=True, exist_ok=True)
        ne_path(cache, name).write_text(json.dumps(data), encoding="utf-8")
    bounds = (37.0, 39.0, 117.0, 56.0)  # Moscow - Beijing: no Overpass query
    assert is_large(bounds) and not is_large((6.8, 45.8, 7.2, 46.1))
    data = LabelData(cache, bounds)
    data.allow_download = False
    features = data.load()
    by_name = {f.name: f for f in features}
    assert set(by_name) == {"Mongolia", "Ulaanbaatar"}  # Chile far away, Darkhan too small
    mongolia = by_name["Mongolia"]
    assert mongolia.kind == "country" and mongolia.display("de") == "Mongolei"
    assert by_name["Ulaanbaatar"].kind == "city" and by_name["Ulaanbaatar"].rank == 4
    from earthling.render.labels import label_texts

    assert label_texts(mongolia, True, "de") == ("MONGOLEI", "")


def test_overpass_timeout_is_not_cached(tmp_path):
    from earthling.data.labels import cache_path

    cache = TileCache(tmp_path)
    bounds = (6.8, 45.8, 7.2, 46.1)
    timeout = {"elements": [], "remark": "runtime error: Query timed out in query"}
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=timeout))
    )
    with pytest.raises(RuntimeError):
        download_labels(cache, bounds, client=client)
    assert not cache_path(cache, bounds).exists()


def test_city_and_country_labels_are_drawn(gl_ctx, tmp_path):
    city = LabelFeature("c", "city", "Town", 7.0, 46.0, ele=3500.0, ground=3500.0,
                        population=2e6, rank=4)  # fmt: skip
    country = LabelFeature("n", "country", "Land", 7.0, 46.0, ele=3500.0, ground=3500.0)
    renderer, _ = render_labels(gl_ctx, tmp_path, [city])
    assert renderer.labels.last_drawn == 0  # off by default
    renderer, _ = render_labels(gl_ctx, tmp_path, [city], labels__cities=True)
    assert renderer.labels.last_drawn == 1
    renderer, _ = render_labels(gl_ctx, tmp_path, [city], labels__cities=True,
                                labels__min_population=5e6)  # fmt: skip
    assert renderer.labels.last_drawn == 0
    renderer, img = render_labels(gl_ctx, tmp_path, [country], labels__countries=True)
    assert renderer.labels.last_drawn == 1 and white_pixels(img) > 30


def test_labels_appear_when_passed(gl_ctx, tmp_path):
    from types import SimpleNamespace

    from earthling.render.labels import LabelLayer
    from earthling.render.progress import ProgressPath
    from earthling.render.shader_library import ShaderLibrary

    # a straight path along x: 0 .. 10 km, a second track continuing from 10 km
    xs = np.linspace(0.0, 5000.0, 51)
    first = SimpleNamespace(enu=np.c_[xs, np.zeros(51), np.zeros(51)], dist=xs, offset_m=0.0)
    second = SimpleNamespace(enu=np.c_[xs + 5000.0, np.zeros(51), np.zeros(51)], dist=xs,
                             offset_m=5000.0)  # fmt: skip
    layer = LabelLayer(gl_ctx, ShaderLibrary(gl_ctx))
    path = SimpleNamespace(tracks=[first, second], total_m=10_000.0)
    path.nearest_distances = lambda xy: ProgressPath.nearest_distances(path, xy)
    layer.path = path
    anchors = np.array([[1000.0, 800.0, 0.0], [7000.0, -300.0, 0.0]])
    assert layer._passed_at(anchors) == pytest.approx([1000.0, 7000.0])
    # without a drawn track the option changes nothing
    renderer, _ = render_labels(gl_ctx, tmp_path, [peak("P")], labels__follow_progress=True)
    assert renderer.labels.last_drawn == 1


def test_shown_labels_keep_their_place_and_fade():
    """A label that is shown is not displaced by one that touches it briefly; changes fade."""
    import numpy as np

    from earthling.render.labels import FADE_S, LabelLayer

    layer = LabelLayer.__new__(LabelLayer)
    layer._shown, layer._shown_time = {}, None
    clash = np.array([[0.0, 1.0], [1.0, 0.0]])  # candidates 0 and 1 overlap
    # only feature 20 (candidate order: by priority) is on screen at first
    layer.time = 0.0
    assert list(layer._stable_placement(np.array([20]), np.array([1.0]), np.zeros((1, 1)), 10)) == [
        1.0
    ]
    # feature 10 (more important) arrives on top of it: 20 stays, 10 waits
    layer.time = 1 / 60
    shown = layer._stable_placement(np.array([10, 20]), np.ones(2), clash, 10)
    assert shown[1] == 1.0 and shown[0] == 0.0
    # a jump in time (a still): plain priority, 10 is shown at once
    layer.time = 5.0
    shown = layer._stable_placement(np.array([10, 20]), np.ones(2), clash, 10)
    assert list(shown) == [1.0, 0.0]
    # 20 alone again: it fades in over FADE_S
    layer.time = 5.0 + FADE_S / 2
    shown = layer._stable_placement(np.array([20]), np.ones(1), np.zeros((1, 1)), 10)
    assert shown[0] == pytest.approx(0.5)


def test_clouds_hide_only_labels_seen_through_a_layer():
    # camera at 7000 m: a summit at 8849 m 20 km away, a valley village at 4800 m under the
    # sea of clouds (4600 - 5400 m), and a summit seen through the cirrus band from below
    camera = np.array([0.0, 0.0, 7000.0])
    rel = np.array([[20000.0, 0.0, 1849.0], [15000.0, 0.0, -2200.0], [20000.0, 0.0, 4000.0]])
    bands = [(4600.0, 5400.0), (9800.0, 10700.0)]
    assert sight_in_clouds(rel, camera, 7000.0, bands).tolist() == [False, True, True]
    assert sight_in_clouds(rel, camera, 7000.0, []).tolist() == [False, False, False]


def test_static_priority():
    from earthling.render.labels import static_priority

    def f(kind, **kw):
        return LabelFeature("x", kind, "X", 7.0, 46.0, **kw)

    p = static_priority([
        f("place", rank=4), f("place", rank=1),
        f("city", population=5e6), f("city", population=1e5),
        f("pass", ele=2500.0), f("pass", ele=1000.0),
        f("hut", track_distance_m=100.0), f("hut", track_distance_m=20_000.0),
        f("peak", prominence=2000.0, ele=4800.0), f("peak", prominence=None, ele=None),
    ])  # fmt: skip
    assert (p[0::2] > p[1::2]).all()  # town over hamlet, big city, high pass, hut on the way
    assert static_priority([]).shape == (0,)


def test_enricher_takes_the_ground_from_the_dem():
    from types import SimpleNamespace

    from earthling.render.labels import LabelLayer

    zooms = []

    def heights_at(lon, lat, z=None):
        zooms.append(z)
        return np.where(np.asarray(lon) < 7.05, 2000.0, np.nan)  # no DEM in the east

    inside = LabelFeature("i", "hut", "Inside", 7.0, 46.0)
    outside = LabelFeature("o", "hut", "Outside", 7.1, 46.0, ele=1800.0)
    segment = SimpleNamespace(lon=np.array([6.9, 7.2]), lat=np.array([46.0, 46.0]))
    run = LabelLayer.make_enricher(None, FRAME, SimpleNamespace(heights_at=heights_at),
                                   [SimpleNamespace(segments=[segment])])  # fmt: skip
    run([inside, outside])
    assert inside.ele == 2000.0 and inside.ground == 2000.0
    assert outside.ground == 1800.0  # no DEM height: the OSM elevation
    assert inside.track_distance_m < 20.0  # (on the track, a chord in the local frame)
    assert 11 in zooms and None in zooms  # (coarse heights for the prominence)
    # without a DEM only the track distances
    lone = LabelFeature("l", "peak", "Lone", 7.0, 46.009)
    LabelLayer.make_enricher(None, FRAME, None, [SimpleNamespace(segments=[segment])])([lone])
    assert lone.ground is None and lone.track_distance_m == pytest.approx(1000.0, rel=0.02)


def test_track_distance_filter_and_distance_fade(gl_ctx, tmp_path):
    far_hut = LabelFeature("h", "hut", "Hut", 7.0, 46.0, ele=3500.0, ground=3500.0,
                           track_distance_m=5000.0)  # fmt: skip
    country = LabelFeature("n", "country", "Land", 7.0, 46.0, ele=3500.0, ground=3500.0,
                           track_distance_m=5000.0)  # fmt: skip
    huts = {"labels__huts": True}
    renderer, _ = render_labels(gl_ctx, tmp_path, [far_hut], labels__max_track_distance=2.0,
                                **huts)  # fmt: skip
    assert renderer.labels.last_drawn == 0
    renderer, _ = render_labels(gl_ctx, tmp_path, [far_hut], labels__max_track_distance=10.0,
                                **huts)  # fmt: skip
    assert renderer.labels.last_drawn == 1
    renderer, _ = render_labels(gl_ctx, tmp_path, [country], labels__countries=True,
                                labels__max_track_distance=2.0)  # fmt: skip
    assert renderer.labels.last_drawn == 1  # countries are not near the track
    renderer, _ = render_labels(gl_ctx, tmp_path, [far_hut], labels__max_distance=10.0, **huts)
    assert renderer.labels.last_drawn == 0  # the camera is ~55 km away: faded out
    renderer.labels.release()


def test_load_sync_for_exports():
    from earthling.render.labels import LabelLayer

    layer = LabelLayer.__new__(LabelLayer)
    layer.data, layer.enricher, layer._version = None, "enrich", -1
    assert not layer.load_sync()

    class Data:
        ready, version, features = False, 0, None

        def load(self, enricher):
            assert enricher == "enrich"
            self.ready, self.version = True, 1
            self.features = [peak("A"), LabelFeature("x", "peak", "No height", 7.0, 46.0)]

    layer.data = Data()
    assert layer.load_sync()
    assert [f.name for f in layer.features] == ["A"]  # (labels need a height to stand on)
    assert layer._ground.tolist() == [3500.0] and layer._lonlat.shape == (1, 2)


def test_a_newcomer_waits_until_a_hidden_label_has_faded():
    from earthling.render.labels import FADE_S, LabelLayer

    layer = LabelLayer.__new__(LabelLayer)
    layer._shown, layer._shown_time, layer.time = {}, None, 0.0
    clash = np.array([[0.0, 1.0], [1.0, 0.0]])
    layer._stable_placement(np.array([20]), np.array([1.0]), np.zeros((1, 1)), 10)
    # feature 20 is hidden now (e.g. behind a ridge); 10 would take its place
    shown = []
    for step in range(1, 40):
        layer.time = step * FADE_S / 10
        shown.append(layer._stable_placement(np.array([10, 20]), np.array([1.0, 0.0]), clash, 10))
    assert shown[0][0] == 0.0 and 0.0 < shown[0][1] < 1.0  # 10 waits while 20 fades out
    first = next(i for i, s in enumerate(shown) if s[0] > 0.0)
    assert shown[first - 1][1] <= 0.02  # 10 comes once 20 is gone
    assert shown[-1][0] == 1.0 and shown[-1][1] == 0.0


def test_a_closed_label_load_stops(tmp_path, monkeypatch, gl_ctx):
    """A project closed or replaced: its label load makes no more requests, enriches and
    publishes nothing; the renderer closes the labels of the project it no longer shows."""
    import httpx

    from helpers import mock_http

    requests = mock_http(monkeypatch, lambda request: httpx.Response(503))
    enriched = []
    data = LabelData(TileCache(tmp_path), (6.9, 45.9, 7.1, 46.1))
    data.close()
    data.start(enricher=enriched.append)
    assert not data.wait(5.0) and data.features is None
    assert requests == [] and enriched == []
    renderer = Renderer(gl_ctx)
    renderer.set_scene(FRAME, [])
    first = LabelData(TileCache(tmp_path), (6.9, 45.9, 7.1, 46.1))
    renderer.set_labels(first)
    renderer.set_labels(LabelData(TileCache(tmp_path), (7.9, 45.9, 8.1, 46.1)))
    assert first._closed.is_set()
    renderer.shutdown()
    assert renderer.labels.data._closed.is_set()
