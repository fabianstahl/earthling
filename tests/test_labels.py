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
from earthling.render.labels import FontAtlas, signed_distance
from earthling.render.renderer import Renderer
from test_terrain import FRAME, FakeTerrainData

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
