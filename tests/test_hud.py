from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from earthling.core.gpx import ascent_descent, cumulative_ascent_descent
from earthling.core.scene import Scene
from earthling.render.hud import HikeStats, HudLayer, format_date, format_number
from earthling.render.labels import FontAtlas
from earthling.render.shader_library import ShaderLibrary


def fake_path(with_time=True):
    """Two tracks of 11 points: day 1 climbs 1000 -> 2000 m, day 2 descends to 1500 m.
    Every point is 1 km apart (the drawn track counts double: exaggeration)."""
    ground = np.arange(22, dtype=float) * 1000.0
    ele = np.r_[np.linspace(1000, 2000, 11), np.linspace(2000, 1500, 11)]
    start1 = datetime(2026, 7, 1, 7, 0, tzinfo=UTC).timestamp()
    start2 = datetime(2026, 7, 2, 6, 0, tzinfo=UTC).timestamp()
    time = np.r_[start1 + np.arange(11) * 1800.0, start2 + np.arange(11) * 1800.0]
    if not with_time:
        time = np.full(22, np.nan)
    return SimpleNamespace(
        dist=ground * 2.0, ground=ground, ele=ele, time=time,
        track_index=np.r_[np.zeros(11, int), np.ones(11, int)],
    )  # fmt: skip


def test_cumulative_ascent_matches_totals():
    ele = np.array([1000, 1003, 1010, 1004, 1020, np.nan, 1008, 1030.0])
    up, down = cumulative_ascent_descent(ele)
    assert (up[-1], down[-1]) == ascent_descent(ele)
    assert (np.diff(up) >= 0).all() and up[0] == 0.0


def test_hike_stats_by_day():
    stats = HikeStats(fake_path(), "Europe/Paris")
    start = stats.at(0.0)
    assert start.day == 1 and start.distance_m == 0.0 and start.time.hour == 9  # CEST
    mid = stats.at(10_000.0)  # progress axis: 5 km walked on day 1
    assert mid.distance_m == pytest.approx(5000.0) and mid.elevation_m == pytest.approx(1500.0)
    assert mid.ascent_m == pytest.approx(500.0)
    day2 = stats.at(2.0 * 16_000.0)  # 5 km into day 2
    assert day2.day == 2 and day2.day_distance_m == pytest.approx(5000.0)
    assert day2.distance_m == pytest.approx(16_000.0)
    assert day2.ascent_m == pytest.approx(1000.0) and day2.day_ascent_m == pytest.approx(0.0)
    assert day2.day_descent_m == pytest.approx(250.0)
    assert day2.time.day == 2 and day2.time.hour == 10
    end = stats.at(1e9)  # clamped
    assert end.distance_m == pytest.approx(21_000.0) and end.day == 2
    ground, ele, prog = stats.profile(day=2)
    assert len(ground) == 11 and ground[0] == 11_000.0


def test_hike_stats_without_times_counts_tracks():
    stats = HikeStats(fake_path(with_time=False))
    assert stats.at(0.0).time is None
    assert stats.at(30_000.0).day == 2


def test_formatting():
    t = datetime(2026, 7, 14, 9, 5)
    assert format_date(t, "long") == "14 July 2026"
    assert format_date(t, "short") == "14 Jul" and format_date(t, "iso") == "2026-07-14"
    assert format_number(12345.678, 1) == "12 345.7"


def test_hud_draws_panel_and_attribution(gl_ctx):
    scene = Scene()
    store = scene.store
    atlas = FontAtlas(gl_ctx)
    hud = HudLayer(gl_ctx, ShaderLibrary(gl_ctx), lambda: atlas)
    hud.attribution = "Imagery: test"
    fbo = gl_ctx.simple_framebuffer((960, 540))
    store.set("stats.scale", 2.0)

    def draw(**props):
        for pid, value in props.items():
            store.set(pid.replace("__", "."), value)
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        hud.render(fbo, 960, 540, store, fake_path(), 20_000.0, "UTC")
        img = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(540, 960, 3)[::-1]
        return img.astype(int)

    img = draw()
    assert img.max() == 0 and hud.last_stats is None  # hidden by default
    img = draw(stats__visible=True)
    assert hud.last_stats.day == 1 and hud.last_stats.distance_m == pytest.approx(10_000.0)
    panel = img[270:, :480]  # bottom left
    assert (panel.sum(axis=2) > 600).sum() > 50  # white text
    accent = (panel[..., 0] > 120) & (panel[..., 2] < 90)  # orange walked profile
    assert accent.sum() > 20
    assert img[:, 480:].max() == 0  # nothing on the right half yet
    img = draw(stats__attribution=True)
    assert img[510:, 760:].max() > 100  # credits bottom right
    img = draw(stats__position="top_right")
    assert img[:270, 480:].sum() > img[270:, :480].sum()


def test_session_attribution():
    from earthling.core.config import Project
    from earthling.core.session import Session

    text = Session(Project.load(Path("examples/alps_demo"))).attribution()
    assert "Esri" in text and "Copernicus" in text and "OpenStreetMap" in text


def test_first_frame_text_survives_atlas_repacks(gl_ctx):
    """New glyphs in later texts repack the atlas; earlier texts must not keep stale uvs."""
    store = Scene().store
    store.set("stats.visible", True)
    store.set("stats.show_time", True)
    hud = HudLayer(gl_ctx, ShaderLibrary(gl_ctx), lambda: atlas)
    atlas = FontAtlas(gl_ctx)
    fbo = gl_ctx.simple_framebuffer((960, 540))

    def draw():
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        hud.render(fbo, 960, 540, store, fake_path(), 20_000.0, "UTC")
        return np.frombuffer(fbo.read(components=3), dtype=np.uint8).astype(int)

    first = draw()  # fresh atlas: glyphs get added (and repacked) while drawing
    second = draw()  # all glyphs known
    assert np.abs(first - second).max() <= 1


def test_german_formatting():
    from earthling.render.hud import word

    t = datetime(2026, 3, 4, 9, 5)
    assert format_date(t, "long", "de") == "4. März 2026"
    assert format_date(t, "short", "de") == "4. Mär."
    assert format_number(12345.678, 1, "de") == "12.345,7"
    assert format_number(2375.0, 0, "de") == "2.375"
    assert word("de", "day").format(n=22) == "Tag 22"
    assert word("de", "time").format(t=t) == "09:05 Uhr"
