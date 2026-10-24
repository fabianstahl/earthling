"""The command line: argument parsing, exit codes and messages, and the offline commands end to
end on a tiny synthetic project (terrain stub instead of tiles; GL tests skip without 4.3)."""

import httpx
import numpy as np
import pytest
from PIL import Image

from earthling import scenekit as sk
from earthling.cli import build_parser, main
from earthling.core.geo import lonlat_to_tile
from earthling.render import lod
from earthling.session import Session
from helpers import FakeTerrainData, gpx_text, make_project, walk
from test_project_sources import jpeg

COMMANDS = ("gui", "plan", "fetch", "render", "stills", "qa", "check-source")


class Terrain(FakeTerrainData):
    """The renderer's terrain stub plus the heights scene scripts ask for."""

    def height_at(self, lon, lat, z=None):
        return 1500.0

    def heights_at(self, lon, lat, z=None):
        return np.full(np.shape(lon), 1500.0)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """No tile data: a terrain stub instead (the network is refused by the conftest)."""
    tx, ty = lonlat_to_tile(7.0, 46.0, 10)
    monkeypatch.setattr(Session, "terrain_data", lambda self, on_demand=True: Terrain())
    monkeypatch.setattr(Session, "terrain_nodes", lambda self: lod.NodeSet({10: [(tx, ty)]}))
    monkeypatch.setattr(Session, "labels", None)  # (no OSM label download)
    yield
    sk.set_current(None)


@pytest.fixture
def project(tmp_path):
    day = walk(46.0, 7.0, "2026-07-01T06:00:00", 30, north_m=40.0)
    return make_project(tmp_path / "tiny", {"day.gpx": gpx_text({"Day": day})})


@pytest.fixture
def scene_file(project):
    """A 1 s scene at 10 fps made with the scene kit: the line is drawn, the camera follows."""
    journey = sk.load_journey(project)
    scene = sk.new_scene(journey, 1.0, fps=10.0)
    head_of = sk.head_keys_for(scene, journey, 1, 0.0, 1.0, realtime=False)
    times = np.arange(0.0, 1.0 + 1e-9, 0.25)
    sk.key_shots(scene, sk.follow_shots(journey, times, head_of, distance=2500.0, pitch=-35.0,
                                        side=30.0, sigma_s=0.3, heading_sigma_s=0.5))  # fmt: skip
    return sk.save(scene, "s01")


# --- parsing -----------------------------------------------------------------------------------
def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit) as exit_:
        main(["-h"])
    assert exit_.value.code == 0
    out = capsys.readouterr().out
    assert all(command in out for command in COMMANDS)


@pytest.mark.parametrize(
    "argv", [["nope"], ["render"], ["fetch", "p", "--kind", "photos"], ["qa", "v", "--frames", "x"]]
)
def test_bad_arguments_exit_with_usage(argv, capsys):
    with pytest.raises(SystemExit) as exit_:
        main(argv)
    assert exit_.value.code == 2
    assert "usage: earthling" in capsys.readouterr().err


def test_parser_defaults():
    args = build_parser().parse_args(["render", "p"])
    assert (args.scene, args.preset, args.start, args.end) == (
        "scene.json",
        "prores4444",
        0.0,
        None,
    )
    args = build_parser().parse_args(["stills", "p", "1", "2.5", "--set", "a=1", "--set", "b=2"])
    assert args.times == [1.0, 2.5] and args.set == ["a=1", "b=2"] and args.size == "960x540"
    assert build_parser().parse_args(["fetch", "p"]).kind == "all"


def test_gui_is_the_default(monkeypatch):
    import earthling.app.main

    calls = []

    def run_gui(project=None, dev_mode=False):
        calls.append((project, dev_mode))
        return 0

    monkeypatch.setattr(earthling.app.main, "run_gui", run_gui)
    assert main([]) == 0
    assert main(["gui", "some/project", "--dev"]) == 0
    assert calls == [(None, False), ("some/project", True)]


# --- plan and fetch ----------------------------------------------------------------------------
def test_plan(project, capsys):
    assert main(["plan", str(project)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("1 track(s), AOI bounds (6.9")
    assert "Esri World Imagery" in out and "Copernicus DEM GLO-30" in out


def test_missing_project_or_tracks(tmp_path, capsys):
    for command in ("plan", "fetch", "render", "stills"):
        assert main([command, str(tmp_path / "nowhere")]) == 1
        assert "contains no earthling.toml" in capsys.readouterr().err
    empty = make_project(tmp_path / "empty", {})
    assert main(["plan", str(empty)]) == 1
    assert "no tracks found" in capsys.readouterr().err
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "earthling.toml").write_text("[project]\ntimezone = 'Mars/Base'\n")
    assert main(["plan", str(tmp_path / "broken")]) == 1
    assert "unknown timezone 'Mars/Base'" in capsys.readouterr().err


class FakeJob:
    def __init__(self, name, failed=0, errors=(), interrupt=False):
        self.name, self.failed, self.errors, self.interrupt = name, failed, list(errors), interrupt
        self.cancelled = False

    def run(self, progress):
        from earthling.data.downloader import DownloadProgress

        if self.interrupt:
            raise KeyboardInterrupt
        progress(DownloadProgress(total=2, done=2, downloaded=2 - self.failed,
                                  failed=self.failed, bytes=3_000_000))  # fmt: skip
        return self

    def cancel(self):
        self.cancelled = True


def test_fetch_runs_the_jobs_and_reports_failures(project, monkeypatch, capsys):
    import earthling.data.jobs

    asked, jobs = [], [FakeJob("imagery"), FakeJob("dem", failed=1, errors=["z12 1/2: 503"])]

    def download_jobs(session, kinds, max_zoom):
        asked.append((kinds, max_zoom))
        return jobs

    monkeypatch.setattr(earthling.data.jobs, "download_jobs", download_jobs)
    assert main(["fetch", str(project)]) == 1  # one tile failed
    out, err = capsys.readouterr()
    assert "cache: " in out and str(project / "cache") in out
    assert "2/2 tiles, 1 new, 0 unavailable, 1 failed, 3.0 MB" in out
    assert "error: z12 1/2: 503" in err
    jobs[1].failed = 0
    assert main(["fetch", str(project), "--kind", "dem", "--max-zoom", "11"]) == 0
    assert asked == [({"imagery", "dem", "labels"}, None), ({"dem"}, 11)]


def test_fetch_can_be_cancelled(project, monkeypatch, capsys):
    import earthling.data.jobs

    job = FakeJob("imagery", interrupt=True)
    monkeypatch.setattr(earthling.data.jobs, "download_jobs", lambda *a: [job])
    assert main(["fetch", str(project)]) == 130
    assert job.cancelled and "cancelled" in capsys.readouterr().out


# --- render, stills, qa ------------------------------------------------------------------------
def test_render_input_errors(project, scene_file, capsys):
    assert main(["render", str(project), "--preset", "gif"]) == 2
    assert "unknown preset 'gif'" in capsys.readouterr().err
    assert main(["render", str(project), "--scene", "scenes/nope.json"]) == 1
    assert "cannot read scene" in capsys.readouterr().err
    for size in ("big", "1920", "0x1080", "axb"):
        assert main(["render", str(project), "--scene", "scenes/s01.json", "--resolution",
                     size]) == 2  # fmt: skip
        assert f"bad resolution '{size}'" in capsys.readouterr().err


def test_render_lists_presets(capsys):
    assert main(["render", "unused", "--list-presets"]) == 0
    out = capsys.readouterr().out
    assert "h264" in out


def test_stills_input_errors(project, scene_file, capsys):
    assert main(["stills", str(project), "--scene", "scenes/s01.json"]) == 2
    assert "no times given" in capsys.readouterr().err
    assert main(["stills", str(project), "0", "--scene", "scenes/s01.json", "--size", "1x"]) == 2
    assert "bad resolution '1x'" in capsys.readouterr().err
    assert main(["stills", str(project), "0"]) == 1  # (the default scene.json is not there)
    assert "cannot read scene" in capsys.readouterr().err


def test_qa_missing_video(tmp_path, capsys):
    assert main(["qa", str(tmp_path / "none.mp4")]) == 1
    assert "no such file" in capsys.readouterr().err


def test_render_then_qa(gl_ctx, project, scene_file, capsys):
    out = project / "renders" / "clip.mp4"
    out.parent.mkdir()
    argv = ["render", str(project), "--scene", "scenes/s01.json", "--preset", "h264",
            "--resolution", "64x36", "--start", "0.2", "--end", "0.6",
            "--out", str(out)]  # fmt: skip
    assert main(argv) == 0
    text = capsys.readouterr().out
    assert "rendering s01.json 0.2-0.6 s at 64x36 10 fps" in text
    assert "wrote 4 frames" in text and out.stat().st_size > 0  # [start, end)
    assert main(["qa", str(out), "--frames", "4", "--out", str(project / "qa")]) == 0
    assert "clip: 4 frames at 10 fps" in capsys.readouterr().out
    assert (project / "qa" / "clip_sheet.jpg").exists()
    # without --out the video lands next to the scene file
    assert main(["render", str(project), "--scene", "scenes/s01.json", "--preset", "h264",
                 "--resolution", "32x18", "--end", "0.1"]) == 0  # fmt: skip
    assert (project / "scenes" / "s01.mp4").exists()


def test_stills(gl_ctx, project, scene_file, tmp_path, capsys):
    out = tmp_path / "stills"
    argv = ["stills", str(project), "0", "0.9", "--scene", "scenes/s01.json", "--size", "64x36",
            "--out", str(out), "--tag", "_a"]  # fmt: skip
    assert main(argv) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == ["s01_a_000.00.png", "s01_a_000.90.png", "s01_a_sheet.jpg"]
    with Image.open(out / "s01_a_000.90.png") as image:
        assert image.size == (64, 36)
        first = np.asarray(image.convert("RGB"), dtype=np.float64)
    assert first.mean() > 5.0  # not black
    # a property fixed for all stills, N evenly spaced times, the default folder
    argv = ["stills", str(project), "--scene", "scenes/s01.json", "--size", "64x36", "--auto",
            "3", "--set", "post.exposure=-4", "--sequence", "0"]  # fmt: skip
    assert main(argv) == 0
    default = project / "renders" / "stills"
    assert sorted(p.name for p in default.glob("s01_*.png")) == [
        "s01_000.00.png", "s01_000.45.png", "s01_000.90.png"]  # fmt: skip
    with Image.open(default / "s01_000.90.png") as image:
        darker = np.asarray(image.convert("RGB"), dtype=np.float64)
    assert darker.mean() < first.mean()
    assert str(default / "s01_sheet.jpg") in capsys.readouterr().out


# --- check-source ------------------------------------------------------------------------------
def test_check_source_imagery(project, monkeypatch, tmp_path, capsys):
    import earthling.data.check

    urls = []

    def tiles(request):
        urls.append(str(request.url))
        return httpx.Response(
            200, content=jpeg((90, 120, 80)), headers={"content-type": "image/jpeg"}
        )

    client = httpx.Client(transport=httpx.MockTransport(tiles))
    monkeypatch.setattr(earthling.data.check, "http_client", lambda: client)
    argv = ["check-source", str(project), "esri_world_imagery", "--at", "7.0,46.0", "--out",
            str(tmp_path / "sheet")]  # fmt: skip
    assert main(argv) == 0
    assert "imagery provider esri_world_imagery" in capsys.readouterr().out
    x, y = (int(v) for v in lonlat_to_tile(7.0, 46.0, 10))
    assert any(f"/10/{y}/{x}" in url for url in urls)  # at the given location
    assert (tmp_path / "sheet" / "check_esri_world_imagery.png").exists()


def test_check_source_dem_at_the_first_track_point(project, monkeypatch, capsys):
    import earthling.data.check

    calls = []

    def fake_check(source, lon, lat, cache):
        calls.append((source.id, lon, lat, cache.root))
        report = earthling.data.check.CheckReport()
        report.problem("no data at z14")
        return report

    monkeypatch.setattr(earthling.data.check, "check_dem", fake_check)
    assert main(["check-source", str(project), "copernicus_glo30"]) == 1  # the report failed
    [(source, lon, lat, root)] = calls
    assert (source, lon, lat) == ("copernicus_glo30", 7.0, 46.0)
    assert root == project / "cache"
    assert "no data at z14" in capsys.readouterr().out


def test_check_source_bad_project(tmp_path, capsys):
    assert main(["check-source", str(tmp_path), "copernicus_glo30"]) == 1
    assert "contains no earthling.toml" in capsys.readouterr().err
