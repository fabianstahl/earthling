import pytest

from earthling.core.config import SAMPLE_CONFIG, Config, ConfigError, Project


def test_sample_config_parses():
    cfg = Config.from_toml(SAMPLE_CONFIG)
    assert cfg.area.border_km == 15.0
    assert cfg.area.max_imagery_zoom == 17
    assert cfg.area.zone_for_distance(1.0).imagery_zoom == 17
    assert cfg.area.zone_for_distance(10.0).imagery_zoom == 13
    assert cfg.area.zone_for_distance(20.0) is None
    assert cfg.project.timezone == "Europe/Paris"


def test_defaults_when_empty():
    cfg = Config.from_toml("")
    assert cfg.sources.dem == ("copernicus_glo30",)


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("[project]\ntimezone = 'Mars/Olympus'", "unknown timezone"),
        ("[area]\nborder_km = 20\n", "outermost zone"),
        (
            "[area]\nborder_km = 1\nzones = [{within_km=5, imagery_zoom=1, dem_zoom=1},"
            "{within_km=2, imagery_zoom=1, dem_zoom=1}]",
            "increasing",
        ),
        ("[area]\nbogus = 1", "bogus"),
        ("not toml [", "invalid TOML"),
    ],
)
def test_invalid_configs(text, fragment):
    with pytest.raises(ConfigError, match=fragment):
        Config.from_toml(text)


def test_project_create_and_load(tmp_path):
    folder = tmp_path / "hike"
    project = Project.create(folder)
    assert project.config_path.exists()
    assert project.gpx_dir == (folder / "gpx").resolve()
    assert project.gpx_dir.is_dir()
    assert Project.load(folder).config == project.config


def test_load_missing_config(tmp_path):
    with pytest.raises(ConfigError, match="contains no"):
        Project.load(tmp_path)
