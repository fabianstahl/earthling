import io
from pathlib import Path

import httpx
import numpy as np
import pytest
import shapely
from PIL import Image

from earthling.core.config import Config, ConfigError, Project
from earthling.data.cache import TileCache
from earthling.data.check import check_dem, check_imagery
from earthling.data.dem import DEM_SOURCES, StacDemSource, WmsDemSource
from earthling.data.project_sources import register_project_sources
from earthling.data.providers import PROVIDERS
from test_wms_dem import RampWms, ramp_server

TOML = """
[sources]
imagery = ["my_ortho", "esri_world_imagery"]
dem = ["my_wms_dem", "copernicus_glo30"]

[[providers]]
id = "my_ortho"
kind = "imagery"
name = "My orthophotos"
url = "https://tiles.example/{z}/{y}/{x}.jpeg"
max_zoom = 18
coverage = "region.geojson"
attribution = "Imagery: example.org"
license = "CC BY 4.0"
placeholder_md5 = ["0123456789abcdef0123456789abcdef"]

[[providers]]
id = "my_wms_dem"
kind = "dem"
type = "wms"
url = "https://wms.example/wms"
layer = "DTM"
coverage = "austria"

[[providers]]
id = "my_stac_dem"
kind = "dem"
type = "stac"
url = "https://stac.example/collections/dtm/items"
asset_suffix = "_1m.tif"
"""


@pytest.fixture
def project(tmp_path):
    (tmp_path / "region.geojson").write_text(
        shapely.to_geojson(shapely.box(11.0, 47.0, 11.5, 47.5)), encoding="utf-8"
    )
    return Project(tmp_path, Config.from_toml(TOML))


def test_project_providers_are_registered(project):
    sources = register_project_sources(project)
    assert [s.id for s in sources] == ["my_ortho", "my_wms_dem", "my_stac_dem"]
    ortho = PROVIDERS["my_ortho"]
    assert ortho.tile_url(15, 1, 2) == "https://tiles.example/15/2/1.jpeg"
    assert ortho.coverage_area.contains_lonlat(11.2, 47.2)  # path relative to the project
    assert not ortho.coverage_area.contains_lonlat(12.0, 47.2)
    assert ortho.license.attribution == "Imagery: example.org"
    assert ortho.is_placeholder(b"") is False
    wms = DEM_SOURCES["my_wms_dem"]
    assert isinstance(wms, WmsDemSource) and wms.layer == "DTM" and wms.oversample == 2
    assert wms.coverage_area.contains_lonlat(11.39, 47.26)  # Innsbruck, shipped "austria"
    stac = DEM_SOURCES["my_stac_dem"]
    assert isinstance(stac, StacDemSource) and stac.asset_suffix == "_1m.tif"


@pytest.mark.parametrize(
    "snippet, message",
    [
        ('kind = "dem"\nurl = "https://x/{z}/{x}/{y}"', "type = 'wms' or 'stac'"),
        ('kind = "imagery"\nurl = "https://x/{z}/{x}"', "placeholders"),
        ('kind = "dem"\ntype = "wms"\nurl = "https://x"', "need a layer"),
    ],
)
def test_invalid_provider_definitions(snippet, message):
    with pytest.raises(ConfigError, match=message):
        Config.from_toml(f'[[providers]]\nid = "bad"\n{snippet}\n')


def jpeg(color, noise=True):
    rng = np.random.default_rng(1)
    arr = np.full((256, 256, 3), color, dtype=np.uint8)
    if noise:
        arr = np.clip(arr + rng.integers(-20, 20, arr.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def test_check_imagery_reports_tiles_placeholders_and_errors(project, tmp_path):
    register_project_sources(project)
    ortho = PROVIDERS["my_ortho"]

    def handler(request):
        z = int(request.url.path.split("/")[1])
        if z == 18:
            return httpx.Response(404)
        if z == 16:
            return httpx.Response(200, content=jpeg((255, 255, 255), noise=False))
        return httpx.Response(
            200, content=jpeg((90, 120, 80)), headers={"content-type": "image/jpeg"}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    report = check_imagery(ortho, 11.2, 47.2, client=client, out_dir=tmp_path / "out")
    text = report.text()
    assert "is inside" in text
    assert not report.ok and any("z18" in p for p in report.problems)
    assert any("uniform tile" in w and "placeholder_md5" in w for w in report.warnings)
    assert (tmp_path / "out" / "check_my_ortho.png").exists()
    outside = check_imagery(ortho, 12.0, 47.2, client=client, zooms=[10])
    assert any("outside the coverage" in p for p in outside.problems)


def test_check_dem_alignment_and_lod(tmp_path):
    source, reference = RampWms(), RampWms()
    reference.id = "test_wms_reference"
    client = httpx.Client(transport=httpx.MockTransport(ramp_server([])))
    report = check_dem(source, 6.87, 45.92, TileCache(tmp_path), client=client,
                       zooms=(12, 13), reference=reference)  # fmt: skip
    text = report.text()
    assert report.ok, text
    assert "100.0 % data" in text
    assert "LOD z13 vs z14" in text and "rms 0.00 m" in text
    assert not report.warnings


def test_check_source_cli_rejects_unknown_ids(capsys):
    from earthling.cli import main

    assert main(["check-source", str(Path("examples/alps_demo")), "nope"]) == 2
    assert "unknown source 'nope'" in capsys.readouterr().err
