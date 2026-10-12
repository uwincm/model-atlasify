import json
from pathlib import Path

import pystac
import pytest

from modelatlasify.core.catalog import _union_bbox, iso_duration
from modelatlasify.core.processing import PipelineOptions, run_pipeline
from modelatlasify.core.validation import check_links, validate_catalog


@pytest.fixture
def atlas(wrf_run, tmp_path) -> Path:
    out = tmp_path / "atlas"
    run_pipeline(PipelineOptions(inputs=[wrf_run], output=out, plots=False), report=lambda m: None)
    return out


def test_catalog_structure(atlas):
    cat = pystac.Catalog.from_file(str(atlas / "catalog.json"))
    (col,) = list(cat.get_children())
    assert isinstance(col, pystac.Collection)
    assert col.id == "run01"
    assert "WRF v4.5.1" in col.description
    assert col.extra_fields["modelatlasify:model"] == "WRF"
    assert col.extra_fields["wrf:map_proj"] == 1
    assert len(col.extent.spatial.bboxes) == 1
    assert col.extent.temporal.intervals[0][0].isoformat().startswith("2026-10-06T00:00")
    assert set(col.item_assets) >= {"surface_wind_speed", "precipitation", "precipitation_interval"}

    items = sorted(col.get_items(), key=lambda i: i.datetime)
    assert [i.id for i in items] == [f"d01-20261006T0{h}0000Z" for h in range(4)]
    first, last = items[0], items[-1]
    assert "precipitation_interval" not in first.assets  # undefined at the first time
    assert set(last.assets) == {"surface_wind_speed", "precipitation", "precipitation_interval"}
    assert last.properties["forecast:horizon"] == "PT3H"
    assert last.properties["forecast:reference_datetime"] == "2026-10-06T00:00:00Z"
    assert last.properties["proj:shape"] == [30, 40]
    assert last.properties["proj:wkt2"].startswith("PROJCRS")
    assert last.geometry["type"] == "Polygon" and len(last.bbox) == 4

    tif = last.assets["precipitation_interval"]
    assert tif.media_type.startswith("image/tiff; application=geotiff")
    assert tif.roles == ["data"]
    assert tif.extra_fields["start_datetime"] == "2026-10-06T02:00:00Z"
    assert tif.extra_fields["raster:bands"][0]["unit"] == "mm"


def test_hrefs_are_relative_and_resolve(atlas):
    item_json = next(atlas.glob("run01/items/*/d01-*.json"))
    raw = json.loads(item_json.read_text())
    assert all(not a["href"].startswith("/") for a in raw["assets"].values())
    n, missing = check_links(atlas / "catalog.json")
    assert n == 1 + 1 + 4 and missing == []


def test_catalog_validates_against_stac_schemas(atlas, network):
    report = validate_catalog(atlas / "catalog.json", schemas=True)
    assert report.schema_skipped_reason is None
    assert report.schema_errors == []
    assert report.ok


def test_plots_are_generated_and_cataloged(wrf_run, tmp_path):
    out = tmp_path / "atlas"
    res = run_pipeline(PipelineOptions(inputs=[wrf_run], output=out, variables=["surface_wind_speed"]),
                       report=lambda m: None)
    assert res.n_plots == 4
    item = next(pystac.Catalog.from_file(str(out / "catalog.json")).get_items(recursive=True))
    png = item.assets["surface_wind_speed_map"]
    assert png.media_type == "image/png" and "visual" in png.roles
    assert Path(png.get_absolute_href()).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_iso_duration():
    from datetime import timedelta

    assert iso_duration(timedelta(hours=36)) == "PT36H"
    assert iso_duration(timedelta(minutes=90)) == "PT1H30M"
    assert iso_duration(timedelta(0)) == "PT0S"


def test_union_bbox_handles_antimeridian():
    d01 = [113.6, -31.6, -65.6, 39.7]  # crosses the antimeridian
    d02 = [154.9, -5.2, 175.0, 5.0]
    assert _union_bbox([d01, d02]) == [113.6, -31.6, -65.6, 39.7]
    assert _union_bbox([[-10, 0, 10, 5], [20, -5, 30, 1]]) == [-10, -5, 30, 5]
