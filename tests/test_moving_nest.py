"""Moving nests: every product follows the nest's position at each time."""

import json
import shutil

import numpy as np
import pytest
import rasterio

from conftest import T0, run_out
from wrf_fixture import hourly, write_wrfout
from modelatlasify.cli import main
from modelatlasify.models.wrf.adapter import WRFAdapter

DX2 = 4000.0
MOVE = (2, -1)  # cols east, rows north per hour


@pytest.fixture
def moving_run(tmp_path):
    run = tmp_path / "run01"
    write_wrfout(run, hourly(T0, 4))
    write_wrfout(run, hourly(T0, 4), domain=2, nx=20, ny=16, dx=DX2, move=MOVE)
    return run


def test_moving_nest_is_discovered_with_every_position(moving_run):
    adapter = WRFAdapter()
    d01, d02 = adapter.discover(moving_run)
    assert adapter.skipped_files == []
    assert not d01.moving and d02.moving and len(d02.valid_times) == 4
    assert len(d02.distinct_grids()) == 4
    for k in range(4):
        t0, tk = d02.grid_at(0).transform, d02.grid_at(k).transform
        assert tk.c - t0.c == pytest.approx(k * MOVE[0] * DX2, abs=1.0)
        assert tk.f - t0.f == pytest.approx(k * MOVE[1] * DX2, abs=1.0)
        assert d02.grid_at(k).pixel_offset(d02.grid_at(0)) == (-k * MOVE[1], k * MOVE[0])


def test_outer_domain_still_rejects_a_shifted_file(tmp_path):
    run = tmp_path / "run"
    write_wrfout(run, hourly(T0, 3))
    shifted = write_wrfout(tmp_path / "moved", hourly(T0, 3), move=(5, 0))
    shutil.copy(shifted[2], run / shifted[2].name)  # replace the 3rd file
    adapter = WRFAdapter()
    (ds,) = adapter.discover(run)
    assert len(ds.valid_times) == 2 and not ds.moving
    assert len(adapter.skipped_files) == 1


def test_products_follow_the_nest(moving_run, tmp_path):
    out = tmp_path / "atlas"
    assert main([str(moving_run), "-o", str(out), "--no-validate"]) == 0
    run_dir = run_out(out, moving_run)
    items = sorted((run_dir / "items").glob("d02-*"))
    assert len(items) == 4
    origins, bboxes = [], []
    for k, item_dir in enumerate(items):
        item = json.loads((item_dir / f"{item_dir.name}.json").read_text())
        with rasterio.open(item_dir / "surface_wind_speed.tif") as src:
            assert list(src.transform)[:6] == pytest.approx(item["properties"]["proj:transform"])
            origins.append((src.transform.c, src.transform.f))
            wind = src.read(1)
        bboxes.append(item["bbox"])
        # Wind is attached to the ground: 5 (1 + i/nx), i on the first lattice.
        np.testing.assert_allclose(wind[0], 5 * (1 + (np.arange(20) + k * MOVE[0]) / 20), rtol=1e-5)
    assert len(set(origins)) == 4 and bboxes[0] != bboxes[-1]

    collection = json.loads((run_dir / "collection.json").read_text())
    (d02,) = [d for d in collection["modelatlasify:domains"] if d["id"] == "d02"]
    assert d02["moving"] and d02["n_positions"] == 4
    assert d02["bbox"][0] <= min(b[0] for b in bboxes) and d02["bbox"][2] >= max(b[2] for b in bboxes)

    # Interval precipitation is differenced at the same place, not the same index.
    with rasterio.open(items[2] / "precipitation_interval.tif") as src:
        amount = src.read(1)
    i_abs = np.arange(20) + 2 * MOVE[0]
    expected = (1 + i_abs / 20) + 0.5  # RAINNC + RAINC over one hour
    np.testing.assert_allclose(amount[:-1, :-MOVE[0]],
                               np.broadcast_to(expected[:-MOVE[0]], (15, 18)), rtol=1e-5)
    # Cells that entered the nest during the hour have no earlier value.
    assert np.isnan(amount[-1, :]).all()  # nest moved south by one row: bottom row is new
    assert np.isnan(amount[:, -MOVE[0]:]).all()  # and east by two columns

    images = list((run_dir / "model-images" / "full").rglob("*.png"))
    assert len(images) == len(list(run_dir.glob("items/d01-*/*.tif")))


def test_composite_task_uses_the_nest_position_at_each_time(moving_run):
    from modelatlasify.core.processing import _composite_tasks

    d01, d02 = WRFAdapter().discover(moving_run)
    meta = type("M", (), {"start_time": None, "end_time": None})()
    products = {}
    for ds in (d01, d02):
        for t in ds.valid_times:
            products[(ds.dataset_id, t, "surface_wind_speed")] = (f"{ds.dataset_id}{t:%H}.tif", meta)
    tasks = _composite_tasks(moving_run, [d01, d02], ["surface_wind_speed"], products, None, "uwincm")
    assert [t.nests[0][1] for t in tasks] == [0, 1, 2, 3]
