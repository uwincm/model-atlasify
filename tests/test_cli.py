import re
import shutil
import subprocess

import pytest

from conftest import T0
from wrf_fixture import hourly, write_wrfout
from modelatlasify.cli import main


def test_cli_end_to_end(wrf_run, tmp_path, capsys):
    out = tmp_path / "atlas"
    rc = main([str(wrf_run), "--output", str(out), "--no-plots", "--no-validate"])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == str(out.resolve() / "catalog.json")
    assert "Found WRF run 'run01'" in captured.err
    assert "d01: 4 file(s), 4 time(s)" in captured.err
    assert re.search(r"georeference fit error [\d.]+ m\.\n", captured.err)
    assert "[run01 d01" not in captured.err  # no per-time lines; a progress bar instead
    assert (out / "run01" / "collection.json").is_file()
    assert len(list(out.glob("run01/items/*/*.tif"))) == 4 + 4 + 3


def test_cli_variable_selection(wrf_run, tmp_path):
    out = tmp_path / "atlas"
    assert main([str(wrf_run), "-o", str(out), "--variables", "wind", "--no-plots",
                 "--no-validate"]) == 0
    assert {p.name for p in out.glob("run01/items/*/*.tif")} == {"surface_wind_speed.tif"}


def test_cli_unknown_variable(wrf_run, tmp_path, capsys):
    assert main([str(wrf_run), "-o", str(tmp_path / "a"), "--variables", "snow"]) == 2
    assert "Unknown variable 'snow'" in capsys.readouterr().err


def test_cli_missing_input_variables_is_a_clean_error(tmp_path, capsys):
    run = tmp_path / "run"
    write_wrfout(run, hourly(T0, 2), include=("U10", "V10"))
    rc = main([str(run), "-o", str(tmp_path / "a"), "--no-plots"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "error: Cannot compute precipitation" in err
    assert "Traceback" not in err


def test_cli_missing_directory(tmp_path, capsys):
    assert main([str(tmp_path / "missing"), "-o", str(tmp_path / "a")]) == 1
    assert "Input directory does not exist" in capsys.readouterr().err


def test_cli_multiple_runs_and_domain_filter(tmp_path):
    for name in ("runA", "runB"):
        write_wrfout(tmp_path / name, hourly(T0, 2))
        write_wrfout(tmp_path / name, hourly(T0, 2), domain=2, nx=20, ny=16, dx=4000.0)
    out = tmp_path / "atlas"
    assert main([str(tmp_path / "runA"), str(tmp_path / "runB"), "-o", str(out),
                 "--domains", "d02", "--no-plots", "--no-validate"]) == 0
    assert (out / "runA" / "collection.json").is_file() and (out / "runB" / "collection.json").is_file()
    assert {p.name.split("-")[0] for p in out.glob("*/items/*")} == {"d02"}


@pytest.mark.skipif(shutil.which("modelatlasify") is None, reason="console script not installed")
def test_console_script(wrf_run, tmp_path):
    res = subprocess.run(["modelatlasify", str(wrf_run), "-o", str(tmp_path / "a"), "--no-plots",
                          "--no-validate"], capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip().endswith("catalog.json")


def test_gdalinfo_reads_output(wrf_run, tmp_path):
    """The GeoTIFFs open in GDAL (what QGIS/ArcGIS use) with the expected CRS."""
    if shutil.which("gdalinfo") is None:
        pytest.skip("gdalinfo not available")
    out = tmp_path / "atlas"
    assert main([str(wrf_run), "-o", str(out), "--variables", "wind", "--no-plots",
                 "--no-validate"]) == 0
    tif = next(out.glob("run01/items/*/surface_wind_speed.tif"))
    res = subprocess.run(["gdalinfo", "-json", str(tif)], capture_output=True, text=True, check=True)
    import json

    info = json.loads(res.stdout)
    assert info["size"] == [40, 30]
    assert "Lambert" in info["coordinateSystem"]["wkt"]
    assert info["bands"][0]["unit"] == "m s-1"
    assert info["metadata"]["IMAGE_STRUCTURE"]["LAYOUT"] == "COG"


def test_cli_rejects_zero_processes(wrf_run, tmp_path):
    with pytest.raises(SystemExit) as exc:
        main([str(wrf_run), "-o", str(tmp_path / "a"), "-np", "0"])
    assert exc.value.code == 2


def _item_json(out):
    import json

    items = {}
    for p in sorted(out.glob("run01/items/*/*.json")):
        d = json.loads(p.read_text())
        d["properties"].pop("created", None)
        d["properties"].pop("updated", None)
        items[p.name] = d
    return items


def test_cli_parallel_matches_serial(wrf_run, tmp_path, capsys):
    import numpy as np
    import rasterio

    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    assert main([str(wrf_run), "-o", str(serial), "--no-validate"]) == 0
    assert main([str(wrf_run), "-o", str(parallel), "-np", "2", "--no-validate"]) == 0
    assert "(2 processes)" in capsys.readouterr().err

    files = {p.relative_to(serial) for p in serial.rglob("*") if p.suffix in (".tif", ".png")}
    assert files and files == {p.relative_to(parallel) for p in parallel.rglob("*")
                               if p.suffix in (".tif", ".png")}
    for rel in (f for f in files if f.suffix == ".tif"):
        with rasterio.open(serial / rel) as a, rasterio.open(parallel / rel) as b:
            np.testing.assert_array_equal(a.read(1), b.read(1))
    assert _item_json(serial) == _item_json(parallel)


def test_cli_parallel_quality_skips_and_errors(tmp_path, capsys):
    import netCDF4

    run = tmp_path / "run"
    paths = write_wrfout(run, hourly(T0, 4))
    with netCDF4.Dataset(paths[2], "a") as nc:  # zero-filled precipitation at one time
        nc["RAINNC"][0] = 0.0
        nc["RAINC"][0] = 0.0
    assert main([str(run), "-o", str(tmp_path / "a"), "-np", "2", "--no-plots",
                 "--no-validate"]) == 0
    err = capsys.readouterr().err
    assert "product(s) failed sanity checks" in err and "lower than at" in err

    bad = tmp_path / "bad"
    write_wrfout(bad, hourly(T0, 2), include=("U10", "V10"))
    assert main([str(bad), "-o", str(tmp_path / "b"), "-np", "2", "--no-plots"]) == 1
    err = capsys.readouterr().err
    assert "error: Cannot compute precipitation" in err and "Traceback" not in err
