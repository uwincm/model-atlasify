import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import T0, run_out
from wrf_fixture import hourly, write_wrfout
from modelatlasify.cli import main


def test_cli_end_to_end(wrf_run, tmp_path, capsys):
    out = tmp_path / "atlas"
    rc = main([str(wrf_run), "--output", str(out), "--no-plots", "--no-validate"])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == str(out.resolve() / "catalog.json")
    assert f"Found WRF run '{wrf_run.relative_to('/')}'" in captured.err
    assert "d01: 4 file(s), 4 time(s)" in captured.err
    assert re.search(r"georeference fit error [\d.]+ m\.\n", captured.err)
    assert "[run01 d01" not in captured.err  # no per-time lines; a progress bar instead
    assert (run_out(out, wrf_run) / "collection.json").is_file()
    assert len(list(run_out(out, wrf_run).glob("items/*/*.tif"))) == 4 + 4 + 3


def test_cli_variable_selection(wrf_run, tmp_path):
    out = tmp_path / "atlas"
    assert main([str(wrf_run), "-o", str(out), "--variables", "wind", "--no-plots",
                 "--no-validate"]) == 0
    assert {p.name for p in run_out(out, wrf_run).glob("items/*/*.tif")} == {"surface_wind_speed.tif"}


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
    for name in ("runA", "runB"):
        assert (run_out(out, tmp_path / name) / "collection.json").is_file()
    assert {p.name.split("-")[0] for p in out.rglob("items/*")} == {"d02"}


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
    tif = next(run_out(out, wrf_run).glob("items/*/surface_wind_speed.tif"))
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
    for p in sorted(out.rglob("items/*/*.json")):
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


def test_parse_sub_areas():
    from modelatlasify.core.areas import SubArea, parse_sub_areas

    assert parse_sub_areas("gulf,-98,-82,18,31, lee ,265,272,24,31") == [
        SubArea("gulf", -98.0, -82.0, 18.0, 31.0), SubArea("lee", -95.0, -88.0, 24.0, 31.0)]
    (pac,) = parse_sub_areas("pacific,170,-170,-10,10")  # crosses the antimeridian
    assert (pac.west, pac.east, pac.center_lon) == (170.0, 190.0, 180.0)


@pytest.mark.parametrize("value", [
    "gulf,-98,-82,18",                     # not a multiple of 5
    "gulf,-98,-82,31,18",                  # LAT1 >= LAT2
    "gulf,-98,-82,18,95",                  # latitude out of range
    "full,-98,-82,18,31",                  # reserved name
    "a,1,2,3,4,a,5,6,7,8",                 # duplicate name
    "bad name,1,2,3,4",                    # unsafe name
    "gulf,west,-82,18,31",                 # not a number
    "gulf,-98,-98,18,31",                  # empty longitude range
])
def test_cli_rejects_bad_sub_areas(wrf_run, tmp_path, value, capsys):
    with pytest.raises(SystemExit) as exc:
        main([str(wrf_run), "-o", str(tmp_path / "a"), f"--plot-sub-areas={value}"])
    assert exc.value.code == 2
    assert "--plot-sub-areas" in capsys.readouterr().err


def test_command_is_recorded_in_each_run_directory(tmp_path):
    for name in ("runA", "runB"):
        write_wrfout(tmp_path / name, hourly(T0, 2))
    out = tmp_path / "atlas"
    argv = [str(tmp_path / "runA"), str(tmp_path / "runB"), "-o", str(out), "--no-plots",
            "--no-validate", "--variables=wind"]
    assert main(argv) == 0
    for name in ("runA", "runB"):
        (record,) = run_out(out, tmp_path / name).glob("command-used.*.txt")
        assert re.fullmatch(r"command-used\.\d{8}T\d{6}Z\.txt", record.name)
        first = record.read_text().splitlines()[0]
        assert shlex.split(first) == ["modelatlasify", *argv]  # the line can be pasted back into a shell


def test_output_mirrors_the_input_path_and_cut_dirs(wrf_run, tmp_path, capsys):
    from modelatlasify.core.validation import validate_catalog

    out = tmp_path / "atlas"
    assert main([str(wrf_run), "-o", str(out), "--no-plots", "--no-validate"]) == 0
    full = out / wrf_run.relative_to("/")
    assert (full / "collection.json").is_file()
    collection = json.loads((full / "collection.json").read_text())
    assert collection["id"] == "--".join(wrf_run.parts[1:])
    assert collection["title"] == "WRF run run01"
    children = [ln["href"] for ln in json.loads((out / "catalog.json").read_text())["links"]
                if ln["rel"] == "child"]
    assert children == [f"./{wrf_run.relative_to('/')}/collection.json"]

    # --cut-dirs=2 drops the first two directories, like wget; runs accumulate.
    assert main([str(wrf_run), "-o", str(out), "--no-plots", "--no-validate",
                 "--cut-dirs=2"]) == 0
    cut = out / Path(*wrf_run.parts[3:])
    assert (cut / "collection.json").is_file()
    assert json.loads((cut / "collection.json").read_text())["id"] == "--".join(wrf_run.parts[3:])
    assert validate_catalog(out / "catalog.json", schemas=False).ok
    assert "WARNING: --cut-dirs" not in capsys.readouterr().err

    # Cutting everything keeps just the run directory's name, with a warning.
    assert main([str(wrf_run), "-o", str(out), "--no-plots", "--no-validate",
                 "--cut-dirs=99"]) == 0
    assert (out / "run01" / "collection.json").is_file()
    assert json.loads((out / "run01" / "collection.json").read_text())["id"] == "run01"
    assert "WARNING: --cut-dirs=99 removes all parent directories" in capsys.readouterr().err


def test_cli_rejects_negative_cut_dirs(wrf_run, tmp_path):
    with pytest.raises(SystemExit) as exc:
        main([str(wrf_run), "-o", str(tmp_path / "a"), "--cut-dirs=-1"])
    assert exc.value.code == 2


def test_same_name_in_different_places_and_repeated_inputs(tmp_path, capsys):
    a, b = tmp_path / "x" / "run", tmp_path / "y" / "run"
    for run in (a, b):
        write_wrfout(run, hourly(T0, 2))
    out = tmp_path / "atlas"
    assert main([str(a), str(b), str(a), "-o", str(out), "--no-plots", "--no-validate"]) == 0
    assert "listed more than once" in capsys.readouterr().err
    ids = {json.loads(p.read_text())["id"] for p in out.rglob("collection.json")}
    assert ids == {"--".join(a.parts[1:]), "--".join(b.parts[1:])}


def test_run_location():
    from modelatlasify.core.processing import run_location

    path, run_id, over = run_location(Path("/data/wrf/x/../my run#1"))
    assert (str(path), run_id, over) == ("data/wrf/my run#1", "data--wrf--my-run-1", False)
    assert run_location(Path("/data/wrf/run01"), 1)[:2] == (Path("wrf/run01"), "wrf--run01")
    assert run_location(Path("/data/wrf/run01"), 2)[2] is False
    assert run_location(Path("/data/wrf/run01"), 3) == (Path("run01"), "run01", True)
