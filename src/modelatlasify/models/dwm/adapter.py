"""DWM (Donelan Wave Model) adapter.

DWM was formerly UMWM (University of Miami Wave Model), and its output files
still use the old name. It runs on its own or as the wave component of a coupled system (e.g.
UWIN-CM), and writes one ``umwmout_<YYYY-MM-DD_HH:MM:SS>[.nc]`` file per output
time under ``output/``. The files carry 2-D ``lon``/``lat`` but no projection
metadata and no time variable, so:

- valid times come from the file names;
- the grid is recovered from lon/lat (see :mod:`modelatlasify.geospatial.gridfit`):
  a Mercator grid on WRF's 6 370 km sphere (as used in UWIN-CM), true scale at
  the equator. Mercator's radius only scales x and y together, so the choice of
  sphere does not move anything;
- the simulation start time, if known, comes from ``namelists/main.nml``.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
from pyproj import CRS

from modelatlasify.core.errors import (DiscoveryError, GeoreferenceError, NoModelOutputError,
                                       VariableUnavailableError)
from modelatlasify.core.model import FieldResult, GridDefinition, ModelAdapter, ModelDataset
from modelatlasify.geospatial.gridfit import fit_regular_grid

log = logging.getLogger(__name__)

OUTPUT_PATTERN = re.compile(
    r"^umwmout_(?P<date>\d{4}-\d{2}-\d{2})_(?P<h>\d{2})[:_](?P<m>\d{2})[:_](?P<s>\d{2})(\.nc)?$")
EARTH_RADIUS_M = 6370000.0
DATASET_ID = "DWM"

#: Standard variable -> DWM output variable and the units it must have.
FIELDS = {
    "significant_wave_height": ("swh", "m"),
    "mean_wave_period": ("mwp", "s"),
    "dominant_wave_period": ("dwp", "s"),
}

#: main.nml settings copied into STAC metadata as ``dwm:<lowercase name>``.
NAMELIST_KEYS = ("isGlobal", "mm", "nm", "om", "pm", "fmin", "fmax", "fprog",
                 "startTimeStr", "stopTimeStr", "dtg", "restart", "gridFromFile", "topoFromFile")


def _file_time(name: str) -> datetime | None:
    m = OUTPUT_PATTERN.match(name)
    if not m:
        return None
    return datetime.strptime(f"{m['date']} {m['h']}:{m['m']}:{m['s']}",
                             "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def read_namelist(path: Path) -> dict[str, Any]:
    """The simple ``key = value`` entries of a Fortran namelist (comments dropped)."""
    out: dict[str, Any] = {}
    for line in path.read_text(errors="replace").splitlines():
        line = line.split("!", 1)[0].strip()
        m = re.match(r"^([A-Za-z_]\w*)\s*=\s*(.+?)\s*,?$", line)
        if not m:
            continue
        key, raw = m[1], m[2].strip()
        if raw[:1] in "'\"":
            value: Any = raw.strip("'\"")
        elif raw.lower() in (".t.", ".true."):
            value = True
        elif raw.lower() in (".f.", ".false."):
            value = False
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw.lower().replace("d", "e"))
                except ValueError:
                    value = raw
        out[key] = value
    return out


class DWMAdapter(ModelAdapter):
    name = "DWM"
    #: Composite maps outline this model's domain (dashed), since its edge is
    #: not otherwise obvious against land and the atmosphere's larger domain.
    outline_domain_edge = True

    def __init__(self) -> None:
        self._handles: dict[Path, netCDF4.Dataset] = {}
        self.skipped_files: list[tuple[Path, str]] = []

    def _open(self, path: Path) -> netCDF4.Dataset:
        nc = self._handles.get(path)
        if nc is None:
            nc = netCDF4.Dataset(path, "r")
            nc.set_auto_mask(True)
            self._handles[path] = nc
        return nc

    def close(self) -> None:
        for nc in self._handles.values():
            try:
                nc.close()
            except Exception:  # pragma: no cover - best effort
                pass
        self._handles.clear()

    def _skip(self, path: Path, reason: str) -> None:
        log.warning("Skipping DWM file %s: %s", path.name, reason)
        self.skipped_files.append((path, reason))

    # ------------------------------------------------------------ discovery
    @staticmethod
    def output_dirs(directory: Path) -> list[Path]:
        """Where DWM output is looked for: ``<dir>/output/`` (DWM's own
        layout, also inside a coupled run directory), then ``<dir>/``."""
        return [d for d in (directory / "output", directory) if d.is_dir()]

    def discover(self, directory: Path) -> list[ModelDataset]:
        directory = Path(directory)
        if not directory.is_dir():
            raise DiscoveryError(f"Input directory does not exist: {directory}")
        candidates = sorted(
            (p for d in self.output_dirs(directory) for p in d.iterdir()
             if p.is_file() and _file_time(p.name)), key=lambda p: (_file_time(p.name), str(p)))
        if not candidates:
            raise NoModelOutputError(
                f"No DWM output files (umwmout_<YYYY-MM-DD_HH:MM:SS>) in {directory}/output "
                f"or {directory}")

        entries = []  # (time, path, corner signature)
        for path in candidates:
            try:
                with netCDF4.Dataset(path, "r") as nc:
                    for name in ("lon", "lat"):
                        if name not in nc.variables:
                            raise ValueError(f"no '{name}' variable; not DWM output?")
                    n_times = len(nc.dimensions["Time"]) if "Time" in nc.dimensions else 1
                    if n_times != 1:
                        raise ValueError(f"{n_times} times in one file, but DWM output has no "
                                         "time variable to tell them apart")
                    sig = tuple(round(float(nc.variables[v][0, j, i]), 3)
                                for v in ("lat", "lon") for j in (0, -1) for i in (0, -1))
            except Exception as exc:  # noqa: BLE001 - a bad file is reported, not fatal
                self._skip(path, str(exc).strip() or exc.__class__.__name__)
                continue
            entries.append((_file_time(path.name), path, sig))
        if not entries:
            raise DiscoveryError(
                f"Found {len(candidates)} DWM-like files in {directory}, but none could be used "
                f"(e.g. {self.skipped_files[0][0].name}: {self.skipped_files[0][1]}).")

        # One fixed grid: the most common lon/lat signature wins.
        sig = Counter(e[2] for e in entries).most_common(1)[0][0]
        kept, seen = [], set()
        for t, path, s in entries:
            if s != sig:
                self._skip(path, "lon/lat differ from the rest of the DWM output (corrupt file?)")
            elif t in seen:
                log.warning("Duplicate DWM valid time %s in %s; using the first.", t, path.name)
            else:
                seen.add(t)
                kept.append((t, path))
        return [self._make_dataset(directory, kept)]

    def _make_dataset(self, directory: Path, entries: list[tuple[datetime, Path]]) -> ModelDataset:
        first = entries[0][1]
        with netCDF4.Dataset(first, "r") as nc:
            lon = np.asarray(nc.variables["lon"][0], dtype=np.float64)
            lat = np.asarray(nc.variables["lat"][0], dtype=np.float64)
        grid = self._grid(lon, lat)

        namelist = self._namelist(directory)
        reference_time = None
        if isinstance(namelist.get("startTimeStr"), str):
            try:
                reference_time = datetime.strptime(namelist["startTimeStr"].strip(),
                                                   "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        meta: dict[str, Any] = {f"dwm:{k.lower()}": namelist[k] for k in NAMELIST_KEYS
                                if k in namelist}
        times = [t for t, _ in entries]
        diffs = Counter(b - a for a, b in zip(times, times[1:]))
        if diffs:
            meta["dwm:output_interval_minutes"] = diffs.most_common(1)[0][0].total_seconds() / 60
        meta["dwm:domain"] = DATASET_ID

        return ModelDataset(
            model=self.name,
            run_id=directory.resolve().name,
            dataset_id=DATASET_ID,
            title=f"DWM wave domain ({grid.width}x{grid.height}, "
                  f"{grid.transform.a / 1000:.1f} km at the equator)",
            files=[p for _, p in entries],
            valid_times=times,
            reference_time=reference_time,
            grid=grid,
            metadata=meta,
            source_dir=directory,
            time_sources=[p for _, p in entries],
            adapter_state={"south_up": bool(lat[0, 0] < lat[-1, 0])},
        )

    @staticmethod
    def _grid(lon: np.ndarray, lat: np.ndarray) -> GridDefinition:
        """Fit the grid as Mercator (UWIN-CM's DWM grids are Mercator)."""
        ny, nx = lon.shape
        lon0 = float((lon[ny // 2, nx // 2] + 180.0) % 360.0 - 180.0)
        crs = CRS.from_proj4(f"+proj=merc +lat_ts=0 +lon_0={lon0:.6f} +x_0=0 +y_0=0 "
                             f"+R={EARTH_RADIUS_M:.0f} +units=m +no_defs +type=crs")
        try:
            return fit_regular_grid(crs, lon, lat, description=(
                f"DWM Mercator grid, {nx} x {ny} cells, on a sphere of radius "
                f"{EARTH_RADIUS_M:.0f} m"))
        except GeoreferenceError as exc:
            raise GeoreferenceError(
                f"The DWM grid is not a Mercator grid ({exc}); other grid types are not "
                "supported yet.") from None

    @staticmethod
    def _namelist(directory: Path) -> dict[str, Any]:
        for path in (directory / "namelists" / "main.nml", directory / "main.nml"):
            if path.is_file():
                try:
                    return read_namelist(path)
                except OSError:
                    return {}
        return {}

    # ---------------------------------------------------------- variables
    def supported_variables(self) -> list[str]:
        return list(FIELDS)

    def compute(self, dataset: ModelDataset, variable: str, time_index: int) -> FieldResult | None:
        if variable not in FIELDS:
            raise VariableUnavailableError(
                f"DWM adapter does not know how to compute '{variable}'. "
                f"Supported: {', '.join(FIELDS)}")
        native, units = FIELDS[variable]
        path = dataset.time_sources[time_index]
        nc = self._open(path)
        if native not in nc.variables:
            raise VariableUnavailableError(f"Variable '{native}' is not present in {path.name}.")
        var = nc.variables[native]
        have = str(getattr(var, "units", "")).strip()
        if have != units:
            raise VariableUnavailableError(
                f"DWM {native} has units '{have}', expected '{units}'; refusing to guess.")
        data = np.ma.filled(np.ma.asarray(var[0], dtype=np.float64), np.nan)
        if "seamask" in nc.variables:  # land is NaN already; make sure
            data[np.asarray(nc.variables["seamask"][0]) == 0] = np.nan
        if dataset.adapter_state["south_up"]:
            data = data[::-1].copy()
        return FieldResult(variable, data, dataset.valid_times[time_index], provenance={
            "source_variables": [native],
            "method": f"DWM output field '{native}' ({getattr(var, 'description', native)}), "
                      "as written; land points missing",
        })
