"""DWM (Donelan Wave Model) adapter.

DWM was formerly UMWM (University of Miami Wave Model), and its output files
still use the old name. It runs on its own or as the wave component of a coupled system (e.g.
UWIN-CM), and writes one ``umwmout_<YYYY-MM-DD_HH:MM:SS>[.nc]`` file per output
time under ``output/``. The files carry 2-D ``lon``/``lat`` but no projection
metadata and no time variable, so:

- valid times come from the file names;
- the grid is recovered from lon/lat (see :mod:`modelatlasify.geospatial.gridfit`).
  The candidates, in order: Mercator on WRF's 6 370 km sphere (as used in
  UWIN-CM with a Mercator atmosphere), true scale at the equator; a regular
  lon/lat grid; and the map projection of each WRF domain in the same run
  directory (UWIN-CM puts the waves on the atmosphere's projection, e.g.
  Lambert conformal). If none fits, the fields are resampled (nearest model
  cell) onto a regular lon/lat grid, with a warning. Maps of a grid that is
  not Mercator or lon/lat are drawn in a lon/lat box around the domain;
- the simulation start time, if known, comes from ``namelists/main.nml``.
"""

from __future__ import annotations

import dataclasses
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
from modelatlasify.core.namelist import read_namelist
from modelatlasify.geospatial.gridfit import fit_regular_grid
from modelatlasify.geospatial.regrid import lonlat_crs, resample_to_lonlat

log = logging.getLogger(__name__)

OUTPUT_PATTERN = re.compile(
    r"^umwmout_(?P<date>\d{4}-\d{2}-\d{2})_(?P<h>\d{2})[:_](?P<m>\d{2})[:_](?P<s>\d{2})(\.nc)?$")
EARTH_RADIUS_M = 6370000.0
DATASET_ID = "DWM"
#: Degrees added on each side of the domain for maps drawn in a lon/lat box.
LONLAT_VIEW_BUFFER_DEG = 1.0
WRFOUT_GLOB = "wrfout_d[0-9][0-9]_*"

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
        grid, how, regrid_index, warnings = self._georeference(directory, lon, lat)

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
        meta["dwm:georeference"] = how

        if regrid_index is not None:
            size = f"resampled to {grid.width}x{grid.height}, {grid.transform.a:.3g} deg lon/lat"
        elif grid.crs.is_geographic:
            size = f"{grid.width}x{grid.height}, {grid.transform.a:.3g} deg lon/lat"
        elif grid.uses_lonlat_view:
            size = f"{grid.width}x{grid.height}, {grid.transform.a / 1000:.1f} km"
        else:
            size = f"{grid.width}x{grid.height}, {grid.transform.a / 1000:.1f} km at the equator"
        return ModelDataset(
            model=self.name,
            run_id=directory.resolve().name,
            dataset_id=DATASET_ID,
            title=f"DWM wave domain ({size})",
            files=[p for _, p in entries],
            valid_times=times,
            reference_time=reference_time,
            grid=grid,
            metadata=meta,
            source_dir=directory,
            time_sources=[p for _, p in entries],
            adapter_state={"south_up": bool(lat[0, 0] < lat[-1, 0]), "regrid": regrid_index},
            warnings=warnings,
        )

    @staticmethod
    def _wrf_projections(directory: Path) -> list[tuple[str, CRS]]:
        """(label, CRS) of each WRF domain's map projection in ``directory``
        (one wrfout file per domain), without repeats."""
        from modelatlasify.models.wrf.projection import MAP_PROJ_NAMES, wrf_crs

        first: dict[str, Path] = {}
        for path in sorted(directory.glob(WRFOUT_GLOB)):
            first.setdefault(path.name[7:10], path)  # "d01"
        out: list[tuple[str, CRS]] = []
        for domain, path in first.items():
            try:
                with netCDF4.Dataset(path, "r") as nc:
                    attrs = {k: nc.getncattr(k) for k in nc.ncattrs()}
                crs = wrf_crs(attrs)
            except Exception as exc:  # noqa: BLE001 - just not a candidate
                log.debug("No WRF projection from %s: %s", path.name, exc)
                continue
            if not crs.is_projected or any(crs == c for _, c in out):
                continue
            name = MAP_PROJ_NAMES.get(int(attrs.get("MAP_PROJ", -1)), "map")
            out.append((f"the {name} projection of WRF {domain}", crs))
        return out

    def _georeference(self, directory: Path, lon: np.ndarray, lat: np.ndarray
                      ) -> tuple[GridDefinition, str, np.ndarray | None, list[str]]:
        """The DWM grid: ``(grid, how it was found, regrid index or None, warnings)``.

        DWM output records no projection, so candidates are tried in turn (see
        the module docstring). The regrid index (only when nothing fits) maps
        each lon/lat cell to the flat index of its model cell, -1 outside.
        """
        ny, nx = lon.shape
        lon0 = float((lon[ny // 2, nx // 2] + 180.0) % 360.0 - 180.0)
        sphere = f"on a sphere of radius {EARTH_RADIUS_M:.0f} m"
        candidates = [
            ("Mercator", CRS.from_proj4(
                f"+proj=merc +lat_ts=0 +lon_0={lon0:.6f} +x_0=0 +y_0=0 "
                f"+R={EARTH_RADIUS_M:.0f} +units=m +no_defs +type=crs"), None),
            ("a regular lon/lat grid", lonlat_crs(EARTH_RADIUS_M), None),
            *((label, crs, LONLAT_VIEW_BUFFER_DEG)
              for label, crs in self._wrf_projections(directory)),
        ]
        tried = []
        for label, crs, buffer in candidates:
            try:
                grid = fit_regular_grid(crs, lon, lat, description=(
                    f"DWM grid in {label}, {nx} x {ny} cells, {sphere}"))
            except GeoreferenceError as exc:
                tried.append(label)
                log.debug("DWM grid is not regular in %s: %s", label, exc)
                continue
            if buffer is not None:
                grid = dataclasses.replace(grid, lonlat_view_buffer_deg=buffer)
            return grid, f"regular grid in {label}", None, []

        grid, index = resample_to_lonlat(lon, lat, EARTH_RADIUS_M)
        grid = dataclasses.replace(grid, lonlat_view_buffer_deg=LONLAT_VIEW_BUFFER_DEG,
                                   description="DWM " + grid.description)
        no_wrf = "" if len(tried) > 2 else " (no WRF output here to take a projection from)"
        warning = (
            f"DWM output records no projection, and the grid is not regular in any of: "
            f"{', '.join(tried)}{no_wrf}. Its fields are resampled (each cell takes the value of the "
            f"nearest model cell) onto a regular {grid.transform.a:.3g} x "
            f"{-grid.transform.e:.3g} deg lon/lat grid for the GeoTIFFs and maps.")
        return grid, "resampled (nearest model cell) to a regular lon/lat grid", index, [warning]

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
        index = dataset.adapter_state.get("regrid")
        if index is not None:  # resample onto the lon/lat grid (already north-up)
            out = np.full(index.shape, np.nan)
            inside = index >= 0
            out[inside] = data.ravel()[index[inside]]
            data = out
        elif dataset.adapter_state["south_up"]:
            data = data[::-1].copy()
        return FieldResult(variable, data, dataset.valid_times[time_index], provenance={
            "source_variables": [native],
            "method": f"DWM output field '{native}' ({getattr(var, 'description', native)}), "
                      + ("resampled to the lon/lat grid (nearest model cell)" if index is not None
                         else "as written") + "; land points missing",
        })
