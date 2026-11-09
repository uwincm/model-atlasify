"""HYCOM (HYbrid Coordinate Ocean Model) adapter.

HYCOM writes "a/b" archive files, one per output time: ``archv.<YYYY_DDD_HH>``
(the full 3-D ocean) and/or ``archs.<YYYY_DDD_HH>`` (the surface layer only),
each a ``.a`` (binary) + ``.b`` (text header) pair, next to
``regional.grid.[ab]`` (the grid's lon/lat). ModelAtlasify reads only the top
layer (see :mod:`.hycom_io`):

- valid times come from the file names (the ``.b`` model day, written to only
  3 decimals, must agree within 2 minutes); when ``archv`` and ``archs`` exist
  for the same time, ``archs`` (smaller) is used;
- the grid is recovered from ``plon``/``plat`` like DWM's: Mercator, else a
  regular lon/lat grid, else resampled (nearest model cell) to lon/lat;
- surface currents are the top layer's total velocity: the baroclinic
  ``u-vel.``/``v-vel.`` plus the barotropic ``u_btrop``/``v_btrop``, moved
  from the C-grid cell faces (u: west face, v: south face) to the cell centres
  by averaging the two faces. Grids rotated from east/north (``pang`` != 0)
  are not supported for currents, rather than risk a wrong rotation;
- in a UWIN-CM run, ``uwin.nml`` gives the start time (``startDate``) and
  whether the ocean was on (``ocean``; off = no HYCOM output).
"""

from __future__ import annotations

import dataclasses
import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import CRS

from modelatlasify.core.errors import (DiscoveryError, FieldQualityError, GeoreferenceError,
                                       NoModelOutputError, VariableUnavailableError)
from modelatlasify.core.model import FieldResult, GridDefinition, ModelAdapter, ModelDataset
from modelatlasify.core.namelist import read_namelist
from modelatlasify.geospatial.gridfit import fit_regular_grid
from modelatlasify.geospatial.regrid import lonlat_crs, resample_to_lonlat
from modelatlasify.models.hycom import hycom_io

log = logging.getLogger(__name__)

DATASET_ID = "HYCOM"
EARTH_RADIUS_M = 6370000.0
LONLAT_VIEW_BUFFER_DEG = 1.0
UWIN_NAMELIST = "uwin.nml"
#: Model day (3 decimals in the .b file) and file name must agree this well.
MAX_TIME_MISMATCH = timedelta(minutes=2)
#: Rotated grids (|pang| above this, radians) get no currents.
MAX_PANG = 1e-6

#: Scalar products: standard variable -> layer-1 archive record.
SCALARS = {"sst": "temp", "surface_salinity": "salin"}
CURRENT_FIELDS = ("surface_u", "surface_v", "surface_currents")


class HYCOMAdapter(ModelAdapter):
    name = "HYCOM"
    #: Outline the ocean domain (dashed) on composites, like DWM's.
    outline_domain_edge = True

    def __init__(self) -> None:
        self.skipped_files: list[tuple[Path, str]] = []
        self._uv_cache: dict = {}

    def _skip(self, path: Path, reason: str) -> None:
        log.warning("Skipping HYCOM file %s: %s", path.name, reason)
        self.skipped_files.append((path, reason))

    # ------------------------------------------------------------ discovery
    def discover(self, directory: Path) -> list[ModelDataset]:
        directory = Path(directory)
        if not directory.is_dir():
            raise DiscoveryError(f"Input directory does not exist: {directory}")
        b_files = sorted(p for kind in ("archs", "archv") for p in directory.glob(f"{kind}.*.b")
                         if hycom_io.filename_time(p.name))
        if not b_files:
            raise NoModelOutputError(
                f"No HYCOM archive files (archv.<YYYY_DDD_HH>.[ab] or archs...) in {directory}")
        off = self._ocean_off_reason(directory)
        if off:
            raise NoModelOutputError(f"HYCOM archives in {directory} ignored: {off}")
        grid_b = directory / "regional.grid.b"
        if not (grid_b.is_file() and grid_b.with_suffix(".a").is_file()):
            raise DiscoveryError(f"HYCOM archives in {directory}, but no regional.grid.[ab] "
                                 "to place them.")

        by_time: dict[datetime, tuple[Path, hycom_io.Header]] = {}
        for b in b_files:  # archs before archv: the surface file wins for a time
            a = b.with_suffix(".a")
            t = hycom_io.filename_time(b.name)
            if t in by_time:
                continue
            try:
                if not a.is_file():
                    raise ValueError(f"no {a.name}")
                header = hycom_io.read_header(b)
                for name in ("temp", "salin", "u-vel.", "v-vel."):
                    if not header.has(name, 1):
                        raise ValueError(f"no layer-1 '{name}' record")
                day_time = hycom_io.model_day_to_datetime(header.records[0].model_day,
                                                          header.yrflag)
                if abs(day_time - t) > MAX_TIME_MISMATCH:
                    raise ValueError(f"model day says {day_time:%Y-%m-%d %H:%M}, the name says "
                                     f"{t:%Y-%m-%d %H:%M}; file content appears inconsistent")
                record_bytes = -(-header.idm * header.jdm // hycom_io.WORD) * hycom_io.WORD * 4
                last = max(header.index(n, 1) for n in ("temp", "salin", "u-vel.", "v-vel."))
                if a.stat().st_size < (last + 1) * record_bytes:
                    raise ValueError(f"{a.name} is shorter than its .b file says (truncated?)")
            except Exception as exc:  # noqa: BLE001 - a bad file is reported, not fatal
                self._skip(a, str(exc).strip() or exc.__class__.__name__)
                continue
            by_time[t] = (a, header)
        if not by_time:
            raise DiscoveryError(
                f"Found {len(b_files)} HYCOM archive(s) in {directory}, but none could be used "
                f"(e.g. {self.skipped_files[0][0].name}: {self.skipped_files[0][1]}).")
        return [self._make_dataset(directory, sorted(by_time.items()))]

    @staticmethod
    def _uwin(directory: Path) -> dict[str, Any]:
        path = directory / UWIN_NAMELIST
        if not path.is_file():
            return {}
        try:
            return {k.lower(): v for k, v in read_namelist(path).items()}
        except OSError:
            return {}

    def _ocean_off_reason(self, directory: Path) -> str | None:
        value = self._uwin(directory).get("ocean")
        return f"{UWIN_NAMELIST} has ocean = .F." if value is False else None

    def _start_time(self, directory: Path) -> datetime | None:
        raw = self._uwin(directory).get("startdate")
        try:
            parts = [int(x) for x in str(raw).replace(",", " ").split()]
            return datetime(*parts[:6], tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return None

    def _make_dataset(self, directory: Path,
                      entries: list[tuple[datetime, tuple[Path, hycom_io.Header]]]
                      ) -> ModelDataset:
        grid_b = directory / "regional.grid.b"
        gh = hycom_io.read_header(grid_b)
        lon = hycom_io.read_record(grid_b.with_suffix(".a"), gh, gh.index("plon"))
        lat = hycom_io.read_record(grid_b.with_suffix(".a"), gh, gh.index("plat"))
        pang = (hycom_io.read_record(grid_b.with_suffix(".a"), gh, gh.index("pang"))
                if gh.has("pang") else np.zeros_like(lon))
        first_header = entries[0][1][1]
        if (gh.idm, gh.jdm) != (first_header.idm, first_header.jdm):
            raise DiscoveryError(f"regional.grid is {gh.idm}x{gh.jdm} but the archives are "
                                 f"{first_header.idm}x{first_header.jdm}.")
        grid, how, regrid_index, warnings = self._georeference(lon, lat)
        rotated = bool(np.nanmax(np.abs(pang)) > MAX_PANG)

        times = [t for t, _ in entries]
        meta: dict[str, Any] = {"hycom:domain": DATASET_ID, "hycom:georeference": how,
                                "hycom:mapflg": gh.mapflg, "hycom:yrflag": first_header.yrflag,
                                "hycom:archive_kinds": sorted({p.name[:5] for _, (p, _) in entries})}
        diffs = Counter(b - a for a, b in zip(times, times[1:]))
        if diffs:
            meta["hycom:output_interval_minutes"] = diffs.most_common(1)[0][0].total_seconds() / 60
        km = grid.transform.a / 1000
        size = (f"{grid.width}x{grid.height}, {grid.transform.a:.3g} deg lon/lat"
                if grid.crs.is_geographic else
                f"{grid.width}x{grid.height}, {km:.1f} km at the equator")
        return ModelDataset(
            model=self.name, run_id=directory.resolve().name, dataset_id=DATASET_ID,
            title=f"HYCOM ocean domain ({size})",
            files=[p for _, (p, _) in entries], valid_times=times,
            reference_time=self._start_time(directory), grid=grid, metadata=meta,
            source_dir=directory, time_sources=[p for _, (p, _) in entries],
            adapter_state={"headers": [h for _, (_, h) in entries], "regrid": regrid_index,
                           "south_up": bool(lat[0, 0] < lat[-1, 0]),
                           "rotated": rotated},
            warnings=warnings + ([
                "the HYCOM grid is rotated from east/north (pang != 0); surface currents "
                "are not produced for it"] if rotated else []),
        )

    def _georeference(self, lon: np.ndarray, lat: np.ndarray
                      ) -> tuple[GridDefinition, str, np.ndarray | None, list[str]]:
        ny, nx = lon.shape
        lon0 = float((lon[ny // 2, nx // 2] + 180.0) % 360.0 - 180.0)
        candidates = [
            ("Mercator", CRS.from_proj4(
                f"+proj=merc +lat_ts=0 +lon_0={lon0:.6f} +x_0=0 +y_0=0 "
                f"+R={EARTH_RADIUS_M:.0f} +units=m +no_defs +type=crs")),
            ("a regular lon/lat grid", lonlat_crs(EARTH_RADIUS_M)),
        ]
        tried = []
        for label, crs in candidates:
            try:
                grid = fit_regular_grid(crs, lon, lat, description=(
                    f"HYCOM grid in {label}, {nx} x {ny} cells"))
            except GeoreferenceError as exc:
                tried.append(label)
                log.debug("HYCOM grid is not regular in %s: %s", label, exc)
                continue
            return grid, f"regular grid in {label}", None, []
        grid, index = resample_to_lonlat(lon, lat, EARTH_RADIUS_M)
        grid = dataclasses.replace(grid, lonlat_view_buffer_deg=LONLAT_VIEW_BUFFER_DEG,
                                   description="HYCOM " + grid.description)
        warning = (f"The HYCOM grid is not regular in {' or '.join(tried)}; its fields are "
                   f"resampled (nearest model cell) onto a regular {grid.transform.a:.3g} deg "
                   "lon/lat grid for the GeoTIFFs and maps.")
        return grid, "resampled (nearest model cell) to a regular lon/lat grid", index, [warning]

    # ---------------------------------------------------------- variables
    def supported_variables(self) -> list[str]:
        return [*SCALARS, *CURRENT_FIELDS]

    def unavailable_variables(self, dataset: ModelDataset) -> dict[str, str]:
        if dataset.adapter_state["rotated"]:
            return {v: "the HYCOM grid is rotated (pang != 0), which is not supported for "
                       "currents" for v in CURRENT_FIELDS}
        return {}

    def _read(self, dataset: ModelDataset, t: int, name: str, layer: int | None) -> np.ndarray:
        header = dataset.adapter_state["headers"][t]
        path = dataset.time_sources[t]
        try:
            index = header.index(name, layer)
        except KeyError as exc:
            raise VariableUnavailableError(f"HYCOM {path.name}: {exc.args[0]}.") from None
        return hycom_io.read_record(path, header, index)

    def _to_grid(self, dataset: ModelDataset, data: np.ndarray) -> np.ndarray:
        """Model array (as stored, south row first) -> the dataset grid (north-up)."""
        index = dataset.adapter_state["regrid"]
        if index is not None:
            out = np.full(index.shape, np.nan)
            inside = index >= 0
            out[inside] = data.ravel()[index[inside]]
            return out
        return data[::-1].copy() if dataset.adapter_state["south_up"] else data

    def _currents(self, dataset: ModelDataset, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Total top-layer (u, v) at the cell centres, as stored (south row first)."""
        key = (dataset.run_id, t)
        if self._uv_cache.get("key") == key:
            return self._uv_cache["value"]
        u = self._read(dataset, t, "u-vel.", 1) + self._read(dataset, t, "u_btrop", 0)
        v = self._read(dataset, t, "v-vel.", 1) + self._read(dataset, t, "v_btrop", 0)
        # C grid: u[j, i] is on the west face of cell (j, i), v[j, i] on its south
        # face; the east (north) face is u[j, i+1] (v[j+1, i]). A face against
        # land has no value; nothing flows through it, so it counts as 0. The
        # last column (row) has no east (north) face in the file: left missing.
        u0, v0 = np.nan_to_num(u), np.nan_to_num(v)
        uc = np.full_like(u, np.nan)
        vc = np.full_like(v, np.nan)
        uc[:, :-1] = 0.5 * (u0[:, :-1] + u0[:, 1:])
        vc[:-1, :] = 0.5 * (v0[:-1, :] + v0[1:, :])
        land = np.isnan(self._read(dataset, t, "temp", 1))
        uc[land] = np.nan
        vc[land] = np.nan
        self._uv_cache = {"key": key, "value": (uc, vc)}
        return uc, vc

    def compute(self, dataset: ModelDataset, variable: str, time_index: int) -> FieldResult:
        valid = dataset.valid_times[time_index]
        path = dataset.time_sources[time_index].name
        resampled = dataset.adapter_state["regrid"] is not None
        how = "; resampled to the lon/lat grid (nearest model cell)" if resampled else ""
        if variable in SCALARS:
            native = SCALARS[variable]
            data = self._read(dataset, time_index, native, 1)
            if np.isnan(data).all():
                raise FieldQualityError("all values are missing")
            return FieldResult(variable, self._to_grid(dataset, data), valid, provenance={
                "source_variables": [native],
                "method": f"HYCOM layer-1 '{native}' from {path}{how}; land missing",
            })
        if variable in CURRENT_FIELDS:
            if dataset.adapter_state["rotated"]:
                raise VariableUnavailableError(
                    "HYCOM grid is rotated (pang != 0); surface currents are not supported.")
            u, v = self._currents(dataset, time_index)
            data = {"surface_u": u, "surface_v": v, "surface_currents": np.hypot(u, v)}[variable]
            return FieldResult(variable, self._to_grid(dataset, data), valid, provenance={
                "source_variables": ["u-vel.", "v-vel.", "u_btrop", "v_btrop"],
                "method": f"HYCOM layer-1 total current (u-vel. + u_btrop, v-vel. + v_btrop) "
                          f"from {path}, averaged from the C-grid faces to the cell centres"
                          + (", speed sqrt(u^2 + v^2)" if variable == "surface_currents" else "")
                          + f"{how}; land missing",
            })
        raise VariableUnavailableError(
            f"HYCOM adapter does not know how to compute '{variable}'. "
            f"Supported: {', '.join(self.supported_variables())}")
