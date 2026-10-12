"""WRF (atmosphere-only) model adapter.

Discovers ``wrfout_d<NN>_<YYYY-MM-DD_HH:MM:SS>`` files, groups them into one
dataset per domain, and computes ModelAtlasify's standard variables from the
native WRF fields.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np

from modelatlasify.core.errors import (DiscoveryError, FieldQualityError, GeoreferenceError,
                                       VariableUnavailableError)
from modelatlasify.core.model import FieldResult, GridDefinition, ModelAdapter, ModelDataset, shift_to_grid
from modelatlasify.models.wrf.projection import MAP_PROJ_NAMES, wrf_grid
from modelatlasify.variables.precipitation import NEGATIVE_TOLERANCE_MM, interval_precipitation
from modelatlasify.variables.wind import destagger, wind_speed

log = logging.getLogger(__name__)

#: An accumulated-precipitation field that is lower than the previous valid
#: output at more than this fraction of grid points is rejected as corrupt (or
#: as an accumulation reset ModelAtlasify does not understand).
MAX_DECREASE_FRACTION = 0.01
#: The same for moving nests. WRF moves a nest several times between outputs,
#: and cells that leave and re-enter it are re-interpolated from the parent,
#: which can lower their accumulation (seen at up to ~4% of a vortex-following
#: d03 per hour). Zero-filled output still fails by a wide margin.
MAX_DECREASE_FRACTION_MOVING = 0.2

WRFOUT_PATTERN = re.compile(r"^wrfout_d(?P<domain>\d{2})_(?P<time>\d{4}-\d{2}-\d{2}_\d{2}[:_]\d{2}[:_]\d{2})")
WRF_TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"

#: Global attributes copied into STAC metadata (as ``wrf:<lowercase name>``)
#: when present. Nothing is invented: absent attributes are simply omitted.
METADATA_ATTRS = (
    "TITLE", "SIMULATION_START_DATE", "START_DATE", "SIMULATION_INITIALIZATION_TYPE",
    "GRID_ID", "PARENT_ID", "PARENT_GRID_RATIO", "I_PARENT_START", "J_PARENT_START",
    "MAP_PROJ", "MAP_PROJ_CHAR", "TRUELAT1", "TRUELAT2", "STAND_LON", "CEN_LAT",
    "CEN_LON", "MOAD_CEN_LAT", "POLE_LAT", "POLE_LON", "DX", "DY", "DT",
    "WEST-EAST_GRID_DIMENSION", "SOUTH-NORTH_GRID_DIMENSION",
    "BOTTOM-TOP_GRID_DIMENSION", "GRIDTYPE", "MP_PHYSICS", "CU_PHYSICS",
    "BL_PBL_PHYSICS", "SF_SFCLAY_PHYSICS", "SF_SURFACE_PHYSICS", "RA_LW_PHYSICS",
    "RA_SW_PHYSICS", "SHCU_PHYSICS", "SF_URBAN_PHYSICS", "SF_OCEAN_PHYSICS",
    "FEEDBACK", "MMINLU", "NUM_LAND_CAT", "BUCKET_MM", "PREC_ACC_DT",
)


def parse_wrf_time(text: str) -> datetime:
    return datetime.strptime(text.strip(), WRF_TIME_FORMAT).replace(tzinfo=timezone.utc)


def _to_python(value: Any) -> Any:
    """netCDF attribute -> JSON-friendly Python value."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
        return value[0] if len(value) == 1 else value
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        # netCDF float32 attributes -> shortest decimal that round-trips float32.
        return float(f"{value:.7g}") if value == value else None
    if isinstance(value, str):
        return value.strip()
    return value


def _wrf_version(title: str | None) -> str | None:
    if not title:
        return None
    m = re.search(r"WRF\s+V?(\d+(?:\.\d+)*)", title, flags=re.IGNORECASE)
    return m.group(1) if m else None


@dataclass(frozen=True)
class _TimeSlot:
    path: Path
    index: int  # index along the file's Time dimension


@dataclass
class _WRFState:
    slots: list[_TimeSlot]  # where each valid time lives
    accumulation_start: datetime | None
    output_interval: timedelta | None  # the run's history-output cadence
    # Result of the accumulated-precipitation check per time (None = passed),
    # filled by WRFAdapter.prepare() so worker processes can look it up.
    accumulation_problems: list[str | None] | None = None


class WRFAdapter(ModelAdapter):
    name = "WRF"

    def __init__(self) -> None:
        self._handles: dict[Path, netCDF4.Dataset] = {}
        self._accum_cache: dict[tuple[str, str, int], tuple[np.ndarray, dict]] = {}
        self._accum_check: dict[tuple[str, str, int], str | None] = {}  # None = passed
        self.skipped_files: list[tuple[Path, str]] = []

    # ------------------------------------------------------------------ I/O
    def _open(self, path: Path) -> netCDF4.Dataset:
        ds = self._handles.get(path)
        if ds is None:
            ds = netCDF4.Dataset(path, "r")
            ds.set_auto_mask(True)  # honour _FillValue / missing_value
            self._handles[path] = ds
        return ds

    def close(self) -> None:
        for ds in self._handles.values():
            try:
                ds.close()
            except Exception:  # pragma: no cover - best effort
                pass
        self._handles.clear()

    def _read_2d(self, dataset: ModelDataset, time_index: int, name: str) -> np.ndarray:
        """Read a native field at one time as float64 with NaN for missing.

        Returned south-up, exactly as stored (callers flip at the end).
        """
        slot = dataset.adapter_state.slots[time_index]
        nc = self._open(slot.path)
        if name not in nc.variables:
            raise VariableUnavailableError(f"Variable '{name}' is not present in {slot.path.name}.")
        var = nc.variables[name]
        if not var.dimensions or var.dimensions[0] != "Time":
            raise VariableUnavailableError(f"'{name}' in {slot.path.name} has no Time dimension.")
        arr = var[slot.index]
        return np.ma.filled(np.ma.asarray(arr, dtype=np.float64), np.nan)

    def _has(self, dataset: ModelDataset, time_index: int, name: str) -> bool:
        slot = dataset.adapter_state.slots[time_index]
        return name in self._open(slot.path).variables

    def _units(self, dataset: ModelDataset, time_index: int, name: str) -> str:
        slot = dataset.adapter_state.slots[time_index]
        return str(getattr(self._open(slot.path).variables[name], "units", "")).strip()

    # ------------------------------------------------------------ discovery
    def _skip(self, path: Path, reason: str) -> None:
        log.warning("Skipping WRF file %s: %s", path.name, reason)
        self.skipped_files.append((path, reason))

    @staticmethod
    def _read_header(path: Path) -> tuple[list[datetime], dict, list[tuple]]:
        """Times, global attributes and grid-corner signature at each time of one file.

        Raises ValueError with a readable reason if the file is unusable.
        """
        with netCDF4.Dataset(path, "r") as nc:
            for name in ("Times", "XLAT", "XLONG"):
                if name not in nc.variables:
                    raise ValueError(f"no '{name}' variable; not WRF history output?")
            raw = np.ma.getdata(nc.variables["Times"][:])
            times = [parse_wrf_time(str(s)) for s in netCDF4.chartostring(np.atleast_2d(raw))]
            attrs = {k: nc.getncattr(k) for k in nc.ncattrs()}

            # Integrity check: XTIME ("minutes since <epoch>") must agree with
            # Times. Damaged files often have zero-filled blocks, which shows
            # up as XTIME = 0 at a later time.
            if "XTIME" in nc.variables:
                xt = nc.variables["XTIME"]
                m = re.match(r"minutes since (\d{4}-\d{2}-\d{2})[ T_](\d{2}:\d{2}:\d{2})",
                             str(getattr(xt, "units", "")))
                if m:
                    epoch = parse_wrf_time(f"{m[1]}_{m[2]}")
                    for t, minutes in zip(times, np.atleast_1d(np.ma.getdata(xt[:]))):
                        implied = epoch + timedelta(minutes=float(minutes))
                        if abs((implied - t).total_seconds()) > 1.0:
                            raise ValueError(
                                f"XTIME={float(minutes):g} min ({implied:%Y-%m-%d %H:%M}) disagrees "
                                f"with Times ({t:%Y-%m-%d %H:%M}); file content appears corrupt"
                            )

            # Grid signature per time: the four corner coordinates (moving
            # nests change position between, and within, files).
            cols = [np.atleast_1d(np.ma.getdata(v[:, j, i])).astype(np.float64)
                    for v in (nc.variables["XLAT"], nc.variables["XLONG"])
                    for j in (0, -1) for i in (0, -1)]
            corners = [tuple(round(float(c[n]), 3) for c in cols) for n in range(len(times))]
        return times, attrs, corners

    def discover(self, directory: Path) -> list[ModelDataset]:
        directory = Path(directory)
        if not directory.is_dir():
            raise DiscoveryError(f"Input directory does not exist: {directory}")
        candidates = sorted(
            p for p in directory.iterdir() if p.is_file() and WRFOUT_PATTERN.match(p.name)
        )
        if not candidates:
            raise DiscoveryError(
                f"No WRF output files (wrfout_d<NN>_<YYYY-MM-DD_HH:MM:SS>) found in {directory}"
            )

        # Times implied by *all* file names (including unusable files), so the
        # run's real output cadence is known even when some files are bad.
        name_times: dict[int, set[datetime]] = defaultdict(set)
        for path in candidates:
            m = WRFOUT_PATTERN.match(path.name)
            name_times[int(m["domain"])].add(
                parse_wrf_time(m["time"][:13] + ":" + m["time"][14:16] + ":" + m["time"][17:19]))

        # Read the light-weight header info of every file.
        per_group: dict[tuple[int, str], list[tuple[datetime, _TimeSlot, dict, tuple]]] = defaultdict(list)
        for path in candidates:
            try:
                times, attrs, corners = self._read_header(path)
            except Exception as exc:  # noqa: BLE001 - a bad file is reported, not fatal
                self._skip(path, str(exc).strip() or exc.__class__.__name__)
                continue
            domain = int(attrs.get("GRID_ID", WRFOUT_PATTERN.match(path.name)["domain"]))
            sim_start = str(attrs.get("SIMULATION_START_DATE", "unknown")).strip()
            for i, t in enumerate(times):
                per_group[(domain, sim_start)].append((t, _TimeSlot(path, i), attrs, corners[i]))

        if not per_group:
            raise DiscoveryError(
                f"Found {len(candidates)} WRF-like files in {directory}, but none could be used "
                f"(e.g. {self.skipped_files[0][0].name}: {self.skipped_files[0][1]})."
            )

        run_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", directory.resolve().name).strip("-") or "wrf-run"
        starts_per_domain = Counter(d for d, _ in per_group)
        datasets = []
        for (domain, sim_start), entries in sorted(per_group.items()):
            entries.sort(key=lambda e: e[0])
            grids = self._time_grids(domain, entries)
            unique: list[tuple] = []
            for e in entries:
                if e[3] not in grids:
                    continue
                if unique and unique[-1][0] == e[0]:
                    log.warning(
                        "Duplicate valid time %s in %s and %s; using the first.",
                        e[0].isoformat(), unique[-1][1].path.name, e[1].path.name,
                    )
                    continue
                unique.append(e)
            if not unique:
                continue
            datasets.append(
                self._make_dataset(directory, run_id, domain, sim_start, unique, grids,
                                   split_by_start=starts_per_domain[domain] > 1,
                                   name_times=name_times.get(domain, set()))
            )
        if not datasets:
            raise DiscoveryError(
                f"No usable WRF output in {directory} (e.g. {self.skipped_files[0][0].name}: "
                f"{self.skipped_files[0][1]})."
            )
        return datasets

    def _time_grids(self, domain: int, entries) -> dict[tuple, GridDefinition]:
        """Grid for each usable corner signature of one domain (time-sorted entries).

        The outer domain cannot move: the most common signature wins and other
        files are corrupt. A nest may move (e.g. following a storm): every
        position whose XLAT/XLONG form a regular grid that is a whole-cell shift
        of the first position is kept. All positions use the first file's
        attributes for the CRS, so they share one CRS (WRF rewrites CEN_LON as
        a nest moves, which would otherwise shift a Mercator CRS).
        """
        sigs = Counter(e[3] for e in entries)
        attrs = entries[0][2]
        if domain == 1:
            ref_sig = sigs.most_common(1)[0][0]
            candidates = [ref_sig]
        else:
            candidates = list(dict.fromkeys(e[3] for e in entries))  # time order

        grids: dict[tuple, GridDefinition] = {}
        reasons: dict[tuple, str] = {}
        ref: GridDefinition | None = None
        for sig in candidates:
            slot = next(e[1] for e in entries if e[3] == sig)
            try:
                with netCDF4.Dataset(slot.path, "r") as nc:
                    xlat = np.asarray(nc.variables["XLAT"][slot.index], dtype=np.float64)
                    xlong = np.asarray(nc.variables["XLONG"][slot.index], dtype=np.float64)
                grid = wrf_grid(attrs, xlat, xlong)
            except GeoreferenceError as exc:
                reasons[sig] = f"XLAT/XLONG are not a valid grid ({exc})"
                continue
            if ref is None:
                ref = grid
            elif grid.pixel_offset(ref) is None:
                reasons[sig] = ("XLAT/XLONG are not a whole-cell shift of the domain's grid "
                                "(corrupt file?)")
                continue
            # Only the first position keeps its (large) lat/lon arrays.
            grids[sig] = grid if not grids else dataclasses.replace(grid, lat=None, lon=None)

        skipped = {p for p, _ in self.skipped_files}
        for e in entries:
            if e[3] in grids or e[1].path in skipped:
                continue
            skipped.add(e[1].path)
            self._skip(e[1].path, reasons.get(
                e[3], "XLAT/XLONG differ from the rest of the domain (corrupt file?)"))
        return grids

    def _make_dataset(self, directory, run_id, domain, sim_start, entries, grids,
                      split_by_start, name_times=frozenset()):
        attrs = entries[0][2]
        positions = list(dict.fromkeys(e[3] for e in entries))
        grid = grids[positions[0]]
        time_grids = [grids[e[3]] for e in entries] if len(positions) > 1 else None

        try:
            reference_time = parse_wrf_time(sim_start)
        except ValueError:
            reference_time = None
        accumulation_start = self._accumulation_start(domain, reference_time, entries)
        all_times = sorted(set(name_times) | {e[0] for e in entries})
        diffs = Counter(b - a for a, b in zip(all_times, all_times[1:]))
        output_interval = diffs.most_common(1)[0][0] if diffs else None

        dataset_id = f"d{domain:02d}"
        if split_by_start:
            dataset_id += "-" + (reference_time.strftime("%Y%m%dT%H%M%SZ") if reference_time else "unknown")

        meta: dict[str, Any] = {}
        for key in METADATA_ATTRS:
            if key in attrs:
                meta["wrf:" + key.lower().replace("-", "_")] = _to_python(attrs[key])
        version = _wrf_version(meta.get("wrf:title"))
        if version:
            meta["wrf:version"] = version
        meta["wrf:domain"] = dataset_id
        if "MAP_PROJ" in attrs:
            meta.setdefault("wrf:map_proj_char", MAP_PROJ_NAMES.get(int(attrs["MAP_PROJ"])))
        if output_interval:
            meta["wrf:output_interval_minutes"] = output_interval.total_seconds() / 60.0
        if accumulation_start:
            meta["wrf:accumulation_start"] = accumulation_start.strftime("%Y-%m-%dT%H:%M:%SZ")

        return ModelDataset(
            model=self.name,
            run_id=run_id,
            dataset_id=dataset_id,
            title=f"WRF domain {domain:02d} ({grid.width}x{grid.height}, {grid.transform.a / 1000:g} km)",
            files=list(dict.fromkeys(e[1].path for e in entries)),
            valid_times=[e[0] for e in entries],
            reference_time=reference_time,
            grid=grid,
            metadata=meta,
            source_dir=directory,
            time_sources=[e[1].path for e in entries],
            time_grids=time_grids,
            adapter_state=_WRFState(slots=[e[1] for e in entries],
                                    accumulation_start=accumulation_start,
                                    output_interval=output_interval),
        )

    @staticmethod
    def _accumulation_start(domain: int, sim_start: datetime | None, entries) -> datetime | None:
        """When this domain's precipitation accumulation began.

        Accumulations normally start at SIMULATION_START_DATE and survive
        restarts. A nest that was started later accumulates from its own start,
        which WRF records as START_DATE. START_DATE is also reset by restarts,
        so it is only trusted for nests (GRID_ID > 1) whose files all agree on
        a START_DATE later than the simulation start. Assumption: such a domain
        was started late rather than only ever written after a restart.
        """
        starts = {str(e[2].get("START_DATE", "")).strip() for e in entries}
        if domain > 1 and len(starts) == 1 and sim_start is not None:
            try:
                start = parse_wrf_time(starts.pop())
            except ValueError:
                return sim_start
            if start > sim_start:
                return start
        return sim_start

    # ---------------------------------------------------------- variables
    def supported_variables(self) -> list[str]:
        return ["surface_wind_speed", "precipitation", "precipitation_interval"]

    def compute(self, dataset: ModelDataset, variable: str, time_index: int) -> FieldResult | None:
        valid = dataset.valid_times[time_index]
        if variable == "surface_wind_speed":
            data, prov = self._surface_wind_speed(dataset, time_index)
            return FieldResult(variable, data[::-1].copy(), valid, provenance=prov)
        if variable == "precipitation":
            data, prov = self._checked_accumulation(dataset, time_index)
            return FieldResult(
                variable, data[::-1].copy(), valid,
                start_time=dataset.adapter_state.accumulation_start, end_time=valid,
                provenance=prov,
            )
        if variable == "precipitation_interval":
            return self._interval_precip(dataset, time_index)
        raise VariableUnavailableError(
            f"WRF adapter does not know how to compute '{variable}'. "
            f"Supported: {', '.join(self.supported_variables())}"
        )

    def _surface_wind_speed(self, dataset, t):
        if self._has(dataset, t, "U10") and self._has(dataset, t, "V10"):
            u = self._read_2d(dataset, t, "U10")
            v = self._read_2d(dataset, t, "V10")
            return wind_speed(u, v), {
                "source_variables": ["U10", "V10"],
                "method": "sqrt(U10^2 + V10^2); U10/V10 are 10 m diagnostics on mass points",
                "height_m": 10,
            }
        if self._has(dataset, t, "U") and self._has(dataset, t, "V"):
            # Fallback: lowest model level, destaggered from cell faces to mass
            # points. This is NOT a 10 m wind (the lowest half-level is
            # typically ~20-50 m AGL); provenance records that.
            log.warning(
                "%s: U10/V10 missing; using lowest-model-level U/V instead of 10 m wind.",
                dataset.dataset_id,
            )
            u = destagger(self._read_2d(dataset, t, "U")[0], axis=1)  # west_east_stag
            v = destagger(self._read_2d(dataset, t, "V")[0], axis=0)  # south_north_stag
            return wind_speed(u, v), {
                "source_variables": ["U", "V"],
                "method": (
                    "sqrt(U^2 + V^2) at the lowest model level after destaggering U "
                    "along west_east_stag and V along south_north_stag to mass points; "
                    "NOT a 10 m wind"
                ),
            }
        raise VariableUnavailableError(
            f"Cannot compute surface_wind_speed for {dataset.dataset_id}: the output has "
            "neither U10/V10 nor U/V."
        )

    def _accumulated_precip(self, dataset, t):
        key = (dataset.run_id, dataset.dataset_id, t)
        if key in self._accum_cache:
            return self._accum_cache[key]

        missing = [n for n in ("RAINC", "RAINNC") if not self._has(dataset, t, n)]
        if missing:
            raise VariableUnavailableError(
                f"Cannot compute precipitation for {dataset.dataset_id}: missing "
                f"{', '.join(missing)} in {dataset.adapter_state.slots[t].path.name}."
            )
        # RAINSH (shallow-cumulus precipitation) is accumulated separately from
        # RAINC/RAINNC in WRF, so it is part of the total when present.
        components = ["RAINC", "RAINNC"] + (["RAINSH"] if self._has(dataset, t, "RAINSH") else [])
        for name in components:
            units = self._units(dataset, t, name)
            if units.lower() != "mm":
                raise VariableUnavailableError(
                    f"{name} has units '{units}', expected 'mm'; refusing to guess a conversion."
                )
        total = sum(self._read_2d(dataset, t, n) for n in components)

        # Precipitation "buckets": with BUCKET_MM > 0, WRF keeps RAINC/RAINNC
        # below BUCKET_MM and counts resets in I_RAINC/I_RAINNC.
        bucket = float(dataset.metadata.get("wrf:bucket_mm") or -1)
        used_buckets = []
        if bucket > 0:
            for name in ("I_RAINC", "I_RAINNC"):
                if not self._has(dataset, t, name):
                    raise VariableUnavailableError(
                        f"BUCKET_MM={bucket} but {name} is missing, so total precipitation "
                        "cannot be reconstructed."
                    )
                total = total + bucket * self._read_2d(dataset, t, name)
                used_buckets.append(name)

        prov = {
            "source_variables": components + used_buckets,
            "method": " + ".join(components)
            + (f" + BUCKET_MM*({' + '.join(used_buckets)})" if used_buckets else "")
            + "; accumulated since the domain's accumulation start "
            f"({dataset.metadata.get('wrf:accumulation_start', 'unknown')})",
        }
        if len(self._accum_cache) > 8:
            self._accum_cache.pop(next(iter(self._accum_cache)))
        self._accum_cache[key] = (total, prov)
        return total, prov

    def _accumulation_problem(self, dataset, t) -> str | None:
        """Why the accumulated precipitation at t is implausible, or None.

        Accumulations can only grow, so the field is compared with the nearest
        earlier output that itself passed this check.
        """
        precomputed = dataset.adapter_state.accumulation_problems
        if precomputed is not None:
            return precomputed[t]
        key = (dataset.run_id, dataset.dataset_id, t)
        if key in self._accum_check:
            return self._accum_check[key]
        total, _ = self._accumulated_precip(dataset, t)
        problem = None
        if not np.isfinite(total).any():
            problem = "all values are missing"
        else:
            p = t - 1
            while p >= 0 and self._accumulation_problem(dataset, p) is not None:
                p -= 1
            if p >= 0:
                prev, _ = self._accumulated_precip(dataset, p)
                prev, _ = self._aligned(dataset, prev, p, t)
                compared = np.isfinite(prev)  # a moving nest's new cells have no past
                frac = float(np.sum(total < prev - NEGATIVE_TOLERANCE_MM)
                             / max(int(compared.sum()), 1))
                limit = MAX_DECREASE_FRACTION_MOVING if dataset.moving else MAX_DECREASE_FRACTION
                if frac > limit:
                    problem = (
                        f"accumulated precipitation is lower than at "
                        f"{dataset.valid_times[p]:%Y-%m-%d %H:%M} UTC at {100 * frac:.1f}% of grid "
                        f"points ({dataset.time_sources[t].name} is probably corrupt)"
                    )
        self._accum_check[key] = problem
        return problem

    @staticmethod
    def _aligned(dataset: ModelDataset, prev: np.ndarray, p: int, t: int
                 ) -> tuple[np.ndarray, tuple[int, int] | None]:
        """A south-up field at time index p, re-indexed to the grid position at
        time t (a moving nest's index (i, j) is a different place at each
        position). Cells outside the earlier position are NaN. Also returns
        the (rows, cols) shift, or None if the grid did not move."""
        if not dataset.moving:
            return prev, None
        src, dst = dataset.grid_at(p), dataset.grid_at(t)
        offset = dst.pixel_offset(src)
        if offset == (0, 0):
            return prev, None
        return shift_to_grid(prev[::-1], src, dst)[::-1], offset

    def prepare(self, dataset: ModelDataset, variables: list[str]) -> None:
        """Run the accumulation check for every time, in order (each check
        compares with the previous valid time, so in-order evaluation recurses
        only one level), and keep the results in the dataset's state."""
        if not {"precipitation", "precipitation_interval"} & set(variables):
            return
        state = dataset.adapter_state
        if state.accumulation_problems is None:
            state.accumulation_problems = [
                self._accumulation_problem(dataset, t) for t in range(len(dataset.valid_times))
            ]

    def _checked_accumulation(self, dataset, t):
        problem = self._accumulation_problem(dataset, t)
        if problem:
            raise FieldQualityError(problem)
        return self._accumulated_precip(dataset, t)

    @staticmethod
    def nominal_output_interval(dataset: ModelDataset) -> timedelta | None:
        """The run's output cadence: most common spacing of all file times,
        including files that were present but unusable."""
        return dataset.adapter_state.output_interval

    def _interval_precip(self, dataset, t):
        """Precipitation between output time t-1 and t.

        Only computed when the two times are consecutive outputs, i.e. their
        spacing equals the run's output interval (inferred from all file names,
        so unusable files still count as outputs). Across a gap (missing or
        unusable files) the difference would cover a longer,
        irregular period, so it is skipped rather than mixed into the series.
        """
        if t == 0:
            return None
        start, end = dataset.valid_times[t - 1], dataset.valid_times[t]
        nominal = self.nominal_output_interval(dataset)
        if end - start != nominal:
            log.info(
                "%s %s: previous output is %s earlier (nominal %s); skipping interval precipitation.",
                dataset.dataset_id, end.isoformat(), end - start, nominal,
            )
            return None
        a_end, prov = self._checked_accumulation(dataset, t)
        try:
            a_start, _ = self._checked_accumulation(dataset, t - 1)
        except FieldQualityError as exc:
            raise FieldQualityError(f"previous output time is unusable: {exc}") from None
        a_start, shift = self._aligned(dataset, a_start, t - 1, t)
        amount, n_bad = interval_precipitation(a_end, a_start)
        if n_bad:
            log.warning(
                "%s %s: accumulated precipitation decreased at %d grid points since %s "
                "(%s); those points are set to missing.",
                dataset.dataset_id, end.isoformat(), n_bad, start.isoformat(),
                "cells re-interpolated as the nest moved" if dataset.moving
                else "accumulation reset?",
            )
        return FieldResult(
            "precipitation_interval", amount[::-1].copy(), end,
            start_time=start, end_time=end,
            provenance={
                "source_variables": prov["source_variables"],
                "method": f"[{prov['method'].split(';')[0]}](t) - [same](t - {end - start}), "
                "differences > -0.01 mm clipped to 0, larger negative differences set missing",
                "negative_points_masked": n_bad,
                **({"moving_nest_shift_cells": {"rows": shift[0], "cols": shift[1]},
                    "moving_nest_note": "the earlier field was shifted to the nest's position "
                                        "at the valid time; cells new to the nest are missing"}
                   if shift else {}),
            },
        )
