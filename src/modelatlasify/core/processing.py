"""End-to-end pipeline: discover -> compute -> GeoTIFF/PNG -> STAC.

Valid times are processed independently (compute, GeoTIFF, map), optionally in
a pool of worker processes; the STAC catalog is assembled in the main process.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

import modelatlasify.variables  # noqa: F401 - registers the standard variables
from modelatlasify.core.catalog import AssetRecord, CatalogBuilder, ItemRecord
from modelatlasify.core.errors import DiscoveryError, FieldQualityError
from modelatlasify.core.model import FieldResult, ModelAdapter, ModelDataset
from modelatlasify.core.variables import VariableDefinition, get_variable
from modelatlasify.geospatial.geotiff import write_geotiff
from modelatlasify.models import get_adapter

log = logging.getLogger(__name__)


@dataclass
class PipelineOptions:
    inputs: list[Path]
    output: Path
    model: str = "wrf"
    variables: list[str] = field(
        default_factory=lambda: ["surface_wind_speed", "precipitation", "precipitation_interval"]
    )
    plots: bool = True
    domains: list[str] | None = None  # e.g. ["d01"]; None = all
    processes: int = 1  # worker processes for the per-time work


@dataclass
class PipelineResult:
    catalog_path: Path
    datasets: list[ModelDataset]
    n_items: int
    n_geotiffs: int
    n_plots: int
    skipped_files: list[tuple[Path, str]]
    quality_skips: list[str]
    seconds: float


def discover(adapter: ModelAdapter, inputs: list[Path], domains: list[str] | None,
             report: Callable[[str], None]) -> dict[str, list[ModelDataset]]:
    """Discover datasets in every input directory, grouped by (unique) run id."""
    runs: dict[str, list[ModelDataset]] = {}
    for directory in inputs:
        found = adapter.discover(directory)
        if domains:
            found = [d for d in found if d.dataset_id.split("-")[0] in domains]
            if not found:
                raise DiscoveryError(f"No output for domain(s) {', '.join(domains)} in {directory}")
        run_id = found[0].run_id
        if run_id in runs:  # two inputs with the same directory name
            n = 2
            while f"{run_id}-{n}" in runs:
                n += 1
            for d in found:
                d.run_id = f"{run_id}-{n}"
            run_id = f"{run_id}-{n}"
        runs[run_id] = found
        report(f"Found {adapter.name} run '{run_id}' in {directory}:")
        for d in found:
            g = d.grid
            report(
                f"  {d.dataset_id}: {len(d.files)} file(s), {len(d.valid_times)} time(s) "
                f"{d.valid_times[0]:%Y-%m-%d %H:%M} -> {d.valid_times[-1]:%Y-%m-%d %H:%M} UTC, "
                f"{g.width}x{g.height} grid, georeference fit error {g.max_georeference_error_m:.1f} m."
            )
    return runs


# --------------------------------------------------------- per-time worker
@dataclass
class _TimeResult:
    """What a worker sends back for one valid time (no grids or field arrays)."""

    run_id: str
    dataset_id: str
    time_index: int
    assets: list[AssetRecord] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    quality_skips: list[str] = field(default_factory=list)
    n_tif: int = 0
    n_png: int = 0


# Set by _init_worker, in each worker process (or in-process when serial).
_worker: dict = {}


def _init_worker(model: str, datasets: dict[tuple[str, str], ModelDataset], output: Path,
                 variables: list[str], plots: bool, log_levels: dict[str, int]) -> None:
    for name, level in log_levels.items():
        if name == "":
            logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
        logging.getLogger(name).setLevel(level)
    plot_field = None
    if plots:
        from modelatlasify.plotting.maps import plot_field  # heavy import; only when needed
    _worker.clear()
    _worker.update(adapter=get_adapter(model), datasets=datasets,
                   builder=CatalogBuilder(output), variables=variables, plot_field=plot_field)


def _close_worker() -> None:
    adapter = _worker.get("adapter")
    if adapter is not None:
        adapter.close()
    _worker.clear()


def _process_time(task: tuple[str, str, int]) -> _TimeResult:
    """Compute, check, write and plot every variable at one valid time."""
    run_id, dataset_id, ti = task
    adapter: ModelAdapter = _worker["adapter"]
    plot_field = _worker["plot_field"]
    ds: ModelDataset = _worker["datasets"][(run_id, dataset_id)]
    valid = ds.valid_times[ti]
    item_dir = _worker["builder"].item_dir(ds, valid)
    res = _TimeResult(run_id, dataset_id, ti)
    for vname in _worker["variables"]:
        var = get_variable(vname)
        try:
            fld = adapter.compute(ds, vname, ti)
            if fld is None:
                continue
            check_field(fld, var)
        except FieldQualityError as exc:
            reason = str(exc)
            if ds.reference_time == valid:
                reason += (" (this is the initialization time, where models often "
                           "write zero-valued diagnostics)")
            res.skipped.append({"variable": vname, "reason": reason})
            res.quality_skips.append(f"{ds.dataset_id} {valid:%Y-%m-%d %H:%M} {vname}: {reason}")
            continue
        tif = item_dir / f"{vname}.tif"
        info = write_geotiff(tif, fld, ds.grid, var, extra_tags={
            "model": ds.model, "run_id": ds.run_id, "dataset": ds.dataset_id,
        })
        if plot_field is not None:
            png = item_dir / f"{vname}.png"
            title, subtitle = _titles(ds, fld, var.long_name, var.units)
            plot_field(png, fld, ds.grid, var, title, subtitle)
        meta = dataclasses.replace(fld, data=None)  # the catalog needs no pixels
        res.assets.append(AssetRecord(vname, tif, vname, "data", meta, info))
        res.n_tif += 1
        if plot_field is not None:
            res.assets.append(AssetRecord(f"{vname}_map", png, vname, "visual", meta))
            res.n_png += 1
    return res


def _log_levels() -> dict[str, int]:
    names = ("", "modelatlasify", "modelatlasify.models")
    return {n: logging.getLogger(n).level for n in names}


# ---------------------------------------------------------------- pipeline
def run_pipeline(options: PipelineOptions, report: Callable[[str], None] = print,
                 progress: bool = True) -> PipelineResult:
    t0 = time.monotonic()
    if options.processes < 1:
        raise ValueError("processes must be >= 1")
    adapter = get_adapter(options.model)
    unsupported = [v for v in options.variables if v not in adapter.supported_variables()]
    if unsupported:
        raise DiscoveryError(
            f"{adapter.name} adapter cannot produce: {', '.join(unsupported)}. "
            f"Supported: {', '.join(adapter.supported_variables())}"
        )

    pool = None
    try:
        runs = discover(adapter, options.inputs, options.domains, report)
        skipped = list(getattr(adapter, "skipped_files", []))
        if skipped:
            report(f"WARNING: skipped {len(skipped)} unusable file(s), e.g. "
                   f"{skipped[0][0].name}: {skipped[0][1]}")

        all_datasets = {(d.run_id, d.dataset_id): d for dss in runs.values() for d in dss}
        for ds in all_datasets.values():
            adapter.prepare(ds, options.variables)
        if options.plots:
            from modelatlasify.plotting.maps import prefetch_geography

            for ds in all_datasets.values():
                prefetch_geography(ds.grid)
        adapter.close()  # workers open their own file handles

        init_args = (options.model, all_datasets, options.output, options.variables,
                     options.plots, _log_levels())
        n_tasks = sum(len(d.valid_times) for d in all_datasets.values())
        processes = min(options.processes, n_tasks)
        if processes > 1:
            # spawn, not fork: forking a process with open netCDF/HDF5 handles is unsafe.
            pool = multiprocessing.get_context("spawn").Pool(
                processes, initializer=_init_worker, initargs=init_args)
            mapper = lambda tasks: pool.imap(_process_time, tasks, chunksize=1)  # noqa: E731
        else:
            _init_worker(*init_args)
            mapper = lambda tasks: map(_process_time, tasks)  # noqa: E731

        builder = CatalogBuilder(options.output)
        n_items = n_tif = n_png = 0
        quality_skips: list[str] = []
        for run_id, datasets in runs.items():
            items = []
            for ds in datasets:
                tasks = [(run_id, ds.dataset_id, ti) for ti in range(len(ds.valid_times))]
                results = tqdm(mapper(tasks), total=len(tasks), desc=f"{run_id} {ds.dataset_id}",
                               unit="time", file=sys.stderr, disable=not progress)
                for res in results:
                    ti = res.time_index
                    rec = ItemRecord(ds, ti, builder.item_dir(ds, ds.valid_times[ti]),
                                     source_file=ds.time_sources[ti].name,
                                     assets=res.assets, skipped=res.skipped)
                    items.append(builder.build_item(rec))
                    n_items += 1
                    n_tif += res.n_tif
                    n_png += res.n_png
                    quality_skips.extend(res.quality_skips)
            builder.add_run(run_id, datasets, items, options.variables)
        catalog_path = builder.save()
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        else:
            _close_worker()
        adapter.close()

    return PipelineResult(
        catalog_path=catalog_path,
        datasets=[d for ds in runs.values() for d in ds],
        n_items=n_items, n_geotiffs=n_tif, n_plots=n_png,
        skipped_files=skipped, quality_skips=quality_skips, seconds=time.monotonic() - t0,
    )


def check_field(fld: FieldResult, var: VariableDefinition) -> None:
    """Model-independent sanity checks before a field is published."""
    finite = fld.data[np.isfinite(fld.data)]
    if finite.size == 0:
        raise FieldQualityError("all values are missing")
    if var.constant_is_suspicious and finite.min() == finite.max():
        raise FieldQualityError(
            f"field is constant ({float(finite.min()):g} everywhere); input data is probably "
            "zero-filled or corrupt"
        )
    if var.max_zero_fraction is not None:
        frac = float(np.mean(finite == 0.0))
        if frac > var.max_zero_fraction:
            raise FieldQualityError(
                f"{100 * frac:.1f}% of grid points are exactly 0; input data is probably "
                "partly zero-filled or corrupt"
            )


def _titles(ds: ModelDataset, fld: FieldResult, long_name: str, units: str) -> tuple[str, str]:
    valid = fld.valid_time
    if fld.variable == "precipitation_interval" and fld.start_time:
        hours = (fld.end_time - fld.start_time).total_seconds() / 3600.0
        title = (f"{hours:g}-h precipitation ({units}), "
                 f"{fld.start_time:%Y-%m-%d %H:%M} to {fld.end_time:%H:%M} UTC")
    elif fld.variable == "precipitation" and ds.reference_time:
        title = f"{long_name} since {ds.reference_time:%Y-%m-%d %H:%M} UTC ({units})"
    else:
        title = f"{long_name} ({units})"
    sub = f"Valid {valid:%Y-%m-%d %H:%M} UTC"
    if ds.reference_time:
        lead = (valid - ds.reference_time).total_seconds() / 3600.0
        sub += f"  |  init {ds.reference_time:%Y-%m-%d %H:%M} UTC (+{lead:g} h)"
    sub += f"  |  {ds.model} run {ds.run_id}, {ds.dataset_id}"
    return title, sub
