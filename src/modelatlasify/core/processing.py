"""End-to-end pipeline: discover -> compute -> GeoTIFF/PNG -> STAC.

Valid times are processed independently (compute, GeoTIFF, map), optionally in
a pool of worker processes; the STAC catalog is assembled in the main process.
With plots on, each run then also gets composite maps (the outermost domain
with its nests overlaid) under ``<run>/model-images/``, outside the catalog,
and a looping HTML animation of each area's maps of each variable.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np
from tqdm import tqdm

import modelatlasify.variables  # noqa: F401 - registers the standard variables
from modelatlasify.core.areas import FULL_AREA, SubArea
from modelatlasify.core.catalog import AssetRecord, CatalogBuilder, ItemRecord
from modelatlasify.core.errors import DiscoveryError, FieldQualityError, NoModelOutputError
from modelatlasify.core.model import FieldResult, ModelAdapter, ModelDataset
from modelatlasify.core.variables import VariableDefinition, get_variable
from modelatlasify.geospatial.geotiff import write_geotiff
from modelatlasify.models import AVAILABLE_MODELS, get_adapter

log = logging.getLogger(__name__)


@dataclass
class PipelineOptions:
    inputs: list[Path]
    output: Path
    #: "auto" (every model whose output is found in an input directory), one
    #: model name, or a comma-separated list, e.g. "wrf,dwm".
    model: str = "auto"
    #: None = every variable the models found can produce.
    variables: list[str] | None = None
    plots: bool = True
    domains: list[str] | None = None  # e.g. ["d01"]; None = all
    processes: int = 1  # worker processes for the per-time work
    image_prefix: str = "uwincm"  # file name prefix of the composite maps
    cut_dirs: int = 0  # leading input-path directories dropped from the output folder
    sub_areas: list[SubArea] = field(default_factory=list)  # zoomed composites, in lon/lat
    full_plots: bool = True  # whole-domain composites (model-images/full/)


@dataclass
class PipelineResult:
    catalog_path: Path
    datasets: list[ModelDataset]
    n_items: int
    n_geotiffs: int
    n_plots: int
    n_composites: int
    n_animations: int
    skipped_files: list[tuple[Path, str]]
    quality_skips: list[str]
    seconds: float


def run_location(directory: Path, cut_dirs: int = 0) -> tuple[PurePosixPath, str, bool]:
    """Where a run goes under the output directory, and its id.

    The run folder mirrors the input directory's absolute path (``..`` is
    normalised; symlinks are kept as typed), minus its first ``cut_dirs``
    directories, like wget's --cut-dirs. Cutting everything keeps just the
    run directory's own name (the third value is then True, for a warning).
    The id is the folder's path with "/" replaced by "--", restricted to
    characters that are safe in STAC ids and URLs.
    """
    parts = Path(os.path.abspath(directory)).parts[1:]
    over_cut = cut_dirs >= len(parts)
    kept = parts[-1:] if over_cut else parts[cut_dirs:]
    run_id = "--".join(re.sub(r"[^A-Za-z0-9_.-]+", "-", p).strip("-") or "x" for p in kept)
    return PurePosixPath(*kept), run_id, over_cut


def model_names(model: str) -> list[str]:
    """Adapter names for PipelineOptions.model ("auto", "wrf", "wrf,dwm", ...)."""
    if model.strip().lower() == "auto":
        return list(AVAILABLE_MODELS)
    return [m.strip().lower() for m in model.split(",") if m.strip()]


def discover(adapters: list[ModelAdapter], inputs: list[Path], domains: list[str] | None,
             report: Callable[[str], None], cut_dirs: int = 0) -> dict[str, list[ModelDataset]]:
    """Discover datasets in every input directory, grouped by run id.

    Every adapter looks at every input directory; a model whose output is
    absent (e.g. a coupled run with the wave model switched off) is skipped.
    """
    runs: dict[str, list[ModelDataset]] = {}
    for directory in inputs:
        run_path, run_id, over_cut = run_location(directory, cut_dirs)
        if run_id in runs:
            report(f"WARNING: {directory} is listed more than once; processing it once.")
            continue
        if over_cut:
            report(f"WARNING: --cut-dirs={cut_dirs} removes all parent directories of "
                   f"{os.path.abspath(directory)}; using '{run_path}' only.")
        if not Path(directory).is_dir():
            raise DiscoveryError(f"Input directory does not exist: {directory}")
        found, absent = [], []
        for adapter in adapters:
            try:
                found += adapter.discover(directory)
            except NoModelOutputError as exc:
                absent.append(str(exc))
        if not found:
            raise DiscoveryError("; ".join(absent))
        if domains:
            wanted = {d.lower() for d in domains}
            found = [d for d in found if d.dataset_id.split("-")[0].lower() in wanted]
            if not found:
                raise DiscoveryError(f"No output for domain(s) {', '.join(domains)} in {directory}")
        for d in found:
            d.run_id, d.run_path = run_id, run_path.as_posix()
        runs[run_id] = found
        models = " + ".join(dict.fromkeys(d.model for d in found))
        report(f"Found {models} run '{run_path}' in {directory}:")
        for d in found:
            g = d.grid
            report(
                f"  {d.dataset_id}: {len(d.files)} file(s), {len(d.valid_times)} time(s) "
                f"{d.valid_times[0]:%Y-%m-%d %H:%M} -> {d.valid_times[-1]:%Y-%m-%d %H:%M} UTC, "
                f"{g.width}x{g.height} grid, georeference fit error {g.max_georeference_error_m:.1f} m"
                + (f", moving nest ({len(d.distinct_grids())} positions)." if d.moving else ".")
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


def _init_worker(models: list[str], datasets: dict[tuple[str, str], ModelDataset], output: Path,
                 variables: list[str], plots: bool, log_levels: dict[str, int]) -> None:
    for name, level in log_levels.items():
        if name == "":
            logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
        logging.getLogger(name).setLevel(level)
    plot_field = plot_composite = None
    if plots:  # heavy imports; only when needed
        from modelatlasify.plotting.maps import plot_composite, plot_field
    _worker.clear()
    adapters = {a.name: a for a in map(get_adapter, models)}
    _worker.update(adapters=adapters, datasets=datasets,
                   builder=CatalogBuilder(output), variables=variables, plot_field=plot_field,
                   plot_composite=plot_composite)


def _close_worker() -> None:
    for adapter in _worker.get("adapters", {}).values():
        adapter.close()
    _worker.clear()


def _process_time(task: tuple[str, str, int]) -> _TimeResult:
    """Compute, check, write and plot every variable at one valid time."""
    run_id, dataset_id, ti = task
    ds: ModelDataset = _worker["datasets"][(run_id, dataset_id)]
    adapter: ModelAdapter = _worker["adapters"][ds.model]
    plot_field = _worker["plot_field"]
    valid = ds.valid_times[ti]
    item_dir = _worker["builder"].item_dir(ds, valid)
    res = _TimeResult(run_id, dataset_id, ti)
    supported = adapter.supported_variables()
    for vname in (v for v in _worker["variables"] if v in supported):
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
        info = write_geotiff(tif, fld, ds.grid_at(ti), var, extra_tags={
            "model": ds.model, "run_id": ds.run_id, "dataset": ds.dataset_id,
        })
        if plot_field is not None:
            png = item_dir / f"{vname}.png"
            title, subtitle = _titles(ds, fld, var.long_name, var.units)
            plot_field(png, fld, ds.grid_at(ti), var, title, subtitle)
        meta = dataclasses.replace(fld, data=None)  # the catalog needs no pixels
        res.assets.append(AssetRecord(vname, tif, vname, "data", meta, info))
        res.n_tif += 1
        if plot_field is not None:
            res.assets.append(AssetRecord(f"{vname}_map", png, vname, "visual", meta))
            res.n_png += 1
    return res


# ------------------------------------------------------ composite maps
def composite_path(run_dir: Path, plot_area: str, variable: str, prefix: str,
                   valid: datetime) -> Path:
    """``<run>/model-images/<area>/<variable>/<prefix>.<area>.<variable>.<time>.png``"""
    name = f"{prefix}.{plot_area}.{variable}.{valid:%Y%m%dT%H%M%SZ}.png"
    return run_dir / "model-images" / plot_area / variable / name


@dataclass
class _CompositeTask:
    """One composite map: the base domain's product plus its nests' (if any)."""

    run_id: str
    variable: str
    path: Path
    area: SubArea | None  # None = the base domain's whole area
    edge: bool  # outline the base domain's edge (dashed)
    base: tuple[str, int, Path, FieldResult]  # (dataset_id, time index, GeoTIFF, metadata)
    # (dataset_id, time index or None for a fixed nest without output then,
    #  GeoTIFF or None = outline only)
    nests: list[tuple[str, int | None, Path | None]]


def _composite_tasks(run_dir: Path, datasets: list[ModelDataset], variables: list[str],
                     products: dict[tuple[str, datetime, str], tuple[Path, FieldResult]],
                     area: SubArea | None, prefix: str, edge: bool = False
                     ) -> list[_CompositeTask]:
    """Composite maps of one run over the outermost domain's whole area, or
    over a lon/lat sub-area.

    ``products`` maps (dataset_id, valid time, variable) to the published
    GeoTIFF. A nest is overlaid only when its product covers the same period as
    the base's (e.g. not when a nest started accumulating later); otherwise, or
    when it has no product at that time, only its outline is drawn. Moving
    nests are drawn where they were at that time; a moving nest without
    output at that time is left out, since its position is unknown.
    """
    base, *nests = sorted(datasets, key=lambda d: d.dataset_id)
    plot_area = FULL_AREA if area is None else area.name
    tasks = []
    nest_times = {n.dataset_id: {t: i for i, t in enumerate(n.valid_times)} for n in nests}
    for bi, valid in enumerate(base.valid_times):
        for vname in variables:
            if (base.dataset_id, valid, vname) not in products:
                continue  # not defined or failed sanity checks
            tif, meta = products[(base.dataset_id, valid, vname)]
            overlays = []
            for nest in nests:
                ni = nest_times[nest.dataset_id].get(valid)
                if ni is None and nest.moving:
                    continue
                p = products.get((nest.dataset_id, valid, vname))
                same_period = p is not None and (p[1].start_time, p[1].end_time) == (
                    meta.start_time, meta.end_time)
                overlays.append((nest.dataset_id, ni, p[0] if same_period else None))
            tasks.append(_CompositeTask(
                base.run_id, vname, composite_path(run_dir, plot_area, vname, prefix, valid),
                area, edge, (base.dataset_id, bi, tif, meta), overlays))
    return tasks


def _area_in_run(area: SubArea, datasets: list[ModelDataset]) -> bool:
    """Whether the sub-area overlaps the run's outermost domain."""
    from modelatlasify.plotting.maps import subarea_view, view_overlaps

    base = min(datasets, key=lambda d: d.dataset_id)
    view = subarea_view(area)
    return any(view_overlaps(view, g) for g in base.distinct_grids())


def _read_tif(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1)


def _plot_composite(task: _CompositeTask) -> Path:
    datasets = _worker["datasets"]
    base_id, bi, base_tif, meta = task.base
    base_ds: ModelDataset = datasets[(task.run_id, base_id)]
    var = get_variable(task.variable)
    nests = []
    for nid, ni, tif in task.nests:
        nest: ModelDataset = datasets[(task.run_id, nid)]
        grid = nest.grid if ni is None else nest.grid_at(ni)
        nests.append((None if tif is None else _read_tif(tif), grid))
    title, subtitle = _titles(base_ds, meta, var.long_name, var.units)
    overlaid = [(nid, grid) for (nid, _, tif), (_, grid) in zip(task.nests, nests)
                if tif is not None]
    if task.area is not None:  # name only the nests that are in the area
        from modelatlasify.plotting.maps import subarea_view, view_overlaps

        view = subarea_view(task.area)
        overlaid = [(nid, g) for nid, g in overlaid if view_overlaps(view, g)]
    subtitle += "".join(f" + {nid}" for nid, _ in overlaid)
    if task.area is not None:
        subtitle += f"  |  area '{task.area.name}'"
    return _worker["plot_composite"](task.path, (_read_tif(base_tif), base_ds.grid_at(bi)),
                                     nests, var, title, subtitle, area=task.area,
                                     domain_edge=task.edge)


def _log_levels() -> dict[str, int]:
    names = ("", "modelatlasify", "modelatlasify.models")
    return {n: logging.getLogger(n).level for n in names}


# ---------------------------------------------------------------- pipeline
def run_pipeline(options: PipelineOptions, report: Callable[[str], None] = print,
                 progress: bool = True) -> PipelineResult:
    t0 = time.monotonic()
    if options.processes < 1:
        raise ValueError("processes must be >= 1")
    names = model_names(options.model)
    adapter_list = [get_adapter(n) for n in names]  # KeyError for an unknown model
    adapters = {a.name: a for a in adapter_list}

    pool = None
    try:
        runs = discover(adapter_list, options.inputs, options.domains, report, options.cut_dirs)
        used = [adapters[m] for m in dict.fromkeys(d.model for ds in runs.values() for d in ds)]
        variables = _variables_for(options.variables, used)
        for run_id in list(runs):  # leave out models that make none of the variables
            keep = [d for d in runs[run_id]
                    if set(variables) & set(adapters[d.model].supported_variables())]
            for d in runs[run_id]:
                if d not in keep:
                    report(f"  ({d.model} {d.dataset_id} of '{d.run_path}' skipped: it provides "
                           "none of the requested variables)")
            if keep:
                runs[run_id] = keep
            else:
                del runs[run_id]
        if not runs:
            raise DiscoveryError("None of the inputs provides the requested variables.")
        skipped = [s for a in adapter_list for s in getattr(a, "skipped_files", [])]
        if skipped:
            report(f"WARNING: skipped {len(skipped)} unusable file(s), e.g. "
                   f"{skipped[0][0].name}: {skipped[0][1]}")

        all_datasets = {(d.run_id, d.dataset_id): d for dss in runs.values() for d in dss}
        for ds in all_datasets.values():
            adapter = adapters[ds.model]
            adapter.prepare(ds, [v for v in variables if v in adapter.supported_variables()])
        if options.plots:
            from modelatlasify.plotting.maps import prefetch_geography

            for ds in all_datasets.values():
                prefetch_geography(ds.grid)
            for area in options.sub_areas:
                prefetch_geography(area)
        for adapter in adapter_list:
            adapter.close()  # workers open their own file handles

        init_args = (names, all_datasets, options.output, variables, options.plots,
                     _log_levels())
        n_tasks = sum(len(d.valid_times) for d in all_datasets.values())
        processes = min(options.processes, n_tasks)
        if processes > 1:
            # spawn, not fork: forking a process with open netCDF/HDF5 handles is unsafe.
            pool = multiprocessing.get_context("spawn").Pool(
                processes, initializer=_init_worker, initargs=init_args)
            mapper = lambda tasks: pool.imap(_process_time, tasks, chunksize=1)  # noqa: E731
            mapper_composite = lambda tasks: pool.imap(  # noqa: E731
                _plot_composite, tasks, chunksize=1)
        else:
            _init_worker(*init_args)
            mapper = lambda tasks: map(_process_time, tasks)  # noqa: E731
            mapper_composite = lambda tasks: map(_plot_composite, tasks)  # noqa: E731

        builder = CatalogBuilder(options.output)
        n_items = n_tif = n_png = n_comp = n_anim = 0
        quality_skips: list[str] = []
        for run_id, datasets in runs.items():
            items = []
            products: dict[tuple[str, datetime, str], tuple[Path, FieldResult]] = {}
            for ds in datasets:
                tasks = [(run_id, ds.dataset_id, ti) for ti in range(len(ds.valid_times))]
                results = tqdm(mapper(tasks), total=len(tasks), desc=f"{ds.run_name} {ds.dataset_id}",
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
                    for a in res.assets:
                        if a.kind == "data":
                            products[(ds.dataset_id, ds.valid_times[ti], a.variable)] = (
                                a.path, a.field)
            if options.plots:
                tasks = []
                for group in _model_groups(datasets):  # each model has its own maps
                    model = group[0].model
                    for area in ([None] if options.full_plots else []) + options.sub_areas:
                        if area is not None and not _area_in_run(area, group):
                            report(f"WARNING: sub-area '{area.name}' is entirely outside the "
                                   f"{model} domain of run '{group[0].run_path}'; no {model} "
                                   "maps made for it.")
                            continue
                        tasks += _composite_tasks(builder.run_dir(group[0]), group, variables,
                                                  products, area, options.image_prefix,
                                                  edge=adapters[model].outline_domain_edge)
                for _ in tqdm(mapper_composite(tasks), total=len(tasks),
                              desc=f"{datasets[0].run_name} composite maps", unit="map", file=sys.stderr,
                              disable=not progress):
                    n_comp += 1
                from modelatlasify.plotting.animation import write_animation

                for var_dir in sorted({t.path.parent for t in tasks}):
                    if write_animation(var_dir) is not None:
                        n_anim += 1
            builder.add_run(run_id, datasets, items,
                            [v for v in variables if any(v in adapters[d.model].supported_variables()
                                                         for d in datasets)])
        catalog_path = builder.save()
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        else:
            _close_worker()
        for adapter in adapter_list:
            adapter.close()

    return PipelineResult(
        catalog_path=catalog_path,
        datasets=[d for ds in runs.values() for d in ds],
        n_items=n_items, n_geotiffs=n_tif, n_plots=n_png, n_composites=n_comp,
        n_animations=n_anim,
        skipped_files=skipped, quality_skips=quality_skips, seconds=time.monotonic() - t0,
    )


def _variables_for(requested: list[str] | None, adapters: list[ModelAdapter]) -> list[str]:
    """The variables to produce: as requested (each made by the models that
    can), or everything the models found can produce."""
    supported = [v for a in adapters for v in a.supported_variables()]
    if requested is None:
        return list(dict.fromkeys(supported))
    missing = [v for v in requested if v not in supported]
    if missing:
        found = ", ".join(f"{a.name} ({', '.join(a.supported_variables())})" for a in adapters)
        raise DiscoveryError(f"No model found in the input can produce: {', '.join(missing)}. "
                             f"Found: {found}")
    return list(requested)


def _model_groups(datasets: list[ModelDataset]) -> list[list[ModelDataset]]:
    """Datasets grouped by model, in order of first appearance."""
    groups: dict[str, list[ModelDataset]] = {}
    for d in datasets:
        groups.setdefault(d.model, []).append(d)
    return list(groups.values())


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
        if fld.provenance.get("hourly_equivalent_scale", 1.0) != 1.0:
            # e.g. a 15-min PREC_ACC accumulation scaled up to mm per hour.
            title = (f"1-h equivalent precipitation ({units}) from {60 * hours:g}-min total, "
                     f"{fld.start_time:%Y-%m-%d %H:%M} to {fld.end_time:%H:%M} UTC")
        else:
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
    sub += f"  |  {ds.model} run {ds.run_name}"
    if ds.dataset_id != ds.model:  # e.g. ", d01" for WRF; nothing extra for DWM
        sub += f", {ds.dataset_id}"
    return title, sub
