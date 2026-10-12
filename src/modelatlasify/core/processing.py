"""End-to-end pipeline: discover -> compute -> GeoTIFF/PNG -> STAC."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import modelatlasify.variables  # noqa: F401 - registers the standard variables
from modelatlasify.core.catalog import AssetRecord, CatalogBuilder, ItemRecord
import numpy as np

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
                f"{g.width}x{g.height} grid, georeference fit error {g.max_georeference_error_m:.1f} m"
            )
    return runs


def run_pipeline(options: PipelineOptions, report: Callable[[str], None] = print) -> PipelineResult:
    t0 = time.monotonic()
    adapter = get_adapter(options.model)
    unsupported = [v for v in options.variables if v not in adapter.supported_variables()]
    if unsupported:
        raise DiscoveryError(
            f"{adapter.name} adapter cannot produce: {', '.join(unsupported)}. "
            f"Supported: {', '.join(adapter.supported_variables())}"
        )
    plot_field = None
    if options.plots:
        from modelatlasify.plotting.maps import plot_field  # heavy import; only when needed

    try:
        runs = discover(adapter, options.inputs, options.domains, report)
        skipped = list(getattr(adapter, "skipped_files", []))
        if skipped:
            report(f"WARNING: skipped {len(skipped)} unusable file(s), e.g. "
                   f"{skipped[0][0].name}: {skipped[0][1]}")

        builder = CatalogBuilder(options.output)
        n_items = n_tif = n_png = 0
        quality_skips: list[str] = []
        for run_id, datasets in runs.items():
            items = []
            for ds in datasets:
                total = len(ds.valid_times)
                for ti, valid in enumerate(ds.valid_times):
                    item_dir = builder.item_dir(ds, valid)
                    rec = ItemRecord(ds, ti, item_dir, source_file=ds.time_sources[ti].name)
                    made = []
                    for vname in options.variables:
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
                            rec.skipped.append({"variable": vname, "reason": reason})
                            quality_skips.append(f"{ds.dataset_id} {valid:%Y-%m-%d %H:%M} {vname}: {reason}")
                            continue
                        tif = item_dir / f"{vname}.tif"
                        info = write_geotiff(tif, fld, ds.grid, var, extra_tags={
                            "model": ds.model, "run_id": ds.run_id, "dataset": ds.dataset_id,
                        })
                        rec.assets.append(AssetRecord(vname, tif, vname, "data", fld, info))
                        n_tif += 1
                        made.append(vname)
                        if plot_field is not None:
                            png = item_dir / f"{vname}.png"
                            title, subtitle = _titles(ds, fld, var.long_name, var.units)
                            plot_field(png, fld, ds.grid, var, title, subtitle)
                            rec.assets.append(AssetRecord(f"{vname}_map", png, vname, "visual", fld))
                            n_png += 1
                    items.append(builder.build_item(rec))
                    n_items += 1
                    report(f"  [{run_id} {ds.dataset_id} {ti + 1}/{total}] "
                           f"{valid:%Y-%m-%d %H:%M} UTC: {', '.join(made) or 'nothing'}"
                           + (f" (skipped {', '.join(x['variable'] for x in rec.skipped)})"
                              if rec.skipped else ""))
            builder.add_run(run_id, datasets, items, options.variables)
        catalog_path = builder.save()
    finally:
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
