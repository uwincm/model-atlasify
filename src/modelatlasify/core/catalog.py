"""Build a STAC catalog describing model runs and their generated assets.

Structure (one Collection per model run, one Item per dataset/domain and
valid time)::

    <output>/catalog.json                                     Catalog
    <output>/<run_path>/collection.json                       Collection (the run)
    <output>/<run_path>/items/<item_id>/<item_id>.json        Item (domain @ valid time)
    <output>/<run_path>/items/<item_id>/<variable>.tif|.png   Assets

``<run_path>`` mirrors the input directory's path (optionally with leading
directories cut), e.g. ``home/me/wrf/run01``; the Collection id is that path
joined with "--" (``home--me--wrf--run01``). Composite maps (all domains of
a run on one map) are written alongside, under ``<output>/<run_path>/model-images/``,
and are not part of the catalog.

The catalog is saved "self-contained": every link and asset href is relative,
so the whole tree can be moved or uploaded (e.g. to object storage) intact.

An output directory accumulates runs: collections already listed in an
existing catalog.json are kept (if their collection.json still exists), and
a run that is generated again keeps its place instead of being listed twice.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pystac
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import DataType, RasterBand, RasterExtension, Statistics

from modelatlasify import __version__
from modelatlasify.core.model import FieldResult, ModelDataset
from modelatlasify.core.variables import get_variable
from modelatlasify.geospatial.footprint import Footprint, grid_footprint
from modelatlasify.geospatial.geotiff import MEDIA_TYPE as COG_MEDIA_TYPE
from modelatlasify.geospatial.geotiff import RasterInfo

log = logging.getLogger(__name__)

FORECAST_EXTENSION = "https://stac-extensions.github.io/forecast/v0.2.0/schema.json"
CATALOG_ID = "modelatlasify"


@dataclass
class AssetRecord:
    key: str  # e.g. "surface_wind_speed" or "surface_wind_speed_map"
    path: Path
    variable: str
    kind: str  # "data" (GeoTIFF) or "visual" (PNG)
    field: FieldResult
    raster: RasterInfo | None = None


@dataclass
class ItemRecord:
    dataset: ModelDataset
    time_index: int
    item_dir: Path
    source_file: str
    assets: list[AssetRecord] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)  # {variable, reason}

    @property
    def valid_time(self) -> datetime:
        return self.dataset.valid_times[self.time_index]


def item_id(dataset: ModelDataset, valid_time: datetime) -> str:
    return f"{dataset.dataset_id}-{valid_time.strftime('%Y%m%dT%H%M%SZ')}"


def iso_duration(delta: timedelta) -> str:
    """timedelta -> ISO 8601 duration (e.g. PT36H, PT90M)."""
    secs = int(round(delta.total_seconds()))
    sign = "-" if secs < 0 else ""
    secs = abs(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    out = "PT" + (f"{h}H" if h else "") + (f"{m}M" if m else "") + (f"{s}S" if s else "")
    return sign + (out if out != "PT" else "PT0S")


def _union_bbox(bboxes: list[list[float]]) -> list[float]:
    """Union of lon/lat bboxes that may cross the antimeridian (west > east).

    Computed both in [-180, 180] and in [0, 360] longitude space; the narrower
    result is the right one.
    """
    south = min(b[1] for b in bboxes)
    north = max(b[3] for b in bboxes)
    candidates = []
    for base in (-180.0, 0.0):
        ws, es = [], []
        for w, _, e, _ in bboxes:
            w2 = (w - base) % 360.0 + base
            e2 = (e - base) % 360.0 + base
            if e2 < w2:  # still wraps in this space
                break
            ws.append(w2)
            es.append(e2)
        else:
            candidates.append((max(es) - min(ws), min(ws), max(es)))
    if not candidates:
        return [-180.0, south, 180.0, north]
    _, w, e = min(candidates)
    wrap = lambda x: ((x + 180.0) % 360.0) - 180.0 if not -180.0 <= x <= 180.0 else x  # noqa: E731
    return [round(v, 6) for v in (wrap(w), south, wrap(e), north)]


class CatalogBuilder:
    def __init__(self, output_dir: Path, title: str = "ModelAtlasify catalog"):
        self.output_dir = Path(output_dir).resolve()
        self.catalog = pystac.Catalog(
            id=CATALOG_ID,
            title=title,
            description=(
                "Catalog of numerical model output converted by ModelAtlasify into "
                "georeferenced, discoverable assets (Cloud-Optimized GeoTIFFs, quick-look "
                "maps) described with STAC."
            ),
        )
        self._footprints: dict[tuple, Footprint] = {}

    def run_dir(self, dataset: ModelDataset) -> Path:
        return self.output_dir / (dataset.run_path or dataset.run_id)

    def item_dir(self, dataset: ModelDataset, valid_time: datetime) -> Path:
        return self.run_dir(dataset) / "items" / item_id(dataset, valid_time)

    def footprint(self, dataset: ModelDataset, time_index: int = 0) -> Footprint:
        """Footprint of the dataset's grid at a valid time (moving nests move)."""
        grid = dataset.grid_at(time_index)
        key = (dataset.run_id, dataset.dataset_id, tuple(grid.transform))
        if key not in self._footprints:
            self._footprints[key] = grid_footprint(grid)
        return self._footprints[key]

    def dataset_bbox(self, dataset: ModelDataset) -> list[float]:
        """bbox of everywhere the dataset's grid is at any valid time."""
        if not dataset.moving:
            return self.footprint(dataset).bbox
        firsts = {}
        for ti, g in enumerate(dataset.time_grids):
            firsts.setdefault(tuple(g.transform), ti)
        return _union_bbox([self.footprint(dataset, ti).bbox for ti in firsts.values()])

    # ------------------------------------------------------------------ items
    def build_item(self, rec: ItemRecord) -> pystac.Item:
        ds = rec.dataset
        fp = self.footprint(ds, rec.time_index)
        valid = rec.valid_time
        props: dict[str, Any] = {
            "title": f"{ds.model} {ds.run_name} {ds.dataset_id} valid {valid:%Y-%m-%d %H:%M} UTC",
            "modelatlasify:model": ds.model,
            "modelatlasify:run_id": ds.run_id,
            "modelatlasify:dataset": ds.dataset_id,
            "modelatlasify:source_files": [rec.source_file],
        }
        if ds.reference_time is not None:
            props["forecast:reference_datetime"] = ds.reference_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            props["forecast:horizon"] = iso_duration(valid - ds.reference_time)
        if rec.skipped:
            props["modelatlasify:skipped_products"] = rec.skipped
        # Per-domain model metadata (grid, projection, physics) as wrf:* etc.
        props.update(ds.metadata)

        item = pystac.Item(
            id=item_id(ds, valid), geometry=fp.geometry, bbox=fp.bbox, datetime=valid,
            properties=props,
        )
        item.stac_extensions.append(FORECAST_EXTENSION)

        g = ds.grid_at(rec.time_index)
        proj = ProjectionExtension.ext(item, add_if_missing=True)
        proj.code = g.crs.to_epsg()  # None for WRF's custom spherical CRS ...
        proj.wkt2 = g.crs.to_wkt()  # ... so the full WKT2 definition is given
        proj.shape = [g.height, g.width]
        proj.transform = list(g.transform)[:6]
        proj.bbox = list(g.native_bounds)

        RasterExtension.add_to(item)
        for a in rec.assets:
            item.add_asset(a.key, self._asset(a))
        item.set_self_href(str(rec.item_dir / f"{item.id}.json"))
        return item

    def _asset(self, a: AssetRecord) -> pystac.Asset:
        var = get_variable(a.variable)
        f = a.field
        extra: dict[str, Any] = {}
        if f.start_time is not None and f.end_time is not None:
            extra["start_datetime"] = f.start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            extra["end_datetime"] = f.end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        period = ""
        if var.name == "precipitation_interval":
            period = f" ({f.start_time:%Y-%m-%d %H:%M} to {f.end_time:%Y-%m-%d %H:%M} UTC)"
        if f.provenance:
            extra["modelatlasify:provenance"] = f.provenance

        if a.kind == "data":
            asset = pystac.Asset(
                href=str(a.path), media_type=COG_MEDIA_TYPE, roles=["data"],
                title=f"{var.long_name} ({var.units}){period}",
                description=var.description, extra_fields=extra,
            )
            r = a.raster
            band = RasterBand.create(nodata="nan", data_type=DataType(r.dtype), unit=var.units)
            if r.statistics:
                band.statistics = Statistics.create(
                    minimum=r.statistics["minimum"], maximum=r.statistics["maximum"],
                    mean=r.statistics["mean"], stddev=r.statistics["stddev"],
                    valid_percent=round(r.valid_percent, 3),
                )
            RasterExtension.ext(asset).bands = [band]
            return asset
        return pystac.Asset(
            href=str(a.path), media_type=pystac.MediaType.PNG, roles=["visual", "overview"],
            title=f"Map: {var.long_name} ({var.units}){period}",
            extra_fields={k: v for k, v in extra.items() if k != "modelatlasify:provenance"},
        )

    # ------------------------------------------------------------ collection
    def add_run(self, run_id: str, datasets: list[ModelDataset], items: list[pystac.Item],
                variables: list[str]) -> pystac.Collection:
        ds_bboxes = [self.dataset_bbox(d) for d in datasets]
        # STAC: the first bbox is the overall extent; sub-extents only when >1.
        bboxes = [ds_bboxes[0]] if len(ds_bboxes) == 1 else (
            [_union_bbox(ds_bboxes)] + ds_bboxes)
        times = sorted(t for d in datasets for t in d.valid_times)
        model = datasets[0].model
        parent = min(datasets, key=lambda d: d.dataset_id)

        domains = []
        for d, bbox in zip(datasets, ds_bboxes):
            g = d.grid
            domains.append({
                "id": d.dataset_id,
                "title": d.title,
                "grid": g.description,
                "shape": [g.height, g.width],
                "resolution_m": [abs(g.transform.a), abs(g.transform.e)],
                "bbox": bbox,  # for a moving nest: everywhere it went
                "crs_wkt2": g.crs.to_wkt(),
                "georeference_max_error_m": round(g.max_georeference_error_m or 0.0, 3),
                "n_times": len(d.valid_times),
                "start_datetime": d.valid_times[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_datetime": d.valid_times[-1].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source_files": [p.name for p in d.files],
                **({"moving": True, "n_positions": len(d.distinct_grids())} if d.moving else {}),
            })

        # Run-level metadata: attributes identical across all domains.
        shared = {
            k: v for k, v in parent.metadata.items()
            if all(d.metadata.get(k) == v for d in datasets)
        }
        extra: dict[str, Any] = {
            "modelatlasify:model": model,
            "modelatlasify:version": __version__,
            "modelatlasify:domains": domains,
            "modelatlasify:run_path": parent.run_path or run_id,
            **shared,
        }
        if parent.source_dir is not None:
            extra["modelatlasify:source_directory"] = str(Path(parent.source_dir).resolve())

        version = shared.get("wrf:version")
        desc = (
            f"Output of {model}{' v' + version if version else ''} run "
            f"'{parent.run_path or run_id}': "
            f"{len(datasets)} domain(s) ({', '.join(d.dataset_id for d in datasets)}), "
            f"{len(times)} domain-times from {times[0]:%Y-%m-%d %H:%M} to "
            f"{times[-1]:%Y-%m-%d %H:%M} UTC"
            + (f", initialized {parent.reference_time:%Y-%m-%d %H:%M} UTC" if parent.reference_time else "")
            + ". Variables: " + ", ".join(get_variable(v).long_name.lower() for v in variables)
            + ". GeoTIFFs are in the model's native map projection."
        )
        collection = pystac.Collection(
            id=run_id,
            title=f"{model} run {parent.run_name}",
            description=desc,
            extent=pystac.Extent(
                pystac.SpatialExtent(bboxes),
                pystac.TemporalExtent([[times[0], times[-1]]]),
            ),
            license="other",
            keywords=[model, "numerical weather prediction", "model output",
                      *[get_variable(v).long_name.lower() for v in variables]],
            extra_fields=extra,
            summaries=pystac.Summaries({"modelatlasify:dataset": [d.dataset_id for d in datasets]}),
        )
        collection.item_assets = self._item_assets(variables)
        collection.set_self_href(str(self.run_dir(parent) / "collection.json"))
        for it in items:
            href = it.get_self_href()
            collection.add_item(it)
            it.set_self_href(href)  # keep each item next to its assets
        self.catalog.add_child(collection)
        return collection

    @staticmethod
    def _item_assets(variables: list[str]) -> dict[str, pystac.ItemAssetDefinition]:
        out = {}
        for v in variables:
            var = get_variable(v)
            out[v] = pystac.ItemAssetDefinition.create(
                title=f"{var.long_name} ({var.units})", description=var.description,
                media_type=COG_MEDIA_TYPE, roles=["data"],
            )
            out[f"{v}_map"] = pystac.ItemAssetDefinition.create(
                title=f"Map: {var.long_name}", description=None,
                media_type=pystac.MediaType.PNG, roles=["visual", "overview"],
            )
        return out

    def save(self) -> Path:
        path = self.output_dir / "catalog.json"
        previous = self._existing_children(path)  # read before it is overwritten
        self.catalog.set_self_href(str(path))
        # Children and items already have explicit self hrefs; keep them.
        self.catalog.make_all_asset_hrefs_relative()
        self.catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
        if previous:
            self._merge_children(path, previous)
        return path

    def _existing_children(self, path: Path) -> list[dict]:
        """Child links of an existing catalog.json whose targets still exist."""
        if not path.is_file():
            return []
        try:
            links = json.loads(path.read_text()).get("links", [])
        except (OSError, ValueError) as exc:
            log.warning("Existing %s is unreadable (%s); it is replaced.", path, exc)
            return []
        return [ln for ln in links
                if ln.get("rel") == "child" and isinstance(ln.get("href"), str)
                and (path.parent / ln["href"]).is_file()]

    @staticmethod
    def _merge_children(path: Path, previous: list[dict]) -> None:
        """Keep previously cataloged collections: old children first, in their
        order, then this run's new ones. Targets are compared by resolved path."""
        def key(link: dict) -> str:
            return os.path.normpath(path.parent / link["href"])

        cat = json.loads(path.read_text())
        current = [ln for ln in cat["links"] if ln.get("rel") == "child"]
        old_keys = {key(ln) for ln in previous}
        children = previous + [ln for ln in current if key(ln) not in old_keys]
        cat["links"] = [ln for ln in cat["links"] if ln.get("rel") != "child"] + children
        path.write_text(json.dumps(cat, indent=2))
