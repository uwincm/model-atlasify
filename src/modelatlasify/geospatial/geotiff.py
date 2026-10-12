"""Write model fields as Cloud-Optimized GeoTIFFs in the model's native CRS."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import rasterio.shutil
from rasterio.io import MemoryFile

from modelatlasify.core.model import FieldResult, GridDefinition
from modelatlasify.core.variables import VariableDefinition

#: Model output is float32 natively; storing float32 loses no information.
DTYPE = "float32"
NODATA = float("nan")
MEDIA_TYPE = "image/tiff; application=geotiff; profile=cloud-optimized"


@dataclass
class RasterInfo:
    path: Path
    dtype: str
    nodata: float
    shape: tuple[int, int]
    statistics: dict[str, float]  # empty if all values are missing
    valid_percent: float


def _iso(t) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ") if t else ""


def write_geotiff(
    path: Path, field: FieldResult, grid: GridDefinition, variable: VariableDefinition,
    extra_tags: dict[str, str] | None = None,
) -> RasterInfo:
    data = np.asarray(field.data)
    if data.shape != grid.shape:
        raise ValueError(f"Field shape {data.shape} does not match grid shape {grid.shape}.")
    data = np.where(np.isfinite(data), data, np.nan).astype(DTYPE)

    tags = {
        "long_name": variable.long_name,
        "units": variable.units,
        "description": variable.description,
        "valid_time": _iso(field.valid_time),
    }
    if variable.standard_name:
        tags["standard_name"] = variable.standard_name
    if variable.cell_methods:
        tags["cell_methods"] = variable.cell_methods
    if field.start_time:
        tags["start_time"] = _iso(field.start_time)
    if field.end_time:
        tags["end_time"] = _iso(field.end_time)
    if field.provenance:
        tags["provenance"] = json.dumps(field.provenance)
    tags.update(extra_tags or {})

    profile = dict(
        driver="GTiff", width=grid.width, height=grid.height, count=1, dtype=DTYPE,
        crs=grid.crs, transform=grid.transform, nodata=NODATA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to memory, then translate to a COG (tiled, overviews, internal
    # compression) so the files are ready for HTTP range requests later.
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(data, 1)
            dst.set_band_description(1, variable.name)
            dst.set_band_unit(1, variable.units)
            dst.update_tags(1, **tags)
            dst.update_tags(AREA_OR_POINT="Area", model_grid=grid.description)
        with mem.open() as src:
            rasterio.shutil.copy(
                src, path, driver="COG", compress="DEFLATE", predictor="FLOATING_POINT",
                blocksize=512, overview_resampling="average",
            )

    finite = data[np.isfinite(data)]
    stats = (
        {
            "minimum": float(finite.min()),
            "maximum": float(finite.max()),
            "mean": float(finite.mean(dtype=np.float64)),
            "stddev": float(finite.std(dtype=np.float64)),
        }
        if finite.size
        else {}
    )
    return RasterInfo(
        path=path, dtype=DTYPE, nodata=NODATA, shape=grid.shape, statistics=stats,
        valid_percent=100.0 * finite.size / data.size,
    )
