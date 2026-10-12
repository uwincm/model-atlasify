"""Model-agnostic data structures and the model adapter interface.

A *model adapter* knows how to find one model's output on disk and how to turn
its native fields into ModelAtlasify's standard variables. Everything
downstream of the adapter (GeoTIFF writing, plotting, STAC) only sees the types
defined here, so adding a new model means writing a new adapter, not touching
the rest of the pipeline.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine
from pyproj import CRS


@dataclass(frozen=True)
class GridDefinition:
    """A regular grid in a projected CRS.

    ``data`` arrays handed to the rest of the pipeline are always oriented
    north-up (row 0 is the northernmost row), matching ``transform``, which
    maps (col, row) pixel *corners* to CRS coordinates.
    """

    crs: CRS
    transform: Affine
    width: int
    height: int
    lat: np.ndarray  # (height, width), north-up, degrees north
    lon: np.ndarray  # (height, width), north-up, degrees east in [-180, 180)
    description: str = ""
    max_georeference_error_m: float | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def native_bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) of the grid's outer pixel edges in the CRS."""
        t = self.transform
        xs = (t.c, t.c + t.a * self.width)
        ys = (t.f, t.f + t.e * self.height)
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class ModelDataset:
    """One spatially consistent time series of model output.

    For WRF this is a single domain (e.g. d01) of a single run. Several datasets
    with the same ``run_id`` (e.g. d01 and d02) belong to the same model run.
    """

    model: str  # e.g. "WRF"
    run_id: str
    dataset_id: str  # unique within the run, e.g. "d01"
    title: str
    files: list[Path]
    valid_times: list[datetime]  # timezone-aware UTC, sorted, unique
    reference_time: datetime | None  # initialization / simulation start
    grid: GridDefinition
    metadata: dict[str, Any] = field(default_factory=dict)  # model-specific, JSON-serializable
    source_dir: Path | None = None
    #: The file holding each valid time (parallel to ``valid_times``).
    time_sources: list[Path] = field(default_factory=list)
    #: Opaque, adapter-private bookkeeping (e.g. which file/index holds each time).
    adapter_state: Any = field(default=None, repr=False)


@dataclass
class FieldResult:
    """A computed 2-D field for one variable at one valid time."""

    variable: str
    data: np.ndarray  # float, (height, width), north-up, NaN = missing
    valid_time: datetime
    # For period quantities (e.g. precipitation over an output interval).
    start_time: datetime | None = None
    end_time: datetime | None = None
    # Free-form provenance: which native variables were used and how.
    provenance: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(abc.ABC):
    """Interface every model adapter implements."""

    #: Short model name, e.g. "WRF".
    name: str = ""

    @abc.abstractmethod
    def discover(self, directory: Path) -> list[ModelDataset]:
        """Find this model's output in ``directory`` and group it into datasets."""

    @abc.abstractmethod
    def supported_variables(self) -> list[str]:
        """Standard variable names this adapter can produce."""

    @abc.abstractmethod
    def compute(
        self, dataset: ModelDataset, variable: str, time_index: int
    ) -> FieldResult | None:
        """Compute ``variable`` at ``dataset.valid_times[time_index]``.

        Returns ``None`` when the variable is legitimately undefined at that time
        (e.g. interval precipitation at the first output time). Raises
        :class:`~modelatlasify.core.errors.VariableUnavailableError` when the
        required model fields are missing.
        """

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release any open file handles."""
