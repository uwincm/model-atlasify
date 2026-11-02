"""Checks on a generated catalog: link integrity and STAC JSON-schema validity."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pystac

from modelatlasify.core import storage


@dataclass
class ValidationReport:
    n_objects: int = 0
    missing_hrefs: list[str] = field(default_factory=list)
    schema_errors: list[str] = field(default_factory=list)
    schema_skipped_reason: str | None = None

    @property
    def ok(self) -> bool:
        return not self.missing_hrefs and not self.schema_errors


def _open_catalog(catalog_path: Path) -> pystac.Catalog:
    return pystac.Catalog.from_file(str(catalog_path), stac_io=storage.FsspecStacIO())


def check_links(catalog_path: Path) -> tuple[int, list[str]]:
    """Resolve every child/item link and asset href; return objects seen and missing files."""
    catalog = _open_catalog(catalog_path)
    missing: list[str] = []
    n = 1
    for child in catalog.get_children():
        n += 1
        existing = _existing_objects(child.get_self_href())
        for item in child.get_items(recursive=True):
            n += 1
            for key, asset in item.assets.items():
                href = asset.get_absolute_href()
                found = (href in existing if existing is not None
                         else storage.as_path(href).is_file())
                if not found:
                    missing.append(f"{item.id}:{key} -> {href}")
    return n, missing


def _existing_objects(collection_href: str) -> set[str] | None:
    """For a collection on S3, every object under its folder, listed once (one
    request per asset would be slow); None for local files."""
    if not storage.is_remote(collection_href):
        return None
    folder = storage.as_path(collection_href).parent
    storage.refresh(folder)
    return {f"{storage.S3_PREFIX}{key}" for key in folder.fs.find(folder.path)}


def validate_schemas(catalog_path: Path) -> list[str]:
    """Validate every object against the STAC core + extension JSON schemas.

    Requires ``jsonschema``; extension schemas are fetched over HTTP by pystac.
    Raises ``ConnectionError``-like exceptions if schemas cannot be fetched.
    """
    catalog = _open_catalog(catalog_path)
    errors: list[str] = []
    objs = [catalog]
    for child in catalog.get_children():
        objs.append(child)
        objs.extend(child.get_items(recursive=True))
    for obj in objs:
        try:
            obj.validate()
        except pystac.errors.STACValidationError as exc:
            errors.append(f"{type(obj).__name__} {obj.id}: {exc}")
    return errors


def validate_catalog(catalog_path: Path, schemas: bool = True) -> ValidationReport:
    report = ValidationReport()
    report.n_objects, report.missing_hrefs = check_links(catalog_path)
    if schemas:
        try:
            report.schema_errors = validate_schemas(catalog_path)
        except ImportError:
            report.schema_skipped_reason = "jsonschema is not installed"
        except OSError as exc:  # schema download failed (offline)
            report.schema_skipped_reason = f"could not fetch STAC schemas ({exc})"
    else:
        report.schema_skipped_reason = "disabled"
    return report
