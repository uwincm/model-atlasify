"""Command-line interface: ``modelatlasify INPUT_DIR [INPUT_DIR ...] --output OUT``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from modelatlasify import __version__
import modelatlasify.variables  # noqa: F401 - registers the standard variables
from modelatlasify.core.errors import ModelAtlasifyError
from modelatlasify.core.variables import all_variables, resolve_names
from modelatlasify.models import AVAILABLE_MODELS


def _status(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    variables = all_variables()
    var_help = "; ".join(f"{v.name} ({v.units})" for v in variables.values())
    p = argparse.ArgumentParser(
        prog="modelatlasify",
        description=(
            "Turn numerical model output into georeferenced GeoTIFFs, quick-look maps "
            "and a STAC catalog. Each input directory is treated as one model run."
        ),
        epilog=(
            "Example: modelatlasify /data/wrf/run01 --output ./atlas\n"
            f"Variables: {var_help}. Aliases: wind, precipitation (= accumulated + interval)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", type=Path, metavar="INPUT_DIR",
                   help="directory containing model output (e.g. wrfout_d01_* files)")
    p.add_argument("-o", "--output", type=Path, default=Path("atlas"),
                   help="output directory for the catalog and assets (default: ./atlas)")
    p.add_argument("--model", default="wrf", choices=AVAILABLE_MODELS,
                   help="model adapter to use (default: wrf)")
    p.add_argument("--variables", default="wind,precipitation",
                   help="comma-separated variables to generate (default: wind,precipitation)")
    p.add_argument("--domains", default=None,
                   help="comma-separated domains to process, e.g. d01 (default: all)")
    p.add_argument("-np", "--processes", type=int, default=1, metavar="N",
                   help="number of worker processes for per-time processing (default: 1)")
    p.add_argument("--no-plots", dest="plots", action="store_false",
                   help="skip PNG map generation")
    p.add_argument("--no-validate", dest="validate", action="store_false",
                   help="skip STAC JSON-schema validation (asset links are always checked)")
    p.add_argument("-v", "--verbose", action="store_true", help="show debug messages")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.processes < 1:
        parser.error("-np/--processes must be at least 1")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    # Library warnings about individual files are summarised by the pipeline.
    if not args.verbose:
        logging.getLogger("modelatlasify.models").setLevel(logging.ERROR)

    try:
        variables = resolve_names(args.variables.split(","))
    except KeyError as exc:
        _status(f"error: {exc.args[0]}")
        return 2
    domains = [d.strip() for d in args.domains.split(",")] if args.domains else None

    from modelatlasify.core.processing import PipelineOptions, run_pipeline
    from modelatlasify.core.validation import validate_catalog

    options = PipelineOptions(
        inputs=args.inputs, output=args.output, model=args.model, variables=variables,
        plots=args.plots, domains=domains, processes=args.processes,
    )
    _status(f"ModelAtlasify {__version__}: {args.model.upper()} -> {args.output.resolve()}")
    _status(f"Variables: {', '.join(variables)}"
            + (f" ({args.processes} processes)" if args.processes > 1 else ""))
    try:
        result = run_pipeline(options, report=_status)
    except ModelAtlasifyError as exc:
        _status(f"error: {exc}")
        return 1

    if result.quality_skips:
        _status(f"WARNING: {len(result.quality_skips)} product(s) failed sanity checks and were "
                "not published (recorded in each item's modelatlasify:skipped_products):")
        for line in result.quality_skips[:10]:
            _status(f"  {line}")
        if len(result.quality_skips) > 10:
            _status(f"  ... and {len(result.quality_skips) - 10} more")
    _status(f"Generated {result.n_items} STAC items, {result.n_geotiffs} GeoTIFFs, "
            f"{result.n_plots} maps in {result.seconds:.0f} s.")
    _status("Checking catalog links" + (" and validating STAC schemas..." if args.validate else "..."))
    report = validate_catalog(result.catalog_path, schemas=args.validate)
    if report.missing_hrefs:
        _status(f"error: {len(report.missing_hrefs)} asset link(s) do not resolve, e.g. "
                f"{report.missing_hrefs[0]}")
    if report.schema_errors:
        _status(f"error: {len(report.schema_errors)} STAC object(s) failed validation, e.g. "
                f"{report.schema_errors[0]}")
    if report.schema_skipped_reason and args.validate:
        _status(f"warning: STAC schema validation skipped: {report.schema_skipped_reason}")
    elif args.validate and report.ok:
        _status(f"STAC validation passed for {report.n_objects} objects; all asset links resolve.")
    print(result.catalog_path)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
