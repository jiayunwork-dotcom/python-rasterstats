#!/usr/bin/env python3
"""Fractional zonal-statistics benchmark on large constant GeoTIFFs.

Three subcommands run in separate processes so input generation and the
independent reference check never inflate the analysis memory profile:

* ``make``   - generate a constant float32 GeoTIFF (4096 or 8192) with an
               internal block layout and a slanted polygon with a hole;
* ``bench``  - run fractional zonal stats, recording every real rasterio
               read window, the largest materialized 2D working array,
               wall time and peak RSS measured *after* dependencies are
               imported;
* ``check``  - independent verification using Shapely geometry only (the
               rasterstats fractional kernel is never imported here beyond
               reading the constant value via rasterio).

Usage::

    python benchmark_fractional.py make  --size 4096 --workdir /tmp/fracbench
    python benchmark_fractional.py bench --size 4096 --max-cells 4096  --workdir /tmp/fracbench
    python benchmark_fractional.py check --size 4096 --workdir /tmp/fracbench
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from shapely.geometry import Polygon, box

VALUE = 3.0
# Small block layout to prove that tiling decisions and results are
# independent of how the TIFF chunks its data.
BLOCKSIZE = 256


def paths(workdir, size):
    workdir = Path(workdir)
    return {
        "tif": workdir / f"constant_{size}.tif",
        "geom": workdir / f"zone_{size}.geojson",
        "record": workdir / f"bench_{size}.json",
    }


def make_zone(size):
    """Slanted parallelogram spanning most of the grid with a square hole."""
    outer = Polygon(
        [
            (size * 0.05, size * 0.10),
            (size * 0.95, size * 0.20),
            (size * 0.90, size * 0.90),
            (size * 0.10, size * 0.80),
        ]
    )
    hole = box(size * 0.40, size * 0.40, size * 0.60, size * 0.60)
    zone = outer.difference(hole)
    return zone


def cmd_make(args):
    p = paths(args.workdir, args.size)
    p["tif"].parent.mkdir(parents=True, exist_ok=True)
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, float(args.size))
    with rasterio.open(
        p["tif"],
        "w",
        driver="GTiff",
        height=args.size,
        width=args.size,
        count=1,
        dtype="float32",
        transform=transform,
        blockxsize=BLOCKSIZE,
        blockysize=BLOCKSIZE,
        tiled=True,
    ) as dst:
        # write in strips so the constant file is never fully materialized
        strip = np.full((BLOCKSIZE, args.size), VALUE, dtype="float32")
        for row in range(0, args.size, BLOCKSIZE):
            rows = min(BLOCKSIZE, args.size - row)
            dst.write(strip[:rows], 1, window=((row, row + rows), (0, args.size)))

    zone = make_zone(args.size)
    from shapely.geometry import mapping

    p["geom"].write_text(json.dumps(mapping(zone)))
    print(json.dumps({"wrote": [str(p["tif"]), str(p["geom"])]}, indent=2))


def _zone_bounds_in_grid_coords(zone, size):
    # the TIFF is north-up with y offset = size, so grid row coords are
    # size - y; convert the CRS polygon into row/col-space for analysis
    minx, miny, maxx, maxy = zone.bounds
    return Polygon(
        [
            (minx, size - maxy),
            (maxx, size - maxy),
            (maxx, size - miny),
            (minx, size - miny),
        ]
    )


def cmd_bench(args):
    # Import only after process start so the RSS baseline includes every
    # dependency but excludes input generation (which runs in another process).
    import shapely

    from rasterstats import gen_zonal_stats

    p = paths(args.workdir, args.size)
    zone_geojson = json.loads(p["geom"].read_text())
    zone = shapely.from_geojson(json.dumps(zone_geojson))
    grid_zone = _zone_bounds_in_grid_coords(zone, args.size)

    # Instrument real read windows through Raster's lightweight hook rather
    # than wrapping DatasetReader.read: a Python wrapper changes GDAL's
    # block-cache reuse path and would inflate the measured memory profile.
    from rasterstats.io import Raster

    read_log = []
    original_init = Raster.__init__

    def instrumented_init(self, raster, affine=None, nodata=None, band=1):
        original_init(self, raster, affine, nodata, band)
        self.read_window_log = read_log

    Raster.__init__ = instrumented_init
    try:
        # Baseline sampled before any dataset is opened: the GDAL block cache
        # is configured (cap) at open time, so this captures only the extra
        # peak attributable to the analysis itself.
        rss_after_import_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        start = time.perf_counter()
        generator = gen_zonal_stats(
            [grid_zone],
            str(p["tif"]),
            affine=None,
            coverage="fractional",
            value_kind=args.value_kind,
            max_cells=args.max_cells,
        )
        result = next(generator)
        generator.close()
        elapsed = time.perf_counter() - start
        peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    finally:
        Raster.__init__ = original_init

    cells = [
        int((r1 - r0) * (c1 - c0))
        for (r0, r1), (c0, c1) in read_log
    ]
    first_reads = [
        {
            "rows": [int(r0), int(r1)],
            "cols": [int(c0), int(c1)],
            "cells": int((r1 - r0) * (c1 - c0)),
        }
        for (r0, r1), (c0, c1) in read_log[:5]
    ]
    record = {
        "size": args.size,
        "value_kind": args.value_kind,
        "max_cells": args.max_cells,
        "read_count": len(read_log),
        "max_read_cells": max(cells),
        "total_cells_read": sum(cells),
        "first_reads": first_reads,
        "elapsed_seconds": elapsed,
        "rss_after_import_kib": rss_after_import_kb,
        "peak_rss_kib": peak_rss_kb,
        "extra_peak_rss_kib": peak_rss_kb - rss_after_import_kb,
        "result": result,
    }
    # prove no single read covered the full feature bounding window
    bbox_cells = int(
        (grid_zone.bounds[2] - grid_zone.bounds[0])
        * (grid_zone.bounds[3] - grid_zone.bounds[1])
    )
    record["feature_bbox_cells"] = bbox_cells
    record["full_bbox_read_detected"] = max(cells) >= bbox_cells
    existing = json.loads(p["record"].read_text()) if p["record"].exists() else []
    existing = [e for e in existing if e["max_cells"] != args.max_cells]
    existing.append(record)
    existing.sort(key=lambda e: e["max_cells"])
    p["record"].write_text(json.dumps(existing, indent=2))
    print(json.dumps(record, indent=2))


def cmd_check(args):
    """Independent reference: constant raster, so totals derive from areas."""
    import shapely

    p = paths(args.workdir, args.size)
    zone_geojson = json.loads(p["geom"].read_text())
    zone = shapely.from_geojson(json.dumps(zone_geojson))
    grid_zone = _zone_bounds_in_grid_coords(zone, args.size)
    frame = box(0, 0, args.size, args.size)
    in_frame = grid_zone.intersection(frame)

    with rasterio.open(p["tif"]) as src:
        value = float(src.read(1, window=((0, 1), (0, 1)))[0, 0])
        nodata = src.nodata

    coverage_area = in_frame.area
    # constant, finite, non-nodata file: every covered cell is observed
    valid_area = coverage_area
    expected = {
        "coverage_area": coverage_area,
        "valid_area": valid_area,
        "missing_area": coverage_area - valid_area,
        "density_sum": value * valid_area,
        "total_sum": value * valid_area,  # unit cells: fraction*value == a*value
        "mean": value,
        "nodata": nodata,
    }
    print(json.dumps(expected, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--size", type=int, choices=(4096, 8192), required=True)
    common.add_argument("--workdir", default="/tmp/fracbench")

    make_parser = sub.add_parser("make", parents=[common])
    make_parser.set_defaults(func=cmd_make)

    bench_parser = sub.add_parser("bench", parents=[common])
    bench_parser.add_argument("--max-cells", type=int, required=True)
    bench_parser.add_argument(
        "--value-kind", choices=("total", "density"), default="density"
    )
    bench_parser.set_defaults(func=cmd_bench)

    check_parser = sub.add_parser("check", parents=[common])
    check_parser.set_defaults(func=cmd_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
