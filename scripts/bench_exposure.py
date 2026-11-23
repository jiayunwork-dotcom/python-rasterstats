"""Long-route exposure benchmark.

Writes an 8192x8192 raster in chunks (never materializing a full-size array
beyond the raster itself), then runs continuous exposure analysis along

* a long diagonal route, and
* a repeatedly folding route whose bounding box covers most of the raster,

with ``read_cells`` set to 64 and 4096. The script records the actual
rasterio read windows, the peak cell-cache occupancy, peak RSS, wall time
and the number of emitted segments, demonstrating that reads expand along
the route's actual supports rather than materializing the large bounding
box.

Run with::

    python scripts/bench_exposure.py
"""

import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from shapely.geometry import LineString

from rasterstats.exposure import CellCache, analyze_component


def write_chunked_raster(path, size=8192, chunk=512):
    """Write a smooth analytic concentration field in row chunks."""
    transform = Affine.identity()
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="float32",
        transform=transform,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        # field: gentle planar trend plus a periodic term so threshold
        # crossings occur repeatedly along any long route
        cols = np.arange(size, dtype="float32")
        for row_start in range(0, size, chunk):
            row_stop = min(size, row_start + chunk)
            rows = np.arange(row_start, row_stop, dtype="float32")[:, None]
            block = (
                1.0
                + 0.01 * cols[None, :]
                + 0.01 * rows
                + 2.0 * np.sin(cols[None, :] / 200.0)
                + 2.0 * np.cos(rows / 200.0)
            ).astype("float32")
            dst.write(block, 1, window=((row_start, row_stop), (0, size)))


def diagonal_route(size):
    margin = 2.0
    return LineString(
        [(margin, margin), (size - margin, size - margin)]
    )


def folding_route(size, folds=12):
    """A zigzag whose bbox covers most of the raster but whose total
    traversed length is much larger than the bbox diagonal."""
    margin = 3.0
    coords = []
    for i in range(folds + 1):
        x = margin + (size - 2 * margin) * i / folds
        y = margin if i % 2 == 0 else size - margin
        coords.append((x, y))
    return LineString(coords)


def run_once(raster_path, line, read_cells, threshold):
    from rasterstats.io import Raster

    with Raster(str(raster_path)) as rast:
        cache = CellCache(rast, read_cells)
        coords = list(line.coords)
        start = time.perf_counter()
        part = analyze_component(
            coords, ~rast.affine, threshold, 1e-8, cache
        )
        elapsed = time.perf_counter() - start
    window_sizes = [
        (r1 - r0) * (c1 - c0) for (r0, r1), (c0, c1) in cache.read_windows
    ]
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "read_cells": read_cells,
        "read_window_count": len(cache.read_windows),
        "max_window_cells": max(window_sizes),
        "total_cells_read": cache.cells_read,
        "max_cached_cells": cache.max_cached,
        "peak_rss_mb": rss_kb / 1024.0,
        "elapsed_seconds": elapsed,
        "segments": len(part["intervals"]),
        "total_length": part["total_length"],
        "above_length": part["above_length"],
        "missing_length": part["missing_length"],
    }


def main():
    out_dir = Path(os.environ.get("BENCH_DIR", "/tmp/exp/bench"))
    out_dir.mkdir(parents=True, exist_ok=True)
    raster_path = out_dir / "big.tif"
    if not raster_path.exists():
        print(f"writing chunked raster {raster_path} ...")
        write_chunked_raster(raster_path)
    size = 8192
    routes = {
        "diagonal": diagonal_route(size),
        "folding": folding_route(size),
    }
    threshold = 30.0
    report = {}
    for name, line in routes.items():
        report[name] = []
        for budget in (64, 4096):
            print(f"running {name} with read_cells={budget} ...")
            metrics = run_once(str(raster_path), line, budget, threshold)
            report[name].append(metrics)
            print(json.dumps(metrics, indent=2))
    report_path = out_dir / "exposure_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"report written to {report_path}")


if __name__ == "__main__":
    main()
