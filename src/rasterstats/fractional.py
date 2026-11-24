"""Fractional area-weighted zonal statistics.

When ``coverage='fractional'`` is requested, raster cells are no longer
treated as all-or-nothing members of a zone.  Instead, for every cell of
area ``A`` overlapping the zone geometry over area ``a``, the coverage
fraction ``a / A`` is used as the cell weight.

Two value semantics are supported:

``value_kind='total'``
    The raster stores a quantity that is uniform over the whole cell
    (e.g. number of people).  The cell contributes ``(a / A) * value`` to
    the summed total.

``value_kind='density'``
    The raster stores a per-unit-area intensity that is constant inside
    the cell (e.g. people per square coordinate unit).  The cell
    contributes ``a * value`` to the summed total.

In both cases ``mean`` is the summed total divided by the valid
(observed) overlap area, i.e. the area-weighted mean intensity of the
zone.  Area units are the square of the coordinate units; the caller is
responsible for putting the raster and the geometries in a single,
planar, projected coordinate reference system.
"""

import math

import numpy as np
import shapely
from affine import Affine
from rasterio import features
from shapely.geometry import Polygon

from rasterstats.io import read_validity

#: Statistics supported by the fractional coverage mode.
FRACTIONAL_STATS = ("count", "sum", "mean")

#: Area fields that are always attached to every fractional result.
AREA_FIELDS = ("coverage_area", "valid_area", "missing_area")

#: GDAL block-cache cap for tiled fractional analysis, expressed in bytes.
#: Modern GDAL interprets GDAL_CACHEMAX values above 10000 as bytes (smaller
#: values as MiB), so a plain small integer such as 4 would mean four bytes
#: and fail to cap anything. 16 MiB bounds the cache while holding several
#: tiles; the default cache is otherwise a fraction of total machine RAM and
#: grows with the number of small window reads.
FRACTIONAL_GDAL_CACHE_BYTES = 16 * 1024 * 1024


def validate_fractional_options(
    coverage,
    value_kind,
    stats,
    categorical,
    add_stats,
    zone_func,
    raster_out,
    all_touched,
):
    """Validate the options that are specific to fractional coverage mode.

    Returns the normalized ``stats`` list.  Raises ``ValueError`` for any
    combination that the fractional mode cannot honour.  When ``coverage``
    is left as ``None`` the function only checks that ``value_kind`` is also
    absent and returns ``stats`` unchanged.
    """
    if coverage is None:
        if value_kind is not None:
            raise ValueError(
                "value_kind can only be used with coverage='fractional'; "
                "leave value_kind as None for ordinary zonal statistics"
            )
        return stats

    if coverage != "fractional":
        raise ValueError(
            f"coverage {coverage!r} is not valid; "
            "use None (ordinary cell inclusion) or 'fractional'"
        )
    if value_kind not in ("total", "density"):
        raise ValueError(
            f"value_kind {value_kind!r} is not valid; "
            "choose 'total' (whole-cell quantities) or 'density' "
            "(per-unit-area intensities) when coverage='fractional'"
        )

    if categorical:
        raise ValueError("categorical statistics are not supported with coverage='fractional'")
    if add_stats is not None:
        raise ValueError("add_stats is not supported with coverage='fractional'")
    if zone_func is not None:
        raise ValueError("zone_func is not supported with coverage='fractional'")
    if raster_out:
        raise ValueError("raster_out is not supported with coverage='fractional'")
    if all_touched:
        raise ValueError(
            "all_touched is an all-or-nothing cell inclusion rule and cannot "
            "be combined with coverage='fractional'"
        )

    if not stats:
        stats = list(FRACTIONAL_STATS)
    elif isinstance(stats, str):
        stats = stats.split()
    for stat in stats:
        if stat not in FRACTIONAL_STATS:
            raise ValueError(
                f"stat {stat!r} is not supported with coverage='fractional'; "
                f"choose from {list(FRACTIONAL_STATS)}"
            )
    return stats


def validate_max_cells(max_cells):
    """Reject booleans, non-integers and non-positive cell budgets."""
    if isinstance(max_cells, bool) or not isinstance(max_cells, (int, np.integer)):
        raise ValueError(
            f"max_cells {max_cells!r} is not valid; "
            "it must be a positive integer (booleans are not accepted)"
        )
    if max_cells <= 0:
        raise ValueError(f"max_cells {max_cells!r} is not valid; it must be positive")
    return int(max_cells)


def validate_affine(affine):
    """The affine must be present, finite and invertible (non-zero determinant)."""
    if affine is None:
        raise ValueError(
            "An affine transform is required for coverage='fractional' "
            "(pass affine=... for ndarray rasters)"
        )
    if not isinstance(affine, Affine):
        raise ValueError(f"affine must be an affine.Affine instance, got {type(affine)!r}")
    coefficients = tuple(affine)[:6]
    if not all(math.isfinite(v) for v in coefficients):
        raise ValueError("affine transform contains non-finite coefficients")
    determinant = affine.determinant
    if not math.isfinite(determinant) or determinant == 0.0:
        raise ValueError("affine transform is not invertible (zero determinant)")


def validate_polygonal(geom, feature_index):
    """Reject non-polygonal and self-invalid polygon geometries.

    For MultiPolygons every component must be valid on its own; overlapping
    components are allowed (the union step makes them count once).
    """
    if geom is None or geom.is_empty:
        return
    geom_type = geom.geom_type
    if geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(
            f"feature {feature_index}: coverage='fractional' only accepts "
            f"Polygon or MultiPolygon geometries, got {geom_type}"
        )
    if geom_type == "Polygon":
        components = [geom]
    else:
        components = list(geom.geoms)
    for component_index, component in enumerate(components):
        if component.is_empty:
            continue
        if not component.is_valid:
            raise ValueError(
                f"feature {feature_index}: polygon component {component_index} "
                "is not valid (self-intersection or degenerate ring); fix the "
                "geometry before requesting fractional coverage statistics"
            )


def _polygon_parts(geom):
    """Flatten a polygonal geometry into a list of valid Polygon objects."""
    if geom is None or geom.is_empty:
        return []
    geom_type = geom.geom_type
    if geom_type == "Polygon":
        return [geom]
    if geom_type == "MultiPolygon":
        return [part for part in geom.geoms if not part.is_empty]
    if geom_type == "GeometryCollection":
        parts = []
        for part in geom.geoms:
            if part.is_empty:
                continue
            if part.geom_type == "Polygon":
                parts.append(part)
            elif part.geom_type == "MultiPolygon":
                parts.extend(p for p in part.geoms if not p.is_empty)
        return parts
    return []


def _frame_polygon(affine, height, width):
    """Polygon describing the outer border of the raster in CRS coordinates."""
    corners = [
        affine @ (0, 0),
        affine @ (width, 0),
        affine @ (width, height),
        affine @ (0, height),
    ]
    return Polygon(corners)


def _bounds_window(bounds, affine, shape):
    """Full-cover rasterio-style window valid for rotated/sheared affines.

    The four corners of the bounds rectangle are mapped through the inverse
    affine; the window spans floor(min) .. ceil(max) of the resulting
    row/column coordinates and is then clamped to the raster extent.
    """
    west, south, east, north = bounds
    inverse = ~affine
    cols = []
    rows = []
    for x, y in (
        (west, south),
        (west, north),
        (east, south),
        (east, north),
    ):
        col, row = inverse @ (x, y)
        cols.append(col)
        rows.append(row)
    height, width = shape
    row_start = max(0, math.floor(min(rows)))
    row_stop = min(height, math.ceil(max(rows)))
    col_start = max(0, math.floor(min(cols)))
    col_stop = min(width, math.ceil(max(cols)))
    return (row_start, row_stop), (col_start, col_stop)


def _window_affine(affine, window):
    """Affine of a rasterio-style window under a possibly rotated/sheared grid.

    ``io.window_bounds`` assumes axis-aligned transforms; composing the base
    affine with the window's row/column translation is exact for any affine.
    """
    (row_start, _), (col_start, _) = window
    xoff, yoff = affine @ (col_start, row_start)
    return Affine(affine.a, affine.b, xoff, affine.d, affine.e, yoff)


def _iter_band_windows(window, max_cells):
    """Split a full window into disjoint 2D tiles of <= ``max_cells`` slots.

    Tiles are near square, which minimizes both the number of GDAL
    read/rasterize calls and the number of cells sitting on a tile border
    (those need exact Shapely clipping rather than full-cell accounting).
    Every tile, and every 2D working array derived from it, stays within the
    ``max_cells`` slot budget.
    """
    (row_start, row_stop), (col_start, col_stop) = window
    row_span = row_stop - row_start
    col_span = col_stop - col_start
    if row_span <= 0 or col_span <= 0:
        return
    # Tile side keeps every materialized data/working 2D array within the
    # slot budget. Border classification is NumPy-only and tile-local.
    side = max(1, math.isqrt(max_cells))
    tile_cols = min(col_span, side)
    tile_rows = min(row_span, max(1, max_cells // tile_cols))
    for rows in range(row_start, row_stop, tile_rows):
        row_end = min(rows + tile_rows, row_stop)
        for cols in range(col_start, col_stop, tile_cols):
            col_end = min(cols + tile_cols, col_stop)
            yield (rows, row_end), (cols, col_end)


def _cell_polygon(affine, row, col):
    """Polygon of one cell (works for arbitrary finite invertible affines)."""
    return Polygon(
        [
            affine @ (col, row),
            affine @ (col + 1, row),
            affine @ (col + 1, row + 1),
            affine @ (col, row + 1),
        ]
    )


def _validity_mask(fsrc):
    """Boolean mask of cells that carry no usable observation.

    Dataset masks, explicit nodata, NaN and +/- infinity all invalidate a
    cell.  Legitimate zeros stay valid.  The shared read layer reports the
    complete mask; directly constructed rasters fall back to the same
    composition used elsewhere.
    """
    return read_validity(fsrc)


def _empty_result(stats, coverage_area=0.0):
    """Result template for zones without any observed (or covered) area."""
    result = {}
    if "count" in stats:
        result["count"] = 0
    if "sum" in stats:
        result["sum"] = 0.0
    if "mean" in stats:
        result["mean"] = None
    result["coverage_area"] = float(coverage_area)
    result["valid_area"] = 0.0
    result["missing_area"] = float(coverage_area)
    return result


def fractional_zonal_stats(geom, rast, stats, value_kind, max_cells):
    """Compute fractional coverage statistics for one zone geometry.

    Parameters
    ----------
    geom : shapely Polygon or MultiPolygon
        Zone geometry, validated by the caller.
    rast : rasterstats.io.Raster
        Open raster (file-backed dataset or ndarray with affine).
    stats : sequence of str
        Subset of :data:`FRACTIONAL_STATS` to return.
    value_kind : {"total", "density"}
        Interpretation of the cell values.
    max_cells : int
        Maximum number of cell slots read or held in any one 2D working
        array.

    Returns
    -------
    dict
    """
    affine = rast.affine
    height, width = rast.shape
    cell_area = abs(affine.determinant)

    if geom is None or geom.is_empty:
        return _empty_result(stats)

    # Union once so overlapping MultiPolygon components count a single time.
    # shapely.union_all preserves holes and leaves a lone Polygon untouched.
    zone = shapely.union_all(geom)
    frame = _frame_polygon(affine, height, width)
    in_frame = zone.intersection(frame)

    coverage_area = float(in_frame.area)
    if in_frame.is_empty or coverage_area == 0.0:
        # Edge-only contact has no area; nothing can be observed either.
        return _empty_result(stats)

    zone_polygons = _polygon_parts(in_frame)
    if not zone_polygons:
        return _empty_result(stats, coverage_area=coverage_area)

    tree = shapely.STRtree(zone_polygons)

    full_window = _bounds_window(in_frame.bounds, affine, rast.shape)

    count = 0.0
    valid_area = 0.0
    weighted_sum = 0.0

    tiles = list(_iter_band_windows(full_window, max_cells))
    if not tiles:
        return _empty_result(stats, coverage_area=coverage_area)

    def _accumulate(overlap, values, invalid_mask):
        nonlocal count, valid_area, weighted_sum
        observed = (~invalid_mask) & (overlap > 0.0)
        if not observed.any():
            return
        fractions = overlap / cell_area
        values64 = values.astype(np.float64, copy=False)
        count += float(fractions[observed].sum())
        chunk_valid_area = float(overlap[observed].sum())
        valid_area += chunk_valid_area
        if value_kind == "total":
            # Whole-cell quantity split by coverage fraction.
            weighted_sum += float((fractions[observed] * values64[observed]).sum())
        else:
            # Constant per-unit-area intensity times the overlap area.
            weighted_sum += float((overlap[observed] * values64[observed]).sum())

    # The GDAL block cache is capped by the caller (gen_zonal_stats enters a
    # rasterio.Env before opening the dataset, since the cap is only honored
    # at open time), so peak RSS stays independent of the number of tiles.
    for tile in tiles:
        (tile_row_start, tile_row_stop), (tile_col_start, tile_col_stop) = tile
        tile_rows = tile_row_stop - tile_row_start
        tile_cols = tile_col_stop - tile_col_start

        # The zone was clipped to the raster frame before windowing, so every
        # requested window lies inside the dataset: boundless=False is always
        # valid here and, unlike boundless reads, goes through the capped GDAL
        # block cache instead of accumulating per-window allocations.
        fsrc = rast.read(window=tile, boundless=False)
        tile_affine = _window_affine(affine, tile)
        invalid, raw_data = _validity_mask(fsrc)

        # A single GDAL rasterize per tile gives every cell the zone fills
        # or touches. Keeping it to one call matters because each rasterizer
        # invocation retains allocations in the GDAL process memory pool.
        touched = features.rasterize(
            [(in_frame, 1)],
            out_shape=(tile_rows, tile_cols),
            transform=tile_affine,
            fill=0,
            dtype="uint8",
            all_touched=True,
        ).astype(bool)

        # Boundary cells = touched cells with an untouched 8-neighbor.
        # The neighbor mask is derived purely with NumPy from the tile
        # itself; cells on the tile's outer border are conservatively
        # treated as boundary and clipped exactly below. Those seam cells
        # are only O(perimeter), so interior bulk cells stay on the cheap
        # path and no array larger than the tile is ever allocated.
        neighbor_inside = touched.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                shifted = np.zeros_like(touched)
                src_r = (max(0, dr), tile_rows + min(0, dr))
                dst_r = (max(0, -dr), tile_rows + min(0, -dr))
                src_c = (max(0, dc), tile_cols + min(0, dc))
                dst_c = (max(0, -dc), tile_cols + min(0, -dc))
                shifted[dst_r[0]:dst_r[1], dst_c[0]:dst_c[1]] = touched[
                    src_r[0]:src_r[1], src_c[0]:src_c[1]
                ]
                neighbor_inside &= shifted
        full_mask = touched & neighbor_inside
        boundary_mask = touched & ~neighbor_inside

        overlap = np.zeros((tile_rows, tile_cols), dtype=np.float64)
        overlap[full_mask] = cell_area

        # Only boundary cells need exact Shapely clipping.
        exact_local = np.nonzero(boundary_mask)
        for lr, lc in zip(exact_local[0].tolist(), exact_local[1].tolist()):
            global_row = tile_row_start + lr
            global_col = tile_col_start + lc
            cell = _cell_polygon(affine, global_row, global_col)
            candidates = tree.query(cell)
            area = 0.0
            for candidate in candidates.tolist():
                area += zone_polygons[candidate].intersection(cell).area
            if area > 0.0:
                overlap[lr, lc] = area

        # Tiles are disjoint cell sets, so every cell is accumulated once.
        _accumulate(overlap, raw_data, invalid)

    result = {}
    if "count" in stats:
        result["count"] = float(count)
    if "sum" in stats:
        result["sum"] = float(weighted_sum)
    if "mean" in stats:
        result["mean"] = None if valid_area == 0.0 else float(weighted_sum / valid_area)
    result["coverage_area"] = coverage_area
    result["valid_area"] = float(valid_area)
    # clamp away cancellation noise so covered-valid never reads as negative
    result["missing_area"] = max(0.0, coverage_area - valid_area)
    return result
