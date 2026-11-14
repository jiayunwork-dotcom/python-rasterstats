"""Exact area-weighted (fractional) coverage statistics.

Unlike the center-point / ``all_touched`` cell selection used by
:func:`rasterstats.zonal_stats`, the functions in this module weight every
raster cell by the *exact* fraction of its area that intersects a polygon.

The coverage fraction is a planar quantity: each raster cell is the real
quadrilateral obtained by applying the raster's :class:`~affine.Affine`
transform to the unit square, and its weight is

    intersection area(cell, polygon) / area(cell)

computed in the (shared, planar) coordinate system of the inputs.  No
reprojection or spherical-area conversion is performed.  Boundary-only
contacts (zero intersection area) contribute nothing, interior rings are
subtracted, and a valid ``MultiPolygon`` is treated as one region.

The module is organized by responsibility:

* input validation helpers (:func:`check_positive_int`, ...),
* polygon validation (:func:`validate_polygon`),
* cell-grid geometry helpers (:func:`grid_bounds`, :func:`cell_polygons`),
* bounded, chunked raster access (:class:`ArrayBlocks` and
  :class:`FileBlocks`), and
* the public generator / list entry points
  (:class:`FractionalStatsGenerator`, :func:`gen_fractional_stats`,
  :func:`fractional_stats`).
"""

import math
import warnings

import numpy as np
from affine import Affine
from rasterio.enums import MaskFlags
from shapely import area, box, intersection, prepare, transform
from shapely.geometry import shape as shapely_shape

from rasterstats.io import DEFAULT_CHUNK_SIZE, read_features

#: The seven keys delivered for every feature, in their documented order.
FRACTIONAL_STAT_KEYS = (
    "coverage",
    "count",
    "nodata",
    "sum",
    "mean",
    "min",
    "max",
)


# ---------------------------------------------------------------------------
# Scalar / raster input validation
# ---------------------------------------------------------------------------


def check_positive_int(value, name):
    """Validate a non-bool, strictly positive integer parameter."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"{name} must be a positive integer (not a bool); got {value!r}"
        )
    return value


def check_affine(affine):
    """Validate a finite, invertible :class:`Affine` grid transform."""
    if not isinstance(affine, Affine):
        raise ValueError(f"affine must be an affine.Affine instance; got {affine!r}")
    if not all(math.isfinite(v) for v in tuple(affine)):
        raise ValueError(f"affine must contain only finite values; got {affine!r}")
    determinant = affine.determinant
    if not math.isfinite(determinant) or determinant == 0.0:
        raise ValueError(
            f"affine must be invertible (non-zero determinant); got {affine!r}"
        )
    return affine


def check_real_2d_array(array):
    """Validate a 2D, real-valued (non-bool, non-complex, non-object) array."""
    if not isinstance(array, np.ndarray):
        raise ValueError(f"raster array must be a numpy.ndarray; got {type(array)!r}")
    if array.ndim != 2:
        raise ValueError(
            f"raster array must be two-dimensional; got {array.ndim} dimensions"
        )
    kind = array.dtype.kind
    if kind == "b":
        raise ValueError("boolean raster arrays are not supported; use an integer dtype")
    if kind not in ("i", "u", "f"):
        raise ValueError(
            "raster array must have an integer or floating-point dtype; "
            f"got {array.dtype!r}"
        )
    return array


# ---------------------------------------------------------------------------
# Polygon validation
# ---------------------------------------------------------------------------


def validate_polygon(geometry):
    """Return a finite, valid, 2D Shapely polygon-like geometry.

    Empty ``Polygon`` / ``MultiPolygon`` geometries are legal.  ``None``,
    non-polygon geometry types, three-dimensional or non-finite coordinates
    and topologically invalid polygons raise :class:`ValueError`.
    """
    if geometry is None:
        raise ValueError("geometry is None; a Polygon or MultiPolygon is required")
    if not hasattr(geometry, "geom_type"):
        # GEOS emits a RuntimeWarning while decoding non-finite ring
        # coordinates; that condition is reported below as a ValueError.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            geometry = shapely_shape(geometry)

    geom_type = geometry.geom_type
    if geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(
            "only Polygon or MultiPolygon geometries are supported; "
            f"got {geom_type}"
        )

    if geometry.has_z:
        raise ValueError("three-dimensional geometries are not supported")

    coords = np.asarray(
        list(
            _all_coords(geometry)
            if geom_type == "MultiPolygon"
            else _polygon_coords(geometry)
        ),
        dtype=np.float64,
    )
    if coords.size and not np.isfinite(coords).all():
        raise ValueError("geometry contains non-finite (NaN / infinite) coordinates")

    # Empty polygons (e.g. an empty polygon ring) are explicitly allowed.
    if geometry.is_empty:
        return geometry

    if not geometry.is_valid:
        raise ValueError(
            "geometry is topologically invalid (e.g. self-intersecting or "
            "overlapping MultiPolygon components); a valid Polygon or "
            "MultiPolygon is required"
        )

    if geom_type == "MultiPolygon":
        for i, part in enumerate(geometry.geoms):
            if part.geom_type != "Polygon" or part.has_z:
                raise ValueError(
                    "MultiPolygon components must be 2D polygons; "
                    f"component {i} is {part.geom_type}"
                )
    return geometry


def _all_coords(multipolygon):
    """Yield every coordinate of a MultiPolygon's rings."""
    for polygon in multipolygon.geoms:
        yield from polygon.exterior.coords
        for interior in polygon.interiors:
            yield from interior.coords


# ---------------------------------------------------------------------------
# Cell-grid geometry
# ---------------------------------------------------------------------------


def grid_bounds(geometry, inverse_affine):
    """Return ``(min_row, max_row, min_col, max_col)`` cell indices covering
    a geometry.

    Indices are in grid (row, column) space obtained with the inverse
    :class:`Affine`.  The returned span may include cells that merely touch
    the geometry; those end up with a zero intersection area and are dropped.
    """
    coordinates = np.asarray(
        list(
            _all_coords(geometry)
            if geometry.geom_type == "MultiPolygon"
            else _polygon_coords(geometry)
        ),
        dtype=np.float64,
    )
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    grid_x = inverse_affine.a * x + inverse_affine.b * y + inverse_affine.c
    grid_y = inverse_affine.d * x + inverse_affine.e * y + inverse_affine.f
    return (
        math.floor(grid_y.min()),
        math.ceil(grid_y.max()),
        math.floor(grid_x.min()),
        math.ceil(grid_x.max()),
    )


def _polygon_coords(polygon):
    yield from polygon.exterior.coords
    for interior in polygon.interiors:
        yield from interior.coords


def cell_polygons(row_offset, col_offset, n_rows, n_cols, affine):
    """Build the cell quadrilaterals of a block as a Shapely geometry array.

    ``row_offset`` / ``col_offset`` are the absolute grid indices of the
    block's top-left cell.  Rotation, shear and negative-axis transforms are
    handled because every unit-square corner is mapped through ``affine``.
    """
    local_cols, local_rows = np.meshgrid(
        np.arange(n_cols, dtype=np.float64),
        np.arange(n_rows, dtype=np.float64),
    )
    abs_cols = (local_cols + col_offset).ravel()
    abs_rows = (local_rows + row_offset).ravel()
    unit_cells = box(abs_cols, abs_rows, abs_cols + 1.0, abs_rows + 1.0)

    def _map(coords):
        x = coords[:, 0]
        y = coords[:, 1]
        mapped_x = affine.a * x + affine.b * y + affine.c
        mapped_y = affine.d * x + affine.e * y + affine.f
        return np.column_stack([mapped_x, mapped_y])

    return transform(unit_cells, _map)


# ---------------------------------------------------------------------------
# Bounded raster block readers
# ---------------------------------------------------------------------------


class ArrayBlocks:
    """Windowed access to an in-memory NumPy array.

    Only ``O(max_cells)`` temporary arrays are ever allocated; the source
    array itself is referenced, never copied wholesale.
    """

    def __init__(self, array, affine, nodata, max_cells):
        self.array = check_real_2d_array(array)
        self.affine = check_affine(affine)
        check_positive_int(max_cells, "max_cells")
        self.shape = self.array.shape
        self.max_cells = max_cells
        self.is_masked = isinstance(array, np.ma.MaskedArray)
        if isinstance(nodata, bool) or (
            nodata is not None
            and not isinstance(nodata, (int, float, np.integer, np.floating))
        ):
            raise ValueError(f"nodata must be a number or None; got {nodata!r}")
        self.sentinel = nodata

    def read_block(self, row_start, row_stop, col_start, col_stop):
        """Return ``(values, validity)`` for a fully in-raster window.

        ``values`` is a float64 copy of the block; ``validity`` is a boolean
        array marking cells that are not invalidated by a mask, a numeric
        sentinel, NaN or infinity.
        """
        values = np.asarray(self.array[row_start:row_stop, col_start:col_stop], dtype=np.float64)
        valid = np.ones(values.shape, dtype=bool)

        if self.is_masked and self.array.mask is not np.ma.nomask:
            valid &= ~np.asarray(self.array.mask[row_start:row_stop, col_start:col_stop])

        if self.sentinel is not None:
            valid &= values != float(self.sentinel)

        if values.dtype.kind == "f":
            valid &= np.isfinite(values)
        return values, valid

    def close(self):
        self._closed = True


class FileBlocks:
    """Windowed, chunked access to an on-disk rasterio dataset.

    Every :meth:`read_block` call asks rasterio for at most ``max_cells``
    pixels, so neither the whole raster nor a feature's full bounding box is
    materialized at once.
    """

    def __init__(self, path, band, nodata, max_cells):
        import rasterio

        check_positive_int(max_cells, "max_cells")
        self.max_cells = max_cells
        self.src = rasterio.open(path, "r")
        try:
            if not isinstance(band, int) or isinstance(band, bool) or band < 1:
                raise ValueError(f"band must be an integer >= 1; got {band!r}")
            if band > self.src.count:
                raise ValueError(
                    f"band {band} does not exist; raster has {self.src.count} band(s)"
                )
            self.band = band
            dtype = np.dtype(self.src.dtypes[band - 1])
            if dtype.kind == "b":
                raise ValueError("boolean rasters are not supported")
            if dtype.kind not in ("i", "u", "f"):
                raise ValueError(
                    f"raster must have an integer or floating-point dtype; got {dtype!r}"
                )
            self.affine = check_affine(self.src.transform)
            self.shape = (self.src.height, self.src.width)
            self.sentinel = float(nodata) if nodata is not None else self.src.nodata
            flags = self.src.mask_flag_enums[band - 1]
            # A per-dataset mask or an alpha band both encode validity
            # independently of any numeric nodata sentinel.  The plain
            # ``nodata`` flag, by contrast, is driven by the sentinel value
            # and therefore *can* be overridden via the nodata argument.
            self.has_independent_mask = any(
                flag in (MaskFlags.per_dataset, MaskFlags.alpha) for flag in flags
            )
        except BaseException:
            self.src.close()
            raise

    def read_block(self, row_start, row_stop, col_start, col_stop):
        window = ((row_start, row_stop), (col_start, col_stop))
        values = self.src.read(self.band, window=window, masked=False).astype(np.float64)
        valid = np.ones(values.shape, dtype=bool)

        if self.has_independent_mask:
            # The dataset's own validity mask cannot be removed by nodata=.
            valid &= self.src.read_masks(self.band, window=window) != 0

        if self.sentinel is not None:
            valid &= values != self.sentinel

        valid &= np.isfinite(values)
        return values, valid

    def close(self):
        if self.src is not None:
            self.src.close()
            self.src = None


# ---------------------------------------------------------------------------
# Per-feature computation
# ---------------------------------------------------------------------------


def _empty_result():
    return {
        "coverage": 0.0,
        "count": 0.0,
        "nodata": 0.0,
        "sum": 0.0,
        "mean": None,
        "min": None,
        "max": None,
    }


def iter_block_windows(row_lo, row_hi, col_lo, col_hi, max_cells):
    """Yield ``(row_start, row_stop, col_start, col_stop)`` windows.

    The candidate span is split two-dimensionally (column strips first,
    then row blocks) so that every window contains at most ``max_cells``
    cells, even when a feature's bounding box is wider than the limit.
    """
    check_positive_int(max_cells, "max_cells")
    for col_start in range(col_lo, col_hi, max_cells):
        col_stop = min(col_start + max_cells, col_hi)
        strip_width = col_stop - col_start
        block_height = max(1, max_cells // strip_width)
        for row_start in range(row_lo, row_hi, block_height):
            row_stop = min(row_start + block_height, row_hi)
            yield row_start, row_stop, col_start, col_stop


def fractional_feature(geometry, blocks, max_cells):
    """Compute the seven fractional statistics for one polygon geometry.

    Workspace is bounded by ``O(max_cells)`` plus the supplied geometry and
    fixed grid metadata: cells are visited block by block and only running
    totals are retained.
    """
    geometry = validate_polygon(geometry)
    if geometry.is_empty:
        return _empty_result()

    prepare(geometry)
    affine = blocks.affine
    cell_area = abs(affine.determinant)
    inverse = ~affine

    n_rows_total, n_cols_total = blocks.shape
    min_row, max_row, min_col, max_col = grid_bounds(geometry, inverse)

    # Clip the candidate span to the finite raster extent.  Coverage beyond
    # the raster is neither counted nor treated as nodata.
    row_lo = max(min_row, 0)
    row_hi = min(max_row, n_rows_total)
    col_lo = max(min_col, 0)
    col_hi = min(max_col, n_cols_total)
    if row_lo >= row_hi or col_lo >= col_hi:
        return _empty_result()

    coverage_total = 0.0
    count_total = 0.0
    nodata_total = 0.0
    sum_total = 0.0
    min_value = None
    max_value = None

    for row_start, row_stop, col_start, col_stop in iter_block_windows(
        row_lo, row_hi, col_lo, col_hi, max_cells
    ):
        # Cheap envelope pre-check: skip (without touching the raster) any
        # block whose axis-aligned extent misses the polygon.  The four block
        # corners are mapped through the (possibly rotated) affine, so this
        # stays conservative and never skips a block that truly intersects.
        corners = np.array(
            [
                (col_start, row_start),
                (col_stop, row_start),
                (col_stop, row_stop),
                (col_start, row_stop),
            ],
            dtype=np.float64,
        )
        map_x = affine.a * corners[:, 0] + affine.b * corners[:, 1] + affine.c
        map_y = affine.d * corners[:, 0] + affine.e * corners[:, 1] + affine.f
        if not geometry.intersects(
            box(map_x.min(), map_y.min(), map_x.max(), map_y.max())
        ):
            continue

        n_rows = row_stop - row_start
        n_cols = col_stop - col_start
        values, valid = blocks.read_block(row_start, row_stop, col_start, col_stop)

        cells = cell_polygons(row_start, col_start, n_rows, n_cols, affine)
        fractions = area(intersection(geometry, cells)).reshape(n_rows, n_cols) / cell_area
        # Guard against tiny floating-point overshoot from the planar math.
        np.clip(fractions, 0.0, 1.0, out=fractions)

        touched = fractions > 0.0
        if not touched.any():
            continue

        coverage_total += float(fractions.sum())
        valid_fractions = fractions * valid
        nodata_fractions = fractions * ~valid
        count_block = float(valid_fractions[touched].sum())
        nodata_block = float(nodata_fractions[touched].sum())
        count_total += count_block
        nodata_total += nodata_block
        # Replace non-finite values before weighting so NaN/inf * 0 cannot
        # leak into the sum; validity already excludes those cells.
        safe_values = np.where(valid, values, 0.0)
        sum_total += float((safe_values * valid_fractions).sum())

        positive_valid = touched & valid
        if positive_valid.any():
            block_min = float(values[positive_valid].min())
            block_max = float(values[positive_valid].max())
            min_value = block_min if min_value is None else min(min_value, block_min)
            max_value = block_max if max_value is None else max(max_value, block_max)

    result = {
        "coverage": coverage_total,
        "count": count_total,
        "nodata": nodata_total,
        "sum": sum_total,
        "mean": (sum_total / count_total) if count_total > 0.0 else None,
        "min": min_value,
        "max": max_value,
    }
    if result["count"] == 0.0:
        result["sum"] = 0.0
    return result


# ---------------------------------------------------------------------------
# Public generator / list entry points
# ---------------------------------------------------------------------------


class FractionalStatsGenerator:
    """Lazy, per-feature generator of exact fractional coverage statistics.

    Instances are iterables that open their data sources lazily on the first
    iteration step, process one feature at a time (a one-shot feature source
    is consumed exactly once) and release every data source they opened when
    exhausted, on error or on an explicit :meth:`close`.
    """

    def __init__(
        self,
        vectors,
        raster,
        *,
        band=1,
        affine=None,
        nodata=None,
        layer=0,
        engine=None,
        chunk_size=DEFAULT_CHUNK_SIZE,
        max_cells=DEFAULT_CHUNK_SIZE,
        prefix=None,
        geojson_out=False,
    ):
        check_positive_int(chunk_size, "chunk_size")
        check_positive_int(max_cells, "max_cells")
        if prefix is not None and not isinstance(prefix, str):
            raise ValueError(f"prefix must be None or a string; got {prefix!r}")
        if not isinstance(geojson_out, bool):
            raise ValueError(f"geojson_out must be a bool; got {geojson_out!r}")
        if not isinstance(band, int) or isinstance(band, bool) or band < 1:
            raise ValueError(f"band must be an integer >= 1; got {band!r}")
        if isinstance(nodata, bool) or (
            nodata is not None
            and not isinstance(nodata, (int, float, np.integer, np.floating))
        ):
            raise ValueError(f"nodata must be a number or None; got {nodata!r}")

        self.vectors = vectors
        self.raster = raster
        self.band = band
        self.affine = affine
        self.nodata = nodata
        self.layer = layer
        self.engine = engine
        self.chunk_size = chunk_size
        self.max_cells = max_cells
        self.prefix = prefix
        self.geojson_out = geojson_out

        self._features = None
        self._blocks = None
        self._feature_index = 0
        self._closed = False

    def _start(self):
        """Lazily open the feature stream and raster block reader."""
        # read_features treats an empty FeatureCollection's falsy feature
        # list as unrecognized; short-circuit to an empty stream here.
        if (
            isinstance(self.vectors, dict)
            and self.vectors.get("type") == "FeatureCollection"
            and not self.vectors.get("features")
        ):
            self._features = iter(())
        else:
            self._features = iter(
                read_features(
                    self.vectors,
                    self.layer,
                    engine=self.engine,
                    chunk_size=self.chunk_size,
                )
            )
        if isinstance(self.raster, np.ndarray):
            if self.band != 1:
                raise ValueError(
                    f"array rasters only support band=1; got band={self.band}"
                )
            if self.affine is None:
                raise ValueError("an affine transform is required when raster is an array")
            self._blocks = ArrayBlocks(
                self.raster, self.affine, self.nodata, self.max_cells
            )
        else:
            if self.affine is not None:
                raise ValueError(
                    "affine can only be supplied together with an array raster; "
                    "file rasters provide their own transform"
                )
            self._blocks = FileBlocks(
                self.raster, self.band, self.nodata, self.max_cells
            )

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        if self._features is None:
            try:
                self._start()
            except BaseException:
                # Failure while opening a source: release whatever opened.
                self.close()
                raise
        try:
            feature = next(self._features)
        except StopIteration:
            self.close()
            raise
        except (ValueError, AttributeError, TypeError) as exc:
            # Feature parsing/decoding problems are data-related and carry
            # the 0-based feature index; file I/O exceptions are not caught
            # here and propagate in their original form.
            self.close()
            raise ValueError(f"feature {self._feature_index}: {exc}") from exc
        except BaseException:
            # A broken feature source: release the raster before propagating.
            self.close()
            raise

        geometry = feature["geometry"]
        if geometry is None:
            self.close()
            raise ValueError(
                f"feature {self._feature_index}: geometry is None; "
                "a Polygon or MultiPolygon is required"
            )
        try:
            stats = fractional_feature(geometry, self._blocks, self.max_cells)
        except ValueError as exc:
            # Data-related errors identify the 0-based feature.  File I/O
            # errors are raised by the block reader and pass through untouched.
            self.close()
            raise ValueError(
                f"feature {self._feature_index}: {exc}"
            ) from exc
        except BaseException:
            self.close()
            raise

        result = self._format(feature, stats)
        self._feature_index += 1
        return result

    def _format(self, feature, stats):
        if self.prefix is not None:
            stats = {f"{self.prefix}{key}": value for key, value in stats.items()}
        if not self.geojson_out:
            return stats

        # Never modify the caller's Feature: copy the dict and its properties.
        out = dict(feature)
        properties = dict(feature.get("properties") or {})
        properties.update(stats)
        out["properties"] = properties
        if "geometry" in feature:
            out["geometry"] = feature["geometry"]
        if "id" in feature:
            out["id"] = feature["id"]
        return out

    def close(self):
        """Release data sources opened by this generator.

        Safe to call more than once and safe to call before iteration starts;
        a closed generator never reads again.  Cleanup failures must not mask
        an exception currently being propagated.
        """
        if self._closed:
            return
        self._closed = True
        if self._blocks is not None:
            try:
                self._blocks.close()
            except Exception:
                pass
        if self._features is not None:
            # Generators from the vector backends release their handles when
            # closed (fiona) or when dropped (pyogrio reads per chunk).
            close = getattr(self._features, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        self._blocks = None
        self._features = None

    def __del__(self):
        # Best-effort cleanup if the caller abandoned the generator.
        try:
            self.close()
        except Exception:
            pass


def gen_fractional_stats(
    vectors,
    raster,
    *,
    band=1,
    affine=None,
    nodata=None,
    layer=0,
    engine=None,
    chunk_size=DEFAULT_CHUNK_SIZE,
    max_cells=DEFAULT_CHUNK_SIZE,
    prefix=None,
    geojson_out=False,
):
    """Generate exact area-weighted (fractional) raster statistics per feature.

    Each raster cell is treated as the real quadrilateral produced by the
    raster's :class:`~affine.Affine` transform.  The weight of a cell for a
    feature is the area of its intersection with the feature polygon divided
    by the cell's own area, measured in the shared planar coordinate system
    of the inputs (no reprojection or spherical-area conversion).  Cells only
    touched along a boundary carry zero weight, interior rings are subtracted
    and a valid ``MultiPolygon`` counts as a single region.  Regions smaller
    than a cell still contribute their actual fractional coverage.

    Parameters
    ----------
    vectors :
        Anything accepted by :func:`rasterstats.io.read_features`: a vector
        file path, a GeoJSON ``FeatureCollection`` / ``Feature`` / geometry
        mapping, a Shapely-like geometry, or a one-shot iterable of
        feature-like objects.  One result is delivered per input feature, in
        input order; duplicate features are computed independently and the
        inputs are never modified.
    raster :
        A path to a GDAL raster source or a two-dimensional real NumPy
        array (``MaskedArray`` supported).  Arrays require ``affine`` and
        only ``band=1``.  Boolean, complex and object dtypes are rejected.
    band : int, keyword-only
        1-based raster band for file rasters.  Arrays only support band 1.
    affine : Affine, keyword-only
        Grid-to-coordinate transform for array rasters.  Must be finite and
        invertible; rotation, shear and negative-axis directions are
        supported.  File rasters use their own transform.
    nodata : float, keyword-only
        Numeric sentinel overriding a file's metadata nodata.  It cannot
        remove a dataset's independent validity mask.  With ``None``, files
        use their metadata sentinel while arrays have no numeric sentinel.
        A plain value such as ``-999`` stays valid when no metadata marks it.
    layer : int or str, keyword-only
        Vector layer index or name for file-based vector sources.
    engine : {"pyogrio", "fiona"} or None, keyword-only
        Vector-reading backend, forwarded to ``read_features``.
    chunk_size : int, keyword-only
        Number of vector features read per batch (pyogrio backend).
    max_cells : int, keyword-only
        Maximum number of raster cells read in one window.  All new arrays,
        weights and intersection bookkeeping fit in ``O(max_cells)``
        workspace; a feature's full bounding box is never read in one go.
    prefix : str or None, keyword-only
        Optional prefix prepended to every output key.
    geojson_out : bool, keyword-only
        When true, yield GeoJSON ``Feature`` dicts preserving the original
        geometry, id and properties (the input feature is not modified).

    Returns
    -------
    generator of dict
        One dictionary per feature with the keys ``coverage``, ``count``,
        ``nodata``, ``sum``, ``mean``, ``min`` and ``max``:

        * ``coverage``: sum of intersection fractions over all in-raster
          cells (the covered area expressed in cell-area units).
        * ``count``: as a float, the sum of fractions of *valid* cells.
        * ``nodata``: sum of fractions of invalid (masked / sentinel /
          NaN / infinite) cells.
        * ``sum``: valid cell values weighted by their coverage fraction.
        * ``mean``: coverage-weighted mean over valid coverage.
        * ``min`` / ``max``: original values of the valid cells with
          positive coverage (extreme values, unweighted).

        Coverage outside the raster extent is ignored and is not nodata.
        With no valid coverage, ``count`` and ``sum`` are ``0.0`` and
        ``mean`` / ``min`` / ``max`` are ``None`` while ``coverage`` and
        ``nodata`` still reflect the actual intersection.  Empty or fully
        disjoint regions yield all-zero coverage quantities.
    """
    return FractionalStatsGenerator(
        vectors,
        raster,
        band=band,
        affine=affine,
        nodata=nodata,
        layer=layer,
        engine=engine,
        chunk_size=chunk_size,
        max_cells=max_cells,
        prefix=prefix,
        geojson_out=geojson_out,
    )


def fractional_stats(
    vectors,
    raster,
    *,
    band=1,
    affine=None,
    nodata=None,
    layer=0,
    engine=None,
    chunk_size=DEFAULT_CHUNK_SIZE,
    max_cells=DEFAULT_CHUNK_SIZE,
    prefix=None,
    geojson_out=False,
):
    """List-returning counterpart of :func:`gen_fractional_stats`.

    All arguments are identical to :func:`gen_fractional_stats`; this
    materializes the generator and returns a list of the per-feature
    statistics dictionaries.
    """
    return list(
        gen_fractional_stats(
            vectors,
            raster,
            band=band,
            affine=affine,
            nodata=nodata,
            layer=layer,
            engine=engine,
            chunk_size=chunk_size,
            max_cells=max_cells,
            prefix=prefix,
            geojson_out=geojson_out,
        )
    )
