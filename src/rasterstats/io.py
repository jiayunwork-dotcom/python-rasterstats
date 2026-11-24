import json
import math
import warnings
from collections.abc import Iterable, Mapping
from json import JSONDecodeError
from os import PathLike

import numpy as np
import pyogrio
import pyogrio.raw
import rasterio
import shapely
from affine import Affine
from rasterio.enums import MaskFlags
from rasterio.transform import guard_transform
from shapely import wkb, wkt

try:
    from shapely.errors import ShapelyError
except ImportError:  # pragma: no cover
    from shapely.errors import ReadingError as ShapelyError


geom_types = [
    "Point",
    "LineString",
    "Polygon",
    "MultiPoint",
    "MultiLineString",
    "MultiPolygon",
]

# Fiona backend


def _fiona_generator(obj, layer=0):
    """Yield GeoJSON-like Feature dicts using fiona (optional engine).

    Raises ImportError with a helpful message if fiona is not installed.
    """
    try:
        import fiona
    except ImportError:
        raise ImportError(
            "fiona is required for engine='fiona'. "
            "Install it with: pip install rasterstats[fiona]"
        )
    try:
        import fiona.model

        with fiona.open(obj, "r", layer=layer) as src:
            for feat in src:
                yield fiona.model.to_dict(feat)
    except ImportError:
        # fiona < 1.9 — no fiona.model
        with fiona.open(obj, "r", layer=layer) as src:
            yield from src


# pyogrio backend

DEFAULT_CHUNK_SIZE = 65536


def _pyogrio_generator(obj, layer=0, chunk_size=DEFAULT_CHUNK_SIZE):
    """Yield GeoJSON-like Feature dicts using pyogrio, reading in chunks."""
    info = pyogrio.read_info(obj, layer=layer)
    field_names = list(info["fields"])

    # Some sources advertise `info["features"]" == -1` (WFS, certain VRT sources).
    # ie they don't support *fast* feature counting, thus return -1
    # But they *do* support iteration, which is all we need.

    skip = 0
    while True:
        _meta, fids, geometries, field_data = pyogrio.raw.read(
            obj,
            layer=layer,
            skip_features=skip,
            max_features=chunk_size,
            return_fids=True,
            # Note do not use_arrow=True, reads all records into mem
            # https://pyogrio.readthedocs.io/en/latest/about.html#how-it-works
        )
        batch_size = len(geometries)
        geoms = shapely.from_wkb(geometries)
        cols = [col.tolist() for col in field_data]
        for i in range(batch_size):
            props = {name: cols[j][i] for j, name in enumerate(field_names)}
            geom = geoms[i]
            fid = str(fids[i])  # matches engine=fiona behavior
            yield {
                "type": "Feature",
                "id": fid,
                "geometry": geom.__geo_interface__ if geom is not None else None,
                "properties": props,
            }
        skip += batch_size
        if batch_size < chunk_size:
            break


# Public dispatcher

DEFAULT_ENGINE = "pyogrio"


def feature_generator(obj, layer=0, engine=None, chunk_size=DEFAULT_CHUNK_SIZE):
    """Yield GeoJSON-like Feature dicts from a file-based vector source.

    Parameters
    ----------
    obj : str or PathLike
        Path to a vector data source supported by pyogrio (default) or fiona.
    layer : int or str, optional
        Layer index or name (default: 0).
    engine : {"pyogrio", "fiona"} or None, optional
        Backend to use for reading. ``None`` selects the default engine
        (``"pyogrio"``). Pass ``"fiona"`` to opt in to the fiona backend
        (requires ``pip install rasterstats[fiona]``).
    chunk_size : int, optional
        Number of features to read per batch when using the pyogrio engine.
        Reduce this value to lower peak memory usage for datasets with
        large, vertex-dense geometries. Default: ``DEFAULT_CHUNK_SIZE``.
        Has no effect when ``engine='fiona'``.

    Yields
    ------
    dict
        GeoJSON-like Feature dicts.
    """
    resolved = engine if engine is not None else DEFAULT_ENGINE
    if resolved == "pyogrio":
        yield from _pyogrio_generator(obj, layer=layer, chunk_size=chunk_size)
    elif resolved == "fiona":
        yield from _fiona_generator(obj, layer=layer)
    else:
        raise ValueError(f"Unknown engine {resolved!r}. Choose 'pyogrio' or 'fiona'.")


def wrap_geom(geom):
    """Wraps a geometry dict in an GeoJSON Feature"""
    return {"type": "Feature", "properties": {}, "geometry": geom}


def parse_feature(obj):
    """Given a python object
    attemp to a GeoJSON-like Feature from it
    """

    # object implementing geo_interface
    if hasattr(obj, "__geo_interface__"):
        gi = obj.__geo_interface__
        if gi["type"] in geom_types:
            return wrap_geom(gi)
        elif gi["type"] == "Feature":
            return gi

    # wkt
    try:
        shape = wkt.loads(obj)
        return wrap_geom(shape.__geo_interface__)
    except (ShapelyError, TypeError, AttributeError):
        pass

    # wkb
    try:
        shape = wkb.loads(obj)
        return wrap_geom(shape.__geo_interface__)
    except (ShapelyError, TypeError):
        pass

    # geojson-like python mapping
    try:
        if obj["type"] in geom_types:
            return wrap_geom(obj)
        elif obj["type"] == "Feature":
            return obj
    except (AssertionError, TypeError):
        pass

    raise ValueError(f"Can't parse {obj} as a geojson Feature object")


def _is_vector_file(path, layer):
    """Return True if ``path`` is a readable vector file with at least one feature.

    Uses pyogrio.read_info for the probe — no fiona required.
    Returns False for invalid paths, non-vector files, and genuinely empty sources.
    A feature count of -1 (unknown, e.g. some GeoJSON drivers) is treated as
    non-empty (True).
    """
    try:
        info = pyogrio.read_info(path, layer=layer)
    # important: don't catch DataLayerErrors, let those bubble up directly
    except pyogrio.errors.DataSourceError:
        return False
    return info["features"] != 0


def read_features(obj, layer=0, engine=None, chunk_size=DEFAULT_CHUNK_SIZE):
    features_iter = None
    if isinstance(obj, (str, PathLike)):
        obj = str(obj)
        if _is_vector_file(obj, layer):
            features_iter = feature_generator(
                obj, layer, engine=engine, chunk_size=chunk_size
            )
        else:
            try:
                mapping = json.loads(obj)
                if "type" in mapping and mapping["type"] == "FeatureCollection":
                    features_iter = mapping["features"]
                elif mapping["type"] in geom_types + ["Feature"]:
                    features_iter = [parse_feature(mapping)]
            except (ValueError, JSONDecodeError):
                # Single feature-like string
                features_iter = [parse_feature(obj)]
    elif isinstance(obj, Mapping):
        if "type" in obj and obj["type"] == "FeatureCollection":
            features_iter = obj["features"]
        else:
            features_iter = [parse_feature(obj)]
    elif isinstance(obj, bytes):
        # Single binary object, probably a wkb
        features_iter = [parse_feature(obj)]
    elif hasattr(obj, "__geo_interface__"):
        mapping = obj.__geo_interface__
        if mapping["type"] == "FeatureCollection":
            features_iter = mapping["features"]
        else:
            features_iter = [parse_feature(mapping)]
    elif isinstance(obj, Iterable):
        # Iterable of feature-like objects
        features_iter = (parse_feature(x) for x in obj)

    if not features_iter:
        raise ValueError("Object is not a recognized source of Features")
    return features_iter


def read_featurecollection(obj, layer=0):
    features = read_features(obj, layer=layer)
    fc = {"type": "FeatureCollection", "features": []}
    fc["features"] = [f for f in features]
    return fc


def rowcol(x, y, affine, op=math.floor):
    """Get row/col for a x/y"""
    r = int(op((y - affine.f) / affine.e))
    c = int(op((x - affine.c) / affine.a))
    return r, c


def bounds_window(bounds, affine):
    """Create a full cover rasterio-style window"""
    w, s, e, n = bounds
    row_start, col_start = rowcol(w, n, affine)
    row_stop, col_stop = rowcol(e, s, affine, op=math.ceil)
    return (row_start, row_stop), (col_start, col_stop)


def window_bounds(window, affine):
    (row_start, row_stop), (col_start, col_stop) = window
    w, s = affine * (col_start, row_stop)
    e, n = affine * (col_stop, row_start)
    return w, s, e, n


def beyond_extent(window, shape):
    """Checks if window references pixels beyond the raster extent"""
    (wr_start, wr_stop), (wc_start, wc_stop) = window
    return wr_start < 0 or wc_start < 0 or wr_stop > shape[0] or wc_stop > shape[1]


def _window_shape(arr, window):
    (wr_start, wr_stop), (wc_start, wc_stop) = window
    if arr.ndim == 3:
        return (arr.shape[0], wr_stop - wr_start, wc_stop - wc_start)
    return (wr_stop - wr_start, wc_stop - wc_start)


def _overlap(arr, window):
    """Return the slice of ``arr`` that falls inside ``window``.

    All returned indices are normalised to the source array, so callers can
    both copy the overlapping pixels and mark the remaining window slots as
    outside the source extent.
    """
    (wr_start, wr_stop), (wc_start, wc_stop) = window
    height, width = arr.shape[-2:]
    olr_start = max(min(wr_start, height), 0)
    olr_stop = max(min(wr_stop, height), 0)
    olc_start = max(min(wc_start, width), 0)
    olc_stop = max(min(wc_stop, width), 0)
    return (olr_start, olr_stop), (olc_start, olc_stop)


def _fill_value(dtype, nodata):
    """A representable fill value for window slots outside the source extent.

    The fill is only a data placeholder: callers receive an explicit
    ``in_bounds`` mask, so the value never has to be interpreted as a
    legitimate observation.  Using the numeric marker when it fits keeps the
    historical masked=False representation where possible; otherwise a
    dtype-safe placeholder is used instead of truncating or wrapping the
    marker.
    """
    dtype = np.dtype(dtype)
    if dtype.kind in "iu":
        if nodata is not None:
            try:
                if isinstance(nodata, (float, np.floating)):
                    if not math.isfinite(nodata) or nodata != math.floor(nodata):
                        return 0
                marker = int(nodata)
                info = np.iinfo(dtype)
                if info.min <= marker <= info.max:
                    return marker
            except (TypeError, ValueError, OverflowError):
                pass
        return 0
    if dtype.kind == "b":
        try:
            if nodata in (0, 1, False, True):
                return bool(nodata)
        except TypeError:
            pass
        return False
    if dtype.kind == "f":
        if nodata is not None:
            try:
                marker = float(nodata)
                if math.isnan(marker):
                    return np.nan
                if math.isfinite(marker):
                    cast = dtype.type(marker)
                    if np.isfinite(cast):
                        return cast
            except (TypeError, ValueError, OverflowError):
                pass
        return np.nan
    return 0


def nodata_matches(values, nodata):
    """Boolean mask of cells equal to the effective numeric marker.

    The comparison is exact for every source dtype: a marker outside the
    representable range of the source dtype matches nothing (it is never
    truncated, wrapped, or allowed to kill another value).  Non-finite
    markers never match, because NaN and +/- infinity are invalid on their
    own and must not be counted as numeric-marker hits.
    """
    values = np.ma.getdata(values)
    dtype = values.dtype
    if nodata is None:
        return np.zeros(values.shape, dtype=bool)

    if dtype.kind in "iu":
        try:
            if isinstance(nodata, (float, np.floating)):
                if not math.isfinite(nodata) or nodata != math.floor(nodata):
                    return np.zeros(values.shape, dtype=bool)
            elif isinstance(nodata, (complex, np.complexfloating)):
                return np.zeros(values.shape, dtype=bool)
            marker = int(nodata)
        except (TypeError, ValueError, OverflowError):
            return np.zeros(values.shape, dtype=bool)
        info = np.iinfo(dtype)
        if marker < info.min or marker > info.max:
            return np.zeros(values.shape, dtype=bool)
        return values == marker

    if dtype.kind == "b":
        try:
            if nodata in (0, 1, False, True):
                return values == bool(nodata)
        except TypeError:
            pass
        return np.zeros(values.shape, dtype=bool)

    if dtype.kind == "f":
        try:
            marker = float(nodata)
        except (TypeError, ValueError, OverflowError):
            return np.zeros(values.shape, dtype=bool)
        if not math.isfinite(marker):
            return np.zeros(values.shape, dtype=bool)
        # A finite marker outside the source float dtype's range cannot be
        # equal to any stored finite value; it must not wrap to infinity and
        # match genuine infinities, nor underflow to zero and kill legitimate
        # zeros.  A non-zero marker that collapses to zero is rejected.
        cast = dtype.type(marker)
        if not np.isfinite(cast):
            return np.zeros(values.shape, dtype=bool)
        if marker != 0.0 and float(cast) == 0.0:
            return np.zeros(values.shape, dtype=bool)
        return values == cast

    return np.zeros(values.shape, dtype=bool)


def nonfinite_mask(values):
    """Cells holding NaN or +/- infinity (always invalid observations)."""
    values = np.ma.getdata(values)
    if values.dtype.kind == "f":
        return ~np.isfinite(values)
    if values.dtype.kind == "c":
        return ~np.isfinite(values)
    return np.zeros(values.shape, dtype=bool)


def _extract_window(arr, window, nodata=None):
    """Copy a (possibly boundless) window out of ``arr`` without converting.

    Returns ``(data, in_bounds)``.  Both arrays are window-sized; the source
    is only sliced, never converted or copied wholesale.
    """
    window_shape = _window_shape(arr, window)
    (wr_start, _), (wc_start, _) = window
    fill = _fill_value(arr.dtype, nodata)
    out = np.full(window_shape, fill, dtype=arr.dtype)
    in_bounds = np.zeros(window_shape, dtype=bool)

    (olr_start, olr_stop), (olc_start, olc_stop) = _overlap(arr, window)
    nr_start = olr_start - wr_start
    nr_stop = nr_start + (olr_stop - olr_start)
    nc_start = olc_start - wc_start
    nc_stop = nc_start + (olc_stop - olc_start)
    if arr.ndim == 3:
        out[:, nr_start:nr_stop, nc_start:nc_stop] = arr[
            :, olr_start:olr_stop, olc_start:olc_stop
        ]
        in_bounds[:, nr_start:nr_stop, nc_start:nc_stop] = True
    else:
        out[nr_start:nr_stop, nc_start:nc_stop] = arr[
            olr_start:olr_stop, olc_start:olc_stop
        ]
        in_bounds[nr_start:nr_stop, nc_start:nc_stop] = True
    return out, in_bounds


def _window_in_bounds(window, shape):
    """Boolean mask of window slots that lie inside a 2D source shape."""
    height, width = shape[-2:]
    (wr_start, wr_stop), (wc_start, wc_stop) = window
    in_bounds = np.zeros((wr_stop - wr_start, wc_stop - wc_start), dtype=bool)
    olr_start = max(min(wr_start, height), 0)
    olr_stop = max(min(wr_stop, height), 0)
    olc_start = max(min(wc_start, width), 0)
    olc_stop = max(min(wc_stop, width), 0)
    nr_start = olr_start - wr_start
    nc_start = olc_start - wc_start
    if olr_stop > olr_start and olc_stop > olc_start:
        in_bounds[
            nr_start : nr_start + (olr_stop - olr_start),
            nc_start : nc_start + (olc_stop - olc_start),
        ] = True
    return in_bounds


def boundless_array(arr, window, nodata, masked=False):
    dim3 = False
    if len(arr.shape) == 3:
        dim3 = True
    elif len(arr.shape) != 2:
        raise ValueError("Must be a 2D or 3D array")

    # unpack for readability
    (wr_start, wr_stop), (wc_start, wc_stop) = window

    # Calculate overlap
    (olr_start, olr_stop), (olc_start, olc_stop) = _overlap(arr, window)

    # Calc dimensions
    overlap_shape = (olr_stop - olr_start, olc_stop - olc_start)
    if dim3:
        window_shape = (arr.shape[0], wr_stop - wr_start, wc_stop - wc_start)
    else:
        window_shape = (wr_stop - wr_start, wc_stop - wc_start)

    # create an array filled with a representable placeholder; slots outside
    # the source are masked below rather than relying on the marker value.
    out = np.full(window_shape, _fill_value(arr.dtype, nodata), dtype=arr.dtype)
    in_bounds = np.zeros(window_shape, dtype=bool)

    # Fill with data where overlapping
    nr_start = olr_start - wr_start
    nr_stop = nr_start + overlap_shape[0]
    nc_start = olc_start - wc_start
    nc_stop = nc_start + overlap_shape[1]
    if dim3:
        out[:, nr_start:nr_stop, nc_start:nc_stop] = arr[
            :, olr_start:olr_stop, olc_start:olc_stop
        ]
        in_bounds[:, nr_start:nr_stop, nc_start:nc_stop] = True
    else:
        out[nr_start:nr_stop, nc_start:nc_stop] = arr[
            olr_start:olr_stop, olc_start:olc_stop
        ]
        in_bounds[nr_start:nr_stop, nc_start:nc_stop] = True

    if masked:
        invalid = ~in_bounds | nodata_matches(out, nodata) | nonfinite_mask(out)
        out = np.ma.MaskedArray(out, mask=invalid)

    return out


class NodataWarning(UserWarning):
    """Retained for backwards compatibility.

    The reader no longer invents a ``-999`` nodata default: an ndarray has no
    implicit marker, so no warning is emitted for one.
    """


class Raster:
    """Raster abstraction for data access to 2/3D array-like things

    Use as a context manager to ensure dataset gets closed properly::

        >>> with Raster(path) as rast:
        ...

    Parameters
    ----------
    raster: 2/3D array-like data source, required
        Currently supports paths to rasterio-supported rasters and
        numpy arrays with Affine transforms.

    affine: Affine object
        Maps row/col to coordinate reference system
        required if raster is ndarray

    nodata: nodata value, optional
        Overrides the datasource's internal nodata if specified.
        For ndarrays this is the only numeric marker considered; no
        implicit marker is assumed.

    band: integer
        raster band number, optional (default: 1)

    Methods
    -------
    index
    read
    """

    def __init__(self, raster, affine=None, nodata=None, band=1):
        self.array = None
        self.src = None
        # Mask carried by an in-memory MaskedArray; preserved through
        # windowed reads so downstream code can treat it like a dataset mask.
        self.parent_mask = None
        # Full validity of a read result, True where the cell carries no
        # valid observation. Populated by ``read`` and consumed by zonal and
        # point queries; it is the single source of truth for missingness.
        self.invalid_mask = None
        # True where the read slot corresponds to a real source cell. Slots
        # outside the dataset's extent are False; they are invalid, but are
        # never counted as numeric-marker or NaN hits by the diagnostics.
        self.in_bounds_mask = None
        # Optional list that, when set by a caller, receives every rasterio
        # read window ((row_start, row_stop), (col_start, col_stop)). Intended
        # for instrumentation/benchmarking; None means no logging overhead.
        self.read_window_log = None

        if isinstance(raster, np.ndarray):
            if affine is None:
                raise ValueError("Specify affine transform for numpy arrays")
            self.array = raster
            self.affine = affine
            self.shape = raster.shape
            self.nodata = nodata
            if np.ma.is_masked(raster):
                self.parent_mask = np.ma.getmaskarray(raster)
        else:
            self.src = rasterio.open(raster, "r")
            self.affine = guard_transform(self.src.transform)
            self.shape = (self.src.height, self.src.width)
            self.band = band

            if nodata is not None:
                # override with specified nodata (kept unrounded so integer
                # markers larger than 2**53 keep their exact identity)
                self.nodata = nodata
            else:
                self.nodata = self.src.nodata

    def index(self, x, y):
        """Given (x, y) in crs, return the (row, column) on the raster"""
        col, row = (math.floor(a) for a in (~self.affine * (x, y)))
        return row, col

    def read(self, bounds=None, window=None, masked=False, boundless=True):
        """Performs a read against the underlying array source

        Parameters
        ----------
        bounds: bounding box
            in w, s, e, n order, iterable, optional
        window: rasterio-style window, optional
            bounds OR window are required,
            specifying both or neither will raise exception
        masked: boolean
            return a masked numpy array, default: False
        boundless: boolean
            allow window/bounds that extend beyond the dataset's extent, default: True
            partially or completely filled arrays will be returned as appropriate.

        Returns
        -------
        Raster object with update affine and array info

        Notes
        -----
        The returned array keeps the source dtype (no conversion to float).
        ``invalid_mask`` always describes missingness completely, whether or
        not ``masked`` is requested:

        * cells outside the dataset extent;
        * cells equal to the effective numeric marker (the explicit
          ``nodata`` override, or else the file's declared nodata);
        * NaN and +/- infinity;
        * cells hidden by an independent dataset/alpha mask or by the
          caller's own MaskedArray.

        An explicit ``nodata`` replaces the file's numeric marker, including
        any mask the file derived solely from its old nodata value; genuinely
        independent dataset/alpha and MaskedArray masks are kept.
        """
        # Calculate the window
        if bounds and window:
            raise ValueError("Specify either bounds or window")

        if bounds:
            win = bounds_window(bounds, self.affine)
        elif window:
            win = window
        else:
            raise ValueError("Specify either bounds or window")

        if not boundless and beyond_extent(win, self.shape):
            raise ValueError(
                "Window/bounds is outside dataset extent, boundless reads are disabled"
            )

        if self.read_window_log is not None:
            self.read_window_log.append(win)

        c, _, _, f = window_bounds(win, self.affine)  # c ~ west, f ~ north
        a, b, _, d, e, _, _, _, _ = tuple(self.affine)
        new_affine = Affine(a, b, c, d, e, f)

        nodata = self.nodata

        if self.array is not None:
            # It's an ndarray already: slice without converting or copying
            # the full source (only the current window is materialized).
            data, in_bounds = _extract_window(self.array, win, nodata)
            invalid = ~in_bounds
            invalid |= nodata_matches(data, nodata)
            invalid |= nonfinite_mask(data)
            if self.parent_mask is not None:
                parent = np.ma.getdata(self.parent_mask).astype(bool, copy=False)
                parent_window, parent_in = _extract_window(parent, win)
                invalid |= parent_window & parent_in
            result = Raster(data, new_affine, nodata)
            result.parent_mask = None
            result.invalid_mask = invalid
            result.in_bounds_mask = in_bounds
            if masked or self.parent_mask is not None:
                result.array = np.ma.MaskedArray(data, mask=invalid)
            return result

        # It's an open rasterio dataset. Raw values are read without
        # conversion; the validity mask is assembled explicitly so an
        # explicit nodata can replace the file's numeric marker without
        # discarding an independent dataset/alpha mask.
        flags = self.src.mask_flag_enums[self.band - 1]
        independent_mask = (
            MaskFlags.per_dataset in flags or MaskFlags.alpha in flags
        )
        if independent_mask and not masked:
            masked = True
            warnings.warn(
                "Setting masked to True because dataset mask has been detected"
            )

        in_bounds = _window_in_bounds(win, self.shape)
        (row_start, row_stop), (col_start, col_stop) = win
        overlap_rows = (max(row_start, 0), min(row_stop, self.shape[0]))
        overlap_cols = (max(col_start, 0), min(col_stop, self.shape[1]))
        has_overlap = (
            overlap_rows[1] > overlap_rows[0]
            and overlap_cols[1] > overlap_cols[0]
        )

        fill = _fill_value(self.src.dtypes[self.band - 1], nodata)
        data = np.full(
            (row_stop - row_start, col_stop - col_start),
            fill,
            dtype=self.src.dtypes[self.band - 1],
        )
        invalid = ~in_bounds
        if has_overlap:
            # Read only the intersecting part and pad ourselves. GDAL's own
            # boundless path can round real integer cell values when the
            # declared nodata is not exactly representable, so we never use
            # it: every stored value is read at full fidelity.
            inner_window = (overlap_rows, overlap_cols)
            inner = self.src.read(
                self.band,
                window=inner_window,
                boundless=False,
                masked=False,
            )
            nr = overlap_rows[0] - row_start
            nc = overlap_cols[0] - col_start
            data[nr : nr + inner.shape[0], nc : nc + inner.shape[1]] = inner
            if independent_mask:
                inner_mask = self.src.read_masks(
                    self.band, window=inner_window, boundless=False
                )
                dataset_mask = np.zeros(data.shape, dtype=inner_mask.dtype)
                dataset_mask[
                    nr : nr + inner_mask.shape[0], nc : nc + inner_mask.shape[1]
                ] = inner_mask
                invalid = invalid | (dataset_mask == 0)

        invalid = invalid | nodata_matches(data, nodata) | nonfinite_mask(data)

        result = Raster(data, new_affine, nodata)
        result.parent_mask = None
        result.invalid_mask = invalid
        result.in_bounds_mask = in_bounds
        if masked:
            result.array = np.ma.MaskedArray(data, mask=invalid)
        return result

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.src is not None:
            # close the rasterio reader
            self.src.close()


def read_validity(fsrc):
    """Return ``(invalid_mask, raw_data)`` for a ``Raster.read`` result.

    This is the single consumption point for both zonal statistics and point
    queries.  A ``Raster`` produced by ``read`` carries a complete
    ``invalid_mask``; directly constructed rasters fall back to composing
    their own MaskedArray mask, numeric marker and non-finite values.
    """
    array = fsrc.array
    data = np.ma.getdata(array)
    invalid = getattr(fsrc, "invalid_mask", None)
    if invalid is not None:
        return invalid, data
    invalid = np.zeros(data.shape, dtype=bool)
    if np.ma.is_masked(array):
        invalid = invalid | np.ma.getmaskarray(array)
    invalid = invalid | nodata_matches(data, fsrc.nodata)
    invalid = invalid | nonfinite_mask(data)
    return invalid, data
