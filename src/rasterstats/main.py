import contextlib
import inspect
import sys
import warnings

import numpy as np
import rasterio
from affine import Affine
from shapely import Geometry
from shapely.geometry import shape

from rasterstats.fractional import (
    FRACTIONAL_GDAL_CACHE_BYTES,
    fractional_zonal_stats,
    validate_affine,
    validate_fractional_options,
    validate_max_cells,
    validate_polygonal,
)
from rasterstats.io import (
    DEFAULT_CHUNK_SIZE,
    Raster,
    nodata_matches,
    read_features,
    read_validity,
)
from rasterstats.utils import (
    boxify_points,
    check_stats,
    get_percentile,
    key_assoc_val,
    rasterize_geom,
    remap_categories,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def raster_stats(*args, **kwargs):
    """Deprecated. Use zonal_stats instead."""
    warnings.warn(
        "'raster_stats' is an alias to 'zonal_stats' and will disappear in 1.0",
        DeprecationWarning,
    )
    return zonal_stats(*args, **kwargs)


def zonal_stats(*args, **kwargs):
    """The primary zonal statistics entry point.

    All arguments are passed directly to ``gen_zonal_stats``.
    See its docstring for details.

    The only difference is that ``zonal_stats`` will
    return a list rather than a generator."""
    progress = kwargs.get("progress")
    if progress:
        if tqdm is None:
            raise ValueError(
                "You specified progress=True, but tqdm is not installed in "
                "the environment. "
                "You can do pip install rasterstats[progress] to install tqdm!"
            )
        stats = gen_zonal_stats(*args, **kwargs)
        total = len(args[0])
        return [stat for stat in tqdm(stats, total=total)]
    else:
        return list(gen_zonal_stats(*args, **kwargs))


def gen_zonal_stats(
    vectors,
    raster,
    layer=0,
    band=1,
    nodata=None,
    affine=None,
    stats=None,
    all_touched=False,
    categorical=False,
    category_map=None,
    add_stats=None,
    zone_func=None,
    raster_out=False,
    prefix=None,
    geojson_out=False,
    boundless=True,
    engine=None,
    chunk_size=DEFAULT_CHUNK_SIZE,
    coverage=None,
    value_kind=None,
    max_cells=65536,
    **kwargs,
):
    """Zonal statistics of raster values aggregated to vector geometries.

    Parameters
    ----------
    vectors : str or PathLike
        Path to an vector source or geo-like python objects.

    raster: array_like, str or PathLike
        NumPy array or path to a GDAL raster source.
        If ndarray is passed, the ``affine`` kwarg is required.

    layer: int or string, optional
        If `vectors` is a path to a fiona source,
        specify the vector layer to use either by name or number.
        defaults to 0

    band: int, optional
        If `raster` is a GDAL source, the band number to use (counting from 1).
        defaults to 1.

    nodata: float, optional
        If `raster` is a GDAL source, this value overrides any NODATA value
        specified in the file's metadata.
        If `None`, the file's metadata's NODATA value (if any) will be used.
        defaults to `None`.

    affine: Affine instance
        required only for ndarrays, otherwise it is read from src

    stats:  list of str, or space-delimited str, optional
        Which statistics to calculate for each zone.
        All possible choices are listed in ``utils.VALID_STATS``.
        defaults to ``DEFAULT_STATS``, a subset of these.

    all_touched: bool, optional
        Whether to include every raster cell touched by a geometry, or only
        those having a center point within the polygon.
        defaults to `False`

    categorical: bool, optional

    category_map: dict
        A dictionary mapping raster values to human-readable categorical names.
        Only applies when categorical is True

    add_stats: dict
        with names and functions of additional stats to compute, optional

    zone_func: callable
        function to apply to zone ndarray prior to computing stats

    raster_out: boolean
        Include the masked numpy array for each feature?, optional

        Each feature dictionary will have the following additional keys:
        mini_raster_array: The clipped and masked numpy array
        mini_raster_affine: Affine transformation
        mini_raster_nodata: NoData Value

    prefix: string
        add a prefix to the keys (default: None)

    geojson_out: boolean
        Return list of GeoJSON-like features (default: False)
        Original feature geometry and properties will be retained
        with zonal stats appended as additional properties.
        Use with `prefix` to ensure unique and meaningful property names.

    boundless: boolean
        Allow features that extend beyond the raster dataset's extent, default: True
        Cells outside dataset extents are treated as nodata.

    engine : {"pyogrio", "fiona"} or None, optional
        Backend to use when reading file-based vector sources.
        ``None`` selects the default engine (``"pyogrio"``).
        Pass ``"fiona"`` to opt in to the fiona backend
        (requires ``pip install rasterstats[fiona]``).

    chunk_size : int, optional
        Number of features to read per batch when using the pyogrio engine.
        Reduce this value to lower peak memory usage for datasets with
        large, vertex-dense geometries. Default: ``DEFAULT_CHUNK_SIZE``.
        Has no effect when ``engine='fiona'`` or when ``vectors`` is not a
        file path.

    coverage: {None, "fractional"}, optional
        Cell inclusion strategy. ``None`` (the default) keeps the historical
        behavior: a cell is included wholesale via center-point or
        ``all_touched`` rasterization. ``"fractional"`` switches to exact
        area-weighted analysis: every cell is weighted by the fraction
        ``a / A`` of its area ``A`` that overlaps the zone's area ``a``.
        Holes are removed and overlapping MultiPolygon components count once.
        Only planar (projected) Polygon/MultiPolygon geometries and finite,
        invertible affines are supported; area units are the square of the
        coordinate units.

    value_kind: {"total", "density"}, optional
        Required when ``coverage='fractional'`` and forbidden otherwise.
        ``"total"`` means each cell stores a whole-cell quantity spread
        uniformly over the cell (e.g. people), so the zone total accumulates
        ``(a / A) * value``. ``"density"`` means each cell stores a constant
        per-unit-area intensity (e.g. people per square unit), so the zone
        total accumulates ``a * value``. In both cases ``mean`` is the
        area-weighted mean intensity, ``sum / valid_area``.

    max_cells: int, optional
        Only used with ``coverage='fractional'``. Upper bound on the number
        of raster cell slots read in any single window and held in any single
        2D working array. Large zones are tiled; results do not depend on
        this value. Must be a positive integer and cannot be a bool.
        Default 65536.

    Returns
    -------
    generator of dicts (if geojson_out is False)
        Each item corresponds to a single vector feature and
        contains keys for each of the specified stats.

    generator of geojson features (if geojson_out is True)
        GeoJSON-like Feature as python dict
    """
    # Fractional area-weighted mode has its own validation rules and code
    # path; ordinary behavior is untouched when coverage is left as None.
    fractional = coverage == "fractional"
    if fractional:
        stats = validate_fractional_options(
            coverage,
            value_kind,
            stats,
            categorical,
            add_stats,
            zone_func,
            raster_out,
            all_touched,
        )
        run_count = False
        max_cells = validate_max_cells(max_cells)
    else:
        stats, run_count = check_stats(stats, categorical)
        validate_fractional_options(
            coverage,
            value_kind,
            stats,
            categorical,
            add_stats,
            zone_func,
            raster_out,
            all_touched,
        )

    # Handle 1.0 deprecations
    transform = kwargs.get("transform")
    if transform:
        warnings.warn(
            "GDAL-style transforms will disappear in 1.0. "
            "Use affine=Affine.from_gdal(*transform) instead",
            DeprecationWarning,
        )
        if not affine:
            affine = Affine.from_gdal(*transform)

    cp = kwargs.get("copy_properties")
    if cp:
        warnings.warn(
            "Use `geojson_out` to preserve feature properties", DeprecationWarning
        )

    band_num = kwargs.get("band_num")
    if band_num:
        warnings.warn("Use `band` to specify band number", DeprecationWarning)
        band = band_num

    # Fractional analysis tiles many small windows; cap GDAL's block cache
    # before the dataset is opened (the cap is only honored at open time) so
    # peak RSS stays independent of the number of tiles.
    env_context = (
        rasterio.Env(GDAL_CACHEMAX=FRACTIONAL_GDAL_CACHE_BYTES)
        if fractional
        else contextlib.nullcontext()
    )
    with env_context, Raster(raster, affine, nodata, band) as rast:
        if fractional:
            validate_affine(rast.affine)
        features_iter = read_features(vectors, layer, engine=engine, chunk_size=chunk_size)
        for feat_index, feat in enumerate(features_iter):
            geom = feat["geometry"]
            if not isinstance(geom, Geometry):
                geom = shape(geom) if geom is not None else None

            if fractional:
                validate_polygonal(geom, feat_index)
                feature_stats = fractional_zonal_stats(
                    geom,
                    rast,
                    stats=stats,
                    value_kind=value_kind,
                    max_cells=max_cells,
                )
            else:
                feature_stats = _ordinary_feature_stats(
                    geom,
                    rast,
                    feat,
                    stats=stats,
                    run_count=run_count,
                    all_touched=all_touched,
                    categorical=categorical,
                    category_map=category_map,
                    add_stats=add_stats,
                    zone_func=zone_func,
                    raster_out=raster_out,
                    boundless=boundless,
                )

            if prefix is not None:
                prefixed_feature_stats = {}
                for key, val in feature_stats.items():
                    newkey = f"{prefix}{key}"
                    prefixed_feature_stats[newkey] = val
                feature_stats = prefixed_feature_stats

            if geojson_out:
                for key, val in feature_stats.items():
                    if "properties" not in feat:
                        feat["properties"] = {}
                    feat["properties"][key] = val
                yield feat
            else:
                yield feature_stats


def _ordinary_feature_stats(
    geom,
    rast,
    feat,
    stats,
    run_count,
    all_touched,
    categorical,
    category_map,
    add_stats,
    zone_func,
    raster_out,
    boundless,
):
    """Classical center-point / all-touched zonal statistics for one feature."""
    if "Point" in geom.geom_type:
        geom = boxify_points(geom, rast)

    geom_bounds = tuple(geom.bounds)

    fsrc = rast.read(bounds=geom_bounds, boundless=boundless)

    # rasterized geometry
    rv_array = rasterize_geom(geom, like=fsrc, all_touched=all_touched)

    # Complete validity from the shared read layer: dataset/MaskedArray
    # masks, the effective numeric marker, NaN/inf and off-extent slots.
    invalid, data = read_validity(fsrc)

    # Mask the source data array
    # mask everything that is not a valid value or not within our geom
    masked = np.ma.MaskedArray(data, mask=(invalid | ~rv_array))

    # If we're on 64 bit platform and the array is an integer type
    # make sure we cast to 64 bit to avoid overflow for certain numpy ops.
    # Unsigned magnitudes above int64 (e.g. uint64 > 2**63) must accumulate
    # as uint64 so they are not reinterpreted as negative values.
    if sys.maxsize > 2**32 and issubclass(masked.dtype.type, np.integer):
        if masked.dtype.kind == "u" and masked.dtype.itemsize >= 8:
            accum_dtype = "uint64"
        else:
            accum_dtype = "int64"
    else:
        accum_dtype = None  # numpy default

    # execute zone_func on masked zone ndarray
    if zone_func is not None:
        if not callable(zone_func):
            raise TypeError(
                "zone_func must be a callable function "
                "which accepts a single `zone_array` arg."
            )
        value = zone_func(masked)

        # check if zone_func has return statement
        if value is not None:
            masked = value

    if masked.compressed().size == 0:
        # nothing here, fill with None and move on
        feature_stats = {stat: None for stat in stats}
        if "count" in stats:  # special case, zero makes sense here
            feature_stats["count"] = 0
    else:
        if run_count:
            keys, counts = np.unique(masked.compressed(), return_counts=True)
            try:
                pixel_count = dict(
                    zip([k.item() for k in keys], [c.item() for c in counts])
                )
            except AttributeError:
                pixel_count = dict(
                    zip(
                        [np.asscalar(k) for k in keys],
                        [np.asscalar(c) for c in counts],
                    )
                )

        if categorical:
            feature_stats = dict(pixel_count)
            if category_map:
                feature_stats = remap_categories(category_map, feature_stats)
        else:
            feature_stats = {}

        if "min" in stats:
            feature_stats["min"] = float(masked.min())
        if "max" in stats:
            feature_stats["max"] = float(masked.max())
        if "mean" in stats:
            feature_stats["mean"] = float(masked.mean(dtype=accum_dtype))
        if "count" in stats:
            feature_stats["count"] = int(masked.count())
        # optional
        if "sum" in stats:
            feature_stats["sum"] = float(masked.sum(dtype=accum_dtype))
        if "std" in stats:
            feature_stats["std"] = float(masked.std())
        if "median" in stats:
            feature_stats["median"] = float(np.median(masked.compressed()))
        if "majority" in stats:
            feature_stats["majority"] = float(key_assoc_val(pixel_count, max))
        if "minority" in stats:
            feature_stats["minority"] = float(key_assoc_val(pixel_count, min))
        if "unique" in stats:
            feature_stats["unique"] = len(list(pixel_count.keys()))
        if "range" in stats:
            try:
                rmin = feature_stats["min"]
            except KeyError:
                rmin = float(masked.min())
            try:
                rmax = feature_stats["max"]
            except KeyError:
                rmax = float(masked.max())
            feature_stats["range"] = rmax - rmin

        for pctile in [s for s in stats if s.startswith("percentile_")]:
            q = get_percentile(pctile)
            pctarr = masked.compressed()
            feature_stats[pctile] = float(np.percentile(pctarr, q))

    if "nodata" in stats or "nan" in stats:
        # Diagnostics count cells inside the zone, but are independent of
        # dataset/MaskedArray masks (those may overlap the counts).  Only
        # real in-extent source cells qualify: the numeric marker for
        # ``nodata`` and stored NaN (not infinity) for ``nan``.
        in_bounds = getattr(fsrc, "in_bounds_mask", None)
        if in_bounds is None:
            in_bounds = np.ones(fsrc.array.shape, dtype=bool)
        region = rv_array & in_bounds

        if "nodata" in stats:
            hits = nodata_matches(data, fsrc.nodata) & region
            feature_stats["nodata"] = float(np.count_nonzero(hits))
        if "nan" in stats:
            if np.issubdtype(data.dtype, np.floating) or np.issubdtype(
                data.dtype, np.complexfloating
            ):
                nans = np.isnan(data) & region
                feature_stats["nan"] = float(np.count_nonzero(nans))
            else:
                feature_stats["nan"] = 0.0

    if add_stats is not None:
        for stat_name, stat_func in add_stats.items():
            n_params = len(inspect.signature(stat_func).parameters.keys())
            if n_params == 3:
                feature_stats[stat_name] = stat_func(
                    masked, feat["properties"], rv_array
                )
            # backwards compatible with two-argument function
            elif n_params == 2:
                feature_stats[stat_name] = stat_func(masked, feat["properties"])
            # backwards compatible with single-argument function
            else:
                feature_stats[stat_name] = stat_func(masked)

    if raster_out:
        feature_stats["mini_raster_array"] = masked
        feature_stats["mini_raster_affine"] = fsrc.affine
        feature_stats["mini_raster_nodata"] = fsrc.nodata

    return feature_stats
