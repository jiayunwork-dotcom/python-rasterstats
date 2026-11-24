import numpy as np
from shapely.geometry import shape
from shapely.ops import transform

from rasterstats.io import DEFAULT_CHUNK_SIZE, Raster, read_features, read_validity


def _window_invalid(rast):
    """Invalidity mask of a read result, including any masked-array mask."""
    invalid, _ = read_validity(rast)
    return invalid


def point_window_unitxy(x, y, affine):
    """Given an x, y and a geotransform
    Returns
        - rasterio window representing 2x2 window whose center points encompass point
        - the cartesian x, y coordinates of the point on the unit square
          defined by the array center points.

    ((row1, row2), (col1, col2)), (unitx, unity)
    """
    fcol, frow = ~affine * (x, y)
    r, c = round(frow), round(fcol)

    # The new source window for our 2x2 array
    new_win = ((r - 1, r + 1), (c - 1, c + 1))

    # the new x, y coords on the unit square
    unitxy = (0.5 - (c - fcol), 0.5 + (r - frow))

    return new_win, unitxy


def _cell_invalid(arr):
    """Boolean mask of unusable cells in a 2x2 support.

    A masked array contributes its own mask; NaN and +/- infinity are always
    invalid; a plain numeric array has no other implicit missingness.
    """
    if np.ma.isMaskedArray(arr):
        invalid = np.ma.getmaskarray(arr)
    else:
        invalid = np.zeros(np.shape(arr), dtype=bool)
    data = np.ma.getdata(arr)
    if data.dtype.kind in "fc":
        invalid = invalid | ~np.isfinite(data)
    return np.asarray(invalid, dtype=bool)


def _nearest_cell(arr, x, y):
    """Original nearest-neighbour fallback within the 2x2 support."""
    value = np.ma.getdata(arr)[round(1 - y), round(x)]
    if _cell_invalid(arr)[round(1 - y), round(x)]:
        return None
    if np.ma.isMaskedArray(value):
        return value.item()
    if hasattr(value, "item"):
        return value.item()
    return value


def bilinear(arr, x, y):
    """Given a 2x2 array, an x, and y, treat center points as a unit square
    return the value for the fractional row/col
    using bilinear interpolation between the cells

        +---+---+
        | A | B |      +----+
        +---+---+  =>  |    |
        | C | D |      +----+
        +---+---+

        e.g.: Center of A is at (0, 1) on unit square, D is at (1, 0), etc

    Interpolation uses all four cells only when all four are observations.
    Otherwise the original nearest-neighbour fallback applies: the nearest
    cell's value when it is valid, else ``None``.  A zero-weight NaN or a
    padded cell therefore never contributes to an otherwise valid centre
    point.
    """
    # for now, only 2x2 arrays
    assert arr.shape == (2, 2)
    data = np.ma.getdata(arr)

    # not valid if not on unit square
    assert 0.0 <= x <= 1.0
    assert 0.0 <= y <= 1.0

    if _cell_invalid(arr).any():
        # at least one nodata/masked/non-finite support cell
        # fall back to nearest neighbor
        return _nearest_cell(arr, x, y)

    ulv, urv, llv, lrv = data[0:2, 0:2].flatten().tolist()

    # bilinear interp on unit square
    return (
        (llv * (1 - x) * (1 - y))
        + (lrv * x * (1 - y))
        + (ulv * (1 - x) * y)
        + (urv * x * y)
    )


def geom_xys(geom):
    """Given a shapely geometry,
    generate a flattened series of 2D points as x,y tuples
    """
    if geom.has_z:
        # convert to 2D
        geom = transform(lambda x, y, z=None: (x, y), geom)

    if hasattr(geom, "geoms"):
        geoms = geom.geoms
    else:
        geoms = [geom]

    for g in geoms:
        if hasattr(g, "exterior"):
            yield from geom_xys(g.exterior)
            for interior in g.interiors:
                yield from geom_xys(interior)
        else:
            yield from g.coords


def point_query(*args, **kwargs):
    """The primary point query entry point.

    All arguments are passed directly to ``gen_point_query``.
    See its docstring for details.

    The only difference is that ``point_query`` will
    return a list rather than a generator."""
    return list(gen_point_query(*args, **kwargs))


def gen_point_query(
    vectors,
    raster,
    band=1,
    layer=0,
    nodata=None,
    affine=None,
    interpolate="bilinear",
    property_name="value",
    geojson_out=False,
    boundless=True,
    engine=None,
    chunk_size=DEFAULT_CHUNK_SIZE,
):
    """
    Given a set of vector features and a raster,
    generate raster values at each vertex of the geometry

    For features with point geometry,
    the values will be a 1D with the index refering to the feature

    For features with other geometry types,
    it effectively creates a 2D list, such that
    the first index is the feature, the second is the vertex within the geometry

    Parameters
    ----------
    vectors: path to an vector source or geo-like python objects

    raster: ndarray or path to a GDAL raster source
        If ndarray is passed, the `transform` kwarg is required.

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

    interpolate: string
        'bilinear' or 'nearest' interpolation

    property_name: string
        name of property key if geojson_out

    geojson_out: boolean
        generate GeoJSON-like features (default: False)
        original feature geometry and properties will be retained
        point query values appended as additional properties.

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

    Returns
    -------
    generator of arrays (if ``geojson_out`` is False)
    generator of geojson features (if ``geojson_out`` is True)
    """
    if interpolate not in ["nearest", "bilinear"]:
        raise ValueError("interpolate must be nearest or bilinear")

    features_iter = read_features(vectors, layer, engine=engine, chunk_size=chunk_size)

    with Raster(raster, nodata=nodata, affine=affine, band=band) as rast:
        for feat in features_iter:
            geom = shape(feat["geometry"])
            vals = []
            for x, y in geom_xys(geom):
                if interpolate == "nearest":
                    r, c = rast.index(x, y)
                    window = ((int(r), int(r + 1)), (int(c), int(c + 1)))
                    result = rast.read(
                        window=window, masked=True, boundless=boundless
                    )
                    if _window_invalid(result)[0, 0]:
                        vals.append(None)
                    else:
                        vals.append(np.ma.getdata(result.array)[0, 0].item())

                elif interpolate == "bilinear":
                    window, unitxy = point_window_unitxy(x, y, rast.affine)
                    result = rast.read(
                        window=window, masked=True, boundless=boundless
                    )
                    vals.append(bilinear(result.array, *unitxy))

            if len(vals) == 1:
                vals = vals[0]  # flatten single-element lists

            if geojson_out:
                if "properties" not in feat:
                    feat["properties"] = {}
                feat["properties"][property_name] = vals
                yield feat
            else:
                yield vals
