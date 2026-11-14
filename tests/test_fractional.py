# -*- coding: utf-8 -*-
"""Tests for exact area-weighted fractional coverage statistics.

The expected values below are derived independently (hand geometry /
reference geometry built directly from the Affine transform) rather than
from the implementation under test.
"""
import copy

import numpy as np
import pytest
import rasterio
from affine import Affine
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely import set_coordinates

from rasterstats import fractional_stats, gen_fractional_stats
from rasterstats.fractional import (
    FRACTIONAL_STAT_KEYS,
    ArrayBlocks,
    FileBlocks,
    FractionalStatsGenerator,
    cell_polygons,
    fractional_feature,
    grid_bounds,
    validate_polygon,
)

DATA = __import__("pathlib").Path(__file__).parent / "data"

# A 4x4 grid, north-up unit cells: cell (r, c) covers x in [c, c+1],
# y in [3-r, 4-r].  Values are 0..15 in row-major order.
AFFINE = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
RASTER = np.arange(16, dtype="float64").reshape(4, 4)
INT_RASTER = np.arange(16, dtype="int32").reshape(4, 4)
EMPTY_RESULT = {
    "coverage": 0.0,
    "count": 0.0,
    "nodata": 0.0,
    "sum": 0.0,
    "mean": None,
    "min": None,
    "max": None,
}


def approx_equal(actual, expected, rtol=1e-9, atol=1e-10):
    """Compare a seven-key result dict, tolerating floats only."""
    assert list(actual.keys()) == list(FRACTIONAL_STAT_KEYS)
    for key, value in expected.items():
        if value is None:
            assert actual[key] is None, key
        elif isinstance(value, float):
            assert np.isclose(actual[key], value, rtol=rtol, atol=atol), (
                key,
                actual[key],
                value,
            )
        else:
            assert actual[key] == value, (key, actual[key], value)


# ---------------------------------------------------------------------------
# Hand-computed expectations
# ---------------------------------------------------------------------------


def test_partial_cells_hand_values():
    # 4 full cells (values 0,1,4,5) + two half cells (values 2 and 6).
    poly = Polygon([(0, 2), (2.5, 2), (2.5, 4), (0, 4)])
    result = fractional_stats([poly], RASTER, affine=AFFINE)[0]
    approx_equal(
        result,
        {
            "coverage": 5.0,
            "count": 5.0,
            "nodata": 0.0,
            "sum": 0 + 1 + 4 + 5 + 0.5 * 2 + 0.5 * 6,  # 14.0
            "mean": 14.0 / 5.0,
            "min": 0.0,
            "max": 6.0,
        },
    )


def test_small_region_smaller_than_cell():
    # 0.4 x 0.4 square entirely inside cell (r=0, c=0), value 0.
    poly = Polygon([(0.2, 3.2), (0.6, 3.2), (0.6, 3.6), (0.2, 3.6)])
    result = fractional_stats([poly], RASTER, affine=AFFINE)[0]
    approx_equal(
        result,
        {
            "coverage": 0.16,
            "count": 0.16,
            "nodata": 0.0,
            "sum": 0.16 * 0.0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
        },
    )


def test_thin_strip_spanning_cells():
    # 0.1-wide vertical strip x in [1, 1.1], full height: 4 cells at 10%.
    poly = Polygon([(1.0, 0.0), (1.1, 0.0), (1.1, 4.0), (1.0, 4.0)])
    result = fractional_stats([poly], RASTER, affine=AFFINE)[0]
    expected_sum = 0.1 * RASTER[:, 1].sum()
    approx_equal(
        result,
        {
            "coverage": 0.4,
            "count": 0.4,
            "nodata": 0.0,
            "sum": expected_sum,
            "mean": RASTER[:, 1].mean(),
            "min": float(RASTER[:, 1].min()),
            "max": float(RASTER[:, 1].max()),
        },
    )


def test_interior_ring_is_subtracted():
    poly = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]],
    )
    result = fractional_stats([poly], RASTER, affine=AFFINE)[0]
    hole_sum = float(RASTER[1:3, 1:3].sum())  # cells 5,6,9,10
    approx_equal(
        result,
        {
            "coverage": 12.0,
            "count": 12.0,
            "nodata": 0.0,
            "sum": float(RASTER.sum()) - hole_sum,
            "mean": (float(RASTER.sum()) - hole_sum) / 12.0,
            "min": 0.0,
            "max": 15.0,
        },
    )


def test_multipolygon_treated_as_one_region():
    region = MultiPolygon(
        [
            Polygon([(0, 3), (1, 3), (1, 4), (0, 4)]),  # cell (0,0) = 0
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),  # cell (3,2) = 14
        ]
    )
    result = fractional_stats([region], RASTER, affine=AFFINE)[0]
    approx_equal(
        result,
        {
            "coverage": 2.0,
            "count": 2.0,
            "nodata": 0.0,
            "sum": 14.0,
            "mean": 7.0,
            "min": 0.0,
            "max": 14.0,
        },
    )


def test_edge_touch_only_has_no_area():
    # A zero-area / empty polygon is legal and contributes nothing.
    assert fractional_stats([Polygon()], RASTER, affine=AFFINE)[0] == EMPTY_RESULT

    # A region whose boundary lies exactly on grid lines must not gain
    # weight from the cells it merely touches: the covered area is exact.
    half = Polygon([(2.0, 2.0), (3.0, 2.0), (3.0, 4.0), (2.0, 4.0)])
    result = fractional_stats([half], RASTER, affine=AFFINE)[0]
    assert result["coverage"] == 2.0
    assert result["count"] == 2.0

    # A sliver crossing one grid line touches two cell columns but only the
    # 0.1-wide column has positive area; boundary contact alone adds nothing.
    sliver = Polygon([(2.0, 2.0), (2.1, 2.0), (2.1, 4.0), (2.0, 4.0)])
    result = fractional_stats([sliver], RASTER, affine=AFFINE)[0]
    assert np.isclose(result["coverage"], 0.2)
    assert np.isclose(result["count"], 0.2)


def test_region_extending_beyond_raster():
    # Only the in-raster quarter (one full cell, value 3) counts; the rest
    # is neither coverage nor nodata.
    poly = Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])
    result = fractional_stats([poly], RASTER, affine=AFFINE)[0]
    approx_equal(
        result,
        {
            "coverage": 1.0,
            "count": 1.0,
            "nodata": 0.0,
            "sum": 3.0,
            "mean": 3.0,
            "min": 3.0,
            "max": 3.0,
        },
    )


def test_empty_and_disjoint_geometries():
    assert fractional_stats([Polygon()], RASTER, affine=AFFINE)[0] == EMPTY_RESULT
    assert (
        fractional_stats([MultiPolygon([])], RASTER, affine=AFFINE)[0]
        == EMPTY_RESULT
    )
    disjoint = Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])
    assert fractional_stats([disjoint], RASTER, affine=AFFINE)[0] == EMPTY_RESULT


def test_all_invalid_coverage():
    raster = RASTER.copy()
    raster[0, :] = np.nan  # whole top row invalid
    poly = Polygon([(0, 3), (4, 3), (4, 4), (0, 4)])
    result = fractional_stats([poly], raster, affine=AFFINE)[0]
    approx_equal(
        result,
        {
            "coverage": 4.0,
            "count": 0.0,
            "nodata": 4.0,
            "sum": 0.0,
            "mean": None,
            "min": None,
            "max": None,
        },
    )


def test_masked_array_and_sentinel():
    masked = np.ma.MaskedArray(RASTER.copy(), mask=np.zeros(RASTER.shape, bool))
    masked.mask[0, 0] = True
    poly = Polygon([(0, 2), (2, 2), (2, 4), (0, 4)])  # 4 full cells
    # Mask alone invalidates one cell.
    result = fractional_stats([poly], masked, affine=AFFINE)[0]
    assert result["coverage"] == 4.0
    assert result["nodata"] == 1.0
    assert result["count"] == 3.0

    # Add a numeric sentinel on a second cell.
    masked[0, 1] = -999.0
    result = fractional_stats([poly], masked, affine=AFFINE, nodata=-999)[0]
    assert result["nodata"] == 2.0
    assert result["count"] == 2.0

    # Without a sentinel the plain -999 value remains valid data.
    result = fractional_stats([poly], masked, affine=AFFINE)[0]
    assert result["nodata"] == 1.0
    assert result["min"] == -999.0


def test_legal_minus_999_without_metadata():
    raster = RASTER.copy()
    raster[0, 0] = -999.0
    result = fractional_stats(
        [Polygon([(0, 3), (1, 3), (1, 4), (0, 4)])], raster, affine=AFFINE
    )[0]
    assert result["count"] == 1.0
    assert result["nodata"] == 0.0
    assert result["min"] == -999.0 and result["max"] == -999.0


def test_nan_and_infinities_are_invalid():
    raster = RASTER.copy()
    raster[0, 0] = np.nan
    raster[0, 1] = np.inf
    raster[1, 0] = -np.inf
    poly = Polygon([(0, 2), (2, 2), (2, 4), (0, 4)])
    result = fractional_stats([poly], raster, affine=AFFINE)[0]
    # Three invalid cells, one valid (value 5).
    approx_equal(
        result,
        {
            "coverage": 4.0,
            "count": 1.0,
            "nodata": 3.0,
            "sum": 5.0,
            "mean": 5.0,
            "min": 5.0,
            "max": 5.0,
        },
    )


def test_integer_raster_returns_floats():
    poly = Polygon([(0, 3), (1, 3), (1, 4), (0, 4)])
    result = fractional_stats([poly], INT_RASTER, affine=AFFINE)[0]
    assert all(
        isinstance(result[key], float)
        for key in ("coverage", "count", "nodata", "sum", "min", "max")
    )
    assert result["sum"] == 0.0


def test_unsigned_integer_and_nomask_arrays():
    unsigned = np.arange(16, dtype="uint8").reshape(4, 4)
    poly = Polygon([(0, 2), (2, 2), (2, 4), (0, 4)])
    result = fractional_stats([poly], unsigned, affine=AFFINE)[0]
    assert result["count"] == 4.0 and result["sum"] == 10.0

    no_mask = np.ma.MaskedArray(RASTER.copy())
    assert no_mask.mask is np.ma.nomask
    result = fractional_stats([poly], no_mask, affine=AFFINE)[0]
    assert result["count"] == 4.0 and result["nodata"] == 0.0


# ---------------------------------------------------------------------------
# Transform handling: rotation, shear, reversed axes, invariance
# ---------------------------------------------------------------------------


def _transform_polygon(polygon, matrix):
    return Polygon([matrix @ point for point in polygon.exterior.coords])


def test_rotation_shear_and_reverse_grids_match_identity():
    polygon = Polygon([(0.3, 2.2), (2.5, 2.2), (2.5, 3.7), (0.3, 3.7)])
    expected = fractional_stats([polygon], RASTER, affine=AFFINE)[0]

    theta = 0.37
    rotation = Affine(
        np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta), 0
    )
    shear = Affine(1.0, 0.4, 0.0, 0.0, 1.0, 0.0)
    reverse = Affine(-1.0, 0.0, 4.0, 0.0, 1.0, 0.0)

    for matrix in (rotation, shear, reverse):
        moved = _transform_polygon(polygon, matrix)
        result = fractional_stats(
            [moved], RASTER, affine=(matrix @ AFFINE)
        )[0]
        for key in ("coverage", "count", "nodata", "sum", "mean"):
            assert np.isclose(result[key], expected[key], rtol=1e-9, atol=1e-10), (
                matrix,
                key,
            )
        assert result["min"] == expected["min"]
        assert result["max"] == expected["max"]


def test_max_cells_does_not_change_results():
    polygon = Polygon([(0.3, 0.4), (3.6, 0.4), (3.6, 3.7), (0.3, 3.7)])
    reference = fractional_stats(
        [polygon], RASTER, affine=AFFINE, max_cells=65536
    )[0]
    for max_cells in (1, 2, 3, 5, 7, 16, 100):
        result = fractional_stats(
            [polygon], RASTER, affine=AFFINE, max_cells=max_cells
        )[0]
        for key in ("coverage", "count", "nodata", "sum", "mean"):
            assert np.isclose(
                result[key], reference[key], rtol=1e-9, atol=1e-10
            ), (max_cells, key)
        assert result["min"] == reference["min"]
        assert result["max"] == reference["max"]


def test_negative_pixel_size_supported():
    # Both axes flipped; the same physical region gives the same statistics.
    flipped = Affine(-1.0, 0.0, 4.0, 0.0, 1.0, 0.0)
    polygon = Polygon([(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)])
    identity_polygon = Polygon(
        [(1.5, 1.5), (3.5, 1.5), (3.5, 3.5), (1.5, 3.5)]
    )
    result = fractional_stats([polygon], RASTER, affine=flipped)[0]
    expected = fractional_stats([identity_polygon], RASTER, affine=AFFINE)[0]
    for key in ("coverage", "count", "sum"):
        assert np.isclose(result[key], expected[key])


# ---------------------------------------------------------------------------
# Additivity over a non-overlapping partition
# ---------------------------------------------------------------------------


def test_partition_additivity():
    region = Polygon([(0.25, 0.5), (3.75, 0.5), (3.75, 3.6), (0.25, 3.6)])
    # Split along an arbitrary line that cuts through cells.
    left = Polygon([(0.25, 0.5), (1.8, 0.5), (1.8, 3.6), (0.25, 3.6)])
    right = Polygon([(1.8, 0.5), (3.75, 0.5), (3.75, 3.6), (1.8, 3.6)])
    # Sanity: the partition is non-overlapping and exhaustive.
    assert abs(left.union(right).symmetric_difference(region).area) < 1e-12

    whole = fractional_stats([region], RASTER, affine=AFFINE)[0]
    parts = fractional_stats([left, right], RASTER, affine=AFFINE)

    for key in ("coverage", "count", "nodata", "sum"):
        assert np.isclose(
            parts[0][key] + parts[1][key], whole[key], rtol=1e-9, atol=1e-10
        ), key
    merged_mean = whole["sum"] / whole["count"]
    assert np.isclose(merged_mean, whole["mean"])


def test_partition_additivity_with_nodata_and_hole():
    raster = RASTER.copy()
    raster[0, 0] = -999.0
    masked = np.ma.MaskedArray(raster, mask=np.zeros(RASTER.shape, bool))
    masked.mask[1, 1] = True
    region = Polygon(
        [(0.25, 2.25), (2.75, 2.25), (2.75, 3.75), (0.25, 3.75)],
        holes=[[(1.25, 2.75), (1.75, 2.75), (1.75, 3.25), (1.25, 3.25)]],
    )
    cutter = Polygon([(1.5, -10), (10, -10), (10, 10), (1.5, 10)])
    left = region.difference(cutter)
    right = region.intersection(cutter)
    assert abs(left.union(right).symmetric_difference(region).area) < 1e-12

    whole = fractional_stats([region], masked, affine=AFFINE, nodata=-999)[0]
    parts = fractional_stats([left, right], masked, affine=AFFINE, nodata=-999)
    for key in ("coverage", "count", "nodata", "sum"):
        assert np.isclose(
            parts[0][key] + parts[1][key], whole[key], rtol=1e-9, atol=1e-10
        ), key
    # Fundamental bookkeeping identity.
    assert np.isclose(whole["coverage"], whole["count"] + whole["nodata"])


def test_partial_nodata_cell_is_split():
    raster = RASTER.copy()
    raster[0, 0] = -999.0
    half = Polygon([(0.5, 3), (1.5, 3), (1.5, 4), (0.5, 4)])
    result = fractional_stats([half], raster, affine=AFFINE, nodata=-999)[0]
    approx_equal(
        result,
        {
            "coverage": 1.0,
            "count": 0.5,
            "nodata": 0.5,
            "sum": 0.5,
            "mean": 1.0,
            "min": 1.0,
            "max": 1.0,
        },
    )


# ---------------------------------------------------------------------------
# File-backed rasters, masks and nodata
# ---------------------------------------------------------------------------


def test_file_matches_array_and_uses_metadata_nodata():
    path = DATA / "slope.tif"
    with rasterio.open(path) as src:
        array = src.read(1)
        transform = src.transform
        nodata = src.nodata
    rows, cols = np.where(array == nodata)
    row, col = rows[0], cols[0]
    west, north = transform @ (col - 0.5, row - 0.5)
    east, south = transform @ (col + 1.5, row + 1.5)
    polygon = box(west, south, east, north)

    from_file = fractional_stats([polygon], path)[0]
    from_array = fractional_stats(
        [polygon], array, affine=transform, nodata=nodata
    )[0]
    for key in FRACTIONAL_STAT_KEYS:
        if from_file[key] is None:
            assert from_array[key] is None
        else:
            assert np.isclose(from_file[key], from_array[key], rtol=1e-9)
    assert from_file["nodata"] > 0.0

    # Without a sentinel the same values are ordinary data on an array.
    plain = fractional_stats([polygon], array, affine=transform)[0]
    assert plain["nodata"] == 0.0
    assert plain["count"] == plain["coverage"]


def test_nodata_override_replaces_sentinel_only():
    path = DATA / "slope.tif"
    with rasterio.open(path) as src:
        array = src.read(1)
        transform = src.transform
    polygon = box(244600, 1000000, 245000, 1000400)
    overridden = fractional_stats([polygon], path, nodata=-1.0)[0]
    assert overridden["nodata"] == 0.0
    assert overridden["count"] == overridden["coverage"]


def test_independent_dataset_mask_cannot_be_cancelled():
    path = DATA / "dataset_mask.tif"
    with rasterio.open(path) as src:
        mask = src.dataset_mask()
        transform = src.transform
    rows, cols = np.where(mask == 0)
    row, col = rows[0], cols[0]
    west, north = transform @ (col - 0.5, row - 0.5)
    east, south = transform @ (col + 1.5, row + 1.5)
    polygon = box(west, south, east, north)

    masked_result = fractional_stats([polygon], path)[0]
    overridden = fractional_stats([polygon], path, nodata=-42.0)[0]
    assert masked_result["nodata"] > 0.0
    assert np.isclose(overridden["nodata"], masked_result["nodata"])
    assert np.isclose(overridden["count"], masked_result["count"])


def test_file_band_validation_and_io_exception():
    path = DATA / "slope.tif"
    polygon = box(244600, 1000000, 245000, 1000400)
    with pytest.raises(ValueError):
        fractional_stats([polygon], path, band=99)
    with pytest.raises(rasterio.RasterioIOError):
        fractional_stats([polygon], DATA / "missing.tif")


# ---------------------------------------------------------------------------
# Feature handling: sources, ordering, duplicates, immutability
# ---------------------------------------------------------------------------


def test_feature_order_duplicates_and_input_immutability():
    first = Polygon([(0, 2), (2, 2), (2, 4), (0, 4)])
    second = Polygon([(0, 3), (1, 3), (1, 4), (0, 4)])
    features = [first, second, first]
    snapshot = [copy.deepcopy(f.__geo_interface__) for f in features]
    results = fractional_stats(features, RASTER, affine=AFFINE)
    assert len(results) == 3
    assert results[0] == results[2]
    assert np.isclose(results[1]["coverage"], 1.0)
    assert [f.__geo_interface__ for f in features] == snapshot


def test_geojson_output_preserves_feature():
    polygon = Polygon([(0, 2), (2, 2), (2, 4), (0, 4)])
    feature = {
        "type": "Feature",
        "id": "F1",
        "properties": {"name": "alpha"},
        "geometry": polygon.__geo_interface__,
    }
    snapshot = copy.deepcopy(feature)
    output = fractional_stats(
        [feature], RASTER, affine=AFFINE, geojson_out=True, prefix="z_"
    )[0]
    assert output["id"] == "F1"
    assert output["geometry"] == feature["geometry"]
    assert output["properties"]["name"] == "alpha"
    assert "z_coverage" in output["properties"]
    assert feature == snapshot  # input untouched


def test_prefix_and_plain_key_sets():
    polygon = Polygon([(0, 2), (2, 2), (2, 4), (0, 4)])
    plain = fractional_stats([polygon], RASTER, affine=AFFINE)[0]
    assert set(plain) == set(FRACTIONAL_STAT_KEYS)
    prefixed = fractional_stats(
        [polygon], RASTER, affine=AFFINE, prefix="p_"
    )[0]
    assert set(prefixed) == {f"p_{key}" for key in FRACTIONAL_STAT_KEYS}


def test_geojson_featurecollection_input(tmp_path):
    polygon = Polygon([(0, 2), (2, 2), (2, 4), (0, 4)])
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": polygon.__geo_interface__,
            }
        ],
    }
    results = fractional_stats(collection, RASTER, affine=AFFINE)
    assert len(results) == 1
    assert results[0]["coverage"] == 4.0


def test_empty_feature_sources():
    empty_fc = {"type": "FeatureCollection", "features": []}
    assert fractional_stats(empty_fc, RASTER, affine=AFFINE) == []
    assert fractional_stats(iter(()), RASTER, affine=AFFINE) == []
    # An empty source never opens the raster dataset.
    generator = gen_fractional_stats(empty_fc, DATA / "slope.tif")
    assert list(generator) == []
    assert generator._closed
    assert generator._blocks is None


def test_one_shot_iterable_consumed_once_and_lazy():
    polygon = Polygon([(0, 2), (2, 2), (2, 4), (0, 4)])
    seen = []

    def source():
        for _ in range(3):
            seen.append(1)
            yield polygon

    generator = gen_fractional_stats(source(), RASTER, affine=AFFINE)
    assert seen == []  # nothing consumed before first next()
    first = next(generator)
    assert len(seen) == 1
    assert first["coverage"] == 4.0
    assert len(list(generator)) == 2
    assert len(seen) == 3


def test_repeated_calls_are_independent():
    polygon = Polygon([(0, 2), (2, 2), (2, 4), (0, 4)])
    first = fractional_stats([polygon], RASTER, affine=AFFINE)
    second = fractional_stats([polygon], RASTER, affine=AFFINE)
    assert first == second


def test_file_vector_source_and_geometry_mapping():
    # A real on-disk vector source, read through read_features.
    shapefile = DATA / "polygons.shp"
    raster = DATA / "slope.tif"
    results = fractional_stats(shapefile, raster)
    assert len(results) == 2
    assert all(set(result) == set(FRACTIONAL_STAT_KEYS) for result in results)
    assert all(result["coverage"] > 0.0 for result in results)

    # A bare GeoJSON geometry mapping is accepted too.
    polygon = Polygon([(0, 2), (2, 2), (2, 4), (0, 4)])
    result = fractional_stats(polygon.__geo_interface__, RASTER, affine=AFFINE)[0]
    assert result["coverage"] == 4.0


def test_cleanup_does_not_mask_original_exception():
    class FailingBlocks:
        affine = AFFINE
        shape = RASTER.shape

        def read_block(self, *args):
            raise RuntimeError("boom from the reader")

        def close(self):
            raise RuntimeError("cleanup also failed")

    generator = FractionalStatsGenerator(
        [Polygon([(0, 2), (2, 2), (2, 4), (0, 4)])],
        RASTER,
        affine=AFFINE,
    )
    next(generator)  # opens via ArrayBlocks; swap in the failing reader.
    generator.close()
    generator = FractionalStatsGenerator(
        [Polygon([(0, 2), (2, 2), (2, 4), (0, 4)])],
        RASTER,
        affine=AFFINE,
    )
    generator._start()
    generator._blocks = FailingBlocks()
    with pytest.raises(RuntimeError, match="boom from the reader"):
        next(generator)
    assert generator._closed


# ---------------------------------------------------------------------------
# Generator lifecycle: close, errors keep earlier results, indexed messages
# ---------------------------------------------------------------------------


def test_early_close_stops_reading_and_is_idempotent():
    polygon = Polygon([(0, 2), (2, 2), (2, 4), (0, 4)])
    generator = gen_fractional_stats(
        [polygon, polygon], RASTER, affine=AFFINE
    )
    generator.close()
    assert list(generator) == []
    generator.close()  # no error


def test_exhaustion_releases_file_dataset():
    generator = gen_fractional_stats(
        [box(244600, 1000000, 245000, 1000400)],
        DATA / "slope.tif",
        max_cells=100,
    )
    list(generator)
    assert generator._closed
    assert generator._blocks is None


def test_error_keeps_delivered_results_and_names_index():
    good = Polygon([(0, 2), (2, 2), (2, 4), (0, 4)])
    generator = gen_fractional_stats(
        [good, Point(0, 0), good], RASTER, affine=AFFINE
    )
    first = next(generator)
    assert first["coverage"] == 4.0
    with pytest.raises(ValueError, match="feature 1"):
        next(generator)
    with pytest.raises(StopIteration):
        next(generator)  # closed after the error


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_cells": 0},
        {"max_cells": -1},
        {"max_cells": True},
        {"max_cells": 1.5},
        {"chunk_size": 0},
        {"chunk_size": False},
        {"prefix": 5},
        {"geojson_out": 1},
        {"band": 0},
        {"band": 1.0},
        {"nodata": "x"},
        {"nodata": True},
    ],
)
def test_scalar_parameter_validation(kwargs):
    polygon = Polygon([(0, 2), (2, 2), (2, 4), (0, 4)])
    with pytest.raises(ValueError):
        fractional_stats([polygon], RASTER, affine=AFFINE, **kwargs)


def test_array_requires_affine_and_band_one():
    polygon = Polygon([(0, 2), (2, 2), (2, 4), (0, 4)])
    with pytest.raises(ValueError):
        fractional_stats([polygon], RASTER)
    with pytest.raises(ValueError):
        fractional_stats([polygon], RASTER, affine=AFFINE, band=2)


def test_reject_non_real_or_non_2d_arrays():
    polygon = Polygon([(0, 2), (1, 2), (1, 3), (0, 3)])
    bad_arrays = [
        np.array([[True, False], [False, True]]),
        np.array([[1 + 2j, 2 + 3j], [3 + 4j, 4 + 5j]]),
        np.empty((2, 2), dtype=object),
        np.arange(4),
    ]
    for array in bad_arrays:
        with pytest.raises(ValueError):
            fractional_stats([polygon], array, affine=AFFINE)


def test_reject_bad_affines():
    polygon = Polygon([(0, 2), (1, 2), (1, 3), (0, 3)])
    singular = Affine(1.0, 2.0, 0.0, 2.0, 4.0, 0.0)
    non_finite = Affine(np.nan, 0.0, 0.0, 0.0, 1.0, 0.0)
    for affine in (singular, non_finite):
        with pytest.raises(ValueError):
            fractional_stats([polygon], RASTER, affine=affine)
    with pytest.raises(ValueError):
        fractional_stats([polygon], RASTER, affine="not-an-affine")


@pytest.mark.parametrize(
    "geometry",
    [
        None,
        Point(1, 1),
        LineString([(0, 0), (1, 1)]),
        GeometryCollection(),
        Polygon([(0, 0, 0), (2, 0, 1), (2, 2, 0), (0, 2, 1)]),
        Polygon([(0, 0), (2, 2), (2, 0), (0, 2)]),  # self-intersecting bowtie
    ],
)
def test_reject_invalid_geometries(geometry):
    with pytest.raises(ValueError):
        fractional_stats([geometry], RASTER, affine=AFFINE)


def test_reject_non_finite_coordinates():
    base = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    nan_polygon = set_coordinates(
        base, np.array([[0, 0], [2, 0], [2, 2], [np.nan, 2], [0, 0]])
    )
    with pytest.raises(ValueError, match="non-finite"):
        fractional_stats([nan_polygon], RASTER, affine=AFFINE)


def test_reject_overlapping_multipolygon():
    overlapping = MultiPolygon(
        [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]),
        ]
    )
    with pytest.raises(ValueError):
        fractional_stats([overlapping], RASTER, affine=AFFINE)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def test_block_windows_never_exceed_max_cells():
    from rasterstats.fractional import iter_block_windows

    # Span far wider than the limit: column strips must kick in.
    windows = list(iter_block_windows(0, 3000, 0, 5001, max_cells=2048))
    assert windows, "at least one window expected"
    for row_start, row_stop, col_start, col_stop in windows:
        cells = (row_stop - row_start) * (col_stop - col_start)
        assert cells <= 2048, (row_start, row_stop, col_start, col_stop)
        assert cells > 0
    # Full coverage of the span, exactly once, with no gaps or overlaps.
    covered = sum(
        (row_stop - row_start) * (col_stop - col_start)
        for row_start, row_stop, col_start, col_stop in windows
    )
    assert covered == 3000 * 5001
    # Boundary case: a single column when max_cells is one.
    windows = list(iter_block_windows(0, 3, 0, 2, max_cells=1))
    assert all(
        (row_stop - row_start) * (col_stop - col_start) == 1
        for row_start, row_stop, col_start, col_stop in windows
    )
    assert len(windows) == 6


def test_wide_region_respects_max_cells_on_file(tmp_path):
    # Instrumented end-to-end check: every rasterio window stays within
    # max_cells even though the region is wider than the limit.
    import rasterio

    profile = dict(
        driver="GTiff",
        height=60,
        width=120,
        count=1,
        dtype="float32",
        transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 60.0),
    )
    path = tmp_path / "wide.tif"
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.arange(60 * 120, dtype="float32").reshape(60, 120), 1)

    seen = []
    original_read = rasterio.io.DatasetReader.read

    def spy_read(self, indexes, window=None, **kwargs):
        rows = window[0][1] - window[0][0]
        cols = window[1][1] - window[1][0]
        seen.append(rows * cols)
        return original_read(self, indexes, window=window, **kwargs)

    rasterio.io.DatasetReader.read = spy_read
    try:
        result = fractional_stats(
            [box(0.5, 10.0, 119.5, 50.0)], path, max_cells=128
        )[0]
    finally:
        rasterio.io.DatasetReader.read = original_read

    assert seen and max(seen) <= 128
    assert result["coverage"] > 0.0


def test_grid_bounds_and_cell_polygons_roundtrip():
    polygon = box(1.2, 1.2, 3.4, 3.4)
    min_row, max_row, min_col, max_col = grid_bounds(polygon, ~AFFINE)
    assert (min_row, max_row, min_col, max_col) == (0, 3, 1, 4)
    cells = cell_polygons(0, 0, 4, 4, AFFINE)
    # The (r=0, c=0) cell quadrilateral matches the expected unit square.
    assert np.isclose(cells[0].area, 1.0)
    assert cells[0].equals(box(0, 3, 1, 4))


def test_validate_polygon_accepts_empty():
    assert validate_polygon(Polygon()).is_empty
    assert validate_polygon(MultiPolygon([])).is_empty


def test_block_readers_are_windowed():
    array_blocks = ArrayBlocks(RASTER, AFFINE, None, max_cells=4)
    values, valid = array_blocks.read_block(0, 2, 1, 3)
    assert values.shape == (2, 2)
    assert valid.all()
    assert np.array_equal(values, RASTER[0:2, 1:3].astype(np.float64))
    array_blocks.close()

    file_blocks = FileBlocks(DATA / "slope.tif", 1, None, max_cells=4)
    values, valid = file_blocks.read_block(0, 2, 0, 2)
    assert values.shape == (2, 2)
    file_blocks.close()


def test_generator_is_the_public_class():
    generator = gen_fractional_stats(
        [Polygon()], RASTER, affine=AFFINE
    )
    assert isinstance(generator, FractionalStatsGenerator)
