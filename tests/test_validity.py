"""Regression tests for the shared raster-read validity contract.

Every expected value below is derived by hand from a synthetic 2x2 grid
(the top-left cell starts at (0, 2), unit cells run downwards), never from
the implementation under test.  The same feature is exercised through an
ndarray, a MaskedArray and a temporary GeoTIFF so that all three read paths
agree on which cells are observations.
"""

import json
import math

import numpy as np
import pytest
import rasterio
from affine import Affine
from click.testing import CliRunner
from shapely.geometry import LineString, MultiPoint, Point, Polygon

from rasterstats import gen_point_query, gen_zonal_stats, point_query, zonal_stats
from rasterstats.cli import NODATA, pointquery, zonalstats
from rasterstats.io import Raster
from rasterstats.point import bilinear, point_window_unitxy

# Top-left cell corner is (0, 2); unit cells run downwards, so row 0 spans
# y in (1, 2) and row 1 spans y in (0, 1).
AFFINE = Affine(1, 0, 0, 0, -1, 2)
WHOLE = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
# Centres of the four cells (row, col) -> (x, y)
CENTRES = {
    (0, 0): (0.5, 1.5),
    (0, 1): (1.5, 1.5),
    (1, 0): (0.5, 0.5),
    (1, 1): (1.5, 0.5),
}

DEFAULT_VALUES = np.array([[-999, 2], [3, 4]])


def write_tif(path, array, nodata=None, mask=None):
    """Write a one-band GeoTIFF with the shared synthetic affine."""
    array = np.asarray(array)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype.name,
        transform=AFFINE,
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)
        if mask is not None:
            dst.write_mask(np.asarray(mask, dtype="uint8"))
    return path


def stats_for(raster, **kwargs):
    return zonal_stats(WHOLE, raster, affine=AFFINE, **kwargs)[0]


def point_for(raster, rc, **kwargs):
    x, y = CENTRES[rc]
    return point_query(Point(x, y), raster, affine=AFFINE, **kwargs)[0]


# --- Reported defects --------------------------------------------------------


def test_no_declared_nodata_keeps_every_cell_observed():
    # -999 is a legitimate observation when no marker is declared.
    result = stats_for(
        DEFAULT_VALUES, stats="count sum min max mean nodata nan"
    )
    assert result["count"] == 4
    assert result["sum"] == -990
    assert result["min"] == -999
    assert result["nodata"] == 0
    assert result["nan"] == 0


def test_uint8_nearest_at_first_cell_centre_does_not_overflow():
    arr = np.array([[0, 2], [3, 4]], dtype="uint8")
    assert point_for(arr, (0, 0), interpolate="nearest") == 0
    assert point_for(arr, (0, 1), interpolate="nearest") == 2


def test_file_overrides_old_nodata_and_points_agree(tmp_path):
    path = write_tif(
        tmp_path / "grid.tif", np.array([[-99, 2], [3, 4]], dtype="int16"),
        nodata=-99,
    )
    # The caller's marker (2) replaces the file's (-99).
    result = stats_for(
        path, nodata=2, stats="count sum min max mean nodata nan"
    )
    assert result["count"] == 3
    assert result["sum"] == -92
    assert result["nodata"] == 1
    # First cell (-99) becomes an observation under the override.
    assert point_for(path, (0, 0), nodata=2, interpolate="nearest") == -99
    assert point_for(path, (0, 1), nodata=2, interpolate="nearest") is None


def test_nan_cell_is_never_returned_by_nearest(tmp_path):
    path = write_tif(
        tmp_path / "nan.tif",
        np.array([[np.nan, 2], [3, 4]], dtype="float32"),
        nodata=-99,
    )
    assert point_for(path, (0, 0), nodata=-99, interpolate="nearest") is None


# --- Dtype and marker identity ----------------------------------------------


def test_uint8_marker_outside_dtype_matches_nothing():
    arr = np.array([[0, 2], [3, 4]], dtype="uint8")
    result = stats_for(arr, nodata=-999, stats="count sum nodata")
    assert result["count"] == 4
    assert result["sum"] == 9
    assert result["nodata"] == 0


def test_negative_integer_values_stay_observations():
    arr = np.array([[-5, -2], [-3, -4]], dtype="int16")
    result = stats_for(arr, stats="count sum min max nodata")
    assert result["count"] == 4
    assert result["sum"] == -14
    assert result["min"] == -5


def test_large_int_marker_is_not_rounded():
    big = 2**53 + 1
    arr = np.array([[big, 2], [3, 4]], dtype="int64")
    # The marker matches the first cell and nothing else.
    result = stats_for(arr, nodata=big, stats="count sum nodata")
    assert result["count"] == 3
    assert result["sum"] == 9
    assert result["nodata"] == 1
    # 2**53 + 1 and 2**53 must not collide.
    other = np.array([[2**53, 2], [3, 4]], dtype="int64")
    result = stats_for(other, nodata=big, stats="count sum nodata")
    assert result["count"] == 4
    assert result["nodata"] == 0


def test_large_int_read_keeps_dtype_and_exact_value():
    big = 2**53 + 1
    arr = np.array([[big, 2], [3, 4]], dtype="int64")
    with Raster(arr, AFFINE, nodata=big) as rast:
        result = rast.read(window=((0, 1), (0, 1)))
    assert result.array.dtype == np.int64
    assert int(result.array[0, 0]) == big
    assert result.invalid_mask[0, 0]
    # Point query returns None, not a rounded float.
    assert point_for(arr, (0, 0), nodata=big, interpolate="nearest") is None


def test_float_nodata_accepts_small_decimal():
    arr = np.array([[2.5, 1.0], [3.0, 4.0]], dtype="float32")
    result = stats_for(arr, nodata=2.5, stats="count sum nodata")
    assert result["count"] == 3
    assert result["sum"] == 8
    assert result["nodata"] == 1
    assert point_for(arr, (0, 0), nodata=2.5, interpolate="nearest") is None


def test_nonfinite_float_marker_matches_nothing_extra():
    arr = np.array([[np.nan, np.inf], [-np.inf, 4.0]])
    result = stats_for(arr, nodata=np.inf, stats="count sum nodata nan")
    # Only the finite cell is an observation; inf marker kills nothing more.
    assert result["count"] == 1
    assert result["sum"] == 4
    assert result["nodata"] == 0
    assert result["nan"] == 1


# --- Independent masks survive an explicit marker override -------------------


def test_override_replaces_old_file_mask_but_keeps_independent_mask(tmp_path):
    # File declaration -99 plus an independent dataset mask at (1, 1).
    path = write_tif(
        tmp_path / "both.tif",
        np.array([[-99, 2], [3, 4]], dtype="int16"),
        nodata=-99,
        mask=np.array([[255, 255], [255, 0]], dtype="uint8"),
    )
    # Without override: -99 and the mask are both invalid.
    result = stats_for(path, stats="count sum nodata")
    assert result["count"] == 2
    assert result["sum"] == 5
    assert result["nodata"] == 1

    # With override: -99 becomes a valid observation again, the independent
    # mask at (1, 1) remains invalid, and 2 is now the marker.
    result = stats_for(path, nodata=2, stats="count sum nodata")
    assert result["count"] == 2
    assert result["sum"] == -96
    assert result["nodata"] == 1

    with Raster(path, nodata=2) as rast:
        result = rast.read(window=((0, 2), (0, 2)))
    assert result.array.dtype == np.dtype("int16")
    assert not result.invalid_mask[0, 0]  # old nodata no longer invalid
    assert result.invalid_mask[0, 1]  # new marker
    assert result.invalid_mask[1, 1]  # independent mask preserved


def test_alpha_mask_is_independent_of_numeric_marker(tmp_path):
    path = tmp_path / "rgba.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=2, width=2, count=4,
        dtype="uint8", transform=AFFINE,
    ) as dst:
        for band in (1, 2, 3):
            dst.write(np.full((2, 2), band * 10, dtype="uint8"), band)
        dst.write(np.array([[255, 255], [255, 0]], dtype="uint8"), 4)
        dst.colorinterp = (
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
            rasterio.enums.ColorInterp.alpha,
        )
    result = stats_for(path, stats="count sum")
    assert result["count"] == 3
    assert result["sum"] == 30


def test_masked_array_mask_is_independent_of_marker():
    mask = np.array([[True, False], [False, False]])
    arr = np.ma.MaskedArray(np.array([[-99, 2], [3, 4]], dtype="int16"), mask=mask)
    # Caller mask at (0, 0) and the marker -99 coincide there; the other
    # three cells are observations.
    result = stats_for(arr, nodata=-99, stats="count sum nodata")
    assert result["count"] == 3
    assert result["sum"] == 9
    assert result["nodata"] == 1
    # Overriding the marker drops the old numeric marker but not the mask:
    # (0, 1) now holds the marker and (0, 0) stays independently masked.
    result = stats_for(arr, nodata=2, stats="count sum nodata")
    assert result["count"] == 2
    assert result["sum"] == 7
    assert result["nodata"] == 1


# --- Bounds, all-invalid and window padding ---------------------------------


def test_out_of_bounds_slots_are_invalid_but_not_diagnostic_hits():
    arr = np.array([[0, 2], [3, 4]], dtype="uint8")
    with Raster(arr, AFFINE) as rast:
        result = rast.read(window=((-1, 1), (-1, 1)))
    assert result.in_bounds_mask.sum() == 1
    assert result.invalid_mask.sum() == 3
    # The only in-bounds cell is the original (0, 0) = 0.
    assert int(result.array[1, 1]) == 0

    # Diagnostics ignore the padded slots even though the marker could
    # coincide with the fill value.
    result = stats_for(arr, nodata=0, stats="count sum nodata")
    assert result["count"] == 3
    assert result["sum"] == 9
    assert result["nodata"] == 1


def test_all_invalid_keeps_empty_result_convention():
    arr = np.full((2, 2), -99, dtype="int16")
    result = stats_for(arr, nodata=-99, stats="count sum min max mean nodata nan")
    assert result["count"] == 0
    assert result["sum"] is None
    assert result["min"] is None
    assert result["max"] is None
    assert result["mean"] is None
    assert result["nodata"] == 4


def test_categorical_and_callbacks_see_same_valid_cells():
    arr = np.array([[-99, 2], [3, 4]], dtype="int16")
    seen = {}

    def capture(zone_array):
        seen["data"] = zone_array.compressed().tolist()
        return np.ma.mean(zone_array)

    result = stats_for(
        arr,
        nodata=-99,
        categorical=True,
        add_stats={"captured": capture},
        stats="count sum",
    )
    assert sorted(seen["data"]) == [2, 3, 4]
    assert result["captured"] == 3
    # categorical keys are the observed values only
    assert result[2.0] == 1
    assert result[3.0] == 1
    assert result[4.0] == 1


def test_zone_func_sees_invalid_mask():
    arr = np.array([[-99, 2], [3, 4]], dtype="int16")

    def keep_zero(zone_array):
        # The caller can rely on the mask matching the observations.
        assert zone_array.mask[0, 0]
        return zone_array * 0

    result = stats_for(arr, nodata=-99, zone_func=keep_zero, stats="count sum")
    assert result["count"] == 3
    assert result["sum"] == 0


def test_raster_read_matches_ordinary_stats():
    arr = np.array([[-99, 2], [3, 4]], dtype="int16")
    result = stats_for(arr, nodata=2, stats="count sum min max mean nodata")
    with Raster(arr, AFFINE, nodata=2) as rast:
        read = rast.read(bounds=(0, 0, 2, 2))
        masked = rast.read(bounds=(0, 0, 2, 2), masked=True)
    assert read.array.dtype == np.dtype("int16")
    np.testing.assert_array_equal(
        read.invalid_mask, [[False, True], [False, False]]
    )
    np.testing.assert_array_equal(
        np.ma.getmaskarray(masked.array), read.invalid_mask
    )
    observed = read.array[~read.invalid_mask]
    assert sorted(observed.tolist()) == [-99, 3, 4]
    assert result["count"] == int(observed.size)
    assert result["sum"] == float(observed.sum())


# --- Diagnostics -------------------------------------------------------------


def test_nan_and_nodata_diagnostics_do_not_impersonate_each_other():
    arr = np.array([[np.nan, np.inf], [-99, 4.0]])
    result = stats_for(
        arr, nodata=-99, stats="count sum nodata nan min max"
    )
    # nan counts stored NaN only (not infinity); nodata counts the marker.
    assert result["nan"] == 1
    assert result["nodata"] == 1
    assert result["count"] == 1
    assert result["sum"] == 4


# --- Point queries -----------------------------------------------------------


def test_nearest_returns_none_for_invalid_and_value_for_observation(tmp_path):
    path = write_tif(
        tmp_path / "grid.tif", np.array([[-99, 2], [3, 4]], dtype="int16"),
        nodata=-99,
    )
    assert point_for(path, (0, 0), interpolate="nearest") is None
    assert point_for(path, (0, 1), interpolate="nearest") == 2
    # Marker override makes -99 observed and 2 invalid.
    assert point_for(path, (0, 0), nodata=2, interpolate="nearest") == -99
    assert point_for(path, (0, 1), nodata=2, interpolate="nearest") is None


def test_bilinear_centre_of_valid_cell_ignores_zero_weight_nan():
    # 3x3 grid so the central cell's interpolation support contains a NaN
    # neighbour at x=0 (zero weight) without touching the array edge.
    arr = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, np.nan],
            [7.0, 8.0, 9.0],
        ]
    )
    # Centre of cell (1, 1) is at (1.5, 0.5).
    assert point_query(Point(1.5, 0.5), arr, affine=AFFINE)[0] == 5.0
    # A non-zero-weight NaN still forces the nearest fallback.
    assert point_query(Point(1.6, 0.4), arr, affine=AFFINE)[0] == 5.0


def test_bilinear_falls_back_to_nearest_valid_or_none():
    valid = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert bilinear(valid, 0.5, 0.5) == 2.5

    # Edge cell centre on a boundless window: three support cells are
    # outside the raster, so the fallback returns the visible cell.
    arr = np.array([[5.0, np.nan], [3.0, 4.0]])
    window, unitxy = point_window_unitxy(*CENTRES[(0, 0)], AFFINE)
    with Raster(arr, AFFINE) as rast:
        read = rast.read(window=window, masked=True)
    assert bilinear(read.array, *unitxy) == 5.0

    # The nearest support cell itself is invalid -> None.
    invalid = np.ma.MaskedArray(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float64"),
        mask=[[False, False], [False, True]],
    )
    # unit square centre: nearest cell is the bottom-right, which is masked
    assert bilinear(invalid, 1.0, 0.0) is None


def test_multipoint_line_and_geojson_out_shapes(tmp_path):
    path = write_tif(
        tmp_path / "grid.tif", np.array([[-99, 2], [3, 4]], dtype="int16"),
        nodata=-99,
    )
    multipoint = MultiPoint([CENTRES[(0, 0)], CENTRES[(0, 1)]])
    values = point_query(multipoint, path, interpolate="nearest")[0]
    assert values == [None, 2]

    line = LineString([CENTRES[(1, 0)], CENTRES[(1, 1)]])
    values = point_query(line, path, interpolate="nearest")[0]
    assert values == [3, 4]

    features = point_query(
        Point(*CENTRES[(0, 1)]),
        path,
        interpolate="nearest",
        property_name="TEST",
        geojson_out=True,
    )
    assert features[0]["properties"]["TEST"] == 2
    assert features[0]["type"] == "Feature"


def test_point_generator_matches_list_entry_point(tmp_path):
    path = write_tif(
        tmp_path / "grid.tif", np.array([[-99, 2], [3, 4]], dtype="int16"),
        nodata=-99,
    )
    points = [Point(*CENTRES[rc]) for rc in CENTRES]
    listed = point_query(points, path, nodata=2, interpolate="nearest")
    generated = list(
        gen_point_query(points, path, nodata=2, interpolate="nearest")
    )
    assert listed == generated == [-99, None, 3, 4]


def test_zonal_generator_matches_list_entry_point():
    arr = np.array([[-99, 2], [3, 4]], dtype="int16")
    kwargs = dict(nodata=-99, stats="count sum nodata")
    listed = zonal_stats(WHOLE, arr, affine=AFFINE, **kwargs)
    generated = list(gen_zonal_stats(WHOLE, arr, affine=AFFINE, **kwargs))
    assert listed == generated == [{"count": 3, "sum": 9.0, "nodata": 1}]


# --- Fractional mode shares the same contract --------------------------------


def test_fractional_uses_shared_validity_for_file_and_mask(tmp_path):
    path = write_tif(
        tmp_path / "both.tif",
        np.array([[-99, 2], [3, 4]], dtype="int16"),
        nodata=-99,
        mask=np.array([[255, 255], [255, 0]], dtype="uint8"),
    )
    result = zonal_stats(
        WHOLE, path, nodata=2, coverage="fractional", value_kind="total"
    )[0]
    # Old marker -99 is observed, new marker 2 and the independent mask at
    # (1, 1) are not.
    assert result["valid_area"] == 2.0
    assert result["count"] == 2.0
    assert result["sum"] == -96.0


def test_large_int_file_roundtrip_keeps_exact_identity(tmp_path):
    # GDAL's own boundless path can round 2**53+1 down when the declared
    # nodata is a float; the reader must preserve the exact stored value.
    big = 2**53 + 1
    path = write_tif(tmp_path / "big.tif", np.array([[big, 2], [3, 4]], dtype="int64"))
    assert stats_for(path, nodata=big)["count"] == 3
    result = stats_for(path, nodata=big, stats="count sum nodata")
    assert result["sum"] == 9
    assert result["nodata"] == 1
    assert point_for(path, (0, 0), nodata=big, interpolate="nearest") is None
    with Raster(path, nodata=big) as rast:
        read = rast.read(window=((0, 2), (0, 2)))
    assert read.array.dtype == np.int64
    assert int(read.array[0, 0]) == big
    # Padding a boundless window never rounds the neighbouring real value.
    with Raster(path, nodata=big) as rast:
        padded = rast.read(window=((-1, 2), (-1, 2)))
    assert int(padded.array[1, 1]) == big


# --- Bounded per-window work -------------------------------------------------


def test_small_window_reads_allocate_only_window_sized_arrays(monkeypatch):
    import rasterstats.io as io_module

    calls = []
    real_full = io_module.np.full

    def recording_full(shape, fill_value, *args, **kwargs):
        calls.append(tuple(shape))
        return real_full(shape, fill_value, *args, **kwargs)

    monkeypatch.setattr(io_module.np, "full", recording_full)

    size = 512
    data = np.ones((size, size), dtype="int32")
    mask = np.zeros((size, size), dtype=bool)
    mask[0, 0] = True
    source = np.ma.MaskedArray(data, mask=mask)

    with Raster(source, AFFINE) as rast:
        # No full-source copy is kept by the reader.
        assert rast.parent_mask is not None
        assert rast.parent_mask.shape == (size, size)
        for row in range(0, 100):
            for col in range(0, 100):
                result = rast.read(window=((row, row + 1), (col, col + 1)))
                assert result.array.shape == (1, 1)

    # Every allocation the reader made (data and in-bounds mask) is at most
    # one pixel; none scales with the source dimensions.
    assert calls
    assert max(np.prod(shape) for shape in calls) <= 1


# --- Inputs are not mutated --------------------------------------------------


def test_inputs_and_masks_are_left_untouched(tmp_path):
    raw = np.array([[-99, 2], [3, 4]], dtype="int16")
    original = raw.copy()
    mask = np.array([[False, True], [False, False]])
    masked = np.ma.MaskedArray(raw.copy(), mask=mask.copy())
    masked_before = np.ma.getmaskarray(masked).copy()

    stats_for(raw, nodata=-99)
    stats_for(masked, nodata=2)
    point_for(raw, (0, 0), nodata=2, interpolate="nearest")

    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(masked.data, original)
    np.testing.assert_array_equal(np.ma.getmaskarray(masked), masked_before)

    path = write_tif(tmp_path / "grid.tif", raw, nodata=-99)
    stats_for(path, nodata=2)
    with rasterio.open(path) as src:
        np.testing.assert_array_equal(src.read(1), original)
        assert src.nodata == -99


# --- CLI parity --------------------------------------------------------------


def _write_feature(geometry, filename):
    feature = {
        "type": "Feature",
        "properties": {"name": "zone"},
        "geometry": geometry,
    }
    path = filename
    path.write_text(json.dumps(feature))
    return str(path), feature


def test_cli_zonal_matches_python_for_collection_and_sequence(tmp_path):
    path = write_tif(
        tmp_path / "grid.tif", np.array([[-99, 2], [3, 4]], dtype="int16"),
        nodata=-99,
    )
    vector, feature = _write_feature(
        WHOLE.__geo_interface__, tmp_path / "zone.geojson"
    )
    runner = CliRunner()
    args = [
        vector,
        "--raster",
        str(path),
        "--nodata",
        "2",
        "--stats",
        "count sum min max mean nodata nan",
        "--prefix",
        "v_",
    ]
    collection = runner.invoke(zonalstats, args)
    sequence = runner.invoke(zonalstats, args + ["--sequence"])
    assert collection.exit_code == 0, collection.output
    assert sequence.exit_code == 0, sequence.output

    expected = zonal_stats(
        [feature],
        str(path),
        nodata=2,
        stats="count sum min max mean nodata nan",
        prefix="v_",
        geojson_out=True,
    )[0]["properties"]
    coll_props = json.loads(collection.output)["features"][0]["properties"]
    seq_props = json.loads(sequence.output.splitlines()[0])["properties"]
    assert coll_props == seq_props == expected
    assert "NaN" not in collection.output
    assert "Infinity" not in collection.output


def test_cli_point_matches_python_for_collection_and_sequence(tmp_path):
    path = write_tif(
        tmp_path / "grid.tif", np.array([[-99, 2], [3, 4]], dtype="int16"),
        nodata=-99,
    )
    vector, feature = _write_feature(
        Point(*CENTRES[(0, 1)]).__geo_interface__, tmp_path / "pt.geojson"
    )
    runner = CliRunner()
    args = [
        vector,
        "--raster",
        str(path),
        "--nodata",
        "2",
        "--interpolate",
        "nearest",
        "--property-name",
        "slope",
    ]
    collection = runner.invoke(pointquery, args)
    sequence = runner.invoke(pointquery, args + ["--sequence"])
    assert collection.exit_code == 0, collection.output
    assert sequence.exit_code == 0, sequence.output

    expected = point_query(
        [feature],
        str(path),
        nodata=2,
        interpolate="nearest",
        property_name="slope",
        geojson_out=True,
    )[0]["properties"]
    coll_props = json.loads(collection.output)["features"][0]["properties"]
    seq_props = json.loads(sequence.output.splitlines()[0])["properties"]
    assert coll_props == seq_props == expected
    assert "NaN" not in collection.output
    assert "Infinity" not in collection.output


def test_cli_nodata_accepts_decimal_and_keeps_large_ints():
    assert NODATA.convert("2", None, None) == 2
    assert isinstance(NODATA.convert("2", None, None), int)
    assert NODATA.convert("2.5", None, None) == 2.5
    big = 2**53 + 1
    converted = NODATA.convert(str(big), None, None)
    assert converted == big and isinstance(converted, int)


def test_cli_point_invalid_property_is_null(tmp_path):
    path = write_tif(
        tmp_path / "grid.tif", np.array([[np.nan, 2], [3, 4]], dtype="float32"),
        nodata=-99,
    )
    vector, _ = _write_feature(
        Point(*CENTRES[(0, 0)]).__geo_interface__, tmp_path / "pt.geojson"
    )
    result = CliRunner().invoke(
        pointquery,
        [vector, "--raster", str(path), "--interpolate", "nearest"],
    )
    assert result.exit_code == 0, result.output
    value = json.loads(result.output)["features"][0]["properties"]["value"]
    assert value is None
    assert "NaN" not in result.output


def test_bounded_and_out_of_bounds_windows_are_checked():
    arr = np.array([[0, 2], [3, 4]], dtype="uint8")
    with Raster(arr, AFFINE) as rast:
        # A fully inside window is allowed without padding.
        inside = rast.read(window=((0, 2), (0, 2)), boundless=False)
        assert inside.in_bounds_mask.all()
        # A window beyond the extent is rejected when boundless=False.
        with pytest.raises(ValueError):
            rast.read(window=((-1, 1), (0, 2)), boundless=False)
        # With boundless=True the padded slots are invalid and out of bounds.
        padded = rast.read(window=((-1, 1), (0, 2)))
        assert padded.in_bounds_mask[0].tolist() == [False, False]
        assert padded.in_bounds_mask[1].tolist() == [True, True]


def test_band_selection_reads_the_requested_band(tmp_path):
    path = tmp_path / "bands.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=2, width=2, count=2,
        dtype="int16", transform=AFFINE,
    ) as dst:
        dst.write(np.array([[1, 2], [3, 4]], dtype="int16"), 1)
        dst.write(np.array([[9, 9], [9, 9]], dtype="int16"), 2)

    assert stats_for(path, band=1, stats="count sum") == {"count": 4, "sum": 10.0}
    result = zonal_stats(WHOLE, str(path), affine=None, band=2, stats="count sum")
    assert result[0] == {"count": 4, "sum": 36.0}


def test_explicit_and_implicit_marker_are_distinct_for_files(tmp_path):
    path = write_tif(
        tmp_path / "grid.tif", np.array([[-99, 2], [3, 4]], dtype="int16"),
        nodata=-99,
    )
    implicit = stats_for(path, stats="count sum nodata")
    assert implicit["count"] == 3
    assert implicit["sum"] == 9
    assert implicit["nodata"] == 1

    explicit = stats_for(path, nodata=-99, stats="count sum nodata")
    assert explicit == implicit

    overridden = stats_for(path, nodata=2, stats="count sum nodata")
    assert overridden["count"] == 3
    assert overridden["sum"] == -92
    assert overridden["nodata"] == 1


# --- Environment-independent sanity -----------------------------------------


def test_synthetic_affine_matches_documented_layout():
    # Guards the hand-computed geometry used by every test above.
    for (row, col), (x, y) in CENTRES.items():
        recovered_col, recovered_row = ~AFFINE * (x, y)
        assert math.floor(recovered_col) == col
        assert math.floor(recovered_row) == row
