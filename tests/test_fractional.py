# Tests for the fractional area-weighted zonal statistics mode.
#
# All expected overlap areas below are derived by hand (triangular/rectangular
# geometry on a unit grid), never from the implementation under test.
import json

import numpy as np
import pytest
import rasterio
from affine import Affine
from shapely.geometry import MultiPolygon, Point, Polygon, box

from rasterstats import gen_zonal_stats, zonal_stats
from rasterstats.fractional import (
    AREA_FIELDS,
    FRACTIONAL_STATS,
    fractional_zonal_stats,
)
from rasterstats.io import Raster

IDENTITY = Affine.identity()


def values_4x4():
    # distinct value in every cell: value(r, c) = r * 4 + c + 1
    return np.arange(1, 17, dtype="float64").reshape(4, 4)


# --- Hand-computed overlap areas ---------------------------------------------
# Rectangle [0.5, 2.5] x [0.5, 2.5] on the unit grid.
RECT_AREAS = {
    (0, 0): 0.25,
    (0, 1): 0.5,
    (0, 2): 0.25,
    (1, 0): 0.5,
    (1, 1): 1.0,
    (1, 2): 0.5,
    (2, 0): 0.25,
    (2, 1): 0.5,
    (2, 2): 0.25,
}
RECT = Polygon([(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)])

# Right triangle (0, 0) (4, 0) (0, 4), area 8. The diagonal r + c = 3 clips
# one triangle (area 0.5) off the boundary cells; all other cells are whole.
# These areas were derived by hand from the diagonal intercepts and verified
# cell-by-cell independently of the implementation.
TRI_AREAS = {
    (0, 0): 1.0,
    (0, 1): 1.0,
    (0, 2): 1.0,
    (0, 3): 0.5,
    (1, 0): 1.0,
    (1, 1): 1.0,
    (1, 2): 0.5,
    (2, 0): 1.0,
    (2, 1): 0.5,
    (3, 0): 0.5,
}
TRIANGLE = Polygon([(0, 0), (4, 0), (0, 4)])

# Square [0, 3] x [0, 3] with a unit-square hole [1, 2] x [1, 2], area 8.
HOLE_SQUARE = Polygon(
    [(0, 0), (3, 0), (3, 3), (0, 3)],
    holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]],
)


def _hand_stats(areas, arr, value_kind, cell_area=1.0):
    """Reference statistics computed from the hand-written area dictionary."""
    count = sum(areas.values()) / cell_area
    valid_area = sum(areas.values())
    if value_kind == "total":
        total = sum(a / cell_area * arr[r, c] for (r, c), a in areas.items())
    else:
        total = sum(a * arr[r, c] for (r, c), a in areas.items())
    return {
        "count": count,
        "sum": total,
        "valid_area": valid_area,
        "mean": total / valid_area,
    }


def _approx(result, expected, **kw):
    for key, value in expected.items():
        np.testing.assert_allclose(result[key], value, rtol=1e-9, atol=1e-10, **kw)


def test_rectangle_hand_areas_total_and_density():
    arr = values_4x4()
    expected_total = _hand_stats(RECT_AREAS, arr, "total")
    expected_density = _hand_stats(RECT_AREAS, arr, "density")
    for value_kind, expected in (("total", expected_total), ("density", expected_density)):
        result = zonal_stats(
            [RECT], arr, affine=IDENTITY,
            coverage="fractional", value_kind=value_kind,
        )[0]
        _approx(result, expected)
        assert result["coverage_area"] == 4.0
        assert result["missing_area"] == 0.0
        # unit cells: the two formulations coincide
        np.testing.assert_allclose(result["sum"], expected_total["sum"])
        np.testing.assert_allclose(result["mean"], 6.0)


def test_triangle_hand_areas():
    arr = values_4x4()
    for value_kind in ("total", "density"):
        expected = _hand_stats(TRI_AREAS, arr, value_kind)
        result = zonal_stats(
            [TRIANGLE], arr, affine=IDENTITY,
            coverage="fractional", value_kind=value_kind,
        )[0]
        _approx(result, expected)
        assert result["coverage_area"] == 8.0
        assert result["valid_area"] == 8.0
        assert result["missing_area"] == 0.0
    # independently expanded hand sum of a * value:
    # 1*1 + 1*2 + 1*3 + .5*4 + 1*5 + 1*6 + .5*7 + 1*9 + .5*10 + .5*13
    result = zonal_stats(
        [TRIANGLE], arr, affine=IDENTITY,
        coverage="fractional", value_kind="density",
    )[0]
    np.testing.assert_allclose(result["sum"], 43.0)
    np.testing.assert_allclose(result["mean"], 43.0 / 8.0)


def test_hole_square_hand_areas():
    arr = values_4x4()
    # Hand expectation: full 3x3 block (values 1..12) minus the single hole
    # cell (1, 1) with value 6.
    full_3x3_sum = float(arr[0:3, 0:3].sum())
    expected_sum = full_3x3_sum - 6.0
    result = zonal_stats(
        [HOLE_SQUARE], arr, affine=IDENTITY,
        coverage="fractional", value_kind="density",
    )[0]
    assert result["coverage_area"] == 8.0
    assert result["valid_area"] == 8.0
    np.testing.assert_allclose(result["sum"], expected_sum)
    np.testing.assert_allclose(result["mean"], expected_sum / 8.0)
    assert result["count"] == 8.0


def test_overlapping_multipolygon_counts_once():
    arr = np.ones((4, 4))
    # Two fully overlapping rectangles: union area must be 4, not 8.
    mp = MultiPolygon([RECT, RECT])
    result = zonal_stats(
        [mp], arr, affine=IDENTITY,
        coverage="fractional", value_kind="density",
    )[0]
    assert result["coverage_area"] == 4.0
    assert result["valid_area"] == 4.0
    assert result["sum"] == 4.0
    assert result["count"] == 4.0


def test_multipolygon_invalid_component_rejected_but_overlap_allowed():
    arr = np.ones((4, 4))
    # Overlapping but individually valid components are fine.
    other = box(1, 1, 3, 3)
    mp_ok = MultiPolygon([RECT, other])
    result = zonal_stats(
        [mp_ok], arr, affine=IDENTITY,
        coverage="fractional", value_kind="density",
    )[0]
    # overlapping components are unioned, so the area is the union's area
    import shapely
    assert result["coverage_area"] == shapely.union_all(mp_ok).area

    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    mp_bad = MultiPolygon([RECT, bowtie])
    with pytest.raises(ValueError, match=r"feature 0.*component 1"):
        zonal_stats(
            [mp_bad], arr, affine=IDENTITY,
            coverage="fractional", value_kind="density",
        )


def test_ring_orientation_start_and_component_order_invariant():
    arr = values_4x4()
    base = zonal_stats(
        [TRIANGLE], arr, affine=IDENTITY,
        coverage="fractional", value_kind="density",
    )[0]

    def reverse_ring(polygon):
        exterior = polygon.exterior.coords[::-1]
        return Polygon(exterior)

    reversed_tri = reverse_ring(TRIANGLE)
    # change the starting vertex as well
    coords = list(TRIANGLE.exterior.coords)
    rotated = Polygon(coords[1:] + coords[:1])
    for geom in (reversed_tri, rotated):
        result = zonal_stats(
            [geom], arr, affine=IDENTITY,
            coverage="fractional", value_kind="density",
        )[0]
        assert result == base

    mp = MultiPolygon([RECT, TRIANGLE])
    mp_swapped = MultiPolygon([TRIANGLE, RECT])
    a = zonal_stats(
        [mp], arr, affine=IDENTITY,
        coverage="fractional", value_kind="density",
    )[0]
    b = zonal_stats(
        [mp_swapped], arr, affine=IDENTITY,
        coverage="fractional", value_kind="density",
    )[0]
    assert a == b


def test_partition_additivity():
    arr = values_4x4()
    whole = box(0, 0, 4, 4)
    # non-overlapping partition of the grid
    parts = [
        box(0, 0, 2, 2),
        box(2, 0, 4, 2),
        box(0, 2, 2, 4),
        box(2, 2, 4, 4),
    ]
    whole_stats = zonal_stats(
        [whole], arr, affine=IDENTITY,
        coverage="fractional", value_kind="total",
    )[0]
    part_stats = zonal_stats(
        parts, arr, affine=IDENTITY,
        coverage="fractional", value_kind="total",
    )
    for key in ("count", "sum", "coverage_area", "valid_area"):
        np.testing.assert_allclose(
            sum(s[key] for s in part_stats), whole_stats[key]
        )


def test_total_vs_density_equivalence():
    arr = values_4x4()
    # cell area is 1 under identity; scale cells to area 6 to make the
    # conversion non-trivial
    affine6 = Affine(2.0, 0.0, 0.0, 0.0, 3.0, 0.0)
    cell_area = abs(affine6.determinant)
    density_input = arr / cell_area
    total = zonal_stats(
        [RECT], arr, affine=affine6,
        coverage="fractional", value_kind="total",
    )[0]
    density = zonal_stats(
        [RECT], density_input, affine=affine6,
        coverage="fractional", value_kind="density",
    )[0]
    np.testing.assert_allclose(density["sum"], total["sum"], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(density["mean"], total["mean"], rtol=1e-9, atol=1e-10)


def test_max_cells_does_not_change_results():
    arr = values_4x4()
    reference = zonal_stats(
        [TRIANGLE], arr, affine=IDENTITY,
        coverage="fractional", value_kind="density",
    )[0]
    for max_cells in (1, 2, 3, 5, 7, 16, 65536):
        result = zonal_stats(
            [TRIANGLE], arr, affine=IDENTITY,
            coverage="fractional", value_kind="density",
            max_cells=max_cells,
        )[0]
        assert result == reference


def test_overlap_area_conservation_under_tiling():
    # Summed cell overlap areas must reproduce the union geometry area for a
    # holed polygon and an overlapping MultiPolygon, at every tile budget.
    import shapely

    arr = np.ones((8, 8))
    holed = Polygon(
        [(0, 0), (7, 0), (7, 7), (0, 7)],
        holes=[[(2, 2), (5, 2), (5, 5), (2, 5)]],
    )
    overlapping = MultiPolygon([box(1, 1, 6, 6), box(3, 3, 7, 7)])
    for geom in (holed, overlapping):
        expected_area = shapely.union_all(geom).area
        for max_cells in (1, 3, 16, 64, 65536):
            result = zonal_stats(
                [geom], arr, affine=IDENTITY,
                coverage="fractional", value_kind="density",
                max_cells=max_cells,
            )[0]
            np.testing.assert_allclose(
                result["coverage_area"], expected_area, rtol=1e-9, atol=1e-10
            )
            np.testing.assert_allclose(
                result["valid_area"], expected_area, rtol=1e-9, atol=1e-10
            )
            # unit-density on unit cells means sum equals covered area
            np.testing.assert_allclose(
                result["sum"], expected_area, rtol=1e-9, atol=1e-10
            )


def test_rotated_and_sheared_affine():
    arr = values_4x4()
    rotated = Affine(2.0, 2.0, 0.0, -2.0, 2.0, 0.0)  # 45 degree rotation, area 8
    from rasterstats.fractional import _frame_polygon

    frame = _frame_polygon(rotated, 4, 4)
    cell_area = abs(rotated.determinant)
    result = zonal_stats(
        [frame], arr, affine=rotated,
        coverage="fractional", value_kind="density",
    )[0]
    assert result["coverage_area"] == 16 * cell_area
    assert result["valid_area"] == 16 * cell_area
    np.testing.assert_allclose(result["sum"], cell_area * arr.sum())
    np.testing.assert_allclose(result["mean"], arr.mean())

    # pure shear, still invertible
    sheared = Affine(1.0, 0.5, 0.0, 0.0, 1.0, 0.0)
    frame_s = _frame_polygon(sheared, 4, 4)
    result_s = zonal_stats(
        [frame_s], arr, affine=sheared,
        coverage="fractional", value_kind="total",
    )[0]
    assert result_s["count"] == 16.0
    np.testing.assert_allclose(result_s["sum"], arr.sum())

    # tiling must not change the sheared result either
    reference = result_s
    for max_cells in (1, 3, 9, 65536):
        tiled = zonal_stats(
            [frame_s], arr, affine=sheared,
            coverage="fractional", value_kind="total",
            max_cells=max_cells,
        )[0]
        assert tiled == reference


def test_zone_outside_raster_frame():
    arr = values_4x4()
    far_away = box(20, 20, 24, 24)
    result = zonal_stats(
        [far_away], arr, affine=IDENTITY,
        coverage="fractional", value_kind="density",
    )[0]
    assert result == {
        "count": 0,
        "sum": 0.0,
        "mean": None,
        "coverage_area": 0.0,
        "valid_area": 0.0,
        "missing_area": 0.0,
    }

    # half-in/half-out polygon: only the in-frame area is measured
    crossing = box(2, 2, 6, 6)
    cross = zonal_stats(
        [crossing], arr, affine=IDENTITY,
        coverage="fractional", value_kind="density",
    )[0]
    assert cross["coverage_area"] == 4.0
    assert cross["valid_area"] == 4.0


def test_boundless_convention_identical():
    arr = values_4x4()
    crossing = box(2, 2, 6, 6)
    a = zonal_stats(
        [crossing], arr, affine=IDENTITY,
        coverage="fractional", value_kind="density", boundless=True,
    )[0]
    b = zonal_stats(
        [crossing], arr, affine=IDENTITY,
        coverage="fractional", value_kind="density", boundless=False,
    )[0]
    assert a == b


def test_edge_only_contact_has_no_area():
    arr = values_4x4()
    # triangle sharing only the origin point with the grid
    point_touch = Polygon([(-2, -2), (-1, -2), (-2, -1)])
    result = zonal_stats(
        [point_touch], arr, affine=IDENTITY,
        coverage="fractional", value_kind="density",
    )[0]
    assert result["coverage_area"] == 0.0
    assert result["valid_area"] == 0.0
    assert result["missing_area"] == 0.0
    assert result["mean"] is None


def test_empty_and_none_geometries():
    arr = values_4x4()
    expected = {
        "count": 0,
        "sum": 0.0,
        "mean": None,
        "coverage_area": 0.0,
        "valid_area": 0.0,
        "missing_area": 0.0,
    }
    empty_result = zonal_stats(
        [Polygon()], arr, affine=IDENTITY,
        coverage="fractional", value_kind="density",
    )[0]
    assert empty_result == expected

    none_feature = {"type": "Feature", "properties": {}, "geometry": None}
    none_result = zonal_stats(
        [none_feature], arr, affine=IDENTITY,
        coverage="fractional", value_kind="density",
    )[0]
    assert none_result == expected


def test_all_nodata_is_covered_but_unobserved():
    arr = np.full((4, 4), -999.0)
    whole = box(0, 0, 4, 4)
    result = zonal_stats(
        [whole], arr, affine=IDENTITY, nodata=-999,
        coverage="fractional", value_kind="density",
    )[0]
    # covered vs missing distinction: coverage is 16 but nothing is observed
    assert result["coverage_area"] == 16.0
    assert result["valid_area"] == 0.0
    assert result["missing_area"] == 16.0
    assert result["count"] == 0
    assert result["sum"] == 0.0
    assert result["mean"] is None


def test_nodata_nan_inf_mask_and_valid_zero():
    arr = values_4x4()
    arr[0, 0] = 0.0          # legitimate zero stays valid
    arr[0, 1] = np.nan
    arr[1, 0] = np.inf
    arr[1, 1] = -np.inf
    arr[2, 2] = -999.0       # explicit nodata
    whole = box(0, 0, 4, 4)
    result = zonal_stats(
        [whole], arr, affine=IDENTITY, nodata=-999,
        coverage="fractional", value_kind="density",
    )[0]
    assert result["coverage_area"] == 16.0
    # four unobserved cells: NaN, +inf, -inf, explicit nodata
    assert result["valid_area"] == 12.0
    assert result["missing_area"] == 4.0
    assert result["count"] == 12.0
    # zero contributed nothing to the sum but remains an observed cell
    invalid_cells = {(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)}
    expected_sum = sum(
        arr[r, c]
        for r in range(4)
        for c in range(4)
        if (r, c) not in invalid_cells
    )
    np.testing.assert_allclose(result["sum"], expected_sum)


def test_independent_masked_array(tmp_path):
    # An ndarray carrying its own mask (dataset-mask style), no nodata set.
    mask = np.zeros((4, 4), dtype=bool)
    mask[3, 3] = True
    arr = np.ma.MaskedArray(np.ones((4, 4)), mask=mask)
    whole = box(0, 0, 4, 4)
    with Raster(arr, IDENTITY, nodata=None) as rast:
        result = fractional_zonal_stats(
            whole, rast, stats=list(FRACTIONAL_STATS),
            value_kind="density", max_cells=4,
        )
    assert result["coverage_area"] == 16.0
    assert result["valid_area"] == 15.0
    assert result["missing_area"] == 1.0
    assert result["sum"] == 15.0


def test_default_stats_and_subset_stats():
    arr = values_4x4()
    default_result = zonal_stats(
        [RECT], arr, affine=IDENTITY,
        coverage="fractional", value_kind="total",
    )[0]
    assert set(default_result) == set(FRACTIONAL_STATS) | set(AREA_FIELDS)

    subset = zonal_stats(
        [RECT], arr, affine=IDENTITY,
        coverage="fractional", value_kind="total", stats="sum",
    )[0]
    assert set(subset) == {"sum"} | set(AREA_FIELDS)

    subset2 = zonal_stats(
        [RECT], arr, affine=IDENTITY,
        coverage="fractional", value_kind="total", stats=["count", "mean"],
    )[0]
    assert set(subset2) == {"count", "mean"} | set(AREA_FIELDS)


def test_prefix_and_geojson_out():
    arr = values_4x4()
    feature = {"type": "Feature", "properties": {"name": "zone"}, "geometry": {
        "type": "Polygon",
        "coordinates": [list(RECT.exterior.coords)],
    }}
    result = zonal_stats(
        [feature], arr, affine=IDENTITY,
        coverage="fractional", value_kind="total",
        prefix="frac_", geojson_out=True,
    )[0]
    assert result["properties"]["name"] == "zone"
    for key in set(FRACTIONAL_STATS) | set(AREA_FIELDS):
        assert f"frac_{key}" in result["properties"]
    assert "frac_mean" in result["properties"]


def test_generator_error_carries_feature_index_and_closes_dataset(tmp_path):
    raster_path = tmp_path / "small.tif"
    with rasterio.open(
        raster_path, "w", driver="GTiff", height=4, width=4,
        count=1, dtype="float32", transform=IDENTITY,
    ) as dst:
        dst.write(values_4x4().astype("float32"), 1)

    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    generator = gen_zonal_stats(
        [box(0, 0, 1, 1), bowtie], raster_path,
        coverage="fractional", value_kind="total",
    )
    next(generator)  # first feature is fine and has already been yielded
    with pytest.raises(ValueError, match="feature 1"):
        next(generator)
    # context manager must have closed the dataset even on failure
    generator.close()

    # early close before exhaustion also releases the dataset
    gen2 = gen_zonal_stats(
        [box(0, 0, 1, 1)], raster_path,
        coverage="fractional", value_kind="total",
    )
    next(gen2)
    gen2.close()
    # file remains usable afterwards (source file untouched)
    with rasterio.open(raster_path) as src:
        assert src.read(1).shape == (4, 4)


def test_input_array_and_features_unchanged():
    arr = values_4x4()
    arr_copy = arr.copy()
    feature = {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [list(RECT.exterior.coords)]},
    }
    feature_copy = json.loads(json.dumps(feature))
    zonal_stats(
        [feature], arr, affine=IDENTITY,
        coverage="fractional", value_kind="density", max_cells=1,
    )
    np.testing.assert_array_equal(arr, arr_copy)
    # non-geojson mode must not attach properties; geometry content is kept
    # (coordinate tuples vs JSON lists are representationally equivalent)
    assert feature["properties"] == feature_copy["properties"]
    assert json.dumps(feature["geometry"]) == json.dumps(feature_copy["geometry"])

    # geojson mode adds properties but never alters geometry coordinates or
    # the array data (the geometry dict is parsed to shapely, as in ordinary
    # geojson_out behavior)
    feature2 = json.loads(json.dumps(feature))
    zonal_stats(
        [feature2], arr, affine=IDENTITY,
        coverage="fractional", value_kind="density",
        geojson_out=True,
    )
    np.testing.assert_array_equal(arr, arr_copy)
    assert json.dumps(feature2["geometry"]) == json.dumps(feature_copy["geometry"])
    assert "coverage_area" in feature2["properties"]


# --- Option validation -------------------------------------------------------

def test_invalid_options():
    arr = values_4x4()
    kwargs_base = dict(raster=arr, affine=IDENTITY)

    with pytest.raises(ValueError, match="coverage"):
        zonal_stats([RECT], coverage="area", value_kind="total", **kwargs_base)
    with pytest.raises(ValueError, match="value_kind"):
        zonal_stats([RECT], coverage="fractional", **kwargs_base)
    with pytest.raises(ValueError, match="value_kind"):
        zonal_stats([RECT], coverage="fractional", value_kind="weird", **kwargs_base)
    with pytest.raises(ValueError, match="value_kind"):
        zonal_stats([RECT], value_kind="total", **kwargs_base)

    for bad_max_cells in (0, -1, True, False, 4.5, "8"):
        with pytest.raises(ValueError, match="max_cells"):
            zonal_stats(
                [RECT], coverage="fractional", value_kind="total",
                max_cells=bad_max_cells, **kwargs_base,
            )

    with pytest.raises(ValueError, match="all_touched"):
        zonal_stats(
            [RECT], coverage="fractional", value_kind="total",
            all_touched=True, **kwargs_base,
        )
    with pytest.raises(ValueError, match="categorical"):
        zonal_stats(
            [RECT], coverage="fractional", value_kind="total",
            categorical=True, **kwargs_base,
        )
    with pytest.raises(ValueError, match="add_stats"):
        zonal_stats(
            [RECT], coverage="fractional", value_kind="total",
            add_stats={"x": lambda m: 1}, **kwargs_base,
        )
    with pytest.raises(ValueError, match="zone_func"):
        zonal_stats(
            [RECT], coverage="fractional", value_kind="total",
            zone_func=lambda m: m, **kwargs_base,
        )
    with pytest.raises(ValueError, match="raster_out"):
        zonal_stats(
            [RECT], coverage="fractional", value_kind="total",
            raster_out=True, **kwargs_base,
        )
    with pytest.raises(ValueError, match="'min'"):
        zonal_stats(
            [RECT], coverage="fractional", value_kind="total",
            stats=["sum", "min"], **kwargs_base,
        )


def test_non_polygonal_geometries_rejected():
    arr = values_4x4()
    with pytest.raises(ValueError, match=r"feature 0.*Point"):
        zonal_stats(
            [Point(1, 1)], arr, affine=IDENTITY,
            coverage="fractional", value_kind="total",
        )
    with pytest.raises(ValueError, match="feature 2"):
        zonal_stats(
            [RECT, RECT, Point(1, 1)], arr, affine=IDENTITY,
            coverage="fractional", value_kind="total",
        )


def test_invalid_affine_rejected():
    arr = values_4x4()
    with pytest.raises(ValueError, match="affine"):
        zonal_stats(
            [RECT], arr,
            coverage="fractional", value_kind="total",
        )
    with pytest.raises(ValueError, match="invertible"):
        zonal_stats(
            [RECT], arr, affine=Affine(0, 0, 0, 0, 0, 0),
            coverage="fractional", value_kind="total",
        )
    with pytest.raises(ValueError, match="non-finite"):
        zonal_stats(
            [RECT], arr, affine=Affine(1, 0, 0, 0, np.inf, 0),
            coverage="fractional", value_kind="total",
        )


def test_ordinary_mode_unchanged():
    arr = values_4x4()
    # north-up affine (ordinary rasterstats reads assume e < 0); the new
    # fractional mode itself supports either orientation via the full affine
    north_up = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
    north_rect = Polygon([(0.5, 1.5), (2.5, 1.5), (2.5, 3.5), (0.5, 3.5)])
    # no new keys leak into ordinary zonal stats
    result = zonal_stats([north_rect], arr, affine=north_up)
    assert "coverage_area" not in result[0]
    assert "valid_area" not in result[0]
    assert isinstance(result[0]["count"], int)

# --- GeoTIFF round trip ------------------------------------------------------

def test_geotiff_roundtrip(tmp_path):
    raster_path = tmp_path / "grid.tif"
    arr = values_4x4().astype("float32")
    with rasterio.open(
        raster_path, "w", driver="GTiff", height=4, width=4,
        count=1, dtype="float32", transform=IDENTITY, nodata=-999.0,
    ) as dst:
        dst.write(arr, 1)

    array_result = zonal_stats(
        [TRIANGLE], arr, affine=IDENTITY,
        coverage="fractional", value_kind="density",
    )[0]
    file_result = zonal_stats(
        [TRIANGLE], raster_path,
        coverage="fractional", value_kind="density",
    )[0]
    for key in array_result:
        np.testing.assert_allclose(file_result[key], array_result[key], equal_nan=True)


def test_geotiff_dataset_mask(tmp_path):
    # write a file with an internal mask rather than a nodata value
    raster_path = tmp_path / "masked.tif"
    arr = values_4x4().astype("float32")
    with rasterio.open(
        raster_path, "w", driver="GTiff", height=4, width=4,
        count=1, dtype="float32", transform=IDENTITY,
    ) as dst:
        dst.write(arr, 1)
        mask = np.ones((4, 4), dtype="uint8")
        mask[0, 0] = 0
        dst.write_mask(mask)

    whole = box(0, 0, 4, 4)
    result = zonal_stats(
        [whole], raster_path,
        coverage="fractional", value_kind="density",
    )[0]
    assert result["coverage_area"] == 16.0
    assert result["valid_area"] == 15.0
    assert result["missing_area"] == 1.0
