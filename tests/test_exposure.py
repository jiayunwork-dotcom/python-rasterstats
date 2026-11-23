"""Tests for continuous along-line exposure analysis.

The expected mileages and integrals in this module are derived by hand (the
bilinear field restricted to a straight segment is a quadratic in arc
length, so crossings and antiderivatives are computed analytically); the
solver under test is never used to generate the reference values.

Test rasters are padded so each hand-computed route lies inside a single
interpolation grid, which keeps the analytic references free of grid-boundary bookkeeping.
"""
import json
import math

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from rasterstats import gen_point_query, point_query
from rasterstats.exposure import (
    ABOVE,
    BELOW,
    MISSING,
    analyze_geometry,
    cut_line_coordinates,
    validate_distance_tol,
    validate_exposure_threshold,
    validate_read_cells,
)
from rasterstats.io import Raster

IDENTITY = Affine.identity()


def raster(array, affine=IDENTITY, nodata=None, masked=False):
    if masked:
        array = np.ma.masked_invalid(array)
    return Raster(array, affine=affine, nodata=nodata)


def analyze(line, array, threshold, affine=IDENTITY, nodata=None, masked=False,
            distance_tol=1e-8, read_cells=4096):
    with raster(array, affine, nodata=nodata, masked=masked) as rast:
        return analyze_geometry(
            line, rast, threshold, distance_tol, read_cells, feature_index=0
        )


def interval_kinds(part):
    return [iv["kind"] for iv in part["intervals"]]


def check_coverage(part):
    """Intervals cover [0, total] exactly, without overlap or gaps."""
    assert part["intervals"]
    assert part["intervals"][0]["start"] == 0.0
    assert part["intervals"][-1]["stop"] == part["total_length"]
    for left, right in zip(part["intervals"][:-1], part["intervals"][1:]):
        assert left["stop"] == right["start"]
    lengths = sum(iv["stop"] - iv["start"] for iv in part["intervals"])
    assert lengths == pytest.approx(part["total_length"], abs=1e-12)
    assert part["valid_length"] == pytest.approx(
        sum(
            iv["stop"] - iv["start"]
            for iv in part["intervals"]
            if iv["kind"] != MISSING
        ),
        abs=1e-12,
    )
    assert part["missing_length"] == pytest.approx(
        sum(
            iv["stop"] - iv["start"]
            for iv in part["intervals"]
            if iv["kind"] == MISSING
        ),
        abs=1e-12,
    )
    assert part["above_length"] == pytest.approx(
        sum(
            iv["stop"] - iv["start"]
            for iv in part["intervals"]
            if iv["kind"] == ABOVE
        ),
        abs=1e-12,
    )
    known_integral = sum(
        iv["integral"] for iv in part["intervals"] if iv["kind"] != MISSING
    )
    assert part["integral"] == pytest.approx(known_integral, abs=1e-12)
    for iv in part["intervals"]:
        if iv["kind"] == MISSING:
            assert iv["integral"] is None


def pad(corners, size=7, fill=1.0):
    """Embed a 2x2 corner pattern at cells (2,2)..(3,3) of a padded raster.

    The surrounding fill cells make every grid touched by the diagonal test
    route valid; routes stay inside grid (2,2) or cross only into filled
    grids, so the hand-computed quadratic over the embedded grid is exact.
    """
    arr = np.full((size, size), fill)
    arr[2, 2], arr[2, 3] = corners[0], corners[1]
    arr[3, 2], arr[3, 3] = corners[2], corners[3]
    return arr


# Diagonal route strictly inside grid (2,2): fractional t in [0.1, 0.9],
# so the route never crosses a grid line and the hand-computed quadratic
# over the embedded 2x2 corners is the exact field on the whole route.
DIAG_START = (2.1, 2.1)
DIAG_END = (2.9, 2.9)
DIAG_T0, DIAG_T1 = 0.1, 0.9
SQRT2 = math.sqrt(2.0)


def diag_line():
    return LineString([DIAG_START, DIAG_END])


# ---------------------------------------------------------------------------
# Headline case: both vertices below, threshold exceeded in the middle
# ---------------------------------------------------------------------------

def test_diagonal_midline_exceedance_both_vertices_below():
    # The headline case: all four corners differ and BOTH input vertices are
    # below the threshold while the interior of the single segment exceeds it.
    # Along fu=fv=t, corners a=4,b=12,c=10,d=5 give
    # p(t) = 4 + 14t - 13t^2, a concave parabola. At t=.1 and t=.9 the values
    # (5.27 and 6.07) are both below 6.2, but the parabola peaks at 7.77, so
    # the threshold is crossed twice strictly between the two vertices.
    arr = pad([4.0, 12.0, 10.0, 5.0])
    threshold = 6.2
    result = analyze(diag_line(), arr, threshold)
    part = result["parts"][0]
    # roots of 4 + 14t - 13t^2 = 6.2  ->  13t^2 - 14t + 2.2 = 0
    discriminant = 14.0**2 - 4.0 * 13.0 * 2.2
    t1 = (14.0 - math.sqrt(discriminant)) / 26.0
    t2 = (14.0 + math.sqrt(discriminant)) / 26.0
    assert DIAG_T0 < t1 < t2 < DIAG_T1
    assert interval_kinds(part) == [BELOW, ABOVE, BELOW]
    assert part["intervals"][1]["start"] == pytest.approx(
        (t1 - DIAG_T0) * SQRT2, abs=1e-10
    )
    assert part["intervals"][1]["stop"] == pytest.approx(
        (t2 - DIAG_T0) * SQRT2, abs=1e-10
    )
    assert part["missing_length"] == 0.0

    def antideriv(t):
        return 4.0 * t + 7.0 * t**2 - 13.0 * t**3 / 3.0

    expected = SQRT2 * (antideriv(DIAG_T1) - antideriv(DIAG_T0))
    assert part["integral"] == pytest.approx(expected, rel=1e-8, abs=1e-10)
    assert part["above_length"] == pytest.approx((t2 - t1) * SQRT2, abs=1e-10)
    check_coverage(part)


def test_threshold_crossed_twice_between_vertices():
    # Symmetric corners give p(t) = 8 - 16t + 16t^2 = 16(t-.5)^2 + 4.
    # With threshold 5 the route is above at both vertices and below in the
    # middle: two strict crossings between the two input vertices.
    arr = pad([8.0, 0.0, 0.0, 8.0])
    result = analyze(diag_line(), arr, 5.0)
    part = result["parts"][0]
    t1, t2 = 0.25, 0.75
    assert interval_kinds(part) == [ABOVE, BELOW, ABOVE]
    assert part["intervals"][1]["start"] == pytest.approx(
        (t1 - DIAG_T0) * SQRT2, abs=1e-10
    )
    assert part["intervals"][1]["stop"] == pytest.approx(
        (t2 - DIAG_T0) * SQRT2, abs=1e-10
    )
    def antideriv(t):
        return 16.0 * t**3 / 3.0 - 8.0 * t**2 + 8.0 * t
    expected = SQRT2 * (antideriv(DIAG_T1) - antideriv(DIAG_T0))
    assert part["integral"] == pytest.approx(expected, rel=1e-8, abs=1e-10)
    check_coverage(part)


def test_internal_tangency_keeps_one_kind_no_zero_record():
    # p(t) = 4 - (t-.5)^2 touches threshold 4 from below at t=.5 only.
    arr = pad([3.75, 4.25, 4.25, 3.75])
    result = analyze(diag_line(), arr, 4.0)
    part = result["parts"][0]
    assert interval_kinds(part) == [BELOW]
    assert len(part["intervals"]) == 1
    assert part["above_length"] == 0.0
    def antideriv(t):
        return 3.75 * t + 0.5 * t**2 - t**3 / 3.0
    expected = SQRT2 * (antideriv(DIAG_T1) - antideriv(DIAG_T0))
    assert part["integral"] == pytest.approx(expected, rel=1e-8, abs=1e-10)
    check_coverage(part)


def test_tangency_from_above_does_not_create_below_record():
    # p(t) = 4 + (t-.5)^2 touches threshold 4 from above at t=.5 only.
    arr = pad([4.25, 3.75, 3.75, 4.25])
    result = analyze(diag_line(), arr, 4.0)
    part = result["parts"][0]
    assert interval_kinds(part) == [ABOVE]
    assert len(part["intervals"]) == 1
    assert part["above_length"] == pytest.approx(part["total_length"])
    check_coverage(part)


# ---------------------------------------------------------------------------
# Constant, planar, masked, out-of-raster and degenerate-shape fields
# ---------------------------------------------------------------------------

def test_constant_field_single_interval():
    arr = np.full((8, 8), 3.0)
    line = LineString([(1.6, 2.25), (4.4, 2.25)])
    result = analyze(line, arr, 2.0)
    part = result["parts"][0]
    assert interval_kinds(part) == [ABOVE]
    assert part["total_length"] == pytest.approx(2.8)
    assert part["above_length"] == pytest.approx(2.8)
    assert part["missing_length"] == 0.0
    assert part["integral"] == pytest.approx(8.4, rel=1e-8, abs=1e-10)
    check_coverage(part)


def test_planar_field_threshold_at_boundary_value():
    # p(x, y) = x - 1 via column values -0.5, 0.5, 1.5, 2.5, 3.5, 4.5
    cols = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    arr = np.repeat(cols[None, :], 6, axis=0)
    line = LineString([(1.6, 2.25), (4.4, 2.25)])
    threshold = 2.0  # equality at the grid line x=2.5 (a cell-center row)
    result = analyze(line, arr, threshold)
    part = result["parts"][0]
    assert [iv["kind"] for iv in part["intervals"]] == [BELOW, ABOVE]
    # strictly greater only after x=2.5
    assert part["intervals"][1]["start"] == pytest.approx(2.5 - 1.6, abs=1e-10)
    assert part["above_length"] == pytest.approx(4.4 - 2.5)
    # integral of (x - 1) over x in [1.6, 4.4], dx == ds on this horizontal
    # route: F(x) = 0.5 x^2 - x, so F(4.4) - F(1.6) = 5.28 - (-1.72) = 7.0
    assert part["integral"] == pytest.approx(7.0, abs=1e-10)
    check_coverage(part)


def test_mask_band_splits_known_segments():
    arr = np.full((8, 8), 1.0)
    arr[3, 4] = 9.0
    arr[3, 5] = 9.0
    masked = np.ma.masked_array(arr, mask=np.zeros_like(arr, dtype=bool))
    masked.mask[3, 4] = True
    masked.mask[3, 5] = True
    # horizontal route on row line 3 with margin on both sides; the invalid
    # cells make every grid using them unknown, splitting the known parts
    line = LineString([(2.6, 3.25), (6.4, 3.25)])
    result = analyze(line, masked, 5.0, masked=True)
    part = result["parts"][0]
    kinds = interval_kinds(part)
    assert kinds[0] == BELOW
    assert MISSING in kinds
    assert kinds[-1] == BELOW
    missing = sum(
        iv["stop"] - iv["start"] for iv in part["intervals"] if iv["kind"] == MISSING
    )
    assert missing == pytest.approx(part["missing_length"])
    check_coverage(part)


def test_nodata_and_nonfinite_both_invalidate():
    arr = np.full((4, 7), 1.0)
    arr[2, 2] = -999.0
    arr[2, 4] = np.nan
    line = LineString([(1.6, 2.25), (6.4, 2.25)])
    result = analyze(line, arr, 0.5, nodata=-999.0)
    part = result["parts"][0]
    assert MISSING in interval_kinds(part)
    assert part["missing_length"] > 0.0
    check_coverage(part)


def test_zero_is_a_valid_observation():
    arr = np.zeros((8, 8))
    line = LineString([(1.6, 2.25), (4.4, 2.25)])
    result = analyze(line, arr, 1.0)
    part = result["parts"][0]
    assert interval_kinds(part) == [BELOW]
    assert part["valid_length"] == pytest.approx(2.8)
    assert part["integral"] == 0.0


def test_negative_concentrations_and_integral():
    # corners a=-4 b=-2 c=-2 d=0 -> p(t) = -4 + 4t (cross term cancels)
    arr = pad([-4.0, -2.0, -2.0, 0.0])
    result = analyze(diag_line(), arr, -1.0)
    part = result["parts"][0]
    t_cross = 0.75
    assert interval_kinds(part) == [BELOW, ABOVE]
    assert part["intervals"][1]["start"] == pytest.approx(
        (t_cross - DIAG_T0) * SQRT2, abs=1e-10
    )
    def antideriv(t):
        return -4.0 * t + 2.0 * t**2
    expected = SQRT2 * (antideriv(DIAG_T1) - antideriv(DIAG_T0))
    assert part["integral"] == pytest.approx(expected, rel=1e-8, abs=1e-10)
    check_coverage(part)


def test_outside_raster_half_cell_is_unknown():
    arr = np.ones((4, 4))
    # grid union reaches x=3.5 at an interior row line; extend beyond it
    line = LineString([(1.6, 2.25), (4.2, 2.25)])
    result = analyze(line, arr, 0.5)
    part = result["parts"][0]
    assert interval_kinds(part)[-1] == MISSING
    assert part["missing_length"] > 0.0
    check_coverage(part)


def test_single_row_raster_entirely_unknown():
    arr = np.ones((1, 6))
    line = LineString([(1.6, 0.5), (5.4, 0.5)])
    result = analyze(line, arr, 0.5)
    part = result["parts"][0]
    assert interval_kinds(part) == [MISSING]
    assert part["valid_length"] == 0.0
    assert part["integral"] == 0.0


def test_single_column_raster_entirely_unknown():
    arr = np.ones((6, 1))
    line = LineString([(0.5, 1.6), (0.5, 5.4)])
    result = analyze(line, arr, 0.5)
    part = result["parts"][0]
    assert interval_kinds(part) == [MISSING]


# ---------------------------------------------------------------------------
# Geometry bookkeeping
# ---------------------------------------------------------------------------

def test_inserted_vertices_do_not_change_result():
    arr = pad([0.0, 4.0, 4.0, 8.0])
    coarse = diag_line()
    dense = LineString(
        [
            DIAG_START,
            (2.3, 2.3),
            (2.5, 2.5),
            (2.7, 2.7),
            DIAG_END,
        ]
    )
    a = analyze(coarse, arr, 6.0)["parts"][0]
    b = analyze(dense, arr, 6.0)["parts"][0]
    assert interval_kinds(a) == interval_kinds(b)
    for key in ("total_length", "valid_length", "above_length", "integral"):
        assert a[key] == pytest.approx(b[key], abs=1e-10)
    for ia, ib in zip(a["intervals"], b["intervals"]):
        assert ia["start"] == pytest.approx(ib["start"], abs=1e-10)
        assert ia["stop"] == pytest.approx(ib["stop"], abs=1e-10)


def test_split_component_matches_joined_one():
    arr = pad([4.0, 12.0, 10.0, 5.0])
    joined = diag_line()
    # split at a point strictly inside the grid (t=.5), so neither component
    # introduces an extra grid-boundary piece
    split = MultiLineString(
        [
            [DIAG_START, (2.5, 2.5)],
            [(2.5, 2.5), DIAG_END],
        ]
    )
    res_joined = analyze(joined, arr, 6.0)
    res_split = analyze(split, arr, 6.0)
    assert len(res_split["parts"]) == 2
    for key in ("total_length", "valid_length", "above_length", "integral"):
        assert res_joined[key] == pytest.approx(res_split[key], abs=1e-10)
    joined_kinds = [iv["kind"] for iv in res_joined["parts"][0]["intervals"]]
    split_kinds = [iv["kind"] for p in res_split["parts"] for iv in p["intervals"]]
    # splitting may duplicate a boundary record but the ordered kinds agree
    # after merging adjacent equal kinds
    def merge(kinds):
        out = []
        for k in kinds:
            if not out or out[-1] != k:
                out.append(k)
        return out
    assert merge(split_kinds) == merge(joined_kinds)


def test_reversed_route_reflects_mileage_keeps_totals():
    arr = pad([0.0, 4.0, 4.0, 8.0])
    forward = diag_line()
    reverse = LineString([DIAG_END, DIAG_START])
    rf = analyze(forward, arr, 6.0)["parts"][0]
    rr = analyze(reverse, arr, 6.0)["parts"][0]
    length = rf["total_length"]
    assert rf["total_length"] == pytest.approx(rr["total_length"])
    assert rf["above_length"] == pytest.approx(rr["above_length"])
    assert rf["integral"] == pytest.approx(rr["integral"], rel=1e-8)
    reflected = [
        (length - iv["stop"], length - iv["start"], iv["kind"])
        for iv in reversed(rf["intervals"])
    ]
    actual = [(iv["start"], iv["stop"], iv["kind"]) for iv in rr["intervals"]]
    for (s0, s1, k0), (t0, t1, k1) in zip(reflected, actual):
        assert s0 == pytest.approx(t0, abs=1e-10)
        assert s1 == pytest.approx(t1, abs=1e-10)
        assert k0 == k1


def test_repeated_traversal_counts_twice():
    arr = np.full((9, 9), 2.0)
    line = LineString(
        [(1.6, 2.25), (5.4, 2.25), (1.6, 2.25)]
    )
    result = analyze(line, arr, 1.0)
    part = result["parts"][0]
    assert part["total_length"] == pytest.approx(7.6)
    assert part["above_length"] == pytest.approx(7.6)
    assert part["integral"] == pytest.approx(15.2)


def test_multilinestring_components_have_independent_mileage():
    arr = np.full((9, 9), 2.0)
    geom = MultiLineString(
        [
            [(1.6, 2.25), (4.4, 2.25)],
            [(2.0, 5.25), (4.8, 5.25)],
        ]
    )
    result = analyze(geom, arr, 1.0)
    assert len(result["parts"]) == 2
    for part in result["parts"]:
        assert part["intervals"][0]["start"] == 0.0
        assert part["total_length"] == pytest.approx(2.8)
    assert result["total_length"] == pytest.approx(5.6)
    assert result["integral"] == pytest.approx(11.2)


def test_duplicate_consecutive_vertices_allowed():
    arr = np.full((9, 9), 2.0)
    line = LineString(
        [(1.6, 2.25), (1.6, 2.25), (5.4, 2.25), (5.4, 2.25)]
    )
    part = analyze(line, arr, 1.0)["parts"][0]
    assert part["total_length"] == pytest.approx(3.8)
    assert interval_kinds(part) == [ABOVE]


def test_empty_and_zero_length_components():
    arr = np.full((9, 9), 2.0)
    geom = MultiLineString(
        [
            [(1.0, 1.0), (1.0, 1.0)],
            [(1.6, 2.25), (4.4, 2.25)],
        ]
    )
    result = analyze(geom, arr, 1.0)
    empty, real = result["parts"]
    assert empty["intervals"] == []
    assert empty["total_length"] == 0.0
    assert empty["integral"] == 0.0
    assert real["total_length"] == pytest.approx(2.8)
    assert result["total_length"] == pytest.approx(2.8)


def test_self_intersecting_route_keeps_branch_mileage():
    arr = np.full((15, 15), 2.0)
    # bow-tie strictly inside the grid union (fractional coords in [1, 13]);
    # the crossing point is reached on two different branches and mileage must
    # follow the input order rather than a map-coordinate reordering
    line = LineString(
        [(2.6, 2.6), (5.4, 5.4), (2.6, 5.4), (5.4, 2.6)]
    )
    part = analyze(line, arr, 1.0)["parts"][0]
    # three legs: two diagonals of length hypot(2.8, 2.8) and one horizontal
    # leg of length 2.8 (the input polyline has three segments, not four)
    diagonal = math.hypot(2.8, 2.8)
    expected = 2 * diagonal + 2.8
    assert part["total_length"] == pytest.approx(expected)
    assert interval_kinds(part) == [ABOVE]
    assert part["above_length"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Rotated / sheared transforms
# ---------------------------------------------------------------------------

def test_rotated_shear_affine_matches_axis_aligned_case():
    arr = pad([4.0, 12.0, 10.0, 5.0])
    theta = 0.37
    # a rigid rotation preserves ground arc lengths, so the mileage and
    # concentration line integral must match the identity case exactly
    affine = Affine(
        math.cos(theta), -math.sin(theta), 10.0,
        math.sin(theta), math.cos(theta), -5.0,
    )
    start = affine @ DIAG_START
    end = affine @ DIAG_END
    result = analyze(LineString([start, end]), arr, 6.0, affine=affine)
    reference = analyze(diag_line(), arr, 6.0)["parts"][0]
    part = result["parts"][0]
    assert part["total_length"] == pytest.approx(reference["total_length"], rel=1e-8)
    assert part["above_length"] == pytest.approx(reference["above_length"], rel=1e-8)
    assert part["integral"] == pytest.approx(reference["integral"], rel=1e-8)
    assert interval_kinds(part) == interval_kinds(reference)
    check_coverage(part)


def test_sheared_affine_keeps_field_classification():
    # a pure shear changes ground lengths (and thus mileages) but the field in
    # fractional row/column space is unchanged, so the kind sequence survives
    arr = pad([4.0, 12.0, 10.0, 5.0])
    shear = Affine(1.0, 0.3, 0.0, 0.0, 1.0, 0.0)
    start = shear @ DIAG_START
    end = shear @ DIAG_END
    result = analyze(LineString([start, end]), arr, 6.0, affine=shear)
    reference = analyze(diag_line(), arr, 6.0)["parts"][0]
    part = result["parts"][0]
    assert interval_kinds(part) == interval_kinds(reference)
    check_coverage(part)


def test_non_invertible_affine_rejected():
    singular = Affine(1.0, 2.0, 0.0, 2.0, 4.0, 0.0)
    arr = np.ones((4, 4))
    with pytest.raises(ValueError):
        analyze(LineString([(1.6, 1.6), (2.4, 2.4)]), arr, 0.5, affine=singular)


# ---------------------------------------------------------------------------
# Validation and error location
# ---------------------------------------------------------------------------

def test_threshold_validation():
    assert validate_exposure_threshold(None) is None
    assert validate_exposure_threshold(3) == 3.0
    assert validate_exposure_threshold(-2.5) == -2.5
    for bad in (True, False, float("nan"), float("inf"), "1.0"):
        with pytest.raises(ValueError):
            validate_exposure_threshold(bad)


def test_distance_tol_validation():
    assert validate_distance_tol(1.0) == 1.0
    for bad in (0.0, -1.0, True, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            validate_distance_tol(bad)


def test_read_cells_validation():
    assert validate_read_cells(4) == 4
    assert validate_read_cells(np.int64(4096)) == 4096
    for bad in (3, 0, -5, True, 1.5):
        with pytest.raises(ValueError):
            validate_read_cells(bad)


def test_non_line_geometry_rejected_with_feature_index():
    arr = np.ones((4, 4))
    with raster(arr) as rast:
        with pytest.raises(ValueError, match="feature 2"):
            analyze_geometry(Point(1, 1), rast, 0.5, 1e-8, 4096, 2)
        with pytest.raises(ValueError, match="feature 0"):
            analyze_geometry(
                Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
                rast, 0.5, 1e-8, 4096, 0,
            )


def test_non_finite_coordinates_located_by_component():
    from rasterstats.exposure import line_components

    # shapely rejects NaN vertices itself, so exercise the component
    # validation through minimal geometry-like objects: the analysis only
    # needs geom_type, is_empty, has_z and per-component .coords.
    class FakeLine:
        is_empty = False
        geom_type = "LineString"
        has_z = False

        def __init__(self, coords):
            self._coords = coords

        @property
        def coords(self):
            return iter(self._coords)

    class FakeMultiLine:
        geom_type = "MultiLineString"

        def __init__(self, components):
            self.geoms = [FakeLine(c) for c in components]

    good = [(2.6, 2.6), (3.4, 3.4)]
    bad = [(2.6, 4.25), (float("nan"), 4.25)]
    with pytest.raises(ValueError, match="feature 0 component 1"):
        line_components(FakeMultiLine([good, bad]), 0)

    # a non-finite vertex in the first component of another feature
    with pytest.raises(ValueError, match="feature 3 component 0"):
        line_components(FakeMultiLine([bad]), 3)


def test_three_dimensional_line_rejected():
    arr = np.ones((4, 4))
    line = LineString([(1.6, 1.6, 0.0), (2.4, 2.4, 0.0)])
    with pytest.raises(ValueError, match="2D"):
        analyze(line, arr, 0.5)


def test_nearest_interpolation_incompatible():
    arr = np.ones((4, 4))
    with pytest.raises(ValueError, match="nearest"):
        point_query(
            LineString([(1.6, 1.6), (2.4, 2.4)]),
            arr,
            affine=IDENTITY,
            exposure=1.0,
            interpolate="nearest",
        )


# ---------------------------------------------------------------------------
# Public API behavior
# ---------------------------------------------------------------------------

def test_point_query_exposure_result_shape():
    arr = np.full((9, 9), 2.0)
    line = LineString([(1.6, 2.25), (6.4, 2.25)])
    [result] = point_query(line, arr, affine=IDENTITY, exposure=1.0)
    assert set(result) == {
        "parts",
        "total_length",
        "valid_length",
        "missing_length",
        "above_length",
        "integral",
    }
    assert set(result["parts"][0]) == {
        "total_length",
        "valid_length",
        "missing_length",
        "above_length",
        "integral",
        "intervals",
    }


def test_point_query_without_exposure_unchanged():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    vals = point_query(
        LineString([(0.5, 0.5), (1.5, 1.5)]), arr, affine=IDENTITY
    )
    assert len(vals) == 1 and len(vals[0]) == 2
    assert all(isinstance(v, float) for v in vals[0])


def test_geojson_out_keeps_original_geometry_and_nests_analysis():
    arr = np.full((9, 9), 2.0)
    feature = {
        "type": "Feature",
        "geometry": LineString([(1.6, 2.25), (6.4, 2.25)]).__geo_interface__,
        "properties": {"name": "route-a"},
    }
    [out] = point_query(
        feature, arr, affine=IDENTITY, exposure=1.0, geojson_out=True
    )
    assert out["geometry"] == feature["geometry"]
    assert out["properties"]["name"] == "route-a"
    assert "parts" in out["properties"]["value"]


def test_generator_closes_dataset_on_early_close(tmp_path):
    import rasterio

    path = tmp_path / "grid.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=8, width=8, count=1,
        dtype="float32", transform=IDENTITY,
    ) as dst:
        dst.write(np.ones((8, 8), dtype="float32"), 1)
    gen = gen_point_query(
        LineString([(1.6, 1.6), (6.4, 6.4)]).__geo_interface__,
        str(path),
        exposure=0.5,
    )
    next(gen)
    gen.close()
    with rasterio.open(path) as src:
        assert src.read(1).shape == (8, 8)


# ---------------------------------------------------------------------------
# Clipping geometry
# ---------------------------------------------------------------------------

def test_cut_line_preserves_turns_and_length():
    coords = [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]  # lengths 3 then 5
    cut = cut_line_coordinates(coords, 2.0, 6.0, 1e-8)
    assert cut[0] == pytest.approx((2.0, 0.0))
    assert (3.0, 0.0) in [tuple(p) for p in cut]
    assert cut[-1] == pytest.approx((3.0, 3.0))
    length = sum(
        math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(cut[:-1], cut[1:])
    )
    assert length == pytest.approx(4.0)


def test_cut_line_full_interval_matches_input():
    coords = [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]
    cut = cut_line_coordinates(coords, 0.0, 8.0, 1e-8)
    assert [tuple(p) for p in cut] == coords


# ---------------------------------------------------------------------------
# Bounded reads / cache
# ---------------------------------------------------------------------------

def test_read_cells_bounds_window_and_cache():
    arr = np.ones((100, 100))
    with raster(arr) as rast:
        from rasterstats.exposure import CellCache, analyze_component

        cache = CellCache(rast, 64)
        analyze_component(
            [(1.6, 1.6), (98.4, 98.4)], ~IDENTITY, 0.5, 1e-8, cache
        )
        assert cache.max_cached <= 64
        window_sizes = [
            (r1 - r0) * (c1 - c0) for (r0, r1), (c0, c1) in cache.read_windows
        ]
        assert max(window_sizes) <= 64
        assert len(cache.read_windows) > 1


def test_read_cells_minimum_four():
    arr = np.ones((6, 6))
    with pytest.raises(ValueError):
        analyze(LineString([(1.6, 1.6), (4.4, 4.4)]), arr, 0.5, read_cells=3)


def test_large_coordinates_error_budget():
    arr = pad([0.0, 4.0, 4.0, 8.0])
    shift = Affine.translation(1.0e7, -1.0e7)
    start = shift @ DIAG_START
    end = shift @ DIAG_END
    result = analyze(
        LineString([start, end]), arr, 6.0, affine=shift, distance_tol=1e-6
    )
    part = result["parts"][0]
    # the field is the linear p(t) = 8t on the diagonal; crossing 6 at t=.75
    t_cross = 0.75
    expected = (t_cross - DIAG_T0) * SQRT2
    assert part["intervals"][1]["start"] == pytest.approx(
        expected, abs=1e-6 + 1e-10 * part["total_length"]
    )
    check_coverage(part)


def test_results_are_json_serializable_without_nan_tokens():
    arr = np.full((5, 5), 1.0)
    arr[2, 2] = np.nan
    line = LineString([(1.6, 2.25), (4.4, 2.25)])
    [result] = point_query(line, arr, affine=IDENTITY, exposure=0.5)
    text = json.dumps(result, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
