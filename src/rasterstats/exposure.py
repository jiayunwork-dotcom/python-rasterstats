"""Continuous along-line exposure analysis.

This module implements the continuous field interpretation used by
:func:`rasterstats.point_query` when ``exposure`` is set.  It is deliberately
separate from the historical vertex sampler:

* Pixel values are treated as observations at **cell centers**.
* On every closed quadrilateral formed by the centers of four cells in two
  adjacent rows and columns, the field is a bilinear surface.
* A grid is usable only when all four observations are valid (no nodata, no
  source mask, all finite; zero is a perfectly valid observation).
* Outside the union of usable grids the field is *unknown*; there is no
  nearest-neighbor fallback.
* Along a polyline the restriction of the bilinear surface is a quadratic in
  arc length, so threshold crossings and the line integral are solved
  analytically rather than by connecting discrete samples.

The line is walked grid by grid in fractional row/column space, which keeps
the analysis exact for rotated and sheared (but finite, invertible) affine
transforms and makes the raster access pattern local.  :class:`CellCache`
serves four-point supports from windows whose size is capped by
``read_cells`` and holds at most ``read_cells`` cells, so very long routes
never materialize their bounding box.
"""

import math
from collections import OrderedDict

import numpy as np
from affine import Affine

#: Output interval kinds.
ABOVE = "above"
BELOW = "below"
MISSING = "missing"


def validate_exposure_affine(affine):
    """The grid-to-world map must be a finite invertible affine."""
    if not isinstance(affine, Affine):
        raise ValueError(
            "exposure analysis requires an affine.Affine transform, got "
            f"{type(affine)!r}"
        )
    if not all(math.isfinite(v) for v in tuple(affine)[:6]):
        raise ValueError(
            "exposure analysis requires a transform with finite coefficients"
        )
    determinant = affine.determinant
    if not math.isfinite(determinant) or determinant == 0.0:
        raise ValueError(
            "exposure analysis requires an invertible (non-zero determinant) "
            "affine transform"
        )


def validate_exposure_threshold(exposure):
    """Normalize the ``exposure`` keyword.

    ``None`` disables the continuous mode (historical vertex query).  Any
    finite real number enables it.  Booleans are rejected explicitly because
    ``isinstance(True, int)`` is True in Python.
    """
    if exposure is None:
        return None
    if isinstance(exposure, bool) or not isinstance(exposure, (int, float)):
        raise ValueError(
            f"exposure {exposure!r} is not valid; it must be None or a finite "
            "real-valued threshold (booleans are not accepted)"
        )
    if not math.isfinite(exposure):
        raise ValueError(
            f"exposure {exposure!r} is not valid; the threshold must be finite"
        )
    return float(exposure)


def validate_distance_tol(distance_tol):
    """The positional error budget must be a positive finite non-bool real."""
    if (
        isinstance(distance_tol, bool)
        or not isinstance(distance_tol, (int, float))
        or not math.isfinite(distance_tol)
        or distance_tol <= 0.0
    ):
        raise ValueError(
            f"distance_tol {distance_tol!r} is not valid; it must be a positive "
            "finite real number (booleans are not accepted)"
        )
    return float(distance_tol)


def validate_read_cells(read_cells):
    """The cell budget must be an integer >= 4 and never a bool."""
    if isinstance(read_cells, bool) or not isinstance(read_cells, (int, np.integer)):
        raise ValueError(
            f"read_cells {read_cells!r} is not valid; it must be an integer "
            "no smaller than 4 (booleans are not accepted)"
        )
    read_cells = int(read_cells)
    if read_cells < 4:
        raise ValueError(
            f"read_cells {read_cells} is not valid; it must be no smaller than 4"
        )
    return read_cells


def line_components(geom, feature_index):
    """Split a (multi)linestring into validated coordinate components.

    Returns a list of ``(x, y)`` tuple lists, one per component in input
    order.  Empty components yield an empty list.  Non-line geometries,
    non-finite coordinates and anything other than plain 2D vertices raise
    ``ValueError`` located by zero-based feature and component indices.
    """
    geom_type = geom.geom_type
    if geom_type == "LineString":
        raw_components = [geom]
    elif geom_type == "MultiLineString":
        raw_components = list(geom.geoms)
    else:
        raise ValueError(
            f"feature {feature_index}: exposure analysis only accepts "
            f"LineString or MultiLineString geometries, got {geom_type}"
        )

    components = []
    for part_index, component in enumerate(raw_components):
        if component.is_empty:
            components.append([])
            continue
        if component.geom_type != "LineString":
            raise ValueError(
                f"feature {feature_index} component {part_index}: expected a "
                f"LineString, got {component.geom_type}"
            )
        if getattr(component, "has_z", False):
            raise ValueError(
                f"feature {feature_index} component {part_index}: exposure "
                "analysis requires finite 2D coordinates; 3D vertices are not "
                "accepted"
            )
        coords = []
        for x, y in component.coords:
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError(
                    f"feature {feature_index} component {part_index}: all "
                    "coordinates must be finite 2D planar values in the same "
                    "projection as the raster"
                )
            coords.append((float(x), float(y)))
        components.append(coords)
    return components


class CellCache:
    """Bounded LRU cache of raster cell values and validity flags.

    Every cache entry is one cell slot keyed by ``(row, col)``; the cache
    never holds more than ``read_cells`` entries and every window read fills
    at most ``read_cells`` new slots.  Windows are always clamped to the
    dataset extent, so out-of-dataset support cells are simply absent and
    invalidate the grid that would need them.

    Instrumentation for resource accounting:

    * ``read_windows`` lists every physical window read;
    * ``cells_read`` counts cell slots returned by reads (with replacement);
    * ``max_cached`` records the peak cache occupancy.
    """

    def __init__(self, rast, read_cells):
        self.rast = rast
        self.read_cells = read_cells
        self.height, self.width = rast.shape
        self._cells = OrderedDict()
        self.read_windows = []
        self.cells_read = 0
        self.max_cached = 0
        # Square tiles match route-like access best: a diagonal or folding
        # line only touches a handful of cells per tile. isqrt(n)^2 <= n and
        # read_cells >= 4, so the smallest tile is 2x2 = 4 cells.
        self.tile_side = max(2, math.isqrt(read_cells))

    def _load_tile(self, row, col):
        side = self.tile_side
        col_start = (col // side) * side
        row_start = (row // side) * side
        row_stop = min(self.height, row_start + side)
        col_stop = min(self.width, col_start + side)
        window = ((row_start, row_stop), (col_start, col_stop))

        tile_keys = [
            (r, c)
            for r in range(row_start, row_stop)
            for c in range(col_start, col_stop)
        ]
        missing = [key for key in tile_keys if key not in self._cells]
        if not missing:
            return
        # Evict in bulk rather than one popitem per slot: this keeps the
        # cache bounded while avoiding O(capacity) dict operations per tile
        # along routes that stream through a large raster.
        overflow = len(self._cells) + len(missing) - self.read_cells
        if overflow > 0:
            for key in list(self._cells.keys())[:overflow]:
                del self._cells[key]

        # boundless reads are unnecessary: the window is clamped to the
        # dataset, and cells beyond the dataset simply never get cached.
        fsrc = self.rast.read(window=window, masked=True, boundless=False)
        values, valid = _values_and_validity(fsrc)
        self.read_windows.append(window)
        self.cells_read += values.size

        # store plain python floats/bools: this avoids the very large
        # per-element overhead of numpy masked arrays on the millions of
        # cell lookups performed by a long route. tile_keys and the array
        # share the same row-major (row then column) ordering.
        flat_values = values.reshape(-1).tolist()
        flat_valid = valid.reshape(-1).tolist()
        for index, key in enumerate(tile_keys):
            # an evicted slot is simply re-inserted after the read
            self._cells[key] = (flat_valid[index], flat_values[index])
        self.max_cached = max(self.max_cached, len(self._cells))

    def cell(self, row, col):
        """Return ``(valid, value)``; absent/out-of-dataset cells are invalid."""
        if row < 0 or col < 0 or row >= self.height or col >= self.width:
            return False, math.nan
        key = (int(row), int(col))
        entry = self._cells.get(key)
        if entry is None:
            self._load_tile(key[0], key[1])
            entry = self._cells.get(key)
            if entry is None:  # pragma: no cover - defensive
                return False, math.nan
        else:
            self._cells.move_to_end(key)
        return entry

    def grid_support(self, base_row, base_col):
        """Four corner observations of grid ``(base_row, base_col)``.

        Returns ``(valid, corners)`` where ``corners`` orders the centers as
        array rows/columns: ``((row, col), (row, col+1), (row+1, col),
        (row+1, col+1))`` -- top-left, top-right, bottom-left, bottom-right.
        """
        corners = (
            (base_row, base_col),
            (base_row, base_col + 1),
            (base_row + 1, base_col),
            (base_row + 1, base_col + 1),
        )
        entries = [self.cell(r, c) for r, c in corners]
        valid = all(flag for flag, _ in entries)
        return valid, [value for _, value in entries]


def _values_and_validity(fsrc):
    """Float64 values plus the strict validity mask of a window read.

    Dataset masks, explicit nodata and non-finite values all invalidate;
    legitimate zeros remain valid.  The window array and its affine describe
    exactly the requested window cells for any finite affine, including
    rotated/sheared ones, because windowed reads only translate the origin.
    """
    array = fsrc.array
    if np.ma.is_masked(array):
        invalid = np.ma.getmaskarray(array).copy()
        data = np.ma.getdata(array)
    else:
        invalid = np.zeros(array.shape, dtype=bool)
        data = array
    data = data.astype(np.float64, copy=False)
    if fsrc.nodata is not None and np.isfinite(fsrc.nodata):
        invalid = invalid | (data == fsrc.nodata)
    invalid = invalid | ~np.isfinite(data)
    return data, ~invalid


def _arc_segments(coords, inv_affine):
    """Return per-input-segment arc-length and fractional (u, v) spans.

    Duplicate consecutive vertices produce zero-length segments and are
    skipped.  Each item is ``(s0, s1, u0, v0, u1, v1)`` with ``s`` measured
    from the component start in coordinate (ground) units.
    """
    segments = []
    cumulative = 0.0
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        length = math.hypot(x1 - x0, y1 - y0)
        if length == 0.0:
            continue
        u0, v0 = inv_affine @ (x0, y0)
        u1, v1 = inv_affine @ (x1, y1)
        segments.append(
            (cumulative, cumulative + length, float(u0), float(v0), float(u1), float(v1))
        )
        cumulative += length
    return segments, cumulative


def _grid_crossings(s0, s1, u0, v0, u1, v1):
    """Internal grid-line crossings of a segment as ``(mileage, axis, index)``.

    ``axis`` is 0 for a fractional-column (u) line and 1 for a fractional-row
    (v) line; ``index`` is the integer line crossed. Only strict interior
    crossings are emitted, so endpoints never produce zero-length pieces.
    """
    events = []
    span = s1 - s0
    if u1 != u0:
        lo = min(u0, u1)
        hi = max(u0, u1)
        for k in range(math.ceil(lo), math.floor(hi) + 1):
            t = (k - u0) / (u1 - u0)
            if 0.0 < t < 1.0:
                events.append((s0 + t * span, 0, k))
    if v1 != v0:
        lo = min(v0, v1)
        hi = max(v0, v1)
        for k in range(math.ceil(lo), math.floor(hi) + 1):
            t = (k - v0) / (v1 - v0)
            if 0.0 < t < 1.0:
                events.append((s0 + t * span, 1, k))
    events.sort(key=lambda item: item[0])
    return events


def _walk_pieces(s0, s1, u0, v0, u1, v1):
    """Split one input segment into pieces lying inside one grid cell each.

    Yields ``(left, right, base_row, base_col)``. The base cell is updated
    directly from the crossing events, so no coordinate is ever floored at a
    boundary position where rounding could choose the wrong side. Several
    events at one mileage (a diagonal crossing a cell corner, or a segment
    lying along a grid line) are all applied before the next piece starts.
    """
    # Initial cell: floor of the start vertex, except when the start sits
    # exactly on a grid line -- then it belongs to the cell the directed
    # segment is about to enter (the line itself is shared, and the first
    # strict interior point lives in that cell).
    base_col = math.floor(u0)
    base_row = math.floor(v0)
    if u0 == base_col and u1 != u0 and u1 < u0:
        base_col -= 1
    if v0 == base_row and v1 != v0 and v1 < v0:
        base_row -= 1
    events = _grid_crossings(s0, s1, u0, v0, u1, v1)
    # Crossings of a cell corner arrive as separate u and v events whose
    # mileages agree to floating-point precision; group anything below the
    # scaled positional tolerance so both cell indices update before the
    # next piece opens.
    group_tol = 1e-12 * (s1 - s0) + 1e-300
    cursor = s0
    index = 0
    while index < len(events):
        mileage = events[index][0]
        if mileage > cursor:
            yield cursor, mileage, base_row, base_col
        while index < len(events) and abs(events[index][0] - mileage) <= group_tol:
            _, axis, line_index = events[index]
            # crossing line k while moving up enters cell k; moving down k-1
            if axis == 0:
                base_col = line_index if u1 > u0 else line_index - 1
            else:
                base_row = line_index if v1 > v0 else line_index - 1
            index += 1
        cursor = mileage
    if s1 > cursor:
        yield cursor, s1, base_row, base_col


def _bilinear_polynomial(corners, fu0, fv0, fu1, fv1, s0, s1):
    """Coefficients of concentration as a quadratic in arc length.

    On a grid with top-left ``a``, top-right ``b``, bottom-left ``c`` and
    bottom-right ``d``::

        p(u, v) = a + (b - a) fu + (c - a) fv + (a - b - c + d) fu fv

    with ``fu``/``fv`` the fractional coordinates relative to the grid's
    lower-index cell. Restricted to the straight segment it is
    ``alpha*s^2 + beta*s + gamma`` in arc length.
    """
    a, b, c, d = corners
    span = s1 - s0
    gu = (fu1 - fu0) / span
    gv = (fv1 - fv0) / span
    gamma = (
        a
        + (b - a) * fu0
        + (c - a) * fv0
        + (a - b - c + d) * fu0 * fv0
    )
    beta = (
        (b - a) * gu
        + (c - a) * gv
        + (a - b - c + d) * (fu0 * gv + fv0 * gu)
    )
    alpha = (a - b - c + d) * gu * gv
    return alpha, beta, gamma


def _quadratic_roots(alpha, beta, gamma, threshold, length, tol):
    """Strictly interior arc-length roots of ``p(s) = threshold``.

    Roots are returned on the open interval ``(0, length)``: a root that lands
    within ``tol`` of a piece boundary is treated as sitting on that boundary
    (a grid-line crossing), so it never creates a zero-length sub-interval and
    the endpoint-sign classification resolves it together with the adjacent
    piece. True interior tangencies produce a double root that survives here
    and whose two sides carry the same sign, so merging removes the cut.
    """
    roots = []
    if alpha == 0.0:
        if beta != 0.0:
            roots.append((threshold - gamma) / beta)
    else:
        # scaled discriminant keeps the computation stable when coefficients
        # differ by orders of magnitude
        scale = max(
            1.0,
            abs(alpha) * length * length,
            abs(beta) * length,
            abs(gamma - threshold),
        )
        aa = alpha / scale
        bb = beta / scale
        cc = (gamma - threshold) / scale
        disc = bb * bb - 4.0 * aa * cc
        # a tiny negative discriminant is a numerical tangency (double root)
        if disc >= 0.0:
            root = math.sqrt(disc)
            roots.append((-bb - root) / (2.0 * aa))
            roots.append((-bb + root) / (2.0 * aa))
        elif disc > -1e-14:
            roots.append(-bb / (2.0 * aa))
    clipped = sorted(r for r in roots if tol < r < length - tol)
    merged = []
    for r in clipped:
        if not merged or r - merged[-1] >= tol:
            merged.append(r)
        # tangencies / double roots and numerically duplicated crossings
        # collapse onto one position, never producing a zero-length interval
    return merged

def _append_interval(intervals, start, stop, kind, integral):
    """Append an interval, merging it onto a neighboring interval of one kind."""
    if stop <= start:
        return
    if intervals and intervals[-1]["kind"] == kind:
        previous = intervals[-1]
        previous["stop"] = stop
        if kind != MISSING:
            previous["integral"] = previous["integral"] + integral
        # a missing interval always carries integral=None
        return
    intervals.append(
        {
            "start": start,
            "stop": stop,
            "kind": kind,
            "integral": None if kind == MISSING else integral,
        }
    )


def analyze_component(coords, inv_affine, threshold, distance_tol, cache):
    """Build the classified interval sequence for one line component.

    The segment is walked from grid cell to grid cell (see
    :func:`_walk_pieces`); each piece lies within a single interpolation
    grid and contributes one quadratic concentration polynomial. Known
    pieces are additionally cut at the exact threshold roots. Adjacent
    pieces of the same kind are merged, so an isolated threshold contact
    (tangency) never appears as a zero-length record and never splits the
    surrounding kind, while an unknown piece always separates known pieces.
    """
    intervals = []
    total_length = 0.0
    valid_length = 0.0
    missing_length = 0.0
    above_length = 0.0
    total_integral = 0.0

    segments, total_length = _arc_segments(coords, inv_affine)
    tol = distance_tol + 1e-12 * max(total_length, 1.0)

    for s0, s1, u0, v0, u1, v1 in segments:
        for left, right, base_row, base_col in _walk_pieces(
            s0, s1, u0, v0, u1, v1
        ):
            piece_length = right - left
            # endpoint coordinates of the piece in fractional row/column space
            frac_l = (left - s0) / (s1 - s0)
            frac_r = (right - s0) / (s1 - s0)
            ul = u0 + frac_l * (u1 - u0)
            vl = v0 + frac_l * (v1 - v0)
            ur = u0 + frac_r * (u1 - u0)
            vr = v0 + frac_r * (v1 - v0)
            valid, corners = cache.grid_support(base_row, base_col)

            if not valid:
                missing_length += piece_length
                _append_interval(intervals, left, right, MISSING, None)
                continue

            valid_length += piece_length
            # fractions are anchored to the supporting grid; a boundary
            # endpoint keeps fractional coordinate exactly 1
            fu0, fv0 = ul - base_col, vl - base_row
            fu1, fv1 = ur - base_col, vr - base_row
            alpha, beta, gamma = _bilinear_polynomial(
                corners, fu0, fv0, fu1, fv1, left, right
            )
            # Strictly interior roots cut the piece into constant-sign
            # sub-intervals; a crossing that sits on a grid boundary is not a
            # root of either adjacent piece and is resolved by the midpoint
            # sign on the neighboring pieces. True tangencies produce a double
            # root whose two sides share a sign, so same-kind merging removes
            # the phantom cut.
            roots = _quadratic_roots(
                alpha, beta, gamma, threshold, piece_length, tol
            )
            cuts = [0.0] + roots + [piece_length]
            for a0, a1 in zip(cuts[:-1], cuts[1:]):
                sub_len = a1 - a0
                if sub_len <= 0.0:
                    continue

                def local_value(position):
                    return (
                        alpha * position * position
                        + beta * position
                        + gamma
                        - threshold
                    )

                # Between consecutive roots the quadratic cannot change sign,
                # so the midpoint sign classifies the whole sub-interval. An
                # endpoint exactly on the threshold (a boundary contact or a
                # tangency) is classified below by the strict-greater rule,
                # and same-kind merging removes the phantom cut whenever the
                # surrounding pieces agree.
                midpoint_value = local_value(0.5 * (a0 + a1))
                kind = ABOVE if midpoint_value > 0.0 else BELOW
                # exact integral over [a0, a1] via the antiderivative of the
                # quadratic concentration field
                integral = _antiderivative(alpha, beta, gamma, a1) - _antiderivative(
                    alpha, beta, gamma, a0
                )
                total_integral += integral
                if kind == ABOVE:
                    above_length += sub_len
                _append_interval(
                    intervals, left + a0, left + a1, kind, integral
                )

    # normalize tiny endpoint drift so the pieces cover the component exactly
    if intervals:
        intervals[0]["start"] = 0.0
        intervals[-1]["stop"] = total_length

    return {
        "total_length": total_length,
        "valid_length": valid_length,
        "missing_length": missing_length,
        "above_length": above_length,
        "integral": total_integral,
        "intervals": intervals,
    }


def _antiderivative(alpha, beta, gamma, s):
    return alpha * s**3 / 3.0 + beta * s**2 / 2.0 + gamma * s


def analyze_geometry(geom, rast, threshold, distance_tol, read_cells, feature_index):
    """Run the full continuous analysis for one (multi)linestring feature."""
    read_cells = validate_read_cells(read_cells)
    validate_exposure_affine(rast.affine)
    components = line_components(geom, feature_index)
    inv_affine = ~rast.affine
    cache = CellCache(rast, read_cells)

    # A single row or column has no two-dimensional interpolation grid: the
    # bilinear field is nowhere defined, so every position is unknown.
    flat_raster = rast.shape[0] < 2 or rast.shape[1] < 2

    parts = []
    totals = {
        "total_length": 0.0,
        "valid_length": 0.0,
        "missing_length": 0.0,
        "above_length": 0.0,
        "integral": 0.0,
    }
    for coords in components:
        if flat_raster:
            _, component_length = _arc_segments(coords, inv_affine)
            intervals = []
            if component_length > 0.0:
                intervals.append(
                    {
                        "start": 0.0,
                        "stop": component_length,
                        "kind": MISSING,
                        "integral": None,
                    }
                )
            part = {
                "total_length": component_length,
                "valid_length": 0.0,
                "missing_length": component_length,
                "above_length": 0.0,
                "integral": 0.0,
                "intervals": intervals,
            }
        else:
            part = analyze_component(
                coords, inv_affine, threshold, distance_tol, cache
            )
        parts.append(part)
        for key in totals:
            totals[key] += part[key]

    return {
        "parts": parts,
        "total_length": totals["total_length"],
        "valid_length": totals["valid_length"],
        "missing_length": totals["missing_length"],
        "above_length": totals["above_length"],
        "integral": totals["integral"],
        "_cache": cache,
    }


def cut_line_coordinates(coords, start, stop, distance_tol):
    """Extract the sub-polyline of one component between two mileages.

    Mileage is measured from the component start, exactly as the analysis
    intervals are. Intermediate vertices are preserved, so the clipped
    geometry keeps every turn and its length matches ``stop - start`` within
    the requested positional tolerance.
    """
    eps = distance_tol

    def point_at(distance):
        cumulative = 0.0
        for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len == 0.0:
                continue
            if distance <= cumulative + seg_len + eps:
                t = (distance - cumulative) / seg_len
                return (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
            cumulative += seg_len
        # exact component endpoint
        return coords[-1]

    points = [point_at(start)]
    cumulative = 0.0
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        seg_len = math.hypot(x1 - x0, y1 - y0)
        cumulative += seg_len
        # every original turn strictly inside the interval is retained
        if start + eps < cumulative < stop - eps:
            points.append((x1, y1))
    points.append(point_at(stop))
    # drop identical consecutive points (zero-length input segments are
    # allowed but must not pollute the clipped LineString)
    deduped = [points[0]]
    for point in points[1:]:
        if point != deduped[-1]:
            deduped.append(point)
    return deduped
