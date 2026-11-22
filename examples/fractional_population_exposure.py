"""Fractional zonal statistics: population counts vs population density.

This worked example shows why boundary cells must be weighted by their real
overlap area, and how the two fractional value kinds relate to each other.

A 10x10 grid in a projected CRS with 100 m x 100 m cells (cell area
10,000 m^2 = 1 ha) is summarized over an administrative polygon that cuts
through cells.

* The "population" grid stores a *total* per cell (people spread uniformly
  over the cell), so a cell contributes ``fraction * people`` to the zone's
  population total.
* The "density" grid stores people per m^2, constant inside each cell, so a
  cell contributes ``overlap_area * density``. Building density as
  ``people / cell_area`` makes the two analyses agree exactly.

Run it with::

    python examples/fractional_population_exposure.py
"""

import numpy as np
from affine import Affine
from shapely.geometry import Polygon

from rasterstats import zonal_stats

# 100 m x 100 m cells, origin at (0, 0), north-up.
CELL_METRES = 100.0
CELL_AREA = CELL_METRES**2
transform = Affine(CELL_METRES, 0.0, 0.0, 0.0, -CELL_METRES, 10.0 * CELL_METRES)

# Whole-cell population counts (people), distinct per row.
people = np.tile(np.arange(100, 110, dtype="float64").reshape(10, 1), (1, 10))

# An administrative boundary that deliberately slices through boundary cells.
# Coordinates are in the same projected CRS (metres).
district = Polygon(
    [
        (150.0, 150.0),
        (850.0, 120.0),
        (880.0, 720.0),
        (240.0, 880.0),
    ]
)

total_result = zonal_stats(
    [district],
    people,
    affine=transform,
    coverage="fractional",
    value_kind="total",
)[0]

# A constant per-unit-area intensity derived from the total grid.
density = people / CELL_AREA
density_result = zonal_stats(
    [district],
    density,
    affine=transform,
    coverage="fractional",
    value_kind="density",
)[0]

print("fractional population exposure")
print("------------------------------")
print(f"coverage_area (m^2, inside raster frame): {total_result['coverage_area']:.4f}")
print(f"valid_area    (m^2 with observations)   : {total_result['valid_area']:.4f}")
print(f"missing_area  (m^2 covered but nodata)  : {total_result['missing_area']:.4f}")
print(f"count         (sum of fractions)        : {total_result['count']:.4f}")
print(f"total-kind sum (people)                 : {total_result['sum']:.4f}")
print(f"density-kind sum (people)               : {density_result['sum']:.4f}")
print(f"area-weighted mean intensity (people/m^2): {total_result['mean']:.6f}")

# The two formulations are dimensionally consistent and numerically equal.
assert np.isclose(
    total_result["sum"], density_result["sum"], rtol=1e-12, atol=1e-12
)
assert np.isclose(
    total_result["mean"], density_result["mean"], rtol=1e-12, atol=1e-12
)
print("\nOK: total and density formulations give the same total and mean.")
