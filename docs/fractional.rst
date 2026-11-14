Fractional (exact area-weighted) statistics
===========================================

``zonal_stats`` selects raster cells using a center-point rule or the
``all_touched`` rasterization strategy; a cell that straddles a region
boundary is therefore counted in full or not at all.  When the boundary runs
through the middle of cells, either choice changes the aggregated values.

:func:`rasterstats.gen_fractional_stats` and its list-returning counterpart
:func:`rasterstats.fractional_stats` instead keep the *actual* contribution
of every boundary cell.

How the coverage fraction is defined
------------------------------------

Each raster cell is the real quadrilateral obtained by applying the raster's
:class:`affine.Affine` transform to the unit square.  For every cell and
every polygon region, the coverage weight is

.. code-block:: text

    weight = area(intersection(cell, region)) / area(cell)

with areas measured in the shared planar coordinate system of the inputs.

* The weight is an exact planar area ratio computed with Shapely; the
  library does not use center-point hits, ``all_touched``, regular sampling
  or any up-sampling of the raster.
* Cells that only touch the region along a boundary (zero intersection
  area) contribute nothing.
* Interior rings are subtracted, so holes reduce the coverage.
* A valid ``MultiPolygon`` is treated as a single region.
* A region smaller than one cell still contributes its (small) fraction.
* Inputs are assumed to already share one planar coordinate system: no
  reprojection and no spherical/geodesic area conversion is performed.
* Rotated, sheared and negative-axis grids are supported as long as the
  ``Affine`` transform is finite and invertible.

Returned statistics
-------------------

Every feature yields a dictionary with exactly these seven keys:

``coverage``
    Sum of the intersection fractions over all cells within the finite
    raster extent (i.e. covered area expressed in units of cell area).

``count``
    As a float, the sum of fractions of *valid* cells.

``nodata``
    Sum of fractions of *invalid* cells.  Invalid coverage is determined by
    more than a single sentinel value: NumPy masks, the dataset's own
    validity mask, a numeric nodata sentinel, and ``NaN`` / positive and
    negative infinity all invalidate a cell.

``sum``
    Valid cell values multiplied by their coverage fraction and summed.

``mean``
    The coverage-weighted mean over valid coverage, i.e. ``sum / count``.

``min`` / ``max``
    The original (unweighted) values of the valid cells that have positive
    coverage.

With no valid coverage, ``count`` and ``sum`` are ``0.0`` and
``mean``, ``min`` and ``max`` are ``None``, while ``coverage`` and
``nodata`` still reflect the actual intersection.  Empty geometries and
fully disjoint regions yield all-zero coverage quantities.

Missing-data semantics
----------------------

A cell's coverage is split between valid (``count``) and invalid
(``nodata``) fractions; ``coverage`` always equals their sum for the
in-raster intersection.

* For file rasters, ``nodata=None`` uses the metadata's numeric sentinel
  (if any).  Passing an explicit ``nodata`` replaces that sentinel but can
  never remove a dataset's independent validity mask.
* For NumPy arrays, ``nodata=None`` means there is no numeric sentinel;
  a value such as ``-999`` is then ordinary, valid data.
* ``MaskedArray`` masks and file dataset masks always apply.
* Non-finite floating-point values (``NaN``, ``+inf``, ``-inf``) are always
  invalid.
* Coverage beyond the raster extent is ignored and is not counted as
  nodata.

Resource bounds
---------------

``max_cells`` caps the number of raster cells read in a single window.
When a feature's candidate extent is larger than this (including when it is
wider than the limit), it is split into column strips and then row blocks so
that every read stays within the cap.  Newly allocated value, weight and
intersection bookkeeping arrays therefore occupy ``O(max_cells)`` space; a
feature's full bounding box is never read in one go, no full intersection
history is retained, and only running totals are kept.  Changing
``max_cells`` or the file's physical block (tiling) layout changes the
processing order but not the results.

The generator works one feature at a time, opens its data sources lazily on
the first iteration step, iterates a one-shot feature source exactly once,
and releases every data source it opened when exhausted, when an error
propagates or on an explicit ``close()``.

Example
-------

.. code-block:: python

    from rasterstats import fractional_stats

    stats = fractional_stats("regions.shp", "elevation.tif")
    stats[0].keys()
    # dict_keys(['coverage', 'count', 'nodata', 'sum', 'mean', 'min', 'max'])

Use ``prefix`` to namespace the keys and ``geojson_out=True`` to receive
GeoJSON ``Feature`` dictionaries that keep the original geometry, id and
properties (input features are never modified).
