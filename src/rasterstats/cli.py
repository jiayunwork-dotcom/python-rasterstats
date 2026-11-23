import logging

import click
import cligj
import simplejson as json
from shapely.geometry import shape

from rasterstats import gen_point_query, gen_zonal_stats
from rasterstats._version import __version__ as version
from rasterstats.exposure import cut_line_coordinates, line_components

SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=SETTINGS)
@cligj.features_in_arg
@click.version_option(version=version, message="%(version)s")
@click.option("--raster", "-r", required=True)
@click.option("--all-touched/--no-all-touched", default=False)
@click.option("--band", type=int, default=1)
@click.option("--categorical/--no-categorical", default=False)
@click.option(
    "--coverage",
    type=click.Choice(["fractional"]),
    default=None,
    help="Enable exact area-weighted (fractional cell coverage) analysis.",
)
@click.option(
    "--value-kind",
    type=click.Choice(["total", "density"]),
    default=None,
    help="With --coverage fractional: 'total' cell quantities or "
    "'density' per-unit-area intensities.",
)
@click.option(
    "--max-cells",
    type=int,
    default=65536,
    show_default=True,
    help="With --coverage fractional: max raster cells read/held per tile.",
)
@click.option("--indent", type=int, default=None)
@click.option("--info/--no-info", default=False)
@click.option("--nodata", type=int, default=None)
@click.option("--prefix", type=str, default="_")
@click.option("--stats", type=str, default=None)
@click.option("--sequence/--no-sequence", type=bool, default=False)
@cligj.use_rs_opt
def zonalstats(
    features,
    raster,
    all_touched,
    band,
    categorical,
    coverage,
    value_kind,
    max_cells,
    indent,
    info,
    nodata,
    prefix,
    stats,
    sequence,
    use_rs,
):
    """zonalstats generates summary statistics of geospatial raster datasets
    based on vector features.

    The input arguments to zonalstats should be valid GeoJSON Features. (see cligj)

    The output GeoJSON will be mostly unchanged but have additional properties per
    feature describing the summary statistics (min, max, mean, etc.) of the underlying
    raster dataset.

    The raster is specified by the required -r/--raster argument.

    Example, calculate rainfall stats for each state and output to file:

    \b
       rio zonalstats states.geojson -r rainfall.tif > mean_rainfall_by_state.geojson
    """
    if info:
        logging.basicConfig(level=logging.INFO)

    if stats is not None:
        stats = stats.split(" ")
        if "all" in [x.lower() for x in stats]:
            stats = "ALL"

    zonal_results = gen_zonal_stats(
        features,
        raster,
        all_touched=all_touched,
        band=band,
        categorical=categorical,
        coverage=coverage,
        value_kind=value_kind,
        max_cells=max_cells,
        nodata=nodata,
        stats=stats,
        prefix=prefix,
        geojson_out=True,
    )

    # Fractional mode never emits NaN/Infinity by construction (mean is null),
    # but pin the serializer anyway so the JSON contract stays explicit.
    json_default = dict(ignore_nan=True) if coverage == "fractional" else {}

    if sequence:
        for feature in zonal_results:
            if use_rs:
                click.echo(b"\x1e", nl=False)
            click.echo(json.dumps(feature, **json_default))
    else:
        click.echo(
            json.dumps(
                {"type": "FeatureCollection", "features": list(zonal_results)},
                **json_default,
            )
        )


def _iter_exposure_records(results, distance_tol, property_name):
    """Turn exposure-mode features into one GeoJSON record per interval.

    Each record's geometry is the original component clipped to the interval's
    mileage range, so intermediate turns are retained and its length equals
    ``stop - start``. Original feature properties are carried along; any
    property sharing a name with a result key is overridden by the result.
    """
    for feature_index, feature in enumerate(results):
        properties = dict(feature.get("properties") or {})
        analysis = properties.pop(property_name)
        geom = shape(feature["geometry"])
        components = line_components(geom, feature_index)
        for part_index, part in enumerate(analysis["parts"]):
            coords = components[part_index]
            for interval in part["intervals"]:
                clipped = cut_line_coordinates(
                    coords,
                    interval["start"],
                    interval["stop"],
                    distance_tol,
                )
                record_props = dict(properties)
                record_props.update(
                    {
                        "feature_index": feature_index,
                        "part_index": part_index,
                        "start": interval["start"],
                        "stop": interval["stop"],
                        "kind": interval["kind"],
                        "integral": interval["integral"],
                    }
                )
                yield {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [list(point) for point in clipped],
                    },
                    "properties": record_props,
                }


@click.command(context_settings=SETTINGS)
@cligj.features_in_arg
@click.version_option(version=version, message="%(version)s")
@click.option("--raster", "-r", required=True)
@click.option("--band", type=int, default=1)
@click.option("--nodata", type=int, default=None)
@click.option("--indent", type=int, default=None)
@click.option("--interpolate", type=str, default="bilinear")
@click.option("--property-name", type=str, default="value")
@click.option(
    "--exposure",
    type=float,
    default=None,
    help="Finite concentration threshold; enables continuous along-line "
    "exposure analysis for LineString/MultiLineString inputs and emits one "
    "GeoJSON feature per classified route segment.",
)
@click.option(
    "--distance-tol",
    type=float,
    default=1e-8,
    show_default=True,
    help="Absolute positional error budget in route length units.",
)
@click.option(
    "--read-cells",
    type=int,
    default=4096,
    show_default=True,
    help="Maximum raster cell slots per window read and in the bounded cell cache.",
)
@click.option("--sequence/--no-sequence", type=bool, default=False)
@cligj.use_rs_opt
def pointquery(
    features,
    raster,
    band,
    indent,
    nodata,
    interpolate,
    property_name,
    exposure,
    distance_tol,
    read_cells,
    sequence,
    use_rs,
):
    """
    Queries the raster values at the points of the input GeoJSON Features.
    The raster values are added to the features properties and output as GeoJSON
    Feature Collection.

    If the Features are Points, the point geometry is used.
    For other Feauture types, all of the vertices of the geometry will be queried.
    For example, you can provide a linestring and get the profile along the line
    if the vertices are spaced properly.

    You can use either bilinear (default) or nearest neighbor interpolation.

    With --exposure, the point sampler is replaced by continuous along-line
    exposure analysis: the output becomes one GeoJSON Feature per classified
    segment (above, below, missing), each clipped from the original route,
    carrying feature_index, part_index, start, stop, kind and integral.
    """

    results = gen_point_query(
        features,
        raster,
        band=band,
        nodata=nodata,
        interpolate=interpolate,
        property_name=property_name,
        geojson_out=True,
        exposure=exposure,
        distance_tol=distance_tol,
        read_cells=read_cells,
    )

    if exposure is not None:
        records = _iter_exposure_records(results, distance_tol, property_name)
        # strict JSON: the analysis can legitimately contain nulls but never
        # NaN or Infinity tokens
        dump_kwargs = dict(ignore_nan=True)
        if sequence:
            for record in records:
                if use_rs:
                    click.echo(b"\x1e", nl=False)
                click.echo(json.dumps(record, **dump_kwargs))
        else:
            click.echo(
                json.dumps(
                    {"type": "FeatureCollection", "features": list(records)},
                    **dump_kwargs,
                )
            )
        return

    if sequence:
        for feature in results:
            if use_rs:
                click.echo(b"\x1e", nl=False)
            click.echo(json.dumps(feature))
    else:
        click.echo(json.dumps({"type": "FeatureCollection", "features": list(results)}))
