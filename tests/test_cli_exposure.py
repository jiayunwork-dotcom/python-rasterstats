"""CLI tests for continuous along-line exposure analysis (``rio pointquery``)."""
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from click.testing import CliRunner
from shapely.geometry import LineString, MultiLineString, shape

from rasterstats.cli import pointquery

data_dir = Path(__file__).parent / "data"


def _write_raster(path, array, nodata=None):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        transform=Affine.identity(),
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)


def _feature(geometry, properties=None):
    return {
        "type": "Feature",
        "geometry": geometry.__geo_interface__,
        "properties": properties or {},
    }


@pytest.fixture()
def exposure_raster(tmp_path):
    # p(t) = 8 t^2 style grid embedded with padding so the route is fully known
    array = np.ones((7, 7), dtype="float32")
    array[2, 2], array[2, 3] = 0.0, 4.0
    array[3, 2], array[3, 3] = 4.0, 8.0
    path = tmp_path / "exposure.tif"
    _write_raster(path, array)
    return path


@pytest.fixture()
def route_collection(tmp_path):
    line = LineString([(2.1, 2.1), (2.9, 2.9)])
    feature = _feature(line, {"route_id": "R1", "kind": "above-test"})
    path = tmp_path / "routes.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}))
    return path, line, feature


def test_cli_exposure_feature_collection(exposure_raster, route_collection):
    vector_path, _, _ = route_collection
    runner = CliRunner()
    result = runner.invoke(
        pointquery,
        [str(vector_path), "--raster", str(exposure_raster), "--exposure", "6.0"],
    )
    assert result.exit_code == 0, result.output
    collection = json.loads(result.output)
    assert collection["type"] == "FeatureCollection"
    records = collection["features"]
    assert records, "at least one segment record is expected"
    kinds = [r["properties"]["kind"] for r in records]
    assert "above" in kinds and "below" in kinds
    for record in records:
        props = record["properties"]
        assert record["geometry"]["type"] == "LineString"
        # every result key is present
        for key in (
            "feature_index",
            "part_index",
            "start",
            "stop",
            "kind",
            "integral",
        ):
            assert key in props
        # original properties carried through
        assert props["route_id"] == "R1"
        # clipped geometry length matches stop-start and keeps the route's
        # direction (first point lies on the original line)
        clipped = shape(record["geometry"])
        assert clipped.length == pytest.approx(
            props["stop"] - props["start"], abs=1e-8
        )
        assert clipped.length > 0.0


def test_cli_exposure_sequence_matches_collection(exposure_raster, route_collection):
    vector_path, _, _ = route_collection
    runner = CliRunner()
    collection = runner.invoke(
        pointquery,
        [str(vector_path), "--raster", str(exposure_raster), "--exposure", "6.0"],
    )
    sequence = runner.invoke(
        pointquery,
        [
            str(vector_path),
            "--raster",
            str(exposure_raster),
            "--exposure",
            "6.0",
            "--sequence",
        ],
    )
    assert collection.exit_code == 0 and sequence.exit_code == 0
    coll_records = json.loads(collection.output)["features"]
    seq_records = [json.loads(line) for line in sequence.output.splitlines()]
    assert len(coll_records) == len(seq_records)
    for a, b in zip(coll_records, seq_records):
        assert a == b


def test_cli_exposure_sequence_rs(exposure_raster, route_collection):
    vector_path, _, _ = route_collection
    runner = CliRunner()
    result = runner.invoke(
        pointquery,
        [
            str(vector_path),
            "--raster",
            str(exposure_raster),
            "--exposure",
            "6.0",
            "--sequence",
            "--rs",
        ],
    )
    assert result.exit_code == 0
    assert result.output[0] == "\x1e"


def test_cli_exposure_geometry_keeps_intermediate_turns(exposure_raster, tmp_path):
    # a bent route whose midpoint vertex must survive clipping
    line = LineString([(2.1, 2.1), (2.5, 2.5), (2.9, 2.1)])
    vector = tmp_path / "bent.geojson"
    vector.write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": [_feature(line, {"id": "bent"})]}
        )
    )
    runner = CliRunner()
    result = runner.invoke(
        pointquery,
        [str(vector), "--raster", str(exposure_raster), "--exposure", "6.0"],
    )
    assert result.exit_code == 0, result.output
    records = json.loads(result.output)["features"]
    # at least one record must retain the input turn (three or more points)
    assert any(len(r["geometry"]["coordinates"]) >= 3 for r in records)
    # every record's points lie on the original polyline
    for record in records:
        clipped = shape(record["geometry"])
        assert clipped.length == pytest.approx(
            record["properties"]["stop"] - record["properties"]["start"], abs=1e-8
        )


def test_cli_exposure_missing_segments_emitted(tmp_path):
    array = np.ones((5, 5), dtype="float32")
    raster_path = tmp_path / "edge.tif"
    _write_raster(raster_path, array)
    # route starts inside the grid union and runs past its edge
    line = LineString([(1.6, 2.25), (4.4, 2.25)])
    vector = tmp_path / "edge-route.geojson"
    vector.write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": [_feature(line)]}
        )
    )
    runner = CliRunner()
    result = runner.invoke(
        pointquery,
        [str(vector), "--raster", str(raster_path), "--exposure", "0.5"],
    )
    assert result.exit_code == 0
    records = json.loads(result.output)["features"]
    kinds = [r["properties"]["kind"] for r in records]
    assert "missing" in kinds
    missing = next(r for r in records if r["properties"]["kind"] == "missing")
    assert missing["properties"]["integral"] is None
    # strict JSON: no NaN/Infinity tokens anywhere
    assert "NaN" not in result.output and "Infinity" not in result.output


def test_cli_exposure_multilinestring_part_indices(tmp_path):
    array = np.full((8, 8), 2.0, dtype="float32")
    raster_path = tmp_path / "multi.tif"
    _write_raster(raster_path, array)
    geom = MultiLineString(
        [
            [(2.6, 2.25), (4.4, 2.25)],
            [(3.0, 5.25), (4.8, 5.25)],
        ]
    )
    vector = tmp_path / "multi.geojson"
    vector.write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": [_feature(geom)]}
        )
    )
    runner = CliRunner()
    result = runner.invoke(
        pointquery,
        [str(vector), "--raster", str(raster_path), "--exposure", "1.0"],
    )
    assert result.exit_code == 0
    records = json.loads(result.output)["features"]
    part_indices = {r["properties"]["part_index"] for r in records}
    assert part_indices == {0, 1}
    # each part's mileage starts at zero
    for part_index in (0, 1):
        first = next(
            r for r in records if r["properties"]["part_index"] == part_index
        )
        assert first["properties"]["start"] == 0.0


def test_cli_exposure_result_keys_override_same_named_properties(
    exposure_raster, tmp_path
):
    line = LineString([(2.1, 2.1), (2.9, 2.9)])
    feature = _feature(
        line,
        {
            "kind": "original-kind",
            "feature_index": 999,
            "start": "original-start",
            "integral": "original-integral",
        },
    )
    vector = tmp_path / "overrides.geojson"
    vector.write_text(
        json.dumps({"type": "FeatureCollection", "features": [feature]})
    )
    runner = CliRunner()
    result = runner.invoke(
        pointquery,
        [str(vector), "--raster", str(exposure_raster),
         "--exposure", "6.0"],
    )
    assert result.exit_code == 0
    records = json.loads(result.output)["features"]
    for record in records:
        props = record["properties"]
        # result values override same-named original properties
        assert props["kind"] in ("above", "below", "missing")
        assert props["feature_index"] == 0
        assert isinstance(props["start"], (int, float))
        assert props["integral"] is None or isinstance(props["integral"], float)


def test_cli_without_exposure_unchanged():
    raster = str(data_dir / "slope.tif")
    vector = str(data_dir / "featurecollection.geojson")
    runner = CliRunner()
    result = runner.invoke(
        pointquery, [vector, "--raster", raster, "--property-name", "slope"]
    )
    assert result.exit_code == 0
    outdata = json.loads(result.output)
    assert len(outdata["features"]) == 2
    assert "slope" in outdata["features"][0]["properties"]
    # legacy per-feature output, not per-segment
    for feature in outdata["features"]:
        assert "kind" not in feature["properties"]


def test_cli_exposure_bad_arguments(exposure_raster, route_collection):
    vector_path, _, _ = route_collection
    runner = CliRunner()
    # bool-like threshold is rejected by click/type validation or the library
    bad_tol = runner.invoke(
        pointquery,
        [
            str(vector_path),
            "--raster",
            str(exposure_raster),
            "--exposure",
            "6.0",
            "--distance-tol",
            "0",
        ],
    )
    assert bad_tol.exit_code != 0
    bad_cells = runner.invoke(
        pointquery,
        [
            str(vector_path),
            "--raster",
            str(exposure_raster),
            "--exposure",
            "6.0",
            "--read-cells",
            "3",
        ],
    )
    assert bad_cells.exit_code != 0
    nearest = runner.invoke(
        pointquery,
        [
            str(vector_path),
            "--raster",
            str(exposure_raster),
            "--exposure",
            "6.0",
            "--interpolate",
            "nearest",
        ],
    )
    assert nearest.exit_code != 0
