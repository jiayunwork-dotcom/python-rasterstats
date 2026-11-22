import json
import warnings
from pathlib import Path

from click.testing import CliRunner

from rasterstats.cli import pointquery, zonalstats

# Some warnings must be ignored to parse output properly
# https://github.com/pallets/click/issues/371#issuecomment-223790894

data_dir = Path(__file__).parent / "data"


def test_cli_feature():
    raster = str(data_dir / "slope.tif")
    vector = str(data_dir / "feature.geojson")
    runner = CliRunner()
    warnings.simplefilter("ignore")
    result = runner.invoke(
        zonalstats, [vector, "--raster", raster, "--stats", "mean", "--prefix", "test_"]
    )
    assert result.exit_code == 0
    outdata = json.loads(result.output)
    assert len(outdata["features"]) == 1
    feature = outdata["features"][0]
    assert "test_mean" in feature["properties"]
    assert round(feature["properties"]["test_mean"], 2) == 14.66
    assert "test_count" not in feature["properties"]


def test_cli_feature_stdin():
    raster = str(data_dir / "slope.tif")
    vector_pth = data_dir / "feature.geojson"

    runner = CliRunner()
    warnings.simplefilter("ignore")
    result = runner.invoke(
        zonalstats,
        ["--raster", raster, "--stats", "all", "--prefix", "test_"],
        input=vector_pth.read_text(),
    )
    assert result.exit_code == 0
    outdata = json.loads(result.output)
    assert len(outdata["features"]) == 1
    feature = outdata["features"][0]
    assert "test_mean" in feature["properties"]
    assert "test_std" in feature["properties"]


def test_cli_features_sequence():
    raster = str(data_dir / "slope.tif")
    vector = str(data_dir / "featurecollection.geojson")
    runner = CliRunner()
    result = runner.invoke(
        zonalstats,
        [
            vector,
            "--raster",
            raster,
            "--stats",
            "mean",
            "--prefix",
            "test_",
            "--sequence",
        ],
    )
    assert result.exit_code == 0
    results = result.output.splitlines()
    for r in results:
        outdata = json.loads(r)
        assert outdata["type"] == "Feature"


def test_cli_features_sequence_rs():
    raster = str(data_dir / "slope.tif")
    vector = str(data_dir / "featurecollection.geojson")
    runner = CliRunner()
    result = runner.invoke(
        zonalstats,
        [
            vector,
            "--raster",
            raster,
            "--stats",
            "mean",
            "--prefix",
            "test_",
            "--sequence",
            "--rs",
        ],
    )
    assert result.exit_code == 0
    # assert result.output.startswith(b'\x1e')
    assert result.output[0] == "\x1e"


def test_cli_featurecollection():
    raster = str(data_dir / "slope.tif")
    vector = str(data_dir / "featurecollection.geojson")
    runner = CliRunner()
    result = runner.invoke(
        zonalstats, [vector, "--raster", raster, "--stats", "mean", "--prefix", "test_"]
    )
    assert result.exit_code == 0
    outdata = json.loads(result.output)
    assert len(outdata["features"]) == 2
    feature = outdata["features"][0]
    assert "test_mean" in feature["properties"]
    assert round(feature["properties"]["test_mean"], 2) == 14.66
    assert "test_count" not in feature["properties"]


def test_cli_fractional_featurecollection(tmp_path):
    import numpy as np
    import rasterio
    from affine import Affine

    raster_path = tmp_path / "grid.tif"
    array = np.arange(1, 17, dtype="float32").reshape(4, 4)
    with rasterio.open(
        raster_path, "w", driver="GTiff", height=4, width=4,
        count=1, dtype="float32", transform=Affine.identity(),
    ) as dst:
        dst.write(array, 1)
    vector = str(data_dir / "featurecollection.geojson")
    runner = CliRunner()
    result = runner.invoke(
        zonalstats,
        [
            vector,
            "--raster",
            str(raster_path),
            "--coverage",
            "fractional",
            "--value-kind",
            "density",
            "--max-cells",
            "4",
            "--prefix",
            "frac_",
        ],
    )
    assert result.exit_code == 0
    outdata = json.loads(result.output)
    assert len(outdata["features"]) == 2
    for feature in outdata["features"]:
        props = feature["properties"]
        for key in (
            "frac_count",
            "frac_sum",
            "frac_mean",
            "frac_coverage_area",
            "frac_valid_area",
            "frac_missing_area",
        ):
            assert key in props
    # strict JSON: NaN / Infinity tokens are never emitted
    assert "NaN" not in result.output
    assert "Infinity" not in result.output


def test_cli_fractional_sequence(tmp_path):
    import numpy as np
    import rasterio
    from affine import Affine

    raster_path = tmp_path / "grid.tif"
    array = np.ones((4, 4), dtype="float32")
    with rasterio.open(
        raster_path, "w", driver="GTiff", height=4, width=4,
        count=1, dtype="float32", transform=Affine.identity(),
    ) as dst:
        dst.write(array, 1)
    vector = str(data_dir / "featurecollection.geojson")
    runner = CliRunner()
    collection = runner.invoke(
        zonalstats,
        [vector, "--raster", str(raster_path), "--coverage", "fractional",
         "--value-kind", "total"],
    )
    sequence = runner.invoke(
        zonalstats,
        [vector, "--raster", str(raster_path), "--coverage", "fractional",
         "--value-kind", "total", "--sequence"],
    )
    assert sequence.exit_code == 0
    collection_props = [
        f["properties"] for f in json.loads(collection.output)["features"]
    ]
    sequence_props = [json.loads(line)["properties"] for line in sequence.output.splitlines()]
    assert collection_props == sequence_props


def test_cli_fractional_requires_value_kind(tmp_path):
    import numpy as np
    import rasterio
    from affine import Affine

    raster_path = tmp_path / "grid.tif"
    with rasterio.open(
        raster_path, "w", driver="GTiff", height=4, width=4,
        count=1, dtype="float32", transform=Affine.identity(),
    ) as dst:
        dst.write(np.ones((4, 4), dtype="float32"), 1)
    vector = str(data_dir / "featurecollection.geojson")
    runner = CliRunner()
    result = runner.invoke(
        zonalstats,
        [vector, "--raster", str(raster_path), "--coverage", "fractional"],
    )
    assert result.exit_code != 0
    assert "value_kind" in str(result.exception)

    result_bad = runner.invoke(
        zonalstats,
        [vector, "--raster", str(raster_path), "--coverage", "fractional",
         "--value-kind", "total", "--max-cells", "0"],
    )
    assert result_bad.exit_code != 0
    assert "max_cells" in str(result_bad.exception)


def test_cli_pointquery():
    raster = str(data_dir / "slope.tif")
    vector = str(data_dir / "featurecollection.geojson")
    runner = CliRunner()
    result = runner.invoke(
        pointquery, [vector, "--raster", raster, "--property-name", "slope"]
    )
    assert result.exit_code == 0
    outdata = json.loads(result.output)
    assert len(outdata["features"]) == 2
    feature = outdata["features"][0]
    assert "slope" in feature["properties"]


def test_cli_point_sequence():
    raster = str(data_dir / "slope.tif")
    vector = str(data_dir / "featurecollection.geojson")
    runner = CliRunner()
    result = runner.invoke(
        pointquery,
        [vector, "--raster", raster, "--property-name", "slope", "--sequence"],
    )
    assert result.exit_code == 0
    results = result.output.splitlines()
    for r in results:
        outdata = json.loads(r)
        assert outdata["type"] == "Feature"


def test_cli_point_sequence_rs():
    raster = str(data_dir / "slope.tif")
    vector = str(data_dir / "featurecollection.geojson")
    runner = CliRunner()
    result = runner.invoke(
        pointquery,
        [vector, "--raster", raster, "--property-name", "slope", "--sequence", "--rs"],
    )
    assert result.exit_code == 0
    assert result.output[0] == "\x1e"
