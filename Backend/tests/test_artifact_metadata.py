import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


class MapArtifactMetadataTests(unittest.TestCase):
    def test_accepts_a_well_formed_map_metadata(self):
        from models.artifact import MapArtifactMetadata

        meta = MapArtifactMetadata(
            bbox=[-75.0, 39.0, -73.0, 41.0],
            variable="TEMPO_NO2",
            units="mol/m^2",
            colorbar={"vmin": 0.0, "vmax": 1.0},
            source_handles=["obs_1"],
        )

        self.assertEqual(meta.bbox, [-75.0, 39.0, -73.0, 41.0])
        self.assertEqual(meta.colorbar, {"vmin": 0.0, "vmax": 1.0})
        self.assertEqual(meta.source_handles, ["obs_1"])

    def test_rejects_bbox_with_wrong_number_of_coordinates(self):
        from pydantic import ValidationError
        from models.artifact import MapArtifactMetadata

        with self.assertRaises(ValidationError):
            MapArtifactMetadata(
                bbox=[-75.0, 39.0, -73.0],
                variable="TEMPO_NO2",
                units="mol/m^2",
                colorbar={"vmin": 0.0, "vmax": 1.0},
            )

    def test_rejects_missing_colorbar_bounds(self):
        from pydantic import ValidationError
        from models.artifact import MapArtifactMetadata

        with self.assertRaises(ValidationError):
            MapArtifactMetadata(
                bbox=[-75.0, 39.0, -73.0, 41.0],
                variable="TEMPO_NO2",
                units="mol/m^2",
                colorbar={"vmin": 0.0},
            )

    def test_source_handles_defaults_to_empty_list(self):
        from models.artifact import MapArtifactMetadata

        meta = MapArtifactMetadata(
            bbox=[-75.0, 39.0, -73.0, 41.0],
            variable="TEMPO_NO2",
            units="mol/m^2",
            colorbar={"vmin": 0.0, "vmax": 1.0},
        )

        self.assertEqual(meta.source_handles, [])


class ComparisonArtifactMetadataTests(unittest.TestCase):
    def test_accepts_a_well_formed_n_panel_comparison(self):
        from models.artifact import ComparisonArtifactMetadata

        meta = ComparisonArtifactMetadata(
            mode="n-panel",
            panels=[
                {"handle": "obs_1", "title": "New Jersey"},
                {"handle": "obs_2", "title": "New York"},
            ],
        )

        self.assertEqual(meta.mode, "n-panel")
        self.assertEqual(len(meta.panels), 2)
        self.assertEqual(meta.panels[0].handle, "obs_1")

    def test_rejects_fewer_than_two_panels(self):
        from pydantic import ValidationError
        from models.artifact import ComparisonArtifactMetadata

        with self.assertRaises(ValidationError):
            ComparisonArtifactMetadata(mode="n-panel", panels=[{"handle": "obs_1"}])

    def test_rejects_unknown_mode(self):
        from pydantic import ValidationError
        from models.artifact import ComparisonArtifactMetadata

        with self.assertRaises(ValidationError):
            ComparisonArtifactMetadata(
                mode="side-by-side",
                panels=[{"handle": "obs_1"}, {"handle": "obs_2"}],
            )


class TimeseriesArtifactMetadataTests(unittest.TestCase):
    def test_accepts_a_single_satellite_series(self):
        from models.artifact import TimeseriesArtifactMetadata

        meta = TimeseriesArtifactMetadata(
            series=[{"label": "TEMPO NO2 mean", "source_kind": "satellite"}],
        )

        self.assertEqual(len(meta.series), 1)
        self.assertEqual(meta.series[0].source_kind, "satellite")
        self.assertIsNone(meta.series[0].station_id)

    def test_accepts_a_ground_series_with_station_id(self):
        from models.artifact import TimeseriesArtifactMetadata

        meta = TimeseriesArtifactMetadata(
            series=[
                {"label": "TEMPO NO2", "source_kind": "satellite"},
                {"label": "EPA monitor", "source_kind": "ground", "station_id": "34-023-0011"},
            ],
        )

        self.assertEqual(meta.series[1].station_id, "34-023-0011")

    def test_rejects_empty_series_list(self):
        from pydantic import ValidationError
        from models.artifact import TimeseriesArtifactMetadata

        with self.assertRaises(ValidationError):
            TimeseriesArtifactMetadata(series=[])

    def test_rejects_unknown_source_kind(self):
        from pydantic import ValidationError
        from models.artifact import TimeseriesArtifactMetadata

        with self.assertRaises(ValidationError):
            TimeseriesArtifactMetadata(series=[{"label": "x", "source_kind": "model"}])


if __name__ == "__main__":
    unittest.main()
