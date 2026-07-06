import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


class BuildArtifactReferenceMapTests(unittest.TestCase):
    def test_builds_a_map_artifact_from_a_heatmap_payload(self):
        from services.artifact_registry import build_artifact_reference

        payload = {
            "chart_id": "map_abc123",
            "type": "heatmap",
            "title": "TEMPO NO2 over New Jersey",
            "variable": "TEMPO_NO2",
            "units": "mol/m^2",
            "vmin": 0.0,
            "vmax": 1.0,
            "bounds": [-75.0, 39.0, -73.0, 41.0],
            "metadata": {"source_handles": ["obs_1"]},
        }

        ref = build_artifact_reference(payload)

        self.assertIsNotNone(ref)
        self.assertEqual(ref.id, "map_abc123")
        self.assertEqual(ref.type, "map")
        self.assertEqual(ref.title, "TEMPO NO2 over New Jersey")
        self.assertEqual(ref.metadata["bbox"], [-75.0, 39.0, -73.0, 41.0])
        self.assertEqual(ref.metadata["colorbar"], {"vmin": 0.0, "vmax": 1.0})
        self.assertEqual(ref.metadata["source_handles"], ["obs_1"])

    def test_rejects_a_heatmap_payload_missing_bounds(self):
        from services.artifact_registry import build_artifact_reference

        payload = {
            "chart_id": "map_bad",
            "type": "heatmap",
            "title": "Broken map",
            "variable": "TEMPO_NO2",
            "units": "mol/m^2",
            "vmin": 0.0,
            "vmax": 1.0,
            "metadata": {},
        }

        with self.assertRaises(Exception):
            build_artifact_reference(payload)

    def test_returns_none_for_a_render_type_with_no_artifact_mapping(self):
        from services.artifact_registry import build_artifact_reference

        ref = build_artifact_reference({"type": "table", "title": "n/a"})

        self.assertIsNone(ref)


class BuildArtifactReferenceComparisonTests(unittest.TestCase):
    def test_builds_a_comparison_artifact_from_a_heatmap_multi_payload(self):
        from services.artifact_registry import build_artifact_reference

        payload = {
            "chart_id": "cmp_abc123",
            "type": "heatmap_multi",
            "title": "TEMPO NO2 Comparison",
            "panels": [
                {"title": "New Jersey", "metadata": {"source_handles": ["obs_1"]}},
                {"title": "New York", "metadata": {"source_handles": ["obs_2"]}},
            ],
            "metadata": {"source_handles": ["obs_1", "obs_2"]},
        }

        ref = build_artifact_reference(payload)

        self.assertEqual(ref.type, "comparison")
        self.assertEqual(ref.metadata["mode"], "n-panel")
        self.assertEqual(len(ref.metadata["panels"]), 2)
        self.assertEqual(ref.metadata["panels"][0]["handle"], "obs_1")
        self.assertEqual(ref.metadata["panels"][0]["title"], "New Jersey")
        self.assertEqual(ref.metadata["source_handles"], ["obs_1", "obs_2"])

    def test_rejects_a_heatmap_multi_payload_with_only_one_panel(self):
        from services.artifact_registry import build_artifact_reference

        payload = {
            "chart_id": "cmp_bad",
            "type": "heatmap_multi",
            "title": "Broken comparison",
            "panels": [{"title": "Solo", "metadata": {"source_handles": ["obs_1"]}}],
            "metadata": {"source_handles": ["obs_1"]},
        }

        with self.assertRaises(Exception):
            build_artifact_reference(payload)


class BuildArtifactReferenceTimeseriesTests(unittest.TestCase):
    def test_builds_a_timeseries_artifact_from_a_timeseries_payload(self):
        from services.artifact_registry import build_artifact_reference

        payload = {
            "chart_id": "ts_abc123",
            "type": "timeseries",
            "title": "TEMPO NO2 mean over New Jersey",
            "variable": "TEMPO_NO2",
            "units": "mol/m^2",
            "stat": "mean",
            "times": ["2024-01-01T00:00:00Z"],
            "values": [1.0],
            "metadata": {"source_handles": ["obs_1"]},
        }

        ref = build_artifact_reference(payload)

        self.assertEqual(ref.type, "timeseries")
        self.assertEqual(len(ref.metadata["series"]), 1)
        self.assertEqual(ref.metadata["series"][0]["source_kind"], "satellite")
        self.assertEqual(ref.metadata["series"][0]["label"], "TEMPO NO2 mean over New Jersey")
        self.assertEqual(ref.metadata["source_handles"], ["obs_1"])


if __name__ == "__main__":
    unittest.main()
