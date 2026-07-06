import json
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


class ArtifactRefsFromContentTests(unittest.TestCase):
    def test_extracts_a_map_artifact_ref_from_tool_content(self):
        from agents.supervisor_agent import _artifact_refs_from_content

        content = json.dumps({
            "type": "heatmap",
            "chart_id": "map_abc123",
            "title": "TEMPO over NJ",
            "_artifact_refs": [{
                "id": "map_abc123",
                "type": "map",
                "title": "TEMPO over NJ",
                "metadata": {
                    "bbox": [-75.0, 39.0, -73.0, 41.0],
                    "variable": "TEMPO_NO2",
                    "units": "mol/m^2",
                    "colorbar": {"vmin": 0.0, "vmax": 1.0},
                    "source_handles": ["obs_1"],
                },
            }],
        })

        refs = _artifact_refs_from_content(content)

        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].id, "map_abc123")
        self.assertEqual(refs[0].type, "map")

    def test_returns_empty_list_for_content_with_no_artifact_refs(self):
        from agents.supervisor_agent import _artifact_refs_from_content

        self.assertEqual(_artifact_refs_from_content("plain text"), [])
        self.assertEqual(_artifact_refs_from_content(json.dumps({"type": "heatmap"})), [])

    def test_extract_artifact_refs_from_messages_uses_the_shared_helper(self):
        from types import SimpleNamespace
        from agents.supervisor_agent import _extract_artifact_refs

        table_ref = {"id": "tbl_1", "type": "table", "title": "EPA Summary"}
        messages = [
            SimpleNamespace(name="find_closest_monitor", content=json.dumps({"_artifact_refs": [table_ref]})),
        ]

        refs = _extract_artifact_refs(messages)

        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].id, "tbl_1")


if __name__ == "__main__":
    unittest.main()
