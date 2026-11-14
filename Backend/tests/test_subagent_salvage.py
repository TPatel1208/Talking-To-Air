import importlib.util
import unittest

_HAS_DEPENDENCIES = all(importlib.util.find_spec(name) is not None for name in ("langchain", "pydantic"))


@unittest.skipUnless(_HAS_DEPENDENCIES, "langchain is not installed")
class FinalizeSubAgentResultSalvageTests(unittest.TestCase):
    """A sub-agent whose final message is prose, not the structured envelope,
    still returns a non-error answer that keeps what its tools produced."""

    def _finalize(self):
        from tta_backend.models import AgentResult, ChartPayload
        from tta_backend.models.artifact import ArtifactReference
        from tta_backend.services.subagent_dispatch import _finalize_sub_agent_result

        raw = AgentResult(
            text="I plotted TROPOMI NO2 over New Jersey for 2024-01-15; the map is attached above.",
            charts=[ChartPayload(type="heatmap", title="TROPOMI NO2 over NJ")],
            artifacts=[ArtifactReference(id="map_salvage_1", type="map", title="TROPOMI NO2 over NJ")],
        )
        return raw, _finalize_sub_agent_result(raw, "earthdata")

    def test_a_prose_final_message_is_salvaged_without_an_error(self):
        _, finalized = self._finalize()

        self.assertFalse(finalized.metadata.get("error"))
        self.assertIs(finalized.metadata.get("salvaged"), True)

    def test_the_salvaged_answer_keeps_the_artifact(self):
        _, finalized = self._finalize()

        self.assertEqual([artifact.id for artifact in finalized.artifacts], ["map_salvage_1"])


if __name__ == "__main__":
    unittest.main()
