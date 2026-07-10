import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


class VariableChoiceRegistryTests(unittest.TestCase):
    """T25: the retrieval composite records the model's chosen science
    variable keyed by the eventual obs_/cube_ handle, so a later plot/stat/
    compare call on that same handle inherits the choice instead of hitting
    AggregationService.to_dataarray's candidate-listing error. In-memory,
    per-process -- mirrors ArtifactStore/GeocodingService's existing
    TTL-bounded caches; a handle is only ever resolved by the backend
    process that minted it."""

    def setUp(self):
        from services import variable_choice_registry

        variable_choice_registry._choices.clear()
        variable_choice_registry._pending.clear()

    def test_get_returns_none_for_a_handle_with_no_recorded_choice(self):
        from services.variable_choice_registry import get

        self.assertIsNone(get("obs_unknown"))

    def test_record_pending_then_finalize_makes_the_choice_available_by_handle(self):
        from services.variable_choice_registry import finalize, get, record_pending

        record_pending("job_1", "vertical_column_troposphere")
        self.assertIsNone(get("obs_1"))  # not yet finalized to a handle

        finalize("job_1", "obs_1")

        self.assertEqual(get("obs_1"), "vertical_column_troposphere")

    def test_finalize_is_a_no_op_when_no_choice_was_pending_for_that_job(self):
        from services.variable_choice_registry import finalize, get

        finalize("job_never_recorded", "obs_2")

        self.assertIsNone(get("obs_2"))

    def test_record_pending_ignores_an_empty_or_none_variable(self):
        """Ambiguous/no-choice submissions (0 or >1 variables requested)
        must not poison the registry with a wrong single guess."""
        from services.variable_choice_registry import finalize, get, record_pending

        record_pending("job_3", None)
        finalize("job_3", "obs_3")

        self.assertIsNone(get("obs_3"))


if __name__ == "__main__":
    unittest.main()
