import unittest


class ScopeRegistryTests(unittest.TestCase):
    def test_records_requested_scope_by_job_then_finalizes_to_the_resolved_handle(self):
        """T46: safe_retrieve knows the requested scope synchronously, but the
        obs_/cube_ handle a plot will read it back by is only known once the
        job reaches 'ready' — the same two-step handoff variable_choice_registry
        uses."""
        from tta_backend.services import scope_registry

        scope = {"location": "California", "time_range": "2024-07-15/2024-07-15"}
        scope_registry.record_pending("job_1", scope)
        scope_registry.finalize("job_1", "obs_abc")

        self.assertEqual(scope_registry.get("obs_abc"), scope)

    def test_get_returns_none_for_an_unknown_handle(self):
        from tta_backend.services import scope_registry

        self.assertIsNone(scope_registry.get("obs_never_recorded"))

    def test_record_pending_is_a_noop_without_a_job_handle_or_scope(self):
        from tta_backend.services import scope_registry

        scope_registry.record_pending("", {"location": "X"})
        scope_registry.record_pending("job_2", {})
        scope_registry.finalize("job_2", "obs_def")

        self.assertIsNone(scope_registry.get("obs_def"))


if __name__ == "__main__":
    unittest.main()
