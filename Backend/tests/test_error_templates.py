import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


class RenderErrorAnswerTests(unittest.TestCase):
    def test_fills_stage_and_detail_into_the_categorys_template(self):
        from config.error_templates import render_error_answer

        text = render_error_answer("no_data", "coverage check", "no granules in the requested window.")

        self.assertIn("coverage check", text)
        self.assertIn("no granules in the requested window.", text)

    def test_missing_detail_falls_back_to_a_stated_fact_not_a_blank(self):
        from config.error_templates import render_error_answer

        text = render_error_answer("contract", "chat turn")

        self.assertIn("No further detail is available.", text)

    def test_default_detail_does_not_run_on_into_the_next_sentence(self):
        """Regression (live 2026-07-16): the contract template continues with
        "This has been logged..." after {detail}, and the unpunctuated default
        produced "...no further detail is available This has been logged"."""
        from config.error_templates import render_error_answer

        text = render_error_answer("contract", "chat turn")

        self.assertNotIn("available This", text)

    def test_an_unpunctuated_caller_detail_gains_terminal_punctuation(self):
        from config.error_templates import render_error_answer

        text = render_error_answer("contract", "chat turn", "socket closed unexpectedly")

        self.assertIn("socket closed unexpectedly. This has been logged", text)

    def test_unrecognized_category_falls_back_to_the_contract_template_instead_of_raising(self):
        from config.error_templates import render_error_answer

        text = render_error_answer("some_future_category_this_backend_does_not_know_yet", "chat turn", "detail")

        self.assertIn("internal error", text)

    def test_every_taxonomy_category_renders_without_error(self):
        from config.error_templates import render_error_answer

        for category in ("user_input", "no_data", "not_found", "too_large", "provider_unavailable", "contract"):
            text = render_error_answer(category, "stage", "detail")
            self.assertTrue(text)


class RenderTurnTimeoutAnswerTests(unittest.TestCase):
    def test_names_no_running_jobs_when_none_were_seen(self):
        from config.error_templates import render_turn_timeout_answer

        text = render_turn_timeout_answer([])

        self.assertIn("ran out of time", text)
        self.assertNotIn("Still running", text)

    def test_names_in_flight_job_handles_so_the_jobs_panel_story_stays_coherent(self):
        from config.error_templates import render_turn_timeout_answer

        text = render_turn_timeout_answer(["job_abc123", "job_def456"])

        self.assertIn("job_abc123", text)
        self.assertIn("job_def456", text)


class RenderScopeNoteTests(unittest.TestCase):
    def test_a_single_day_request_answered_by_a_monthly_mean_gets_a_disclosure_note(self):
        """T46 story #2: the substitution the researcher reads must say so —
        a one-day request answered with the monthly mean names both scopes."""
        from config.error_templates import render_scope_note

        note = render_scope_note(
            {"location": "California", "time_range": "2024-07-15/2024-07-15"},
            {
                "region_name": "California",
                "start_date": "2024-07-01T00:00:00",
                "end_date": "2024-07-31T23:59:59",
                "cadence": "monthly",
            },
        )

        self.assertIsNotNone(note)
        self.assertIn("2024-07-15", note)
        self.assertIn("monthly", note.lower())

    def test_an_exact_match_adds_no_note(self):
        """Regression: don't nag when delivered scope equals the request."""
        from config.error_templates import render_scope_note

        note = render_scope_note(
            {"location": "California", "time_range": "2024-07-01/2024-07-31"},
            {
                "region_name": "California",
                "start_date": "2024-07-01T00:00:00",
                "end_date": "2024-07-31T00:00:00",
                "cadence": "monthly",
            },
        )

        self.assertIsNone(note)

    def test_canonicalization_suffix_is_not_a_false_substitution(self):
        """Finding #10: the geocoder canonicalizes "California" to
        "California, United States". Same place, extra detail — the delivered
        region only appends a canonicalization suffix, so it must NOT read as a
        substitution ("you asked about California, but ... California, United
        States")."""
        from config.error_templates import render_scope_note

        note = render_scope_note(
            {"location": "California", "time_range": "2024-07-01/2024-07-31"},
            {
                "region_name": "California, United States",
                "start_date": "2024-07-01T00:00:00",
                "end_date": "2024-07-31T00:00:00",
                "cadence": "monthly",
            },
        )

        self.assertIsNone(note)

    def test_an_abbreviated_component_is_not_a_false_substitution(self):
        """Finding #10: "Los Angeles, CA" and the canonical "Los Angeles,
        California, United States" name the same place — the leading locality
        matches, so the mid-string CA→California expansion is not a
        substitution."""
        from config.error_templates import render_scope_note

        note = render_scope_note(
            {"location": "Los Angeles, CA", "time_range": "2024-07-01/2024-07-31"},
            {
                "region_name": "Los Angeles, California, United States",
                "start_date": "2024-07-01T00:00:00",
                "end_date": "2024-07-31T00:00:00",
                "cadence": "monthly",
            },
        )

        self.assertIsNone(note)

    def test_a_genuine_region_substitution_still_warns(self):
        """Finding #10 must not silence real substitutions: a request for one
        place answered with a different place still names both."""
        from config.error_templates import render_scope_note

        note = render_scope_note(
            {"location": "California", "time_range": "2024-07-01/2024-07-31"},
            {
                "region_name": "Nevada, United States",
                "start_date": "2024-07-01T00:00:00",
                "end_date": "2024-07-31T00:00:00",
                "cadence": "monthly",
            },
        )

        self.assertIsNotNone(note)
        self.assertIn("California", note)
        self.assertIn("Nevada", note)


if __name__ == "__main__":
    unittest.main()
