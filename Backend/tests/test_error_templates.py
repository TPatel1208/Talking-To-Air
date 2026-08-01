import unittest


class RenderErrorAnswerTests(unittest.TestCase):
    def test_fills_stage_and_detail_into_the_categorys_template(self):
        from tta_backend.config.error_templates import render_error_answer

        text = render_error_answer("no_data", "coverage check", "no granules in the requested window.")

        self.assertIn("coverage check", text)
        self.assertIn("no granules in the requested window.", text)

    def test_missing_detail_falls_back_to_a_stated_fact_not_a_blank(self):
        from tta_backend.config.error_templates import render_error_answer

        text = render_error_answer("contract", "chat turn")

        self.assertIn("No further detail is available.", text)

    def test_default_detail_does_not_run_on_into_the_next_sentence(self):
        """Regression (live 2026-07-16): the contract template continues with
        "This has been logged..." after {detail}, and the unpunctuated default
        produced "...no further detail is available This has been logged"."""
        from tta_backend.config.error_templates import render_error_answer

        text = render_error_answer("contract", "chat turn")

        self.assertNotIn("available This", text)

    def test_an_unpunctuated_caller_detail_gains_terminal_punctuation(self):
        from tta_backend.config.error_templates import render_error_answer

        text = render_error_answer("contract", "chat turn", "socket closed unexpectedly")

        self.assertIn("socket closed unexpectedly. This has been logged", text)

    def test_unrecognized_category_falls_back_to_the_contract_template_instead_of_raising(self):
        from tta_backend.config.error_templates import render_error_answer

        text = render_error_answer("some_future_category_this_backend_does_not_know_yet", "chat turn", "detail")

        self.assertIn("internal error", text)

    def test_every_taxonomy_category_renders_without_error(self):
        from tta_backend.config.error_templates import render_error_answer

        for category in ("user_input", "no_data", "not_found", "too_large", "provider_unavailable", "contract"):
            text = render_error_answer(category, "stage", "detail")
            self.assertTrue(text)

    def test_rate_limited_answer_tells_the_user_to_wait_not_to_shrink_the_request(self):
        """A language-model quota/429 is a 'wait and retry' condition, not an
        'internal error' or a data-size problem — its answer must say so, so a
        researcher does not respond by needlessly shrinking a request that was
        never too large (the June 2023 AOD wildfire session, where every failure
        rendered the generic contract text and the agent kept suggesting a
        smaller region)."""
        from tta_backend.config.error_templates import CATEGORY_RATE_LIMITED, render_error_answer

        text = render_error_answer(CATEGORY_RATE_LIMITED, "earthdata agent")

        self.assertNotIn("internal error", text)
        self.assertIn("rate", text.lower())
        self.assertIn("try again", text.lower())

    def test_recursion_exhausted_answer_points_at_splitting_the_request(self):
        """A recursion-limit stop means the workflow ran out of step budget, not
        that the data or the question was bad — the answer must suggest fewer
        periods/regions per request rather than 'internal error'."""
        from tta_backend.config.error_templates import CATEGORY_RECURSION_EXHAUSTED, render_error_answer

        text = render_error_answer(CATEGORY_RECURSION_EXHAUSTED, "earthdata agent")

        self.assertNotIn("internal error", text)
        self.assertIn("step", text.lower())


class RenderTurnTimeoutAnswerTests(unittest.TestCase):
    def test_names_no_running_jobs_when_none_were_seen(self):
        from tta_backend.config.error_templates import render_turn_timeout_answer

        text = render_turn_timeout_answer([])

        self.assertIn("ran out of time", text)
        self.assertNotIn("Still running", text)

    def test_names_in_flight_job_handles_so_the_jobs_panel_story_stays_coherent(self):
        from tta_backend.config.error_templates import render_turn_timeout_answer

        text = render_turn_timeout_answer(["job_abc123", "job_def456"])

        self.assertIn("job_abc123", text)
        self.assertIn("job_def456", text)


class RenderScopeNoteTests(unittest.TestCase):
    def test_a_single_day_request_answered_by_a_monthly_mean_gets_a_disclosure_note(self):
        """T46 story #2: the substitution the researcher reads must say so —
        a one-day request answered with the monthly mean names both scopes."""
        from tta_backend.config.error_templates import render_scope_note

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
        from tta_backend.config.error_templates import render_scope_note

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
        from tta_backend.config.error_templates import render_scope_note

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
        from tta_backend.config.error_templates import render_scope_note

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
        from tta_backend.config.error_templates import render_scope_note

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

    def test_variable_note_names_the_chosen_field_and_alternatives_when_ambiguous(self):
        """T48: a high-ambiguity auto-pick discloses the chosen product AND the
        ranked alternatives, deterministically (no model prose), so a sensor
        fork is transparent and redirectable."""
        from tta_backend.config.error_templates import render_variable_note

        note = render_variable_note(
            "Terra MODIS Dark Target AOD 550",
            ["Aqua MODIS Dark Target AOD 550", "SNPP VIIRS Deep Blue AOD 550"],
        )

        self.assertIsNotNone(note)
        self.assertIn("Terra MODIS Dark Target AOD 550", note)
        self.assertIn("Aqua MODIS Dark Target AOD 550", note)
        self.assertIn("SNPP VIIRS Deep Blue AOD 550", note)

    def test_variable_note_is_a_brief_note_without_alternatives(self):
        """A medium-confidence lone pick gets a brief note naming only the
        chosen field -- no fork to list."""
        from tta_backend.config.error_templates import render_variable_note

        note = render_variable_note("Aerosol Optical Depth", [], ambiguous=False)

        self.assertIsNotNone(note)
        self.assertIn("Aerosol Optical Depth", note)
        self.assertNotIn("Other products available", note)

    def test_variable_note_is_none_without_a_chosen_label(self):
        from tta_backend.config.error_templates import render_variable_note

        self.assertIsNone(render_variable_note(None, ["a", "b"]))


if __name__ == "__main__":
    unittest.main()
