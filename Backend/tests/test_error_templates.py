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


class RenderVariableNoteTests(unittest.TestCase):
    def test_variable_note_names_the_chosen_field_and_alternatives_when_ambiguous(self):
        """A high-ambiguity auto-pick discloses the chosen product AND the
        ranked alternatives, deterministically (no model prose), so a sensor
        fork is transparent and redirectable."""
        from tta_backend.domain.aggregation.variable_resolver import render_variable_note

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
        from tta_backend.domain.aggregation.variable_resolver import render_variable_note

        note = render_variable_note("Aerosol Optical Depth", [], ambiguous=False)

        self.assertIsNotNone(note)
        self.assertIn("Aerosol Optical Depth", note)
        self.assertNotIn("Other products available", note)

    def test_variable_note_is_none_without_a_chosen_label(self):
        from tta_backend.domain.aggregation.variable_resolver import render_variable_note

        self.assertIsNone(render_variable_note(None, ["a", "b"]))


if __name__ == "__main__":
    unittest.main()
