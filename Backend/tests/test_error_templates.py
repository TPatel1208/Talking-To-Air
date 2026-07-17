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


if __name__ == "__main__":
    unittest.main()
