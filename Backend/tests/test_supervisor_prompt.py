import unittest


class SupervisorPromptT25Phase4Tests(unittest.TestCase):
    """T25 Phase 4: the supervisor prompt states the same asymmetry as the
    earthdata agent's — satellite is universal over gridded Earthdata
    collections, ground/cross-source confirmation is air-quality-only — so
    routing never promises a ground check for a non-AQ satellite query."""

    def test_prompt_states_satellite_is_not_limited_to_air_quality(self):
        from tta_backend.config.supervisor_prompt import SUPERVISOR_PROMPT

        prompt = SUPERVISOR_PROMPT.lower()

        self.assertIn("any regularly-gridded", prompt)
        self.assertIn("not just air quality", prompt)

    def test_prompt_states_ground_is_air_quality_pollutants_only(self):
        from tta_backend.config.supervisor_prompt import SUPERVISOR_PROMPT

        prompt = SUPERVISOR_PROMPT.lower()

        self.assertIn("air quality pollutants", prompt)
        self.assertIn("no2, pm2.5, o3, so2, co", prompt)

    def test_prompt_forbids_promising_ground_confirmation_for_non_aq_queries(self):
        from tta_backend.config.supervisor_prompt import SUPERVISOR_PROMPT

        # Collapse whitespace so line wrapping in the prompt can't hide a phrase.
        prompt = " ".join(SUPERVISOR_PROMPT.lower().split())

        self.assertIn("never call ground, and never tell the user", prompt)
        self.assertIn("soil moisture", prompt)
        self.assertIn("land surface temperature", prompt)

    def test_ground_plus_satellite_routing_is_scoped_to_air_quality(self):
        from tta_backend.config.supervisor_prompt import SUPERVISOR_PROMPT

        prompt = " ".join(SUPERVISOR_PROMPT.lower().split())
        routing_idx = prompt.index("ground + satellite: user explicitly requests")

        self.assertIn("air-quality pollutants only", prompt[routing_idx:routing_idx + 300])


class SupervisorPlaceNameTests(unittest.TestCase):
    """The supervisor expanded "the OTC region" into "the Ozone Transport
    Commission (OTC) region (northeastern US)" in its delegation, and the
    subagent sent a spelling the region vocabulary does not hold. Since T64
    the rule lives on the location field the model fills: the Phase 0 gate
    saw it keep every location verbatim with no prompt rule at all."""

    def test_every_request_kinds_location_field_says_to_write_it_as_the_user_did(self):
        from tta_backend.models import dispatch_json_schema

        defs = dispatch_json_schema()["$defs"]

        for kind in ("SatelliteRequest", "GroundRequest"):
            with self.subTest(kind=kind):
                doc = defs[kind]["properties"]["location"]["description"].lower()
                self.assertIn("exactly as the researcher wrote it", doc)
                self.assertIn("do not expand acronyms", doc)


class SupervisorPromptDispatchTests(unittest.TestCase):
    """T64: the supervisor's only sub-agent tool is dispatch. A prompt still
    naming the retired ask_* tools, or the once-per-agent STOP rule the
    executor replaced, contradicts the tool the model actually has."""

    def test_prompt_names_only_the_dispatch_tool(self):
        from tta_backend.config.supervisor_prompt import SUPERVISOR_PROMPT

        self.assertIn("dispatch", SUPERVISOR_PROMPT)
        self.assertNotRegex(SUPERVISOR_PROMPT, r"ask_\w+_agent")

    def test_prompt_drops_the_once_per_agent_stop_rule(self):
        from tta_backend.config.supervisor_prompt import SUPERVISOR_PROMPT

        prompt = " ".join(SUPERVISOR_PROMPT.lower().split())

        self.assertNotIn("exactly once", prompt)
        self.assertNotIn("already retrieved", prompt)

    def test_prompt_leaves_the_data_sources_header_to_the_server(self):
        """D14: the header is computed from what ran; a model told to write
        one can claim a dataset or check that never happened."""
        from tta_backend.config.supervisor_prompt import SUPERVISOR_PROMPT

        prompt = " ".join(SUPERVISOR_PROMPT.lower().split())

        self.assertNotIn("must start with", prompt)
        self.assertIn("do not write a \"data sources\" line", prompt)

    def test_a_plot_of_an_air_quality_pollutant_is_a_satellite_request_alone(self):
        """Phase 0 F5: asked to plot ozone over the OTC, a lite model added an
        unrequested ground request."""
        from tta_backend.config.supervisor_prompt import SUPERVISOR_PROMPT

        prompt = " ".join(SUPERVISOR_PROMPT.lower().split())

        self.assertIn("even for an air-quality pollutant", prompt)

    def test_a_satellite_vs_ground_co_location_is_a_satellite_compare_request(self):
        """The satellite agent's ground tools run only for intent compare, so
        the supervisor must know that is how to ask for them."""
        from tta_backend.config.supervisor_prompt import SUPERVISOR_PROMPT

        prompt = " ".join(SUPERVISOR_PROMPT.lower().split())

        self.assertIn("satellite request with intent compare", prompt)

    def test_a_dependent_task_goes_in_a_second_round_with_prior_findings(self):
        from tta_backend.config.supervisor_prompt import SUPERVISOR_PROMPT

        prompt = " ".join(SUPERVISOR_PROMPT.lower().split())

        self.assertIn("second dispatch call", prompt)
        self.assertIn("prior_findings", prompt)


if __name__ == "__main__":
    unittest.main()
