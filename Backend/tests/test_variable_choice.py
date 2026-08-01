import importlib.util
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


class VariableChoiceModelTests(unittest.TestCase):
    def test_agent_result_variable_choice_round_trips_through_json(self):
        """Tracer: the deterministic picker payload survives the same
        agent_result_to_json -> parse_agent_result trip the SSE layer runs it
        through, so a low/medium-confidence turn's candidate list reaches the
        frontend byte-for-byte as the resolver computed it."""
        from tta_backend.models import parse_agent_result
        from tta_backend.models.agent_result import (
            AgentResult,
            VariableChoice,
            VariableChoiceOption,
            agent_result_to_json,
        )

        result = AgentResult(
            text="",
            variable_choice=VariableChoice(
                message="This dataset has 3 variables I can't confidently narrow down — pick one:",
                candidates=[
                    VariableChoiceOption(
                        name="Terra/AOD_550/Mean",
                        category="distinct",
                        units="1",
                        valid_fraction=0.82,
                        reasons=["has units", "aggregated mean field"],
                        prompt="plot AOD over New Jersey last week using Terra/AOD_550/Mean",
                    ),
                ],
            ),
        )

        restored = parse_agent_result(agent_result_to_json(result))

        self.assertIsNotNone(restored.variable_choice)
        self.assertEqual(
            restored.variable_choice.message,
            "This dataset has 3 variables I can't confidently narrow down — pick one:",
        )
        self.assertEqual(len(restored.variable_choice.candidates), 1)
        opt = restored.variable_choice.candidates[0]
        self.assertEqual(opt.name, "Terra/AOD_550/Mean")
        self.assertEqual(opt.category, "distinct")
        self.assertEqual(opt.units, "1")
        self.assertEqual(opt.valid_fraction, 0.82)
        self.assertEqual(opt.reasons, ["has units", "aggregated mean field"])
        self.assertEqual(
            opt.prompt, "plot AOD over New Jersey last week using Terra/AOD_550/Mean",
        )

    def test_agent_result_defaults_variable_choice_to_none(self):
        """The overwhelming majority of turns (high-confidence or non-variable)
        carry no picker — the field is optional and omitted, never an empty
        payload the frontend must special-case."""
        from tta_backend.models import parse_agent_result
        from tta_backend.models.agent_result import AgentResult, agent_result_to_json

        restored = parse_agent_result(agent_result_to_json(AgentResult(text="ok")))

        self.assertIsNone(restored.variable_choice)


@unittest.skipIf(importlib.util.find_spec("xarray") is None, "xarray is not installed")
class BuildVariableChoiceTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import xarray as xr

        self.np = np
        self.xr = xr

    def _var(self, values, attrs=None):
        return (("lat", "lon"), self.np.array(values, dtype=float), attrs or {})

    def test_low_confidence_refusal_builds_a_picker_over_all_candidates(self):
        """Tracer: a genuinely ambiguous file (two weak, name-only distinct
        products) resolves to no name; the builder turns the resolver's own
        ranked candidates into a picker payload whose message names the count
        and whose options carry every non-empty candidate."""
        from tta_backend.models.agent_result import VariableChoice
        from tta_backend.preprocessing.variable_choice_builder import build_variable_choice
        from tta_backend.preprocessing.variable_resolver import resolve

        ds = self.xr.Dataset({
            "DT_AOD_550_AVG": self._var([[0.1, 0.2]]),
            "COMBINE_AOD_550_AVG": self._var([[0.3, 0.4]]),
        })
        res = resolve(ds)
        self.assertIsNone(res.name)  # low-confidence refusal, precondition

        choice = build_variable_choice(res, ds)

        self.assertIsInstance(choice, VariableChoice)
        names = {c.name for c in choice.candidates}
        self.assertEqual(names, {"DT_AOD_550_AVG", "COMBINE_AOD_550_AVG"})
        self.assertIn("2", choice.message)

    def test_each_option_carries_units_read_from_the_file_attrs(self):
        """A picker row shows units so the researcher can tell the fields
        apart -- read straight off each variable's CF ``units`` attr, never
        invented; a variable with no units attr yields None, not a guess."""
        from tta_backend.preprocessing.variable_choice_builder import build_variable_choice
        from tta_backend.preprocessing.variable_resolver import resolve

        ds = self.xr.Dataset({
            "AOD_550_AVG": self._var([[0.1, 0.2]], {"units": "1"}),
            "COMBINE_AOD_550_AVG": self._var([[0.3, 0.4]]),  # no units attr
        })
        choice = build_variable_choice(resolve(ds), ds)

        by_name = {c.name: c for c in choice.candidates}
        self.assertEqual(by_name["AOD_550_AVG"].units, "1")
        self.assertIsNone(by_name["COMBINE_AOD_550_AVG"].units)

    def test_a_pathologically_wide_file_shows_every_candidate_uncapped(self):
        """The AERDA shape (432 collided leaves) is why T49 exists: the picker
        shows every real candidate, grouped/searchable, never a 20-item prose
        excerpt. All 400 non-empty distinct fields must be present."""
        from tta_backend.preprocessing.variable_choice_builder import build_variable_choice
        from tta_backend.preprocessing.variable_resolver import resolve

        ds = self.xr.Dataset({
            f"field_{i:03d}": self._var([[float(i), float(i) + 1]]) for i in range(400)
        })
        choice = build_variable_choice(resolve(ds), ds)

        self.assertEqual(len(choice.candidates), 400)

    def test_empty_candidates_are_excluded_and_counted_in_the_message(self):
        """A field with no data over the range would plot blank, so it is not
        offered -- but the count of such exclusions is disclosed, so the list
        reads as the populated subset, not the whole file."""
        from tta_backend.preprocessing.variable_choice_builder import build_variable_choice
        from tta_backend.preprocessing.variable_resolver import resolve

        ds = self.xr.Dataset({
            "DT_AOD_550_AVG": self._var([[0.1, 0.2]]),
            "COMBINE_AOD_550_AVG": self._var([[0.3, 0.4]]),
            "Empty_AOD_550_AVG": self._var([[self.np.nan, self.np.nan]]),
        })
        choice = build_variable_choice(resolve(ds), ds)

        names = {c.name for c in choice.candidates}
        self.assertEqual(names, {"DT_AOD_550_AVG", "COMBINE_AOD_550_AVG"})
        self.assertIn("1 variable excluded", choice.message)

    def test_medium_confidence_message_names_the_auto_pick(self):
        """When the resolver auto-picked at medium confidence, the picker frames
        the list as 'showing X, pick another if not' -- the answer already
        arrived; the picker is the one-click override, not a block."""
        from tta_backend.preprocessing.variable_choice_builder import build_variable_choice
        from tta_backend.preprocessing.variable_resolver import resolve

        # A single weakly-signalled pickable field -> medium confidence, auto-pick.
        ds = self.xr.Dataset({
            "aerosol_optical_depth_average": self._var([[0.1, 0.2]]),
        })
        res = resolve(ds)
        self.assertEqual(res.resolution_confidence, "medium")

        choice = build_variable_choice(res, ds)

        self.assertIn("aerosol_optical_depth_average", choice.message)
        self.assertIn(
            "aerosol_optical_depth_average", {c.name for c in choice.candidates},
        )

    def test_fill_prompts_reconstructs_the_full_original_request(self):
        """Clicking a candidate re-sends the WHOLE original request with the
        variable appended -- a self-contained turn carrying its own region/time,
        never a bare 'use <var>' that leans on cross-turn context carryover."""
        from tta_backend.preprocessing.variable_choice_builder import build_variable_choice, fill_prompts
        from tta_backend.preprocessing.variable_resolver import resolve

        ds = self.xr.Dataset({
            "DT_AOD_550_AVG": self._var([[0.1, 0.2]]),
            "COMBINE_AOD_550_AVG": self._var([[0.3, 0.4]]),
        })
        choice = fill_prompts(
            build_variable_choice(resolve(ds), ds),
            "plot AOD over New Jersey last week",
        )

        prompts = {c.name: c.prompt for c in choice.candidates}
        self.assertEqual(
            prompts["DT_AOD_550_AVG"],
            "plot AOD over New Jersey last week using DT_AOD_550_AVG",
        )
        self.assertEqual(
            prompts["COMBINE_AOD_550_AVG"],
            "plot AOD over New Jersey last week using COMBINE_AOD_550_AVG",
        )


    def test_emit_variable_choice_payload_publishes_the_built_picker(self):
        """The tool-layer seam: on catching the short-circuit, the tool builds
        the picker from the signal's resolution and emits it out-of-band. Assert
        the emitted payload is the deterministic candidate list, not model text."""
        from tta_backend.preprocessing.variable_choice_builder import emit_variable_choice_payload
        from tta_backend.preprocessing.variable_resolver import resolve
        from tta_backend.utils import streaming

        ds = self.xr.Dataset({
            "DT_AOD_550_AVG": self._var([[0.1, 0.2]]),
            "COMBINE_AOD_550_AVG": self._var([[0.3, 0.4]]),
        })
        captured = []
        token = streaming._variable_choice_emitter.set(lambda p: captured.append(p))
        try:
            emit_variable_choice_payload(resolve(ds), ds)
        finally:
            streaming._variable_choice_emitter.reset(token)

        self.assertEqual(len(captured), 1)
        names = {c["name"] for c in captured[0]["candidates"]}
        self.assertEqual(names, {"DT_AOD_550_AVG", "COMBINE_AOD_550_AVG"})

    def test_medium_confidence_pick_stashes_a_picker_in_the_aggregate_meta(self):
        """A medium-confidence auto-pick (one weakly-signalled science field
        beside implementation plumbing) delivers its answer AND rides an
        override picker out in meta.variable_resolution.variable_choice -- the
        channel the tool copies into chart provenance and dispatch lifts into
        AgentResult.variable_choice."""
        from tta_backend.preprocessing.aggregation_service import AggregationService

        ds = self.xr.Dataset(
            {
                "aerosol_optical_depth_average": (("lat", "lon"), [[0.1, 0.2]], {}),
                "solar_zenith_angle": (("lat", "lon"), [[30.0, 31.0]], {}),
            },
            coords={"lat": [40.0], "lon": [-75.0, -74.0]},
        )

        meta = AggregationService().aggregate(ds, stat="mean").meta

        resolution = meta["variable_resolution"]
        self.assertEqual(resolution["resolution_confidence"], "medium")
        self.assertIn("variable_choice", resolution)
        picker = resolution["variable_choice"]
        names = {c["name"] for c in picker["candidates"]}
        self.assertIn("aerosol_optical_depth_average", names)
        # Prompts are left blank here -- the dispatch layer fills them from the
        # original request (it holds the researcher's actual wording).
        self.assertTrue(all(c["prompt"] == "" for c in picker["candidates"]))


if __name__ == "__main__":
    unittest.main()
