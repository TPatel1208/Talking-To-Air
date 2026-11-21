"""Checking answers to a paused run's question, and how a capability's question takes its answer."""
import unittest
from typing import Any

from tests.test_harness_workflow_runner import DATASETS, Script, fake_registry, make, spec
from tta_backend.harness.capabilities import Question
from tta_backend.harness.workflows.answers import (
    MAX_ITEMS,
    MAX_TEXT_CHARS,
    AnswerRejected,
    answerable,
    apply_rule,
    capability_rule,
    check_answer,
    item_type,
    question_payload,
)
from tta_backend.harness.workflows.loader import SpecErrors, load_specs

RUN = "run_" + "c" * 32


def asked(kind: str, **offered: Any) -> dict[str, Any]:
    return question_payload(RUN, "step", kind, "Which?", **offered)  # type: ignore[arg-type]


class CheckTests(unittest.TestCase):
    def test_pick_one_takes_an_option_as_given_or_by_its_text(self):
        question = asked("pick_one", options=["TEMPO_NO2", {"label": "OMI NO2", "value": "OMI"}, 3, True])
        self.assertEqual(check_answer(question, "tempo_no2 "), "TEMPO_NO2")
        self.assertEqual(check_answer(question, "omi  no2"), {"label": "OMI NO2", "value": "OMI"})
        self.assertEqual(check_answer(question, 3), 3)
        self.assertIs(check_answer(question, True), True)
        for bad in ("TROPOMI", ["TEMPO_NO2"], 1, "", {"label": "other"}):
            with self.subTest(bad=bad), self.assertRaises(AnswerRejected) as caught:
                check_answer(question, bad)
            self.assertEqual(caught.exception.reason,
                             'The answer must be exactly one of the options: TEMPO_NO2 | OMI NO2 | 3 | true.')

    def test_the_wording_lists_ten_options_then_how_many_more(self):
        question = asked("pick_one", options=[f"o{n}" for n in range(14)])
        with self.assertRaises(AnswerRejected) as caught:
            check_answer(question, "x")
        self.assertTrue(caught.exception.reason.endswith("o8 | o9 and 4 more."), caught.exception.reason)

    def test_pick_many_takes_a_non_empty_subset_once_each(self):
        question = asked("pick_many", options=["a", "b", "c"])
        self.assertEqual(check_answer(question, ["C", "a"]), ["c", "a"])
        self.assertEqual(check_answer(question, "b"), ["b"])
        for bad in ([], ["a", "A"], ["d"], [["a"]]):
            with self.subTest(bad=bad), self.assertRaises(AnswerRejected):
                check_answer(question, bad)

    def test_confirm_list_keeps_the_type_of_the_offered_items(self):
        dates = asked("confirm_list", items=["2024-06-01", "2024-06-02"])
        self.assertEqual(check_answer(dates, ["2024-06-02", " 2024-06-09", "2024-06-02"]), ["2024-06-02", "2024-06-09"])
        for bad in (["2024-02-30"], ["June 9"], [20240609], "2024-06-01", [], ["2024-06-01"] * 0):
            with self.subTest(bad=bad), self.assertRaises(AnswerRejected) as caught:
                check_answer(dates, bad)
            self.assertEqual(caught.exception.reason,
                             f"The answer must be a list of 1 to {MAX_ITEMS} ISO dates, like 2024-06-01.")
        with self.assertRaises(AnswerRejected):
            check_answer(dates, ["2024-06-01"] * (MAX_ITEMS + 1))

        numbers = asked("confirm_list", items=[1, 2.5])
        self.assertEqual(check_answer(numbers, [3, 1]), [3, 1])
        for bad in ([True], ["3"], [float("nan")]):
            with self.subTest(bad=bad), self.assertRaises(AnswerRejected):
                check_answer(numbers, bad)
        sites = asked("confirm_list", items=[{"id": "a", "name": "A"}])
        self.assertEqual(check_answer(sites, [{"id": "b", "name": "B"}]), [{"id": "b", "name": "B"}])
        with self.assertRaises(AnswerRejected):
            check_answer(sites, [{"id": "b"}])

    def test_item_types(self):
        cases = [
            (["2024-06-01"], "date"), ([True, False], "boolean"), ([1, 2.0], "number"), (["a", "2024-06-01"], "text"),
            ([{"a": 1}, {"a": 2}], "object"), ([{"a": 1}, {"b": 2}], "any"), ([1, "a"], "any"), ([], "any"),
        ]
        for items, expected in cases:
            with self.subTest(items=items):
                self.assertEqual(item_type(items), expected)

    def test_text_is_trimmed_bounded_and_plain(self):
        question = asked("text")
        self.assertEqual(check_answer(question, "  Newark  "), "Newark")
        self.assertEqual(len(str(check_answer(question, "x" * MAX_TEXT_CHARS))), MAX_TEXT_CHARS)
        for bad in ("", "  ", "x" * (MAX_TEXT_CHARS + 1), "a\x00b", 3, ["a"]):
            with self.subTest(bad=bad), self.assertRaises(AnswerRejected) as caught:
                check_answer(question, bad)
            self.assertEqual(caught.exception.reason, f"The answer must be text of 1 to {MAX_TEXT_CHARS} characters.")

    def test_the_payload_and_which_questions_take_an_answer(self):
        self.assertEqual(asked("pick_one", options=["a"]), {
            "run_id": RUN, "step": "step", "kind": "pick_one", "question": "Which?", "prompt": "Which?", "options": ["a"],
        })
        self.assertEqual(asked("text"), {"run_id": RUN, "step": "step", "kind": "text", "question": "Which?",
                                         "prompt": "Which?"})
        self.assertTrue(answerable(asked("text")))
        self.assertTrue(answerable(asked("text"), RUN))
        self.assertFalse(answerable(asked("text"), "run_" + "d" * 32))
        self.assertFalse(answerable({"kind": "inputs", "inputs": []}))
        self.assertFalse(answerable({"kind": "text", "prompt": "Where?", "input": "area"}))
        self.assertFalse(answerable(None))


class CapabilityRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = fake_registry(Script())

    def test_a_question_naming_its_parameter_passes_the_answer_as_it(self):
        rule = capability_rule(self.registry.get("ops.ask"), Question(
            kind="pick_one", prompt="Which?", options=("a",), param="level",
        ))
        assert rule is not None
        self.assertEqual(apply_rule(rule, "a"), {"level": "a"})
        # A parameter the capability does not declare cannot take it.
        self.assertIsNone(capability_rule(self.registry.get("ops.ask"), Question(
            kind="pick_one", prompt="Which?", options=("a",), param="nothing",
        )))

    def test_the_size_confirmation_sets_confirmed_or_stops(self):
        question = Question(kind="pick_one", prompt="Large.", options=("confirm", "narrow"))
        rule = capability_rule(self.registry.get("ops.size"), question)
        assert rule is not None
        self.assertEqual(apply_rule(rule, "confirm"), {"confirmed": True})
        self.assertIsNone(apply_rule(rule, "narrow"))
        # The same options on a capability without confirmed have no rule.
        self.assertIsNone(capability_rule(self.registry.get("ops.ask_blind"), question))

    def test_an_unset_param_is_not_part_of_the_question_dump(self):
        self.assertNotIn("param", Question(kind="text", prompt="Where?").model_dump(mode="json"))
        self.assertEqual(Question(kind="text", prompt="Where?", param="place").model_dump()["param"], "place")


class LoaderTests(unittest.TestCase):
    def test_a_step_may_read_an_ask_steps_answer_as_a_value_but_not_a_use_steps_handle(self):
        registry = fake_registry(Script())
        load_specs({"ok": spec(
            [{"id": "pick", "ask": {"kind": "text", "question": "Which?"}}, make("after", product="$pick")],
            {"data": ["$after"]}, inputs={},
        )}, registry, DATASETS)
        with self.assertRaises(SpecErrors) as caught:
            load_specs({"bad": spec(
                [make("base"), make("after", product="$base")], {"data": ["$after"]}, inputs={},
            )}, registry, DATASETS)
        self.assertIn("with.product takes a value, not a handle", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
