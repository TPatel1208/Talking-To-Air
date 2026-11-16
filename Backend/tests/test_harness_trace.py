"""Trace entry and run validation, without a store."""
import datetime
import unittest

from tta_backend.harness.release import HARNESS_RELEASE
from tta_backend.harness.results import RUN_ID_PATTERN, Mode
from tta_backend.harness.trace import (
    MAX_JSON_BYTES,
    FailureCategory,
    InvalidTrace,
    NewEntry,
    Run,
    new_run_id,
    serialize_json,
)

T0 = datetime.datetime(2026, 11, 11, 14, 2, 7, tzinfo=datetime.UTC)
RUN_ID = "run_" + "1" * 32


def entry(**over):
    base = {"run_id": RUN_ID, "kind": "use", "capability": "operations.anomaly",
            "status": "done", "started": T0, "seconds": 1.0}
    return NewEntry(**{**base, **over})


def nested(depth: int) -> object:
    value: object = "x"
    for _ in range(depth):
        value = [value]
    return value


class EntryValidationTests(unittest.TestCase):
    def test_plain_json_is_accepted(self):
        entry(params={"event": "grid_41", "n": 3, "xs": [1.5, None, True]}, result="grid_63")

    def test_non_json_values_are_refused(self):
        for bad in (b"bytes", object(), {1: "x"}, {"a": {1, 2}}, (1, 2), float("nan"),
                    float("inf"), {"a": [float("-inf")]}):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidTrace):
                    entry(params={"x": bad})
                with self.assertRaises(InvalidTrace):
                    entry(result=bad)

    def test_oversized_values_are_refused_with_a_clear_message(self):
        with self.assertRaisesRegex(InvalidTrace, "params is .* bytes.*handle"):
            entry(params={"rows": "x" * (MAX_JSON_BYTES + 1)})
        with self.assertRaisesRegex(InvalidTrace, "result is .* bytes"):
            entry(result=["x" * 100] * 100)
        with self.assertRaisesRegex(InvalidTrace, "result is .* bytes"):
            entry(result=[0] * MAX_JSON_BYTES)

    def test_value_at_the_cap_is_accepted(self):
        entry(params={"k": "x" * (MAX_JSON_BYTES - 10)})

    def test_nesting_within_the_byte_cap_is_accepted(self):
        value = nested(MAX_JSON_BYTES // 2 - 10)
        self.assertLessEqual(len(serialize_json(value, "result")), MAX_JSON_BYTES)

    def test_deep_and_cyclic_values_fail_as_invalid_trace(self):
        cyclic: list[object] = []
        cyclic.append(cyclic)
        cases = {"at the node budget": nested(MAX_JSON_BYTES), "far past it": nested(100_000),
                 "cyclic": cyclic}
        for name, value in cases.items():
            with self.subTest(name), self.assertRaises(InvalidTrace):
                entry(result=value)

    def test_nul_characters_are_refused_everywhere(self):
        for over in ({"params": {"a": "x\x00"}}, {"params": {"k\x00": 1}},
                     {"result": ["\x00"]}, {"reason": "r\x00", "status": "waiting"},
                     {"step": "s\x00"}, {"capability": "c\x00"}):
            with self.subTest(over=over), self.assertRaisesRegex(InvalidTrace, "NUL"):
                entry(**over)

    def test_failed_needs_a_category_and_done_cannot_have_one(self):
        with self.assertRaises(InvalidTrace):
            entry(status="failed")
        with self.assertRaises(InvalidTrace):
            entry(status="done", failure_category=FailureCategory.NO_DATA)
        entry(status="failed", failure_category=FailureCategory.NO_DATA)
        entry(status="waiting", failure_category=FailureCategory.SOURCE_UNAVAILABLE)

    def test_use_needs_a_capability(self):
        with self.assertRaises(InvalidTrace):
            entry(capability=None)
        entry(kind="ask", capability=None)

    def test_other_fields(self):
        with self.assertRaises(InvalidTrace):
            entry(started=T0.replace(tzinfo=None))
        with self.assertRaises(InvalidTrace):
            entry(seconds=-1.0)
        with self.assertRaises(InvalidTrace):
            entry(kind="magic")
        with self.assertRaises(InvalidTrace):
            entry(reason="x" * 2000)

    def test_failure_categories_are_the_design_table_plus_cancelled_and_input_evicted(self):
        self.assertEqual({c.value for c in FailureCategory}, {
            "bad_input", "no_data", "request_too_large", "source_unavailable", "source_down",
            "credential_invalid", "unsupported_shape", "internal_error", "model_failure",
            "budget_exhausted", "cancelled", "input_evicted",
        })


def explore(status="done", result="default", **over):
    params = {"objective": "Where did NO2 rise while the rest of the state fell?",
              "bundle": "analyse", "inputs": {"handles": ["grid_41"], "location": "Texas"}}
    outcome = {"status": "done", "model": "sonnet", "model_calls": 4, "tool_calls": 7,
               "input_tokens": 1200, "output_tokens": 300, "cache_read_tokens": 900, "cache_creation_tokens": 200}
    base = {"kind": "explore", "capability": None, "status": status, "params": params,
            "result": outcome if result == "default" else result}
    return entry(**{**base, **over})


class ExploreEntryTests(unittest.TestCase):
    def test_a_task_and_its_outcome_are_accepted(self):
        explore()
        for bundle in ("analyse", "investigate", "find-data"):
            explore(params={"objective": "x", "bundle": bundle, "inputs": {}})

    def test_the_result_is_optional(self):
        explore(result=None)
        explore("failed", result=None, failure_category=FailureCategory.CANCELLED)

    def test_params_are_checked(self):
        good = {"objective": "x", "bundle": "analyse", "inputs": {"variable": "no2"}}
        bad = [
            {k: v for k, v in good.items() if k != "inputs"},
            {**good, "extra": 1},
            {**good, "objective": ""}, {**good, "objective": "  "}, {**good, "objective": 3},
            {**good, "bundle": "everything"}, {**good, "bundle": None},
            {**good, "inputs": ["grid_1"]}, {**good, "inputs": {"h": 3}},
            {**good, "inputs": {"h": [1]}}, {**good, "inputs": {"h": [["a"]]}},
        ]
        for params in bad:
            with self.subTest(params=params), self.assertRaises(InvalidTrace):
                explore(params=params)

    def test_result_is_checked(self):
        good = {"status": "done", "model": "sonnet", "model_calls": 1, "tool_calls": 0,
                "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_creation_tokens": 0}
        bad = [
            "done", {}, {**good, "extra": 1}, {k: v for k, v in good.items() if k != "model"},
            {**good, "status": "running"}, {**good, "status": "skipped"},
            {**good, "model": ""}, {**good, "model": "m" * 101}, {**good, "model": 3},
            {**good, "model_calls": -1}, {**good, "tool_calls": 1.5},
            {**good, "input_tokens": True}, {**good, "output_tokens": "9"},
            {k: v for k, v in good.items() if k != "cache_read_tokens"},
            {k: v for k, v in good.items() if k != "cache_creation_tokens"},
            {**good, "cache_read_tokens": -1}, {**good, "cache_creation_tokens": 2.0},
            {**good, "cache_read_tokens": None}, {**good, "cache_creation_tokens": False},
        ]
        for result in bad:
            with self.subTest(result=result), self.assertRaises(InvalidTrace):
                explore(result=result)

    def test_result_status_must_match_the_entry_status(self):
        outcome = {"model": "sonnet", "model_calls": 1, "tool_calls": 1,
                   "input_tokens": 1, "output_tokens": 1, "cache_read_tokens": 0, "cache_creation_tokens": 0}
        mapping = {"done": "done", "incomplete": "done", "insufficient_evidence": "done",
                   "waiting": "waiting", "needs_input": "waiting",
                   "failed": "failed", "cancelled": "failed"}
        for result_status, entry_status in mapping.items():
            for status in ("done", "waiting", "failed"):
                kwargs = ({"failure_category": FailureCategory.INTERNAL_ERROR}
                          if status == "failed" else {})
                result = {**outcome, "status": result_status}
                with self.subTest(result_status=result_status, status=status):
                    if status == entry_status:
                        explore(status, result=result, **kwargs)
                    else:
                        with self.assertRaises(InvalidTrace):
                            explore(status, result=result, **kwargs)

    def test_an_explore_entry_cannot_be_skipped(self):
        with self.assertRaises(InvalidTrace):
            explore("skipped")

    def test_an_oversized_objective_is_refused_not_truncated(self):
        with self.assertRaisesRegex(InvalidTrace, "params is .* bytes"):
            explore(params={"objective": "x" * MAX_JSON_BYTES, "bundle": "analyse",
                            "inputs": {}})

    def test_nul_in_the_task_is_refused(self):
        for params in ({"objective": "x\x00", "bundle": "analyse", "inputs": {}},
                       {"objective": "x", "bundle": "analyse", "inputs": {"a": ["b\x00"]}}):
            with self.subTest(params=params), self.assertRaisesRegex(InvalidTrace, "NUL"):
                explore(params=params)


class RunTests(unittest.TestCase):
    def test_incomplete_is_a_run_status(self):
        run = Run(id=RUN_ID, user_id="u", conversation_id="c",
                  mode=Mode.exploratory(), started=T0, status="incomplete")
        self.assertEqual(run.status, "incomplete")

    def test_defaults(self):
        run = Run(id=new_run_id(), user_id="u", conversation_id="c",
                  mode=Mode.exploratory(), started=T0)
        self.assertEqual((run.status, run.harness_release), ("running", HARNESS_RELEASE))
        self.assertRegex(run.id, RUN_ID_PATTERN)

    def test_waiting_is_a_run_status(self):
        run = Run(id=RUN_ID, user_id="u", conversation_id="c",
                  mode=Mode.exploratory(), started=T0, status="waiting")
        self.assertEqual(run.status, "waiting")

    def test_validation(self):
        base = {"id": RUN_ID, "user_id": "u", "conversation_id": "c",
                "mode": Mode.validated("map", 1), "started": T0}
        Run(**base)
        for over in ({"id": "run_1"}, {"id": "run_child"}, {"id": RUN_ID.upper()},
                     {"id": RUN_ID + "\n"}, {"parent_run_id": "run_parent"},
                     {"mode": "validated"}, {"mode": Mode.validated("map\x00", 1)},
                     {"status": "weird"}, {"started": T0.replace(tzinfo=None)},
                     {"user_id": ""}, {"user_id": "u\x00"}, {"conversation_id": "c\x00"}):
            with self.subTest(over=over), self.assertRaises(InvalidTrace):
                Run(**{**base, **over})


if __name__ == "__main__":
    unittest.main()
