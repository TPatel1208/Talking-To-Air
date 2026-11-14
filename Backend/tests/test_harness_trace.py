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

    def test_failure_categories_are_the_design_table_plus_cancelled(self):
        self.assertEqual({c.value for c in FailureCategory}, {
            "bad_input", "no_data", "request_too_large", "source_unavailable", "source_down",
            "credential_invalid", "unsupported_shape", "internal_error", "model_failure",
            "budget_exhausted", "cancelled",
        })


class RunTests(unittest.TestCase):
    def test_defaults(self):
        run = Run(id=new_run_id(), user_id="u", conversation_id="c",
                  mode=Mode.exploratory(), started=T0)
        self.assertEqual((run.status, run.harness_release), ("running", HARNESS_RELEASE))
        self.assertRegex(run.id, RUN_ID_PATTERN)

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
