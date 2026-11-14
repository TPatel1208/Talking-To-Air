"""The result envelope validates its invariants and labels its mode."""
import unittest

from pydantic import ValidationError

from tta_backend.harness.release import HARNESS_RELEASE
from tta_backend.harness.results import Finding, Gap, Mode, Result, WorkflowRef
from tta_backend.harness.trace import new_run_id


def _finding(**over):
    base = {"name": "no2_ratio", "value": 1.8, "unit": "x", "source_handle": "grid_63"}
    return Finding(**{**base, **over})


class ModeTests(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(Mode.exploratory().label(), "Exploratory")
        self.assertEqual(Mode.draft("map", 1).label(), "Draft workflow: map v1")
        self.assertEqual(Mode.validated("map", 2).label(), "map v2")

    def test_exploratory_names_no_workflow(self):
        with self.assertRaises(ValidationError):
            Mode(kind="exploratory", workflow=WorkflowRef(name="map", version=1))
        with self.assertRaises(ValidationError):
            Mode(kind="exploratory", child_workflows=(WorkflowRef(name="a", version=1),))

    def test_workflow_modes_need_a_workflow(self):
        for kind in ("validated", "draft"):
            with self.assertRaises(ValidationError):
                Mode(kind=kind)

    def test_version_starts_at_one(self):
        with self.assertRaises(ValidationError):
            Mode.validated("map", 0)

    def test_children_are_kept(self):
        child = WorkflowRef(name="event-vs-baseline", version=1)
        mode = Mode.validated("ozone-event", 3, (child,))
        self.assertEqual(mode.child_workflows, (child,))
        self.assertEqual(mode.label(), "ozone-event v3")


class FindingTests(unittest.TestCase):
    def test_uncertainty_is_opaque_json(self):
        for u in (0.2, {"lo": 1, "hi": 2}, [1, 2], "high"):
            self.assertEqual(_finding(uncertainty=u).uncertainty, u)

    def test_optional_fields_default_to_none(self):
        f = _finding()
        self.assertIsNone(f.uncertainty)
        self.assertIsNone(f.region)
        self.assertIsNone(f.period)

    def test_value_must_be_finite(self):
        with self.assertRaises(ValidationError):
            _finding(value=float("nan"))

    def test_source_handle_required(self):
        with self.assertRaises(ValidationError):
            _finding(source_handle="")


class ResultTests(unittest.TestCase):
    def _result(self, **over):
        base = {"status": "done", "mode": Mode.exploratory(), "trace_id": new_run_id()}
        return Result(**{**base, **over})

    def test_minimal_result(self):
        r = self._result()
        self.assertEqual((r.findings, r.handles, r.charts, r.disclosures, r.gaps), ((),) * 5)

    def test_harness_release_defaults_to_the_current_release(self):
        self.assertEqual(self._result().harness_release, HARNESS_RELEASE)
        self.assertEqual(self._result(harness_release="0.0.9").harness_release, "0.0.9")
        with self.assertRaises(ValidationError):
            self._result(harness_release="")

    def test_trace_id_uses_the_run_id_format(self):
        for bad in ("abc", "run_ab12", "run_child", "run_" + "A" * 32, "run_" + "a" * 33,
                    "run_" + "a" * 32 + "\n"):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                self._result(trace_id=bad)
        good = "run_" + "0123456789abcdef" * 2
        self.assertEqual(self._result(trace_id=good).trace_id, good)

    def test_round_trips_through_json(self):
        r = self._result(
            findings=[_finding(region="NJ", period="2024-01-15")],
            gaps=[Gap(step="data", reason="no valid pixels")],
            handles=["grid_63"],
        )
        self.assertEqual(Result.model_validate_json(r.model_dump_json()), r)

    def test_status_and_trace_id_are_checked(self):
        with self.assertRaises(ValidationError):
            self._result(status="cancelled")
        with self.assertRaises(ValidationError):
            self._result(trace_id="abc")

    def test_unknown_fields_are_refused(self):
        with self.assertRaises(ValidationError):
            self._result(extra=1)


if __name__ == "__main__":
    unittest.main()
