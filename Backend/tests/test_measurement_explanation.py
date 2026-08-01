"""
tests/test_measurement_explanation.py
========================================
PRD T36 Phase 3 -- the read-only ``explain_measurement`` accessor. It bridges
P2's persisted ``provenance.evidence`` to a grounded explanation by returning a
compact reliability subset, with NO recomputation and NO error path:

- a payload carrying evidence yields ``has_evidence: true`` with the evidence
  list and the masking qa_status/qa_source;
- a science-only payload (no evidence) and an unknown chart_id both yield
  ``has_evidence: false`` with a plain reason, never raising;
- the returned evidence is the stored ``provenance.evidence`` *verbatim* (guards
  against a second, drifting evidence path);
- a get_chart lookup that raises degrades to ``has_evidence: false``.
"""
import asyncio
import importlib.util
import json
import os
import sys
import unittest
from unittest import mock

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


def _evidence_payload():
    """A TEMPO-O3-shaped persisted chart payload as P2 would store it."""
    return {
        "chart_id": "map_abc123",
        "variable": "column_amount_o3",
        "units": "DU",
        "provenance": {
            "variable": "column_amount_o3",
            "units": "DU",
            "region_name": "California",
            "evidence": [
                {"name": "main_data_quality_flag", "role": "quality", "stat": "pass_rate",
                 "value": 0.93, "units": "", "coverage": 1.0},
                {"name": "radiative_cloud_frac", "role": "context", "stat": "mean",
                 "value": 0.098, "units": "1", "coverage": 0.974},
            ],
            "masking": {"qa_status": "verified", "qa_source": "collections_yaml"},
        },
    }


class SummarizeMeasurementEvidenceTests(unittest.TestCase):
    def test_payload_with_evidence_returns_evidence_and_qa_status(self):
        from tta_backend.services.measurement_explanation import summarize_measurement_evidence

        result = summarize_measurement_evidence(_evidence_payload())

        self.assertTrue(result["has_evidence"])
        self.assertEqual(result["variable"], "column_amount_o3")
        self.assertEqual(result["units"], "DU")
        self.assertEqual(result["masking"]["qa_status"], "verified")
        self.assertEqual(result["masking"]["qa_source"], "collections_yaml")
        names = {f["name"] for f in result["evidence"]}
        self.assertEqual(names, {"main_data_quality_flag", "radiative_cloud_frac"})
        # has_evidence true -> no reason.
        self.assertNotIn("reason", result)

    def test_returned_evidence_is_the_stored_provenance_evidence_verbatim(self):
        """No recomputation: the returned facts equal the stored list exactly,
        so a confidence explanation is auditable back to P2's single source."""
        from tta_backend.services.measurement_explanation import summarize_measurement_evidence

        payload = _evidence_payload()
        result = summarize_measurement_evidence(payload)

        self.assertEqual(result["evidence"], payload["provenance"]["evidence"])

    def test_science_only_payload_reports_no_evidence_with_a_reason(self):
        from tta_backend.services.measurement_explanation import summarize_measurement_evidence

        payload = {
            "chart_id": "map_no2only",
            "variable": "vertical_column_troposphere",
            "units": "molecules/cm^2",
            "provenance": {
                "variable": "vertical_column_troposphere",
                "units": "molecules/cm^2",
                "evidence": [],
                "masking": {"qa_status": "not applied — semantics unknown", "qa_source": "none"},
            },
        }
        result = summarize_measurement_evidence(payload)

        self.assertFalse(result["has_evidence"])
        self.assertIn("reason", result)
        self.assertNotIn("evidence", result)
        self.assertEqual(result["variable"], "vertical_column_troposphere")
        # The masking disclosure still travels even with empty evidence.
        self.assertEqual(result["masking"]["qa_status"], "not applied — semantics unknown")

    def test_unknown_chart_returns_no_evidence_and_does_not_raise(self):
        from tta_backend.services.measurement_explanation import summarize_measurement_evidence

        result = summarize_measurement_evidence(None)

        self.assertFalse(result["has_evidence"])
        self.assertIn("reason", result)
        self.assertNotIn("evidence", result)

    def test_payload_without_provenance_reports_no_evidence(self):
        from tta_backend.services.measurement_explanation import summarize_measurement_evidence

        result = summarize_measurement_evidence({"chart_id": "map_x", "variable": "AOD", "units": "1"})

        self.assertFalse(result["has_evidence"])
        self.assertIn("reason", result)
        self.assertEqual(result["variable"], "AOD")

    def test_empty_but_present_provenance_variable_is_preserved(self):
        """provenance is the source of truth. An empty-string variable (a chart
        built from an unnamed DataArray: plot_tools stores ``da.name or ""``) is
        a present value and must not be silently overridden by the payload's
        top-level variable — same ``is not None`` semantics the units field has."""
        from tta_backend.services.measurement_explanation import summarize_measurement_evidence

        payload = _evidence_payload()
        payload["provenance"]["variable"] = ""
        payload["variable"] = "column_amount_o3"

        result = summarize_measurement_evidence(payload)
        self.assertEqual(result["variable"], "")


class ExplainMeasurementCompositeTests(unittest.TestCase):
    def test_composite_returns_the_stored_subset_for_a_known_chart(self):
        from tta_backend.services.measurement_explanation import explain_measurement

        payload = _evidence_payload()

        async def fake_get_chart(chart_id):
            return payload if chart_id == "map_abc123" else None

        result = asyncio.run(explain_measurement("map_abc123", get_chart=fake_get_chart))
        self.assertTrue(result["has_evidence"])
        self.assertEqual(result["evidence"], payload["provenance"]["evidence"])

    def test_composite_unknown_id_is_no_evidence_not_error(self):
        from tta_backend.services.measurement_explanation import explain_measurement

        async def fake_get_chart(chart_id):
            return None

        result = asyncio.run(explain_measurement("nope", get_chart=fake_get_chart))
        self.assertFalse(result["has_evidence"])
        self.assertIn("reason", result)

    def test_composite_lookup_failure_degrades_to_no_evidence(self):
        from tta_backend.services.measurement_explanation import explain_measurement

        async def boom(chart_id):
            raise RuntimeError("db down")

        result = asyncio.run(explain_measurement("map_abc123", get_chart=boom))
        self.assertFalse(result["has_evidence"])
        self.assertIn("reason", result)


class ExplainMeasurementOwnershipTests(unittest.TestCase):
    """chart_id is a free-text, LLM-supplied tool argument. Like every HTTP
    chart endpoint (api.py _get_owned_chart), a chart owned by someone else
    must read exactly as not-found — nothing about it may leak."""

    def _owned_payload(self):
        payload = _evidence_payload()
        payload["user_id"] = "owner-1"
        return payload

    def test_another_users_chart_reads_as_not_found(self):
        from tta_backend.services.measurement_explanation import explain_measurement

        payload = self._owned_payload()

        async def fake_get_chart(chart_id):
            return payload

        result = asyncio.run(
            explain_measurement("map_abc123", get_chart=fake_get_chart, user_id="intruder-2")
        )
        self.assertFalse(result["has_evidence"])
        self.assertIn("reason", result)
        # Not-found means NOT-FOUND: no evidence, variable, units, or masking leak.
        self.assertNotIn("evidence", result)
        self.assertNotIn("variable", result)
        self.assertNotIn("units", result)
        self.assertNotIn("masking", result)

    def test_owner_gets_their_evidence(self):
        from tta_backend.services.measurement_explanation import explain_measurement

        payload = self._owned_payload()

        async def fake_get_chart(chart_id):
            return payload

        result = asyncio.run(
            explain_measurement("map_abc123", get_chart=fake_get_chart, user_id="owner-1")
        )
        self.assertTrue(result["has_evidence"])
        self.assertEqual(result["evidence"], payload["provenance"]["evidence"])

    def test_unbound_caller_cannot_read_an_owned_chart(self):
        """Fail closed: an owned chart with no caller user bound is not-found,
        never a bypass."""
        from tta_backend.services.measurement_explanation import explain_measurement

        payload = self._owned_payload()

        async def fake_get_chart(chart_id):
            return payload

        result = asyncio.run(explain_measurement("map_abc123", get_chart=fake_get_chart))
        self.assertFalse(result["has_evidence"])
        self.assertNotIn("evidence", result)


@unittest.skipIf(importlib.util.find_spec("langchain") is None, "langchain is not installed")
class ExplainMeasurementToolOwnershipTests(unittest.TestCase):
    def test_tool_scopes_lookup_to_the_bound_user(self):
        """The LangChain tool must pass the session's bound user id through to
        the ownership check — the same current_user_id() context every other
        user-scoped tool path uses."""
        from tta_backend.tools.satellite_tools.retrieval_tools import make_explain_measurement
        from tta_backend.utils.streaming import user_id_context

        payload = _evidence_payload()
        payload["user_id"] = "owner-1"

        async def fake_get_chart(chart_id):
            return payload

        tool = make_explain_measurement()
        with mock.patch("tta_backend.services.measurement_explanation.chart_repository.get_chart", fake_get_chart):
            with user_id_context("intruder-2"):
                denied = json.loads(asyncio.run(tool.ainvoke({"chart_id": "map_abc123"})))
            with user_id_context("owner-1"):
                allowed = json.loads(asyncio.run(tool.ainvoke({"chart_id": "map_abc123"})))

        self.assertFalse(denied["has_evidence"])
        self.assertNotIn("evidence", denied)
        self.assertTrue(allowed["has_evidence"])


if __name__ == "__main__":
    unittest.main()
