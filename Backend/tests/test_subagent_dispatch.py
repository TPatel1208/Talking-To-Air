import asyncio
import importlib.util
import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


@unittest.skipIf(importlib.util.find_spec("langchain") is None, "langchain is not installed")
class HelperTests(unittest.TestCase):
    def test_ground_tool_failure_detects_provider_tool_call_errors(self):
        from services.subagent_dispatch import _is_ground_tool_failure

        self.assertTrue(_is_ground_tool_failure("Error: Failed to call a function. failed_generation"))
        self.assertTrue(_is_ground_tool_failure("tool call validation failed: parameters for tool did not match schema"))
        self.assertFalse(_is_ground_tool_failure("The monitor is Rutgers University."))

    def test_ground_context_extracts_and_injects_monitor_facts(self):
        from services.subagent_dispatch import _extract_ground_monitor_context, _inject_ground_context

        context = _extract_ground_monitor_context(
            "The closest NO2 monitor is Rutgers University with station_id 34-023-0011 "
            "located at coordinates (40.462182, -74.429439)."
        )

        self.assertEqual(context["name"], "Rutgers University")
        self.assertEqual(context["site_id"], "34-023-0011")
        self.assertEqual(context["latitude"], "40.462182")
        self.assertEqual(context["longitude"], "-74.429439")
        # Pollutant is intentionally NOT extracted — the user's request is always
        # authoritative and injecting a stale pollutant caused param_code confusion.
        self.assertNotIn("pollutant", context)
        enriched = _inject_ground_context("Give quarterly summary for Q1 2024.", context)
        self.assertIn("station_id=34-023-0011", enriched)
        self.assertIn("coordinates=(40.462182, -74.429439)", enriched)
        self.assertNotIn("pollutant=", enriched)

    def test_ground_context_does_not_bleed_pollutant_across_requests(self):
        from services.subagent_dispatch import _extract_ground_monitor_context, _inject_ground_context

        # Simulate a failure response that mentions multiple pollutants — the first
        # match used to be SO2, which then poisoned a subsequent NO2 request.
        failure_text = (
            "The Chester, NJ monitor has no daily SO2 records for the requested period. "
            "No NO2 or PM2.5 data was found either."
        )
        context = _extract_ground_monitor_context(failure_text)
        self.assertNotIn("pollutant", context)

        enriched = _inject_ground_context("Give daily NO2 summary for NJ in Jan 2024.", context)
        self.assertNotIn("pollutant=", enriched)

    def test_finalize_sub_agent_result_resolves_matching_artifact_ids_and_handles(self):
        from services.subagent_dispatch import _finalize_sub_agent_result
        from models import AgentResult
        from models.artifact import ArtifactReference

        raw = AgentResult(
            text='{"summary": "Found the closest monitor.", "artifact_ids": ["art_1"], "handles": ["obs_1"]}',
            artifacts=[
                ArtifactReference(id="art_1", type="table"),
                ArtifactReference(id="art_2", type="table"),
            ],
        )

        finalized = _finalize_sub_agent_result(raw, "ground sensor")

        self.assertEqual(finalized.text, "Found the closest monitor.")
        self.assertEqual([a.id for a in finalized.artifacts], ["art_1"])
        self.assertEqual(finalized.handles, ["obs_1"])

    def test_finalize_sub_agent_result_drops_unknown_artifact_ids(self):
        from services.subagent_dispatch import _finalize_sub_agent_result
        from models import AgentResult

        raw = AgentResult(text='{"summary": "ok", "artifact_ids": ["missing"], "handles": []}')

        finalized = _finalize_sub_agent_result(raw, "ground sensor")

        self.assertEqual(finalized.artifacts, [])

    def test_finalize_sub_agent_result_salvages_prose_with_collected_artifacts(self):
        from services.subagent_dispatch import _finalize_sub_agent_result
        from models import AgentResult, ChartPayload
        from models.artifact import ArtifactReference

        raw = AgentResult(
            text="I found 3 monitors near Newark, NJ.",
            charts=[ChartPayload(type="heatmap", title="NO2")],
            artifacts=[ArtifactReference(id="art_1", type="table")],
        )

        finalized = _finalize_sub_agent_result(raw, "earthdata")

        # A malformed envelope no longer discards ninety seconds of correct
        # work over a formatting technicality (T15) — the raw prose becomes
        # the summary and everything collected from the tool stream survives.
        self.assertEqual(finalized.text, raw.text)
        self.assertEqual(len(finalized.charts), 1)
        self.assertEqual([a.id for a in finalized.artifacts], ["art_1"])
        self.assertIsNone(finalized.metadata.get("error"))
        self.assertTrue(finalized.metadata.get("salvaged"))
        self.assertIn("raw_preview", finalized.metadata)

    def test_finalize_sub_agent_result_salvage_attaches_handles_from_artifact_metadata(self):
        from services.subagent_dispatch import _finalize_sub_agent_result
        from models import AgentResult
        from models.artifact import ArtifactReference

        raw = AgentResult(
            text="Plotted TROPOMI NO2 over New Jersey.",
            artifacts=[ArtifactReference(
                id="map_1", type="map", metadata={"source_handles": ["obs_1"]},
            )],
        )

        finalized = _finalize_sub_agent_result(raw, "earthdata")

        # Handles named in the collected artifacts' own metadata are
        # unambiguous — they come from the tool results, not the prose.
        self.assertEqual(finalized.handles, ["obs_1"])

    def test_finalize_sub_agent_result_is_a_structured_failure_on_empty_text(self):
        from services.subagent_dispatch import _finalize_sub_agent_result
        from models import AgentResult

        raw = AgentResult(text="   ")

        finalized = _finalize_sub_agent_result(raw, "earthdata")

        # Salvage requires something to salvage — empty text stays a
        # structured failure rather than inventing a cause for it.
        self.assertEqual(finalized.metadata.get("error"), "invalid_envelope")
        self.assertFalse(finalized.metadata.get("salvaged"))
        self.assertIn("earthdata", finalized.text.lower())

    def test_finalize_sub_agent_result_parses_an_envelope_longer_than_the_display_limit(self):
        from services.subagent_dispatch import _finalize_sub_agent_result
        from models import AgentResult
        from models.artifact import ArtifactReference

        long_summary = "A" * 2500
        raw_text = json.dumps({
            "summary": long_summary,
            "artifact_ids": ["art_1"],
            "handles": ["obs_1"],
        })
        self.assertGreater(len(raw_text), 2000)

        raw = AgentResult(
            text=raw_text,
            artifacts=[ArtifactReference(id="art_1", type="table")],
        )

        finalized = _finalize_sub_agent_result(raw, "earthdata")

        self.assertNotEqual(finalized.metadata.get("error"), "invalid_envelope")
        self.assertEqual([a.id for a in finalized.artifacts], ["art_1"])
        self.assertEqual(finalized.handles, ["obs_1"])
        # The extracted summary is truncated for display only after parsing.
        self.assertEqual(len(finalized.text), 2000)

    def test_satellite_retry_task_names_only_the_sanctioned_toolset(self):
        from services.subagent_dispatch import _satellite_retry_task
        from tools.satellite_tools.factory import sanctioned_tool_names

        retry_task = _satellite_retry_task("Plot TROPOMI NO2 over New Jersey")

        for name in sanctioned_tool_names():
            self.assertIn(name, retry_task)
        # Demoted/removed tools must never appear in retry guidance.
        for stale in ("geocode_location", "get_retrieval_status", "summarize_dataset"):
            self.assertNotIn(stale, retry_task)
        self.assertIn("Task: Plot TROPOMI NO2 over New Jersey", retry_task)

    def test_ground_retry_guidance_names_only_ground_tools(self):
        from services.subagent_dispatch import _GROUND_RETRY_TOOL_GUIDANCE
        from tools import GROUND_TOOLS

        for tool in GROUND_TOOLS:
            self.assertIn(tool.name, _GROUND_RETRY_TOOL_GUIDANCE)
        self.assertNotIn("geocode_location", _GROUND_RETRY_TOOL_GUIDANCE)

    def test_extracts_a_map_artifact_ref_from_tool_content(self):
        from services.subagent_dispatch import _artifact_refs_from_content

        content = json.dumps({
            "type": "heatmap",
            "chart_id": "map_abc123",
            "title": "TEMPO over NJ",
            "_artifact_refs": [{
                "id": "map_abc123",
                "type": "map",
                "title": "TEMPO over NJ",
                "metadata": {
                    "bbox": [-75.0, 39.0, -73.0, 41.0],
                    "variable": "TEMPO_NO2",
                    "units": "mol/m^2",
                    "colorbar": {"vmin": 0.0, "vmax": 1.0},
                    "source_handles": ["obs_1"],
                },
            }],
        })

        refs = _artifact_refs_from_content(content)

        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].id, "map_abc123")
        self.assertEqual(refs[0].type, "map")

    def test_returns_empty_list_for_content_with_no_artifact_refs(self):
        from services.subagent_dispatch import _artifact_refs_from_content

        self.assertEqual(_artifact_refs_from_content("plain text"), [])
        self.assertEqual(_artifact_refs_from_content(json.dumps({"type": "heatmap"})), [])

    def test_extract_artifact_refs_from_messages_uses_the_shared_helper(self):
        from services.subagent_dispatch import _extract_artifact_refs

        table_ref = {"id": "tbl_1", "type": "table", "title": "EPA Summary"}
        messages = [
            SimpleNamespace(name="find_closest_monitor", content=json.dumps({"_artifact_refs": [table_ref]})),
        ]

        refs = _extract_artifact_refs(messages)

        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].id, "tbl_1")


@unittest.skipIf(importlib.util.find_spec("langchain") is None, "langchain is not installed")
class RepromptFinalEnvelopeTests(unittest.IsolatedAsyncioTestCase):
    """T15 retry demotion: a refusal-marked first result is recovered with
    one structured-output model call, never a second tool-workflow run."""

    async def test_reprompts_via_the_model_factorys_structured_output_hook(self):
        from services.subagent_dispatch import _reprompt_final_envelope
        from models import SubAgentEnvelope

        envelope = SubAgentEnvelope(summary="Recovered answer.", artifact_ids=[], handles=[])

        class FakeBoundModel:
            def __init__(self):
                self.calls = []

            async def ainvoke(self, text):
                self.calls.append(text)
                return envelope

        bound = FakeBoundModel()

        class FakeModel:
            def with_structured_output(self, schema):
                self.schema = schema
                return bound

        fake_model = FakeModel()
        agent = SimpleNamespace(subagent_model=fake_model)

        result = await _reprompt_final_envelope(agent, "Retry: find the closest monitor.", "ground_sensor")

        self.assertEqual(fake_model.schema, SubAgentEnvelope)
        self.assertEqual(bound.calls, ["Retry: find the closest monitor."])
        self.assertEqual(json.loads(result.text)["summary"], "Recovered answer.")

    async def test_returns_empty_text_when_the_model_call_raises(self):
        from services.subagent_dispatch import _reprompt_final_envelope

        class FakeBoundModel:
            async def ainvoke(self, text):
                raise RuntimeError("provider rejected the request")

        class FakeModel:
            def with_structured_output(self, schema):
                return FakeBoundModel()

        agent = SimpleNamespace(subagent_model=FakeModel())

        result = await _reprompt_final_envelope(agent, "Retry task", "satellite")

        self.assertEqual(result.text, "")

    async def test_returns_empty_text_when_the_agent_has_no_model_attached(self):
        from services.subagent_dispatch import _reprompt_final_envelope

        result = await _reprompt_final_envelope(SimpleNamespace(), "Retry task", "satellite")

        self.assertEqual(result.text, "")


@unittest.skipIf(importlib.util.find_spec("langchain") is None, "langchain is not installed")
class RunGroundTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_ground_reads_and_merges_persisted_monitor_context(self):
        from services import subagent_dispatch

        envelope = json.dumps({"summary": "Ground summary.", "artifact_ids": [], "handles": []})

        class FakeGroundAgent:
            def __init__(self):
                self.invoked_with = None

            async def ainvoke(self, input_, config):
                self.invoked_with = input_["messages"][0].content
                return {"messages": [SimpleNamespace(content=envelope, type="ai")]}

        ground_agent = FakeGroundAgent()
        subagent_dispatch._ground_call_count.set(0)

        with patch.object(
            subagent_dispatch, "get_ground_monitor_context", AsyncMock(return_value={"site_id": "34-023-0011"})
        ), patch.object(subagent_dispatch, "save_ground_monitor_context", AsyncMock()) as save_mock:
            result = await subagent_dispatch.run_ground(ground_agent, "quarterly summary", "thread-1")

        self.assertIn("station_id=34-023-0011", ground_agent.invoked_with)
        self.assertEqual(result.text, "Ground summary.")
        save_mock.assert_not_called()  # the reply didn't mention a new monitor

    async def test_run_ground_persists_newly_discovered_monitor_context(self):
        from services import subagent_dispatch

        envelope = json.dumps({
            "summary": "The closest monitor is Rutgers University with station_id 34-023-0011.",
            "artifact_ids": [],
            "handles": [],
        })

        class FakeGroundAgent:
            async def ainvoke(self, input_, config):
                return {"messages": [SimpleNamespace(content=envelope, type="ai")]}

        subagent_dispatch._ground_call_count.set(0)

        with patch.object(subagent_dispatch, "get_ground_monitor_context", AsyncMock(return_value={})), \
             patch.object(subagent_dispatch, "save_ground_monitor_context", AsyncMock()) as save_mock:
            await subagent_dispatch.run_ground(FakeGroundAgent(), "nearest monitor", "thread-1")

        save_mock.assert_awaited_once()
        saved_thread_id, saved_context = save_mock.await_args.args
        self.assertEqual(saved_thread_id, "thread-1")
        self.assertEqual(saved_context["site_id"], "34-023-0011")

    async def test_run_ground_second_call_in_the_same_task_is_budget_blocked(self):
        from services import subagent_dispatch

        class FakeGroundAgent:
            async def ainvoke(self, input_, config):
                return {"messages": [SimpleNamespace(
                    content=json.dumps({"summary": "ok", "artifact_ids": [], "handles": []}), type="ai",
                )]}

        subagent_dispatch._ground_call_count.set(0)
        with patch.object(subagent_dispatch, "get_ground_monitor_context", AsyncMock(return_value={})), \
             patch.object(subagent_dispatch, "save_ground_monitor_context", AsyncMock()):
            await subagent_dispatch.run_ground(FakeGroundAgent(), "task 1", "thread-1")
            second = await subagent_dispatch.run_ground(FakeGroundAgent(), "task 2", "thread-1")

        self.assertIn("already returned a result", second.text)

    async def test_run_ground_refusal_triggers_one_reprompt_not_a_second_workflow_run(self):
        from services import subagent_dispatch

        class FakeBoundModel:
            def __init__(self):
                self.call_count = 0

            async def ainvoke(self, text):
                self.call_count += 1
                from models import SubAgentEnvelope
                return SubAgentEnvelope(summary="Recovered via reprompt.", artifact_ids=[], handles=[])

        class FakeModel:
            def __init__(self, bound):
                self._bound = bound

            def with_structured_output(self, schema):
                return self._bound

        bound_model = FakeBoundModel()

        class FakeGroundAgent:
            def __init__(self):
                self.ainvoke_call_count = 0
                self.subagent_model = FakeModel(bound_model)

            async def ainvoke(self, input_, config):
                self.ainvoke_call_count += 1
                return {"messages": [SimpleNamespace(
                    content="Error: Failed to call a function. failed_generation", type="ai",
                )]}

        ground_agent = FakeGroundAgent()
        subagent_dispatch._ground_call_count.set(0)
        with patch.object(subagent_dispatch, "get_ground_monitor_context", AsyncMock(return_value={})), \
             patch.object(subagent_dispatch, "save_ground_monitor_context", AsyncMock()):
            result = await subagent_dispatch.run_ground(ground_agent, "task 1", "thread-1")

        # Exactly one further model interaction (the reprompt) — the tool
        # workflow itself never ran a second time.
        self.assertEqual(ground_agent.ainvoke_call_count, 1)
        self.assertEqual(bound_model.call_count, 1)
        self.assertEqual(result.text, "Recovered via reprompt.")


@unittest.skipIf(importlib.util.find_spec("langchain") is None, "langchain is not installed")
class RunSatelliteTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_satellite_forwards_events_via_on_event(self):
        from services import subagent_dispatch

        envelope = json.dumps({"summary": "Plotted NO2 over NJ.", "artifact_ids": [], "handles": ["obs_1"]})

        class FakeSatelliteAgent:
            async def astream(self, input_, config, stream_mode):
                yield "updates", {
                    "agent": {"messages": [
                        SimpleNamespace(tool_calls=[{"id": "tc1", "name": "plot_singular", "args": {}}], content=""),
                    ]},
                }
                await asyncio.sleep(0)
                yield "messages", (SimpleNamespace(content=envelope, type="ai", tool_calls=None), {})

        subagent_dispatch._satellite_call_count.set(0)
        forwarded = []

        async def on_event(event_type, data):
            forwarded.append((event_type, data))

        result = await subagent_dispatch.run_satellite(
            FakeSatelliteAgent(), "Plot NO2 over NJ", "thread-1", on_event=on_event,
        )

        self.assertEqual(result.text, "Plotted NO2 over NJ.")
        forwarded_types = [event_type for event_type, _ in forwarded]
        self.assertIn("tool_call", forwarded_types)

    async def test_run_satellite_second_call_in_the_same_task_is_budget_blocked(self):
        from services import subagent_dispatch

        envelope = json.dumps({"summary": "ok", "artifact_ids": [], "handles": []})

        class FakeSatelliteAgent:
            async def astream(self, input_, config, stream_mode):
                yield "messages", (SimpleNamespace(content=envelope, type="ai", tool_calls=None), {})

        subagent_dispatch._satellite_call_count.set(0)
        await subagent_dispatch.run_satellite(FakeSatelliteAgent(), "task 1", "thread-1")
        second = await subagent_dispatch.run_satellite(FakeSatelliteAgent(), "task 2", "thread-1")

        self.assertIn("already been called", second.text)

    async def test_run_satellite_refusal_triggers_one_reprompt_not_a_second_workflow_run(self):
        from services import subagent_dispatch

        class FakeBoundModel:
            def __init__(self):
                self.call_count = 0

            async def ainvoke(self, text):
                self.call_count += 1
                from models import SubAgentEnvelope
                return SubAgentEnvelope(summary="Recovered via reprompt.", artifact_ids=[], handles=[])

        class FakeModel:
            def __init__(self, bound):
                self._bound = bound

            def with_structured_output(self, schema):
                return self._bound

        bound_model = FakeBoundModel()

        class FakeSatelliteAgent:
            def __init__(self):
                self.astream_call_count = 0
                self.subagent_model = FakeModel(bound_model)

            async def astream(self, input_, config, stream_mode):
                self.astream_call_count += 1
                yield "messages", (
                    SimpleNamespace(content="The necessary tools are not present.", type="ai", tool_calls=None),
                    {},
                )

        satellite_agent = FakeSatelliteAgent()
        subagent_dispatch._satellite_call_count.set(0)
        result = await subagent_dispatch.run_satellite(satellite_agent, "Plot NO2 over NJ", "thread-1")

        # Exactly one further model interaction (the reprompt) — the tool
        # workflow itself never streamed a second time.
        self.assertEqual(satellite_agent.astream_call_count, 1)
        self.assertEqual(bound_model.call_count, 1)
        self.assertEqual(result.text, "Recovered via reprompt.")


if __name__ == "__main__":
    unittest.main()
