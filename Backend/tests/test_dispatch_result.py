"""T64 D11/D12: the dispatch tool's batch envelope, and the one parser every
reader of a supervisor ToolMessage goes through -- so a batch and a legacy
single-AgentResult checkpoint read the same way."""
import unittest


def _batch_json():
    from tta_backend.models import AgentResult, DispatchItemResult, DispatchResult

    return DispatchResult(results=[
        DispatchItemResult(index=0, agent="satellite", intent="visualize", status="ok",
                           result=AgentResult(text="NJ map", charts=[{"type": "heatmap", "chart_id": "c-nj"}])),
        DispatchItemResult(index=1, agent="satellite", intent="visualize", status="rejected",
                           reason="per-turn budget exhausted"),
        DispatchItemResult(index=2, agent="ground", intent="readings", status="duplicate",
                           result=AgentResult(text="NJ readings")),
    ]).model_dump_json(exclude_none=True)


class ParseToolResultsTests(unittest.TestCase):
    def test_a_batch_yields_each_items_result_in_input_order_skipping_items_that_never_ran(self):
        from tta_backend.models import parse_tool_results

        results = parse_tool_results(_batch_json())

        self.assertEqual([r.text for r in results], ["NJ map", "NJ readings"])
        self.assertEqual(results[0].charts[0].chart_id, "c-nj")

    def test_a_legacy_single_agent_result_still_parses(self):
        from tta_backend.models import AgentResult, agent_result_to_json, parse_tool_results

        results = parse_tool_results(agent_result_to_json(AgentResult(text="legacy answer")))

        self.assertEqual([r.text for r in results], ["legacy answer"])

    def test_unstructured_content_parses_to_nothing(self):
        from tta_backend.models import parse_tool_results

        for content in ("plain prose", '{"type": "heatmap"}', None, ""):
            with self.subTest(content=content):
                self.assertEqual(parse_tool_results(content), [])


def _echo_saved_chart(thread_id, payload, user_id):
    return {**payload, "thread_id": thread_id, "user_id": user_id}


def _picker(name):
    return {"message": f"Which {name}?", "candidates": [
        {"name": name, "category": "science", "prompt": f"plot {name}"},
    ]}


class SupervisorStreamBatchTests(unittest.IsolatedAsyncioTestCase):
    """The live supervisor route reading a dispatch batch off a tool_result."""

    async def _stream(self, tool_result_content):
        from unittest.mock import AsyncMock, patch

        from tta_backend.services.chart_service import ChartService
        from tta_backend.services.chat_stream_service import ChatStreamService

        async def fake_stream_response(agent, message, thread_id, **kwargs):
            yield "tool_result", {"name": "dispatch", "content": tool_result_content}
            yield "text", "Agent consulted: SATELLITE\n\nBoth maps are ready."

        service = ChatStreamService(ChartService(), long_request_seconds=999)
        with patch("tta_backend.services.chat_stream_service.stream_response", fake_stream_response), \
             patch("tta_backend.services.chart_service.chart_repository.get_chart", AsyncMock(return_value=None)), \
             patch("tta_backend.services.chart_service.chart_repository.save_chart",
                   AsyncMock(side_effect=_echo_saved_chart)):
            return [
                event
                async for event in service.stream_chat_events(object(), None, None, "hi", "thread-1", "user-1", "req-1")
            ]

    async def test_every_items_chart_reaches_the_stream(self):
        import json

        from tta_backend.models import AgentResult, DispatchItemResult, DispatchResult

        batch = DispatchResult(results=[
            DispatchItemResult(index=i, agent="satellite", intent="visualize", status="ok",
                               result=AgentResult(text=f"map {i}", charts=[{"type": "heatmap", "chart_id": f"c-{i}"}]))
            for i in range(2)
        ]).model_dump_json(exclude_none=True)

        events = await self._stream(batch)

        chart_ids = [json.loads(e.split("data: ", 1)[1])["chart_id"] for e in events if e.startswith("event: chart")]
        self.assertEqual(chart_ids, ["c-0", "c-1"])

    async def test_a_batch_tool_result_is_parsed_once(self):
        """The header and the chart/artifact events both read the batch; a
        six-item envelope should not be validated twice per event."""
        from unittest.mock import patch

        from tta_backend.models import AgentResult, DispatchItemResult, DispatchResult

        batch = DispatchResult(results=[
            DispatchItemResult(index=0, agent="satellite", intent="visualize", status="ok",
                               result=AgentResult(text="map", charts=[{"type": "heatmap", "chart_id": "c-0"}])),
        ]).model_dump_json(exclude_none=True)

        real = DispatchResult.model_validate_json
        with patch.object(DispatchResult, "model_validate_json", side_effect=real) as parse:
            await self._stream(batch)

        # The answer text is also offered to the parser (and rejected); only
        # parses of the batch itself count.
        self.assertEqual(sum(1 for c in parse.call_args_list if c.args[0] == batch), 1)

    async def test_the_done_event_merges_suggestions_and_keeps_the_first_picker(self):
        """D13: suggestions concatenate in request order, deduped, capped at
        the envelope's own limit of two; one picker can be shown, so the
        first item's wins."""
        import json

        from tta_backend.models import AgentResult, DispatchItemResult, DispatchResult, VariableChoice

        batch = DispatchResult(results=[
            DispatchItemResult(index=0, agent="satellite", intent="visualize", status="ok", result=AgentResult(
                text="a", suggested_followups=["Compare with July?"],
                variable_choice=VariableChoice.model_validate(_picker("NO2_column")))),
            DispatchItemResult(index=1, agent="satellite", intent="visualize", status="ok", result=AgentResult(
                text="b", suggested_followups=["Compare with July?", "Show a time series?"],
                variable_choice=VariableChoice.model_validate(_picker("O3_column")))),
            DispatchItemResult(index=2, agent="ground", intent="readings", status="ok", result=AgentResult(
                text="c", suggested_followups=["Find the nearest monitor?"])),
        ]).model_dump_json(exclude_none=True)

        events = await self._stream(batch)

        done = json.loads(next(e for e in events if e.startswith("event: done")).split("data: ", 1)[1])
        self.assertEqual(done["suggested_followups"], ["Compare with July?", "Show a time series?"])
        self.assertEqual(done["variable_choice"]["candidates"][0]["name"], "NO2_column")


class HistoryBatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_reloading_a_thread_restores_every_items_charts_and_artifacts(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch

        from tta_backend.models import (
            AgentResult,
            ArtifactReference,
            DispatchItemResult,
            DispatchResult,
            chart_reference,
        )
        from tta_backend.services.chart_service import ChartService
        from tta_backend.services.history_service import HistoryService

        def item(i):
            return DispatchItemResult(index=i, agent="satellite", intent="visualize", status="ok", result=AgentResult(
                text=f"map {i}",
                charts=[chart_reference({"chart_id": f"c-{i}", "type": "heatmap"})],
                artifacts=[ArtifactReference(id=f"c-{i}", type="map", title=f"map {i}")],
            ))

        tool_content = DispatchResult(results=[item(0), item(1)]).model_dump_json(exclude_none=True)

        class FakeAgent:
            async def aget_state(self, config):
                return SimpleNamespace(values={"messages": [
                    SimpleNamespace(type="human", content="plot NO2 over NJ and over NY"),
                    SimpleNamespace(type="ai", content="", tool_calls=[{"id": "tc1", "name": "dispatch", "args": {}}]),
                    SimpleNamespace(type="tool", name="dispatch", content=tool_content),
                    SimpleNamespace(type="ai", content="Both maps are ready.", tool_calls=[]),
                ]})

        stored = {f"c-{i}": {"chart_id": f"c-{i}", "type": "heatmap", "user_id": "user-1", "lats": [1]} for i in (0, 1)}
        with patch("tta_backend.services.chart_service.chart_repository.get_charts", AsyncMock(return_value=stored)):
            messages = await HistoryService(ChartService()).build_history(FakeAgent(), "thread-1", "user-1")

        assistant = messages[-1]
        self.assertEqual([c["chart_id"] for c in assistant["charts"]], ["c-0", "c-1"])
        self.assertEqual([a["id"] for a in assistant["artifacts"]], ["c-0", "c-1"])


class CheckpointBatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_checkpointed_batch_keeps_chart_references_never_grids(self):
        """Same contract test_checkpoint_chart_refs pins for a single result:
        the checkpoint is rewritten whole at every step, so a grid kept here
        is stored again on every later turn."""
        from unittest.mock import AsyncMock, patch

        from tta_backend.models import (
            AgentResult,
            ChartPayload,
            DispatchItemResult,
            DispatchResult,
            parse_dispatch_result,
        )
        from tta_backend.services.chart_service import ChartService

        def grid(name):
            return ChartPayload.model_validate({"type": "heatmap", "title": name, "lats": [1.0], "lons": [2.0],
                                                "values": [[3.0]]})

        batch = DispatchResult(results=[
            DispatchItemResult(index=0, agent="satellite", intent="visualize", status="ok",
                               result=AgentResult(text="NJ", charts=[grid("NJ")])),
            DispatchItemResult(index=1, agent="satellite", intent="visualize", status="rejected", reason="budget"),
            DispatchItemResult(index=2, agent="satellite", intent="visualize", status="ok",
                               result=AgentResult(text="NY", charts=[grid("NY")])),
        ])
        saved = iter(["c-nj", "c-ny"])

        async def save_chart(thread_id, payload, user_id):
            return {**payload, "chart_id": next(saved), "user_id": user_id}

        with patch("tta_backend.services.chart_service.chart_repository.get_chart", AsyncMock(return_value=None)), \
             patch("tta_backend.services.chart_service.chart_repository.save_chart", AsyncMock(side_effect=save_chart)):
            raw = await ChartService().checkpoint_dispatch_envelope(batch, "thread-1", "user-1")

        self.assertNotIn('"lats"', raw)
        checkpointed = parse_dispatch_result(raw)
        self.assertEqual([i.status for i in checkpointed.results], ["ok", "rejected", "ok"])
        self.assertEqual(checkpointed.results[1].reason, "budget")
        refs = [c.model_dump() for i in checkpointed.results if i.result for c in i.result.charts]
        self.assertEqual([(r["chart_id"], r["reference"]) for r in refs], [("c-nj", True), ("c-ny", True)])


if __name__ == "__main__":
    unittest.main()
