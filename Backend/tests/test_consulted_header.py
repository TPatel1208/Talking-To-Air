"""T64 D14: the "Data sources: ..." header is computed from what the turn's
dispatch calls actually ran, never written by the model. It names the
datasets a satellite run plotted and EPA AQS for a ground run."""
import unittest


def _chart(dataset):
    from tta_backend.models import ChartPayload

    return ChartPayload.model_validate({"type": "heatmap", "provenance": {"dataset": dataset}})


def _result(agent, i, datasets=()):
    from tta_backend.models import AgentResult

    return AgentResult(text=f"{agent} {i}", charts=[_chart(d) for d in datasets])


def _result_with(charts):
    from tta_backend.models import AgentResult

    return AgentResult(text="x", charts=charts)


def _batch(*items):
    """Each item is (agent, status) or (agent, status, datasets plotted). The
    datasets are recorded as the item's sources, as the executor records
    them; the result carries no chart, so nothing needs persisting."""
    from tta_backend.models import DispatchItemResult, DispatchResult
    from tta_backend.services.consulted_header import result_sources

    results = []
    for i, (agent, status, *rest) in enumerate(items):
        result = None if status == "rejected" else _result(agent, i)
        sources = list(rest[0]) if rest else result_sources(agent, result) if result is not None else []
        results.append(DispatchItemResult(
            index=i, agent=agent, intent="visualize" if agent == "satellite" else "readings", status=status,
            result=result, sources=sources,
        ))
    return DispatchResult(results=results).model_dump_json(exclude_none=True)


class ResultSourcesTests(unittest.TestCase):
    def test_a_satellite_run_names_each_dataset_it_plotted_once_in_order(self):
        from tta_backend.services.consulted_header import result_sources

        result = _result("satellite", 0, ["TEMPO_NO2_L3", "S5P_L2__NO2___", "TEMPO_NO2_L3"])
        self.assertEqual(result_sources("satellite", result), ["TEMPO_NO2_L3", "S5P_L2__NO2___"])

    def test_a_comparison_names_the_dataset_behind_each_panel(self):
        from tta_backend.models import AgentResult, ChartPayload
        from tta_backend.services.consulted_header import result_sources

        chart = ChartPayload.model_validate({
            "type": "heatmap_multi",
            "provenance": {"dataset": "TEMPO_NO2_L3"},
            "panels": [{"provenance": {"dataset": "TEMPO_NO2_L3"}}, {"provenance": {"dataset": "S5P_L2__NO2___"}}],
        })
        self.assertEqual(result_sources("satellite", AgentResult(text="x", charts=[chart])),
                         ["TEMPO_NO2_L3", "S5P_L2__NO2___"])

    def test_a_satellite_run_with_no_dataset_on_record_falls_back_to_nasa_earthdata(self):
        from tta_backend.services.consulted_header import result_sources

        self.assertEqual(result_sources("satellite", _result("satellite", 0)), ["NASA Earthdata"])

    def test_a_ground_run_is_epa_aqs(self):
        from tta_backend.services.consulted_header import result_sources

        self.assertEqual(result_sources("ground", _result("ground", 0)), ["EPA AQS"])


    def test_a_satellite_comparison_against_monitors_names_epa_aqs_too(self):
        """validate_against_ground fetches EPA AQS series inside a satellite
        run; its artifacts tag each series with its source_kind."""
        from tta_backend.models import AgentResult, ArtifactReference
        from tta_backend.services.consulted_header import result_sources

        validation = ArtifactReference.model_validate({
            "id": "ts_1", "type": "timeseries",
            "metadata": {"series": [
                {"label": "NO2 (satellite)", "source_kind": "satellite"},
                {"label": "EPA 34-013-0003", "source_kind": "ground", "station_id": "34-013-0003"},
            ]},
        })
        result = AgentResult(text="x", charts=[_chart("TEMPO_NO2_L3")], artifacts=[validation])
        self.assertEqual(result_sources("satellite", result), ["TEMPO_NO2_L3", "EPA AQS"])

    def test_a_run_that_failed_drew_on_no_source(self):
        """A rate-limited or crashed run is salvaged into an error answer with
        nothing produced; naming a data source above it would claim data that
        was never retrieved."""
        from tta_backend.models import AgentResult
        from tta_backend.services.consulted_header import result_sources

        for metadata in ({"salvaged": True}, {"error": "invalid_envelope"}):
            with self.subTest(metadata=metadata):
                self.assertEqual(result_sources("satellite", AgentResult(text="rate-limited", metadata=metadata)), [])
                self.assertEqual(result_sources("ground", AgentResult(text="rate-limited", metadata=metadata)), [])

    def test_a_salvaged_run_that_still_produced_output_keeps_its_sources(self):
        from tta_backend.models import AgentResult
        from tta_backend.services.consulted_header import result_sources

        result = AgentResult(text="prose", charts=[_chart("TEMPO_NO2_L3")], metadata={"salvaged": True})
        self.assertEqual(result_sources("satellite", result), ["TEMPO_NO2_L3"])

    def test_provenance_is_read_without_copying_the_chart(self):
        """A heatmap carries its whole grid; dumping it to read one dict
        copies every value on the dispatch path."""
        from unittest.mock import patch

        from tta_backend.models import ChartPayload
        from tta_backend.services.consulted_header import result_sources

        chart = _chart("TEMPO_NO2_L3")
        with patch.object(ChartPayload, "model_dump", side_effect=AssertionError("model_dump called")):
            self.assertEqual(result_sources("satellite", _result_with([chart])), ["TEMPO_NO2_L3"])


class ConsultedSourcesTests(unittest.TestCase):
    def test_only_items_that_ran_count(self):
        from tta_backend.services.consulted_header import consulted_sources

        self.assertEqual(
            consulted_sources(_batch(("satellite", "ok", ["TEMPO_NO2_L3"]), ("ground", "rejected"))),
            ["TEMPO_NO2_L3"],
        )
        self.assertEqual(
            consulted_sources(_batch(("ground", "error"), ("satellite", "duplicate", ["TEMPO_NO2_L3"]))),
            ["TEMPO_NO2_L3", "EPA AQS"],
        )

    def test_satellite_datasets_come_before_ground_whatever_order_they_ran_in(self):
        from tta_backend.services.consulted_header import consulted_sources

        self.assertEqual(
            consulted_sources(_batch(("ground", "ok"), ("satellite", "ok", ["TEMPO_NO2_L3"]))),
            ["TEMPO_NO2_L3", "EPA AQS"],
        )

    def test_a_batch_saved_before_sources_were_recorded_falls_back_to_the_agent(self):
        """Checkpoints keep only chart references, so a batch written before
        items carried ``sources`` cannot recover its dataset names."""
        import json

        from tta_backend.services.consulted_header import consulted_sources

        legacy = json.loads(_batch(("satellite", "ok", ["TEMPO_NO2_L3"]), ("ground", "ok")))
        for item in legacy["results"]:
            item.pop("sources")
        self.assertEqual(consulted_sources(json.dumps(legacy)), ["NASA Earthdata", "EPA AQS"])

    def test_a_tool_message_that_is_not_a_dispatch_batch_names_no_source(self):
        from tta_backend.models import AgentResult, agent_result_to_json
        from tta_backend.services.consulted_header import consulted_sources

        self.assertEqual(consulted_sources(agent_result_to_json(AgentResult(text="legacy"))), [])
        self.assertEqual(consulted_sources("plain text"), [])

    def test_merging_keeps_first_seen_order_and_drops_repeats(self):
        from tta_backend.services.consulted_header import merge_sources

        self.assertEqual(merge_sources(["EPA AQS"], ["TEMPO_NO2_L3", "EPA AQS"]), ["TEMPO_NO2_L3", "EPA AQS"])


class SourcesHeaderTests(unittest.TestCase):
    def test_the_header_names_what_ran_and_replaces_any_the_model_wrote(self):
        from tta_backend.services.consulted_header import with_sources_header

        self.assertEqual(
            with_sources_header("Agent consulted: SATELLITE\n\nNO2 peaked near Newark.", ["TEMPO_NO2_L3", "EPA AQS"]),
            "Data sources: TEMPO_NO2_L3 · EPA AQS\n\nNO2 peaked near Newark.",
        )
        self.assertEqual(
            with_sources_header("Data sources: MODIS\n\nNO2 peaked near Newark.", ["TEMPO_NO2_L3"]),
            "Data sources: TEMPO_NO2_L3\n\nNO2 peaked near Newark.",
        )
        self.assertEqual(with_sources_header("Ozone was moderate.", ["EPA AQS"]),
                         "Data sources: EPA AQS\n\nOzone was moderate.")

    def test_text_the_model_wrote_above_its_own_header_is_dropped(self):
        """The old _strip_supervisor_preamble's job: a model that restates raw
        data before its header line. On the live stream the server's header
        comes first, so everything before the LAST header line goes."""
        from tta_backend.services.consulted_header import with_sources_header

        for text in (
            "Here is the NO2 data: 4.1e15, 3.9e15\nAgent consulted: SATELLITE\n\nNO2 peaked near Newark.",
            "Data sources: TEMPO_NO2_L3\n\nHere is the data: 4.1e15\nAgent consulted: SATELLITE\n\nNO2 peaked near Newark.",
        ):
            with self.subTest(text=text[:30]):
                self.assertEqual(
                    with_sources_header(text, ["TEMPO_NO2_L3"]),
                    "Data sources: TEMPO_NO2_L3\n\nNO2 peaked near Newark.",
                )

    def test_a_sources_line_inside_the_answer_does_not_drop_the_text_above_it(self):
        """Models cite with "Data sources:" mid-answer; only a leading header,
        or the legacy "Agent consulted" marker, may cut text above it."""
        from tta_backend.services.consulted_header import with_sources_header

        text = "NO2 peaked near Newark.\nData sources: TEMPO NO2 and EPA AQS.\nOzone stayed below 70 ppb."
        self.assertEqual(
            with_sources_header(text, ["TEMPO_NO2_L3"]),
            "Data sources: TEMPO_NO2_L3\n\n" + text,
        )

    def test_a_sources_line_the_model_writes_at_the_end_does_not_swallow_the_answer(self):
        from tta_backend.services.consulted_header import with_sources_header

        self.assertEqual(
            with_sources_header("NO2 peaked near Newark.\n\nData sources: TEMPO", ["TEMPO_NO2_L3"]),
            "Data sources: TEMPO_NO2_L3\n\nNO2 peaked near Newark.",
        )

    def test_with_nothing_consulted_no_header_is_claimed(self):
        from tta_backend.services.consulted_header import with_sources_header

        self.assertEqual(with_sources_header("Agent consulted: GROUND\n\nHello!", []), "Hello!")


class SupervisorStreamHeaderTests(unittest.IsolatedAsyncioTestCase):
    async def _stream(self, *events):
        import json
        from unittest.mock import AsyncMock, patch

        from tta_backend.services.chart_service import ChartService
        from tta_backend.services.chat_stream_service import ChatStreamService

        async def fake_stream_response(agent, message, thread_id, **kwargs):
            for event in events:
                yield event

        service = ChatStreamService(ChartService(), long_request_seconds=999)
        with patch("tta_backend.services.chat_stream_service.stream_response", fake_stream_response), \
             patch("tta_backend.services.chart_service.chart_repository.get_chart", AsyncMock(return_value=None)):
            sse = [e async for e in service.stream_chat_events(object(), None, None, "hi", "thread-1", "user-1", "r")]
        texts = [json.loads(e.split("data: ", 1)[1])["content"] for e in sse if e.startswith("event: text")]
        done = json.loads(next(e for e in sse if e.startswith("event: done")).split("data: ", 1)[1])
        return texts, done["response"]

    async def test_the_answer_streams_under_a_header_naming_what_ran(self):
        texts, response = await self._stream(
            ("tool_call", {"name": "dispatch", "args": {}}),
            ("tool_result", {"name": "dispatch", "content": _batch(("satellite", "ok"), ("ground", "rejected"))}),
            ("text", "Ozone peaked "),
            ("text", "over Philadelphia."),
        )

        self.assertEqual(texts, ["Data sources: NASA Earthdata\n\n", "Ozone peaked ", "over Philadelphia."])
        self.assertEqual(response, "Data sources: NASA Earthdata\n\nOzone peaked over Philadelphia.")

    async def test_a_header_the_model_wrote_is_replaced_by_what_ran(self):
        _, response = await self._stream(
            ("tool_call", {"name": "dispatch", "args": {}}),
            ("tool_result", {"name": "dispatch", "content": _batch(("satellite", "ok", ["TEMPO_NO2_L3"]))}),
            ("text", "Agent consulted: GROUND + SATELLITE\n\nOzone peaked over Philadelphia."),
        )

        self.assertEqual(response, "Data sources: TEMPO_NO2_L3\n\nOzone peaked over Philadelphia.")

    async def test_a_turn_that_dispatched_nothing_carries_no_header(self):
        texts, response = await self._stream(("text", "Hello! Ask me about air quality."))

        self.assertEqual(texts, ["Hello! Ask me about air quality."])
        self.assertEqual(response, "Hello! Ask me about air quality.")


class HistoryHeaderTests(unittest.IsolatedAsyncioTestCase):
    async def _history(self, messages):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch

        from tta_backend.services.chart_service import ChartService
        from tta_backend.services.history_service import HistoryService

        class FakeAgent:
            async def aget_state(self, config):
                return SimpleNamespace(values={"messages": messages})

        with patch("tta_backend.services.chart_service.chart_repository.get_charts", AsyncMock(return_value={})):
            return await HistoryService(ChartService()).build_history(FakeAgent(), "thread-1", "user-1")

    async def test_a_reloaded_dispatch_turn_carries_the_same_header_as_it_streamed_with(self):
        from types import SimpleNamespace

        history = await self._history([
            SimpleNamespace(type="human", content="ozone over the OTC and the nearest monitor to Camden"),
            SimpleNamespace(type="ai", content="", tool_calls=[{"id": "c1", "name": "dispatch", "args": {}}]),
            SimpleNamespace(type="tool", name="dispatch",
                            content=_batch(("satellite", "ok", ["TEMPO_NO2_L3"]), ("ground", "ok"))),
            SimpleNamespace(type="ai", content="Ozone peaked over Philadelphia; Camden Spruce St is nearest.",
                            tool_calls=[]),
            SimpleNamespace(type="human", content="thanks"),
            SimpleNamespace(type="ai", content="You're welcome.", tool_calls=[]),
        ])

        answers = [m["content"] for m in history if m["role"] == "assistant"]
        self.assertEqual(answers, [
            "Data sources: TEMPO_NO2_L3 · EPA AQS\n\nOzone peaked over Philadelphia; Camden Spruce St is nearest.",
            "You're welcome.",
        ])

    async def test_a_legacy_turn_keeps_the_header_it_was_saved_with(self):
        from types import SimpleNamespace

        from tta_backend.models import AgentResult, agent_result_to_json

        saved = "Agent consulted: SATELLITE\n\nNO2 peaked near Newark."
        history = await self._history([
            SimpleNamespace(type="human", content="plot NO2 over NJ"),
            SimpleNamespace(type="ai", content="", tool_calls=[{"id": "c1", "name": "ask_earthdata_agent", "args": {}}]),
            SimpleNamespace(type="tool", name="ask_earthdata_agent", content=agent_result_to_json(AgentResult(text="x"))),
            SimpleNamespace(type="ai", content=saved, tool_calls=[]),
        ])

        self.assertEqual([m["content"] for m in history if m["role"] == "assistant"], [saved])


class DispatchExecutorSourcesTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from tta_backend.utils.streaming import _call_budget

        self._token = _call_budget.set({})

    def tearDown(self):
        from tta_backend.utils.streaming import _call_budget

        _call_budget.reset(self._token)

    async def test_each_item_records_its_sources_before_charts_become_references(self):
        from tta_backend.models import GroundRequest, SatelliteRequest
        from tta_backend.services.dispatch_executor import execute_dispatch

        async def run_one(request):
            return _result(request.agent, 0, ["TEMPO_NO2_L3"] if request.agent == "satellite" else ())

        requests = [
            SatelliteRequest.model_validate({"intent": "visualize", "variable": "NO2", "location": "New Jersey"}),
            GroundRequest.model_validate({"intent": "readings", "pollutant": "NO2", "location": "Newark"}),
        ]
        batch = await execute_dispatch(requests, run_one)

        self.assertEqual([item.sources for item in batch.results], [["TEMPO_NO2_L3"], ["EPA AQS"]])

    async def test_a_run_that_raised_records_no_source(self):
        from tta_backend.models import SatelliteRequest
        from tta_backend.services.dispatch_executor import execute_dispatch

        async def run_one(request):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

        request = SatelliteRequest.model_validate({"intent": "visualize", "variable": "NO2", "location": "Newark"})
        batch = await execute_dispatch([request], run_one)

        self.assertEqual((batch.results[0].status, batch.results[0].sources), ("error", []))

    async def test_a_turn_whose_runs_all_failed_carries_no_header(self):
        from tta_backend.models import AgentResult, DispatchItemResult, DispatchResult

        failed = DispatchResult(results=[DispatchItemResult(
            index=0, agent="satellite", intent="visualize", status="ok",
            result=AgentResult(text="rate-limited"), sources=[],
        )]).model_dump_json(exclude_none=True)
        texts, response = await SupervisorStreamHeaderTests._stream(
            self,
            ("tool_call", {"name": "dispatch", "args": {}}),
            ("tool_result", {"name": "dispatch", "content": failed}),
            ("text", "The request was rate-limited."),
        )
        self.assertEqual(response, "The request was rate-limited.")


if __name__ == "__main__":
    unittest.main()
