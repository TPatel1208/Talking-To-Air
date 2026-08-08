import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


def _echo_saved_chart(thread_id, payload, user_id):
    """chart_repository.save_chart stub -- echo the payload back with its
    ownership stamped on, the way the real repository does."""
    return {**payload, "thread_id": thread_id, "user_id": user_id}


class ExtractedServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_export_service_safe_name_and_missing_export(self):
        from tta_backend.services.export_service import ExportService

        service = ExportService(csv_export_max_granules=2)

        self.assertEqual(service.safe_export_name({"title": "TEMPO over Texas!"}, "csv"), "tempo-over-texas.csv")
        with self.assertRaisesRegex(ValueError, "full-resolution export metadata"):
            list(service.iter_chart_csv_rows({}))

    def test_chart_service_parses_agent_result_and_persists(self):
        from tta_backend.models import AgentResult, ChartPayload, agent_result_to_json
        from tta_backend.services.chart_service import ChartService

        service = ChartService()
        raw = agent_result_to_json(AgentResult(text="done", charts=[ChartPayload(type="heatmap", title="Map")]))

        text, charts = service.parse_charts(raw)

        self.assertEqual(text, "done")
        self.assertEqual(charts[0].type, "heatmap")

    async def test_chart_service_reuses_owned_stored_chart(self):
        from tta_backend.models import ChartPayload
        from tta_backend.services.chart_service import ChartService

        stored = {"chart_id": "chart-1", "user_id": "user-1"}
        with patch("tta_backend.services.chart_service.chart_repository.get_chart", AsyncMock(return_value=stored)):
            result = await ChartService().persist_chart_payload(
                "thread-1",
                ChartPayload(type="heatmap", chart_id="chart-1"),
                "user-1",
            )

        self.assertEqual(result, stored)

    async def test_history_service_surfaces_a_map_artifact_from_a_tool_message(self):
        from tta_backend.services.chart_service import ChartService
        from tta_backend.services.history_service import HistoryService

        tool_content = json.dumps({
            "type": "heatmap",
            "title": "TEMPO over NJ",
            "chart_id": "map_abc123",
            "vmin": 0.0,
            "vmax": 1.0,
            "bounds": [-75.0, 39.0, -73.0, 41.0],
            "variable": "TEMPO_NO2",
            "units": "mol/m^2",
            "metadata": {"source_handles": ["obs_1"]},
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

        class FakeAgent:
            async def aget_state(self, config):
                return SimpleNamespace(values={"messages": [
                    SimpleNamespace(type="human", content="plot TEMPO NO2 over NJ"),
                    SimpleNamespace(type="ai", content="", tool_calls=[{"id": "tc1", "name": "plot_singular", "args": {}}]),
                    SimpleNamespace(type="tool", name="plot_singular", content=tool_content),
                    SimpleNamespace(type="ai", content="Here is the map.", tool_calls=[]),
                ]})

        from tta_backend.services.artifact_store import artifact_store

        with patch("tta_backend.services.chart_service.chart_repository.get_chart", AsyncMock(return_value=None)), \
             patch("tta_backend.services.chart_service.chart_repository.save_chart", AsyncMock(side_effect=_echo_saved_chart)), \
             patch.object(artifact_store, "claim", side_effect=AssertionError("map artifacts must not go through the table artifact_store")) as claim:
            messages = await HistoryService(ChartService()).build_history(FakeAgent(), "thread-1", "user-1")

        claim.assert_not_called()
        assistant = messages[-1]
        self.assertEqual(len(assistant["charts"]), 1)
        self.assertEqual(assistant["charts"][0]["chart_id"], "map_abc123")
        self.assertEqual(len(assistant["artifacts"]), 1)
        self.assertEqual(assistant["artifacts"][0]["id"], "map_abc123")
        self.assertEqual(assistant["artifacts"][0]["type"], "map")

    async def test_history_service_builds_plain_history(self):
        from tta_backend.services.chart_service import ChartService
        from tta_backend.services.history_service import HistoryService

        class FakeAgent:
            async def aget_state(self, config):
                return SimpleNamespace(values={"messages": [
                    SimpleNamespace(type="human", content="hi"),
                    SimpleNamespace(type="ai", content="hello", tool_calls=[]),
                ]})

        messages = await HistoryService(ChartService()).build_history(FakeAgent(), "thread-1", "user-1")

        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["content"], "hello")

    async def test_chat_stream_service_emits_done_event(self):
        from tta_backend.services.chat_stream_service import ChatStreamService
        from tta_backend.services.chart_service import ChartService

        async def fake_stream_response(agent, message, thread_id, **kwargs):
            yield "text", "hello"

        service = ChatStreamService(ChartService(), long_request_seconds=999)
        with patch("tta_backend.services.chat_stream_service.stream_response", fake_stream_response):
            events = [
                event
                async for event in service.stream_chat_events(object(), None, None, "hi", "thread-1", "user-1", "req-1")
            ]

        self.assertIn("event: text", events[0])
        self.assertIn('"content": "hello"', events[0])
        self.assertIn("event: done", events[-1])
        self.assertIn('"response": "hello"', events[-1])

    async def test_chat_stream_service_forwards_job_progress_events(self):
        from tta_backend.services.chat_stream_service import ChatStreamService
        from tta_backend.services.chart_service import ChartService

        job_event = {
            "job_handle": "job_1",
            "status": "processing",
            "progress": 40,
            "phase": "materializing",
            "message": "40% complete",
        }

        async def fake_stream_response(agent, message, thread_id, **kwargs):
            yield "job_progress", job_event

        service = ChatStreamService(ChartService(), long_request_seconds=999)
        with patch("tta_backend.services.chat_stream_service.stream_response", fake_stream_response):
            events = [
                event
                async for event in service.stream_chat_events(object(), None, None, "hi", "thread-1", "user-1", "req-1")
            ]

        self.assertIn("event: job_progress", events[0])
        self.assertIn('"job_handle": "job_1"', events[0])
        self.assertIn('"status": "processing"', events[0])

    async def test_chat_stream_service_does_not_warn_for_plain_tool_result(self):
        from tta_backend.services.chat_stream_service import ChatStreamService
        from tta_backend.services.chart_service import ChartService

        service = ChatStreamService(ChartService(), long_request_seconds=999)

        with self.assertNoLogs("tta_backend.services.chat_stream_service", level="WARNING"):
            events = [
                event
                async for event in service._tool_result_events(
                    "plain tool text",
                    "thread-1",
                    "user-1",
                    [],
                )
            ]

        self.assertEqual(events, [])

    async def test_chat_stream_service_emits_artifact_refs(self):
        from tta_backend.services.artifact_store import artifact_store
        from tta_backend.services.chat_stream_service import ChatStreamService
        from tta_backend.services.chart_service import ChartService

        ref = artifact_store.put_table(
            "Sample Table",
            ["date", "value"],
            [{"date": "2024-01-01", "value": 10}],
        )
        content = json.dumps({"Header": [{"rows": 1}], "Body": [], "_artifact_refs": [ref.model_dump()]})
        service = ChatStreamService(ChartService(), long_request_seconds=999)

        with patch("tta_backend.services.artifact_store.artifact_repository.save_artifact", AsyncMock()), \
             patch("tta_backend.services.artifact_store.artifact_repository.delete_expired_unclaimed", AsyncMock()):
            events = [
                event
                async for event in service._tool_result_events(
                    content,
                    "thread-1",
                    "user-1",
                    [],
                    [],
                )
            ]

        self.assertEqual(len(events), 1)
        self.assertIn("event: artifact", events[0])
        self.assertIn(ref.id, events[0])
        page = await artifact_store.get_page(ref.id, "user-1")
        self.assertEqual(page["rows"], [{"date": "2024-01-01", "value": 10}])

    async def test_chat_stream_service_emits_both_chart_and_artifact_for_a_map_payload(self):
        from tta_backend.services.chat_stream_service import ChatStreamService
        from tta_backend.services.chart_service import ChartService

        content = json.dumps({
            "type": "heatmap",
            "title": "TEMPO over NJ",
            "chart_id": "map_abc123",
            "vmin": 0.0,
            "vmax": 1.0,
            "bounds": [-75.0, 39.0, -73.0, 41.0],
            "variable": "TEMPO_NO2",
            "units": "mol/m^2",
            "metadata": {"source_handles": ["obs_1"]},
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
        service = ChatStreamService(ChartService(), long_request_seconds=999)

        with patch("tta_backend.services.chart_service.chart_repository.get_chart", AsyncMock(return_value=None)), \
             patch("tta_backend.services.chart_service.chart_repository.save_chart", AsyncMock(side_effect=_echo_saved_chart)):
            events = [
                event
                async for event in service._tool_result_events(content, "thread-1", "user-1", [], [])
            ]

        self.assertEqual(len(events), 2)
        self.assertIn("event: chart", events[0])
        self.assertIn('"chart_id": "map_abc123"', events[0])
        self.assertIn("event: artifact", events[1])
        self.assertIn('"id": "map_abc123"', events[1])
        self.assertIn('"type": "map"', events[1])

    async def test_chat_stream_service_warns_for_malformed_chart_payload(self):
        from tta_backend.services.chat_stream_service import ChatStreamService
        from tta_backend.services.chart_service import ChartService

        service = ChatStreamService(ChartService(), long_request_seconds=999)

        with self.assertLogs("tta_backend.services.chat_stream_service", level="WARNING") as captured:
            events = [
                event
                async for event in service._tool_result_events(
                    '{"type":',
                    "thread-1",
                    "user-1",
                    [],
                )
            ]

        self.assertEqual(events, [])
        self.assertIn("chart_payload_parse_failure", captured.output[0])

    async def test_chat_stream_service_persists_and_emits_a_bubbled_chart_payload_event(self):
        """T13: emit_chart's ("chart_payload", dict) event (bubbled up from a
        sub-agent's own stream, mirroring job_progress) is persisted and
        emitted as a "chart" SSE event directly, without waiting for the
        sub-agent's final tool_result envelope."""
        from tta_backend.services.chat_stream_service import ChatStreamService
        from tta_backend.services.chart_service import ChartService

        chart_payload = {"type": "heatmap", "chart_id": "map_abc123", "title": "TEMPO over NJ"}

        async def fake_stream_response(agent, message, thread_id, **kwargs):
            yield "chart_payload", chart_payload

        service = ChatStreamService(ChartService(), long_request_seconds=999)
        with patch("tta_backend.services.chat_stream_service.stream_response", fake_stream_response), \
             patch("tta_backend.services.chart_service.chart_repository.get_chart", AsyncMock(return_value=None)), \
             patch("tta_backend.services.chart_service.chart_repository.save_chart", AsyncMock(side_effect=_echo_saved_chart)):
            events = [
                event
                async for event in service.stream_chat_events(object(), None, None, "hi", "thread-1", "user-1", "req-1")
            ]

        chart_events = [e for e in events if e.startswith("event: chart")]
        self.assertEqual(len(chart_events), 1)
        self.assertIn('"chart_id": "map_abc123"', chart_events[0])

    async def test_chat_stream_service_never_emits_the_same_chart_id_twice(self):
        """A chart bubbled via chart_payload and the same chart_id later
        embedded in the sub-agent's tool_result envelope (AgentResult.charts)
        must not double-render in the UI (Frontend/src/hooks/useChat.js just
        appends every "chart" event to a list, with no dedup of its own)."""
        from tta_backend.models import AgentResult, ChartPayload, agent_result_to_json
        from tta_backend.services.chat_stream_service import ChatStreamService
        from tta_backend.services.chart_service import ChartService

        chart_payload = {"type": "heatmap", "chart_id": "map_abc123", "title": "TEMPO over NJ"}
        envelope = agent_result_to_json(AgentResult(
            text="Plotted NO2.",
            charts=[ChartPayload(type="heatmap", chart_id="map_abc123", title="TEMPO over NJ")],
        ))

        async def fake_stream_response(agent, message, thread_id, **kwargs):
            yield "chart_payload", chart_payload
            yield "tool_result", {"name": "ask_earthdata_agent", "content": envelope}

        service = ChatStreamService(ChartService(), long_request_seconds=999)
        with patch("tta_backend.services.chat_stream_service.stream_response", fake_stream_response), \
             patch("tta_backend.services.chart_service.chart_repository.get_chart", AsyncMock(return_value=None)), \
             patch("tta_backend.services.chart_service.chart_repository.save_chart", AsyncMock(side_effect=_echo_saved_chart)):
            events = [
                event
                async for event in service.stream_chat_events(object(), None, None, "hi", "thread-1", "user-1", "req-1")
            ]

        chart_events = [e for e in events if e.startswith("event: chart")]
        self.assertEqual(len(chart_events), 1)

    async def test_chat_stream_service_ends_the_turn_with_a_timeout_error_then_done_when_the_budget_expires(self):
        """T38: a wedged model stream must not hang the turn forever — once
        chat_turn_timeout_seconds elapses, the SSE sequence ends with the
        timeout `error` (naming no in-flight jobs here) then `done`, and the
        underlying agent stream is actually cancelled, not just abandoned."""
        from tta_backend.services.chat_stream_service import ChatStreamService
        from tta_backend.services.chart_service import ChartService

        class HangingAgent:
            def __init__(self):
                self.cancelled = False

            async def astream(self, input_, config, stream_mode):
                try:
                    await asyncio.sleep(999)
                    yield "messages", (SimpleNamespace(content="unreachable", type="ai", tool_calls=None), {})
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise

        agent = HangingAgent()
        service = ChatStreamService(ChartService(), long_request_seconds=999, chat_turn_timeout_seconds=0.05)

        events = [
            event
            async for event in service.stream_chat_events(agent, None, None, "hi", "thread-1", "user-1", "req-1")
        ]
        # The generator's own finally block only *schedules* the producer
        # task's cancellation (Task.cancel()) — give the loop one tick to
        # actually deliver it before checking the fake's flag.
        await asyncio.sleep(0)

        joined = "".join(events)
        self.assertIn("event: error", joined)
        self.assertIn("event: done", joined)
        self.assertLess(joined.index("event: error"), joined.index("event: done"))
        self.assertIn("ran out of time", joined)
        self.assertTrue(agent.cancelled, "the hanging astream must actually be cancelled, not just abandoned")

    async def test_chat_stream_service_completes_normally_when_slowness_is_within_the_turn_budget(self):
        """A turn whose only slowness is legitimate work that finishes under
        the turn budget must not be misread as a hang."""
        from tta_backend.services.chat_stream_service import ChatStreamService
        from tta_backend.services.chart_service import ChartService

        async def fake_stream_response(agent, message, thread_id, **kwargs):
            await asyncio.sleep(0.01)
            yield "text", "hello"

        service = ChatStreamService(ChartService(), long_request_seconds=999, chat_turn_timeout_seconds=5)
        with patch("tta_backend.services.chat_stream_service.stream_response", fake_stream_response):
            events = [
                event
                async for event in service.stream_chat_events(object(), None, None, "hi", "thread-1", "user-1", "req-1")
            ]

        joined = "".join(events)
        self.assertNotIn("event: error", joined)
        self.assertIn("event: done", joined)

    async def test_find_closest_monitor_accepts_string_k(self):
        from tta_backend.tools.ground_sensor_tools import epa_aqs_tools

        monitors = [
            {
                "state_code": "01",
                "county_code": "001",
                "site_number": "0001",
                "latitude": "40.0",
                "longitude": "-74.0",
                "local_site_name": "Near",
            },
            {
                "state_code": "01",
                "county_code": "001",
                "site_number": "0002",
                "latitude": "40.1",
                "longitude": "-74.1",
                "local_site_name": "Far",
            },
        ]

        with patch.object(
            epa_aqs_tools.geocoding_service,
            "ageocode",
            AsyncMock(return_value={"latitude": 40.0, "longitude": -74.0, "bbox": [39.9, 40.1, -74.1, -73.9]}),
        ), patch(
            "tta_backend.tools.ground_sensor_tools.epa_aqs_tools._fetch_active_monitors",
            AsyncMock(return_value=monitors),
        ) as fetch:
            result = await epa_aqs_tools.find_closest_monitor.ainvoke({
                "location": "Newark, NJ",
                "bdate": "2024-01-01",
                "edate": "2024-01-01",
                "k": "2",
            })

        self.assertEqual(fetch.await_args.args[-1], 2)
        self.assertEqual(result["Header"][0]["rows"], 2)
        self.assertEqual(result["Body"][0]["station_name"], "Near")
        self.assertEqual(result["Body"][0]["monitor_name"], "Near")

    async def test_find_closest_monitor_by_coords_accepts_string_k(self):
        from tta_backend.tools.ground_sensor_tools import epa_aqs_tools

        monitors = [
            {
                "state_code": "01",
                "county_code": "001",
                "site_number": "0001",
                "latitude": "40.0",
                "longitude": "-74.0",
                "local_site_name": "Near",
            },
            {
                "state_code": "01",
                "county_code": "001",
                "site_number": "0002",
                "latitude": "40.1",
                "longitude": "-74.1",
                "local_site_name": "Far",
            },
        ]

        with patch(
            "tta_backend.tools.ground_sensor_tools.epa_aqs_tools._fetch_active_monitors",
            AsyncMock(return_value=monitors),
        ) as fetch:
            result = await epa_aqs_tools.find_closest_monitor_by_coords.ainvoke({
                "latitude": "40.0",
                "longitude": "-74.0",
                "bdate": "2024-01-01",
                "edate": "2024-01-01",
                "k": "2",
            })

        self.assertEqual(fetch.await_args.args[-1], 2)
        self.assertEqual(result["Header"][0]["rows"], 2)
        self.assertEqual(result["Body"][0]["station_name"], "Near")
        self.assertEqual(result["Body"][0]["monitor_name"], "Near")

    async def test_aqs_get_deduplicates_identical_requests(self):
        from tta_backend.tools.ground_sensor_tools import epa_aqs_tools
        from unittest.mock import MagicMock

        call_count = 0

        async def fake_get(url, params):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(return_value={"Header": [{"status": "success"}], "Data": []})
            return resp

        epa_aqs_tools._reset_aqs_request_state()
        self.addCleanup(epa_aqs_tools._reset_aqs_request_state)

        with patch("tta_backend.tools.ground_sensor_tools.epa_aqs_tools.httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=fake_get)
            mock_client_cls.return_value = mock_client

            result1 = await epa_aqs_tools._aqs_get("dailyData/bySite", {"state": "48", "county": "453", "site": "0014"})
            result2 = await epa_aqs_tools._aqs_get("dailyData/bySite", {"state": "48", "county": "453", "site": "0014"})

        self.assertEqual(call_count, 1, "Expected exactly 1 HTTP call for duplicate params")
        self.assertIs(result1, result2, "Both calls should return the same cached object")

    async def test_aqs_get_reuses_settled_data_across_requests(self):
        """Replaces a test that asserted the opposite — that each new request
        re-fetched — which was the ContextVar defect written down as a
        requirement. EPA measurements outside the publication lag never
        change, and the credential is shared app-wide, so re-fetching them
        per request spent a rate-limit slot for identical bytes."""
        from tta_backend.tools.ground_sensor_tools import epa_aqs_tools
        from unittest.mock import MagicMock

        call_count = 0

        async def fake_get(url, params):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(return_value={"Header": [{"status": "success"}], "Data": []})
            return resp

        epa_aqs_tools._reset_aqs_request_state()
        self.addCleanup(epa_aqs_tools._reset_aqs_request_state)

        with patch("tta_backend.tools.ground_sensor_tools.epa_aqs_tools.httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=fake_get)
            mock_client_cls.return_value = mock_client

            await epa_aqs_tools._aqs_get("dailyData/bySite", {"state": "48", "edate": "20200107"})
            # A later turn, a different researcher — same settled query.
            await epa_aqs_tools._aqs_get("dailyData/bySite", {"state": "48", "edate": "20200107"})

        self.assertEqual(call_count, 1, "Settled measurements should be served from cache")

    def test_summary_filter_rejects_literal_site_id_placeholder(self):
        from tta_backend.tools.ground_sensor_tools import epa_aqs_tools

        with self.assertRaisesRegex(ValueError, "placeholder"):
            epa_aqs_tools._resolve_filter(
                "dailyData",
                "34",
                "19",
                "site_id",
                None,
                None,
                None,
                None,
                None,
            )

    def test_summary_filter_accepts_station_id_as_site_number(self):
        from tta_backend.tools.ground_sensor_tools import epa_aqs_tools

        endpoint, params = epa_aqs_tools._resolve_filter(
            "dailyData",
            None,
            None,
            "34-019-0007",
            None,
            None,
            None,
            None,
            None,
        )

        self.assertEqual(endpoint, "dailyData/bySite")
        self.assertEqual(params, {"state": "34", "county": "019", "site": "0007"})

    async def test_daily_summary_returns_bounded_per_site_aggregates(self):
        from tta_backend.tools.ground_sensor_tools import epa_aqs_tools

        records = []
        for idx in range(30):
            records.append({
                "state_code": "01",
                "county_code": "001",
                "site_number": f"{idx:04d}",
                "date_local": "2024-01-01",
                "arithmetic_mean": idx + 1,
                "maximum_value": idx + 2,
                "first_max_value": idx + 2.5,
                "first_max_hour": 13,
                "units_of_measure": "ppb",
                "sample_duration": "1 HOUR",
                "pollutant_standard": "NO2 1-hour 2010",
                "observation_count": 24,
                "observation_percent": 100,
                "local_site_name": f"Site {idx}",
            })

        with patch(
            "tta_backend.tools.ground_sensor_tools.epa_aqs_tools._fetch_summary",
            AsyncMock(return_value=(records, "dailyData/byState", {"state": "01"})),
        ):
            result = await epa_aqs_tools.get_daily_summary.ainvoke({
                "state_code": "01",
                "bdate": "2024-01-01",
                "edate": "2024-01-01",
                "pollutant_standard": "NO2 1-hour 2010",
            })

        self.assertEqual(result["Header"][0]["rows"], 25)
        self.assertEqual(result["Header"][0]["total_sites_matched"], 30)
        self.assertEqual(result["Header"][0]["sites_returned"], 25)
        self.assertEqual(result["Header"][0]["total_periods_matched"], 30)
        self.assertEqual(result["Header"][0]["periods_returned"], 25)
        self.assertEqual(result["Header"][0]["granularity"], "daily")
        self.assertEqual(result["Header"][0]["artifact_count"], 1)
        self.assertEqual(len(result["Body"]), 25)
        self.assertEqual(result["Body"][0]["site_id"], "01-001-0000")
        self.assertEqual(result["Body"][0]["period"], "2024-01-01")
        self.assertEqual(result["Body"][0]["date"], "2024-01-01")
        self.assertEqual(result["Body"][0]["monitor_name"], "Site 0")
        self.assertEqual(result["Body"][0]["n_periods"], 1)
        self.assertEqual(result["Body"][0]["mean"], 1.0)
        # dailyData supplies no minimum_value; min must be null, never a
        # copy of the mean (fabricated-stats regression).
        self.assertIsNone(result["Body"][0]["min"])
        self.assertEqual(result["Body"][0]["max"], 2.0)
        self.assertEqual(result["Body"][0]["peak"], {"value": 2.5, "date": "2024-01-01", "first_max_hour": 13})
        artifact_id = result["_artifact_refs"][0]["id"]
        page = await epa_aqs_tools.artifact_store.get_page(artifact_id, "user-1")
        self.assertEqual(page["total_rows"], 25)

    async def test_daily_summary_returns_one_row_per_day_for_site(self):
        from tta_backend.tools.ground_sensor_tools import epa_aqs_tools

        records = [
            {
                "state_code": "01",
                "county_code": "001",
                "site_number": "0001",
                "date_local": "2024-01-01",
                "arithmetic_mean": 10,
                "maximum_value": 12,
                "first_max_value": 14,
                "first_max_hour": 11,
                "units_of_measure": "ppb",
                "sample_duration": "1 HOUR",
                "pollutant_standard": "NO2 1-hour 2010",
                "observation_count": 24,
                "observation_percent": 100,
                "local_site_name": "Downtown",
            },
            {
                "state_code": "01",
                "county_code": "001",
                "site_number": "0001",
                "date_local": "2024-01-02",
                "arithmetic_mean": 20,
                "maximum_value": 21,
                "first_max_value": 25,
                "first_max_hour": 15,
                "units_of_measure": "ppb",
                "sample_duration": "1 HOUR",
                "pollutant_standard": "NO2 1-hour 2010",
                "observation_count": 24,
                "observation_percent": 100,
                "local_site_name": "Downtown",
            },
        ]

        with patch(
            "tta_backend.tools.ground_sensor_tools.epa_aqs_tools._fetch_summary",
            AsyncMock(return_value=(records, "dailyData/bySite", {"state": "01"})),
        ):
            result = await epa_aqs_tools.get_daily_summary.ainvoke({
                "state_code": "01",
                "county_code": "001",
                "site_number": "0001",
                "bdate": "2024-01-01",
                "edate": "2024-01-02",
                "pollutant_standard": "NO2 1-hour 2010",
            })

        self.assertEqual(result["Header"][0]["total_sites_matched"], 1)
        self.assertEqual(result["Header"][0]["sites_returned"], 1)
        self.assertEqual(result["Header"][0]["total_periods_matched"], 2)
        self.assertEqual(result["Header"][0]["periods_returned"], 2)
        self.assertEqual(len(result["Body"]), 2)
        self.assertEqual([row["period"] for row in result["Body"]], ["2024-01-01", "2024-01-02"])
        self.assertEqual(result["Body"][0]["mean"], 10.0)
        self.assertEqual(result["Body"][1]["mean"], 20.0)
        self.assertEqual(result["Body"][1]["peak"], {"value": 25.0, "date": "2024-01-02", "first_max_hour": 15})

    async def test_daily_summary_long_range_returns_quarterly_rows(self):
        from tta_backend.tools.ground_sensor_tools import epa_aqs_tools

        records = [{
            "state_code": "01",
            "county_code": "001",
            "site_number": "0001",
            "year": 2024,
            "quarter": 1,
            "arithmetic_mean": 10,
            "minimum_value": 3,
            "maximum_value": 12,
            "units_of_measure": "ppb",
            "sample_duration": "1 HOUR",
            "pollutant_standard": "NO2 1-hour 2010",
            "observation_count": 100,
            "observation_percent": 80,
            "local_site_name": "Downtown",
        }]
        fetch_summary = AsyncMock(return_value=(records, "quarterlyData/bySite", {"state": "01"}))

        with patch("tta_backend.tools.ground_sensor_tools.epa_aqs_tools._fetch_summary", fetch_summary):
            result = await epa_aqs_tools.get_daily_summary.ainvoke({
                "state_code": "01",
                "county_code": "001",
                "site_number": "0001",
                "bdate": "2024-01-01",
                "edate": "2024-12-31",
                "pollutant_standard": "NO2 1-hour 2010",
            })

        self.assertEqual(fetch_summary.await_args.args[0], "quarterlyData")
        self.assertEqual(result["Header"][0]["granularity"], "quarterly")
        self.assertIn("exceeds 31 days", result["Header"][0]["note"])
        self.assertEqual(result["Body"][0]["period"], "2024-Q1")

    async def test_quarterly_summary_returns_one_row_per_quarter(self):
        from tta_backend.tools.ground_sensor_tools import epa_aqs_tools

        records = [
            {
                "state_code": "01",
                "county_code": "001",
                "site_number": "0001",
                "year": 2024,
                "quarter": 1,
                "arithmetic_mean": 10,
                "minimum_value": 3,
                "maximum_value": 12,
                "units_of_measure": "ppb",
                "sample_duration": "1 HOUR",
                "pollutant_standard": "NO2 1-hour 2010",
                "observation_count": 100,
                "observation_percent": 80,
                "local_site_name": "Downtown",
            },
            {
                "state_code": "01",
                "county_code": "001",
                "site_number": "0001",
                "year": 2024,
                "quarter": 2,
                "arithmetic_mean": 20,
                "minimum_value": 4,
                "maximum_value": 31,
                "units_of_measure": "ppb",
                "sample_duration": "1 HOUR",
                "pollutant_standard": "NO2 1-hour 2010",
                "observation_count": 110,
                "observation_percent": 90,
                "local_site_name": "Downtown",
            },
        ]

        with patch(
            "tta_backend.tools.ground_sensor_tools.epa_aqs_tools._fetch_summary",
            AsyncMock(return_value=(records, "quarterlyData/bySite", {"state": "01"})),
        ):
            result = await epa_aqs_tools.get_quarterly_summary.ainvoke({
                "state_code": "01",
                "county_code": "001",
                "site_number": "0001",
                "bdate": "2024-01-01",
                "edate": "2024-12-31",
                "pollutant_standard": "NO2 1-hour 2010",
            })

        self.assertEqual(result["Header"][0]["rows"], 2)
        self.assertEqual(result["Header"][0]["total_sites_matched"], 1)
        self.assertEqual(result["Header"][0]["sites_returned"], 1)
        self.assertEqual(result["Header"][0]["total_periods_matched"], 2)
        self.assertEqual(result["Header"][0]["periods_returned"], 2)
        self.assertEqual(result["Header"][0]["granularity"], "quarterly")

        first, second = result["Body"]
        self.assertEqual(first["period"], "2024-Q1")
        self.assertEqual(first["year"], 2024)
        self.assertEqual(first["quarter"], 1)
        self.assertEqual(first["n_periods"], 1)
        self.assertEqual(first["mean"], 10.0)
        self.assertEqual(first["min"], 3.0)
        self.assertEqual(first["max"], 12.0)
        self.assertEqual(first["observation_count"], 100)
        self.assertEqual(first["observation_percent"], 80.0)
        self.assertEqual(first["peak"], {"value": 12.0, "year": 2024, "quarter": 1})
        self.assertEqual(second["period"], "2024-Q2")
        self.assertEqual(second["mean"], 20.0)
        self.assertEqual(second["max"], 31.0)
        self.assertEqual(second["peak"], {"value": 31.0, "year": 2024, "quarter": 2})

    async def test_annual_summary_returns_one_row_per_year(self):
        from tta_backend.tools.ground_sensor_tools import epa_aqs_tools

        records = [
            {
                "state_code": "01",
                "county_code": "001",
                "site_number": "0001",
                "year": 2023,
                "arithmetic_mean": 8,
                "minimum_value": 1,
                "maximum_value": 19,
                "units_of_measure": "ppb",
                "sample_duration": "1 HOUR",
                "pollutant_standard": "NO2 1-hour 2010",
                "observation_count": 200,
                "observation_percent": 75,
                "valid_day_count": 250,
                "required_day_count": 300,
                "local_site_name": "Downtown",
            },
            {
                "state_code": "01",
                "county_code": "001",
                "site_number": "0001",
                "year": 2024,
                "arithmetic_mean": 12,
                "minimum_value": 2,
                "maximum_value": 22,
                "units_of_measure": "ppb",
                "sample_duration": "1 HOUR",
                "pollutant_standard": "NO2 1-hour 2010",
                "observation_count": 210,
                "observation_percent": 85,
                "valid_day_count": 260,
                "required_day_count": 300,
                "local_site_name": "Downtown",
            },
        ]

        with patch(
            "tta_backend.tools.ground_sensor_tools.epa_aqs_tools._fetch_summary",
            AsyncMock(return_value=(records, "annualData/bySite", {"state": "01"})),
        ):
            result = await epa_aqs_tools.get_annual_summary.ainvoke({
                "state_code": "01",
                "county_code": "001",
                "site_number": "0001",
                "bdate": "2023-01-01",
                "edate": "2024-12-31",
                "pollutant_standard": "NO2 1-hour 2010",
            })

        self.assertEqual(result["Header"][0]["rows"], 2)
        self.assertEqual(result["Header"][0]["total_sites_matched"], 1)
        self.assertEqual(result["Header"][0]["sites_returned"], 1)
        self.assertEqual(result["Header"][0]["total_periods_matched"], 2)
        self.assertEqual(result["Header"][0]["periods_returned"], 2)
        self.assertEqual(result["Header"][0]["granularity"], "annual")

        first, second = result["Body"]
        self.assertEqual(first["period"], "2023")
        self.assertEqual(first["year"], 2023)
        self.assertEqual(first["n_periods"], 1)
        self.assertEqual(first["mean"], 8.0)
        self.assertEqual(first["valid_day_count"], 250)
        self.assertEqual(first["required_day_count"], 300)
        self.assertEqual(first["peak"], {"value": 19.0, "year": 2023})
        self.assertEqual(second["period"], "2024")
        self.assertEqual(second["mean"], 12.0)
        self.assertEqual(second["valid_day_count"], 260)
        self.assertEqual(second["required_day_count"], 300)
        self.assertEqual(second["peak"], {"value": 22.0, "year": 2024})


if __name__ == "__main__":
    unittest.main()
