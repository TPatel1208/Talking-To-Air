"""Runs benchmark cases through the current system.

Every case enters through ChatStreamService.stream_chat_events with the real
intent router, the real checkpointed supervisor (so it needs Postgres), the
real ground agent and the real satellite agent. The satellite agent gets the
workspace-bound production toolset, pointed at the fake earthdata service.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from contextlib import ExitStack
from datetime import datetime, timezone
from typing import Any, Callable
from unittest.mock import AsyncMock, Mock, patch

from benchmark.cases import BenchmarkCase, PatchSpec
from benchmark.outcome import BoundaryCall, Outcome, handles_from_calls

_USER_ID = "benchmark-user"
_AQS_GET_TARGET = "tta_backend.tools.ground_sensor_tools.epa_aqs_tools._aqs_get"
_TOKEN_RECORDER_TARGET = "tta_backend.infrastructure.metrics.record_llm_tokens"
_EMPTY_AQS_RESPONSE = {"Header": [{"status": "success"}], "Data": []}
_SOURCES_HEADER = re.compile(r"\A\s*Data sources:[ \t]*([^\n]+)")


def parse_sse(joined: str) -> list[tuple[str, Any]]:
    """The (event name, decoded data) pairs of a joined SSE stream, in order."""
    events: list[tuple[str, Any]] = []
    for block in joined.split("\n\n"):
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                try:
                    data = json.loads(line[len("data:"):].strip())
                except ValueError:
                    data = None
        if name is not None:
            events.append((name, data))
    return events


def sources_in_header(answer_text: str) -> set[str] | None:
    """The sources named in the answer's opening "Data sources:" line, or None without one."""
    match = _SOURCES_HEADER.match(answer_text)
    if match is None:
        return None
    return {source.strip() for source in match.group(1).split("·") if source.strip()}


def outcome_from_sse(
    joined: str,
    *,
    boundary_calls: list[BoundaryCall],
    aqs_calls: list[tuple[str, dict]],
    clock,
    elapsed_seconds: float,
    clock_end=None,
) -> Outcome:
    """Build an Outcome from a joined chat event stream and what the fake services saw."""
    outcome = Outcome(
        boundary_calls=list(boundary_calls),
        aqs_calls=list(aqs_calls),
        handles=handles_from_calls(boundary_calls),
        clock=clock,
        clock_end=clock_end,
        elapsed_seconds=elapsed_seconds,
    )
    streamed_text: list[str] = []
    final_text: str | None = None
    for name, data in parse_sse(joined):
        payload = data if isinstance(data, dict) else {}
        if name == "chart":
            outcome.charts.append({key: payload.get(key) for key in ("type", "title", "chart_id") if key in payload})
        elif name == "artifact":
            artifact_id = payload.get("id")
            if isinstance(artifact_id, str) and artifact_id not in outcome.artifact_ids:
                outcome.artifact_ids.append(artifact_id)
        elif name == "text":
            streamed_text.append(str(payload.get("content", "")))
        elif name == "status":
            if payload.get("stage"):
                outcome.stages.append(payload["stage"])
        elif name == "tool_call":
            if payload.get("name"):
                outcome.tool_calls.append(payload["name"])
        elif name == "done":
            outcome.done = True
            final_text = payload.get("response")
        elif name == "error" and outcome.error is None:
            outcome.error = str(payload.get("detail") or "error event")
    outcome.answer_text = final_text if isinstance(final_text, str) and final_text else "".join(streamed_text)
    outcome.sources_header = sources_in_header(outcome.answer_text)
    return outcome


def counting_tokens(sink: dict[tuple[str, str, str], int], record: Callable[..., None]) -> Callable[..., None]:
    """Wrap the app's token recorder so each count also lands in ``sink``.

    The app's metrics still receive every count; the benchmark only listens.
    """

    def counted(model: str, agent_type: str, kind: str, tokens: int) -> None:
        key = (model, agent_type, kind)
        sink[key] = sink.get(key, 0) + tokens
        record(model, agent_type, kind, tokens)

    return counted


def _mock_for(spec: PatchSpec):
    return AsyncMock(return_value=spec.return_value) if spec.is_async else Mock(return_value=spec.return_value)


class CurrentSystemAdapter:
    def __init__(self, model: str | None = None):
        self.model = model

    async def run(self, case: BenchmarkCase, volume) -> Outcome:
        from benchmark.fake_earthdata import FakeEarthdataMCPServer, build_fake_mcp
        from tta_backend.agents.earthdata_agent import build_earthdata_agent
        from tta_backend.agents.ground_sensor_agent import build_ground_agent
        from tta_backend.agents.supervisor_agent import build_agent as build_supervisor_agent
        from tta_backend.infrastructure.settings import Settings
        from tta_backend.earthdata_mcp.toolset import load_earthdata_tools
        from tta_backend.services.chart_service import ChartService
        from tta_backend.services.chat_stream_service import ChatStreamService
        from tta_backend.earthdata_mcp.tool_cache import clear_tool_cache
        from tta_backend.services.jobs_service import clear_terminal_status_cache
        from tta_backend.infrastructure import metrics
        from tta_backend.utils.streaming import get_call_budget

        # Cases run back to back in one process. The call budget, the tool
        # cache and the job status cache are process-global; left alone, a
        # later case asking for the same area or dataset is answered from the
        # cache, never reaches the fake service and fails its boundary checks.
        get_call_budget().clear()
        clear_tool_cache()
        clear_terminal_status_cache()

        boundary_calls: list[BoundaryCall] = []
        tokens: dict[tuple[str, str, str], int] = {}
        aqs_calls: list[tuple[str, dict]] = []

        async def capturing_aqs_get(endpoint: str, params: dict) -> dict:
            aqs_calls.append((endpoint, dict(params)))
            if case.aqs_get is None:
                return dict(_EMPTY_AQS_RESPONSE)
            return await case.aqs_get(endpoint, params)

        # A fresh conversation each run, so the checkpointed supervisor never sees an earlier run's history.
        thread_id = f"benchmark-{case.id}-{uuid.uuid4().hex[:8]}"
        server = FakeEarthdataMCPServer(build_fake_mcp(case.handlers, recorder=boundary_calls))
        server.start()
        try:
            settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
            tools = await load_earthdata_tools(settings, lambda: _USER_ID)
            ground_agent = build_ground_agent(model=self.model)
            satellite_agent = build_earthdata_agent(model=self.model, mcp_tools=tools)
            supervisor_agent = await build_supervisor_agent(
                model=self.model, ground_agent=ground_agent, satellite_agent=satellite_agent,
            )
            service = ChatStreamService(ChartService(), long_request_seconds=999)

            clock = datetime.now(timezone.utc).date()
            events: list[str] = []
            started = time.monotonic()
            with ExitStack() as stack:
                stack.enter_context(patch(_AQS_GET_TARGET, AsyncMock(side_effect=capturing_aqs_get)))
                stack.enter_context(patch(_TOKEN_RECORDER_TARGET, counting_tokens(tokens, metrics.record_llm_tokens)))
                for spec in case.patches:
                    stack.enter_context(patch(spec.target, _mock_for(spec)))
                async for event in service.stream_chat_events(
                    supervisor_agent, ground_agent, satellite_agent, case.prompt, thread_id, _USER_ID, thread_id,
                ):
                    events.append(event)
            elapsed_seconds = time.monotonic() - started
            clock_end = datetime.now(timezone.utc).date()
        finally:
            server.stop()

        outcome = outcome_from_sse(
            "".join(events),
            boundary_calls=boundary_calls,
            aqs_calls=aqs_calls,
            clock=clock,
            clock_end=clock_end,
            elapsed_seconds=elapsed_seconds,
        )
        outcome.tokens = tokens
        return outcome
