"""Runs benchmark cases through the agent harness's workflows, with no model.

A case that names a workflow is run directly through ``WorkflowRunner`` on the
benchmark's fake data boundary: the earthdata MCP fake behind the app's own
session, EPA AQS behind a capturing fetch, memory stores and an in-memory
checkpointer. The ``Result`` envelope is turned into an ``Outcome`` the
existing checks score. Nothing here turns a prompt into inputs; the case does.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from contextlib import ExitStack
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

from benchmark.cases import BenchmarkCase, PatchSpec
from benchmark.outcome import BoundaryCall, Outcome, handles_from_calls

_USER_ID = "benchmark-user"
_CONVERSATION_ID = "benchmark-conversation"
_PACKS_ROOT = Path(__file__).resolve().parents[2] / "packs"
_AQS_FETCHER_TARGET = "tta_backend.domain.monitors.aqs.fetcher"
_GEOCODER_TARGET = "tta_backend.utils.plotting.GeocodingService"
_EMPTY_AQS_RESPONSE = {"Header": [{"status": "success"}], "Data": []}

# Connector secrets the fakes accept; they never leave the process.
EARTHDATA_SECRET = "benchmark-earthdata-secret"
AQS_EMAIL = "benchmark@example.test"
AQS_KEY = "benchmark-aqs-key"
CREDENTIAL_STRINGS = (EARTHDATA_SECRET, AQS_EMAIL, AQS_KEY)

# Offline geocoder answers for the places cases name, in Nominatim's shape
# (bbox is south, north, west, east); any other place geocodes to nothing.
_NEWARK = {
    "latitude": 40.7357, "longitude": -74.1724, "display_name": "Newark, New Jersey, United States",
    "polygon": None, "bbox": [40.6718, 40.7885, -74.2515, -74.1252],
}
_PLACES: dict[str, dict[str, Any]] = {
    "houston": {
        "latitude": 29.7604, "longitude": -95.3698, "display_name": "Houston, Texas, United States",
        "polygon": None, "bbox": [29.5370, 30.1107, -95.9097, -94.9446],
    },
    "newark nj": _NEWARK,
    "newark new jersey": _NEWARK,
    "death valley": {
        "latitude": 36.5054, "longitude": -117.0794, "display_name": "Death Valley, California, United States",
        "polygon": None, "bbox": [35.6, 37.4, -117.9, -116.3],
    },
}


class NotCovered(ValueError):
    """The case names no workflow, so this adapter does not run it."""


def _geocode(location_name: str) -> dict[str, Any] | None:
    """The hit for exactly this wording, as Nominatim answers a place name it knows; anything else finds nothing."""
    key = " ".join(re.sub(r"[^a-z0-9 ]", " ", location_name.lower()).split())
    hit = _PLACES.get(key)
    return None if hit is None else dict(hit)


async def _ageocode(self: Any, location_name: str) -> dict[str, Any] | None:
    return _geocode(location_name)


def _geocode_sync(self: Any, location_name: str) -> dict[str, Any] | None:
    return _geocode(location_name)


def _mock_for(spec: PatchSpec) -> AsyncMock | Mock:
    return AsyncMock(return_value=spec.return_value) if spec.is_async else Mock(return_value=spec.return_value)


class _SystemClock:
    def today(self) -> date:
        return datetime.now(timezone.utc).date()

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


_MISSING_CONNECTOR = re.compile(r"^This workflow needs the .+ connectors?\.")


def _misconfigured(result: Any) -> bool:
    """Whether a needs_input result means the adapter left out a connector or an input.

    Any other needs_input, such as a refused area, is the workflow's own answer.
    """
    question = result.question
    if isinstance(question, dict) and question.get("kind") == "inputs":
        return True
    return bool(_MISSING_CONNECTOR.match(result.reason or ""))


def compose_answer(result: Any) -> str:
    """The envelope's own wording as text: mode, findings, disclosures, gaps and, unless done, the reason."""
    lines = [result.mode.label()]
    for finding in result.findings:
        unit = f" {finding.unit}" if finding.unit else ""
        lines.append(f"{finding.name}: {finding.value}{unit}")
    lines.extend(result.disclosures)
    lines.extend(f"{gap.step}: {gap.reason}" for gap in result.gaps)
    if result.status != "done" and result.reason:
        lines.append(result.reason)
    return "\n".join(lines)


class WorkflowAdapter:
    """Runs the case's workflow and reports what crossed the boundary and what the user would be shown."""

    def __init__(self) -> None:
        self._catalogue: Any = None
        self._registry: Any = None

    @staticmethod
    def covers(case: BenchmarkCase) -> bool:
        return case.workflow is not None

    def _load(self) -> tuple[Any, Any]:
        if self._catalogue is None:
            from tta_backend.datasets.registry import load_registry
            from tta_backend.harness.capabilities.catalogue import build_registry
            from tta_backend.harness.workflows.loader import load_packs

            self._registry = build_registry()
            self._catalogue = load_packs(_PACKS_ROOT, self._registry, load_registry())
        return self._registry, self._catalogue

    async def run(self, case: BenchmarkCase, volume: Any) -> Outcome:
        if case.workflow is None:
            raise NotCovered(case.id)

        from langgraph.checkpoint.memory import InMemorySaver

        from benchmark.fake_earthdata import FakeEarthdataMCPServer, build_fake_mcp
        from tta_backend.earthdata_mcp.client import load_raw_mcp_tools
        from tta_backend.harness.run_context import Identity, RunContext
        from tta_backend.harness.storage import memory_stores
        from tta_backend.harness.testing import (
            MemoryArtifacts,
            RecordingEvents,
            StaticCredentials,
            env_factory,
        )
        from tta_backend.harness.workflows import WorkflowRunner
        from tta_backend.harness.workflows.loader import UnknownWorkflow
        from tta_backend.harness.workflows.compile import NotRunnable
        from tta_backend.infrastructure.settings import Settings, get_settings
        from tta_backend.services.harness_host import (
            AdmissionCompute,
            McpEarthdataSession,
            ResolverPlaces,
            limits_from_settings,
            open_retrieval_over,
        )
        from tta_backend.utils.plotting import RegionResolver

        registry, catalogue = self._load()
        boundary_calls: list[BoundaryCall] = []
        aqs_calls: list[tuple[str, dict]] = []

        def capturing_fetcher(credential: Any, *, transport: Any = None) -> Any:
            async def fetch(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
                aqs_calls.append((endpoint, dict(params)))
                if case.aqs_get is None:
                    return dict(_EMPTY_AQS_RESPONSE)
                return await case.aqs_get(endpoint, params)

            return fetch

        clock = _SystemClock()
        events = RecordingEvents()
        artifacts = MemoryArtifacts()
        stores = memory_stores()
        credentials = StaticCredentials({
            "earthdata": EARTHDATA_SECRET,
            "epa_aqs": json.dumps({"email": AQS_EMAIL, "key": AQS_KEY}),
        })
        context = RunContext(
            identity=Identity(user_id=_USER_ID, conversation_id=_CONVERSATION_ID),
            events=events, artifacts=artifacts, credentials=credentials, storage=stores, clock=clock,
        )

        run_day = clock.today()
        started = time.monotonic()
        result = None
        error: str | None = None
        server = FakeEarthdataMCPServer(build_fake_mcp(case.handlers, recorder=boundary_calls))
        server.start()
        try:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, {"CUBE_STORE_DIR": os.path.join(tmp, "cubes")}))
                get_settings.cache_clear()
                stack.callback(get_settings.cache_clear)
                stack.enter_context(patch(f"{_GEOCODER_TARGET}.ageocode", _ageocode))
                stack.enter_context(patch(f"{_GEOCODER_TARGET}.geocode", _geocode_sync))
                stack.enter_context(patch(_AQS_FETCHER_TARGET, capturing_fetcher))
                for spec in case.patches:
                    stack.enter_context(patch(spec.target, _mock_for(spec)))

                settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
                tools = await load_raw_mcp_tools(settings)
                build = env_factory(
                    os.path.join(tmp, "payloads"),
                    earthdata=McpEarthdataSession(tools),
                    compute=AdmissionCompute(),
                    limits=limits_from_settings(settings),
                    open_retrieval=open_retrieval_over(tools, _USER_ID),
                    places=ResolverPlaces(RegionResolver()),
                )
                runner = WorkflowRunner(registry, catalogue, InMemorySaver())
                try:
                    version = catalogue.latest(case.workflow).version
                    started = time.monotonic()
                    result = await runner.start(context, build, case.workflow, version, case.inputs_for(run_day))
                except (UnknownWorkflow, NotRunnable) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                except Exception as exc:
                    # The type name only: an exception's text can carry values or secrets.
                    error = f"the workflow run raised {type(exc).__name__}"
                elapsed_seconds = time.monotonic() - started
                trace = (
                    await stores.traces.entries(_USER_ID, result.trace_id) if result is not None else []
                )
        finally:
            server.stop()

        outcome = Outcome(
            boundary_calls=list(boundary_calls),
            aqs_calls=list(aqs_calls),
            handles=handles_from_calls(boundary_calls),
            charts=[dict(chart) for chart in events.charts],
            artifact_ids=list(artifacts.stored),
            stages=[event["stage"] for event in events.progress_events if event["stage"]],
            clock=run_day,
            clock_end=clock.today(),
            elapsed_seconds=elapsed_seconds,
            tool_calls=[entry.capability for entry in trace if entry.capability and entry.status != "skipped"],
        )
        if result is None:
            outcome.error = error
            return outcome
        outcome.mode = result.mode.label()
        outcome.findings = [finding.model_dump(mode="json") for finding in result.findings]
        outcome.answer_text = compose_answer(result)
        if result.status == "failed":
            outcome.error = result.reason or "the workflow failed"
        elif result.status == "needs_input" and _misconfigured(result):
            outcome.error = f"the adapter did not supply what the workflow needs: {result.reason}"
        outcome.done = result.status != "waiting" and outcome.error is None
        return outcome
