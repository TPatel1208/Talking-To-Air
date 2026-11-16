"""The fake data boundary the harness adapters run on.

The earthdata MCP fake behind the app's own session, EPA AQS behind a
capturing fetch, an offline geocoder, memory stores and an in-memory
checkpointer. ``harness_boundary`` sets it up for one case and yields what the
adapters read afterwards.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, Mock, patch

from benchmark.cases import BenchmarkCase, PatchSpec
from benchmark.outcome import BoundaryCall

USER_ID = "benchmark-user"
_CONVERSATION_ID = "benchmark-conversation"
PACKS_ROOT = Path(__file__).resolve().parents[2] / "packs"
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
_LOS_ANGELES = {
    "latitude": 34.0537, "longitude": -118.2428, "display_name": "Los Angeles, California, United States",
    "polygon": None, "bbox": [33.7037, 34.3373, -118.6682, -118.1553],
}
_PLACES: dict[str, dict[str, Any]] = {
    "houston": {
        "latitude": 29.7604, "longitude": -95.3698, "display_name": "Houston, Texas, United States",
        "polygon": None, "bbox": [29.5370, 30.1107, -95.9097, -94.9446],
    },
    "newark nj": _NEWARK,
    "newark new jersey": _NEWARK,
    "los angeles ca": _LOS_ANGELES,
    "los angeles california": _LOS_ANGELES,
    "death valley": {
        "latitude": 36.5054, "longitude": -117.0794, "display_name": "Death Valley, California, United States",
        "polygon": None, "bbox": [35.6, 37.4, -117.9, -116.3],
    },
}


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


class SystemClock:
    def today(self) -> date:
        return datetime.now(timezone.utc).date()

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


@dataclass
class Boundary:
    """One case's run context and environment factory, and what crossed the boundary."""

    context: Any
    build_env: Any
    events: Any
    artifacts: Any
    stores: Any
    boundary_calls: list[BoundaryCall]
    aqs_calls: list[tuple[str, dict]]
    clock: SystemClock
    run_day: date


@asynccontextmanager
async def harness_boundary(case: BenchmarkCase, *, models: Any = None) -> AsyncIterator[Boundary]:
    """The fake boundary for ``case``, torn down on exit.

    ``models`` is what the run environment offers a specialist; None offers none.
    """
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
    from tta_backend.infrastructure.settings import Settings, get_settings
    from tta_backend.services.harness_host import (
        AdmissionCompute,
        McpEarthdataSession,
        ResolverPlaces,
        limits_from_settings,
        open_retrieval_over,
    )
    from tta_backend.utils.plotting import RegionResolver

    boundary_calls: list[BoundaryCall] = []
    aqs_calls: list[tuple[str, dict]] = []

    def capturing_fetcher(credential: Any, *, transport: Any = None) -> Any:
        async def fetch(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
            aqs_calls.append((endpoint, dict(params)))
            if case.aqs_get is None:
                return dict(_EMPTY_AQS_RESPONSE)
            return await case.aqs_get(endpoint, params)

        return fetch

    clock = SystemClock()
    events = RecordingEvents()
    artifacts = MemoryArtifacts()
    stores = memory_stores()
    credentials = StaticCredentials({
        "earthdata": EARTHDATA_SECRET,
        "epa_aqs": json.dumps({"email": AQS_EMAIL, "key": AQS_KEY}),
    })
    context = RunContext(
        identity=Identity(user_id=USER_ID, conversation_id=_CONVERSATION_ID),
        events=events, artifacts=artifacts, credentials=credentials, storage=stores, clock=clock,
    )
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
                open_retrieval=open_retrieval_over(tools, USER_ID),
                places=ResolverPlaces(RegionResolver()),
                models=models,
            )
            yield Boundary(
                context=context, build_env=build, events=events, artifacts=artifacts, stores=stores,
                boundary_calls=boundary_calls, aqs_calls=aqs_calls, clock=clock, run_day=clock.today(),
            )
    finally:
        server.stop()
