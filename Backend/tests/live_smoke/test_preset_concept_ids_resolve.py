"""
tests/live_smoke/test_preset_concept_ids_resolve.py
====================================================
Live-MCP regression guard for the AOD preset-misrouting bug (2026-07-11).

The bug: the preset table told the agent to search with synthetic labels
('MODIS_AOD_TERRA', 'OMI_NO2', ...) that `search_datasets` resolved to ZERO
results, so the agent free-ranged onto unsupported products (HDF4 MCD19A2CMG,
MERRA-2) for AOD. The fix grounds every preset on its CMR concept_id, which
resolves to exactly that one collection.

This test pins the behavioral half the offline test_preset_collections.py
can't: that each preset concept_id actually resolves — as the top hit — when
sent to the *live* search_datasets. Opt-in via the same env gate as the rest
of tests/live_smoke/.

    EARTHDATA_MCP_URL=http://localhost:8765/mcp pytest -m live_mcp \
        tests/live_smoke/test_preset_concept_ids_resolve.py
"""
from __future__ import annotations

import os
import sys
import uuid

import pytest
import pytest_asyncio

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


def _mcp_url() -> str | None:
    return os.environ.get("EARTHDATA_MCP_URL")


pytestmark = [
    pytest.mark.live_mcp,
    pytest.mark.skipif(not _mcp_url(), reason="EARTHDATA_MCP_URL is not set — skipping preset-resolution smoke"),
]


@pytest_asyncio.fixture
async def invoke():
    from config.settings import Settings
    from earthdata_mcp.client import load_raw_mcp_tools
    from earthdata_mcp.results import parse_tool_result

    workspace_id = f"preset_resolve_{uuid.uuid4().hex[:8]}"
    settings = Settings(earthdata_mcp_url=_mcp_url(), earthdata_mcp_token=os.environ.get("EARTHDATA_MCP_TOKEN"))
    tools = await load_raw_mcp_tools(settings)

    async def _invoke(tool_name: str, **kwargs) -> dict:
        raw = await tools[tool_name].ainvoke({**kwargs, "workspace_id": workspace_id})
        return parse_tool_result(raw)

    yield _invoke


def _rows(result: dict):
    rows = result.get("datasets") or result.get("results") or []
    return [rows] if isinstance(rows, dict) else list(rows)


def _preset_params():
    from datasets.preset_collections import PRESET_COLLECTIONS

    return [pytest.param(p, id=p["short_name"]) for p in PRESET_COLLECTIONS]


@pytest.mark.asyncio
@pytest.mark.parametrize("preset", _preset_params())
async def test_every_preset_concept_id_resolves_to_its_collection_at_rank_0(invoke, preset):
    concept_id = preset["concept_id"]
    result = await invoke("search_datasets", query=concept_id, filters=None)
    rows = _rows(result)
    assert rows, (
        f"{preset['short_name']}: search_datasets({concept_id!r}) returned ZERO results — "
        "this is the exact failure mode the fix removed; the agent would free-range from here"
    )
    top_concept_id = (rows[0].get("summary") or {}).get("concept_id")
    assert top_concept_id == concept_id, (
        f"{preset['short_name']}: concept_id {concept_id!r} did not resolve to itself as the top hit "
        f"(got {top_concept_id!r})"
    )
