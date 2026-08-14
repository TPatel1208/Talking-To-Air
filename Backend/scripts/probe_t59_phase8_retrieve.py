"""T59 Phase 8 -- retrieve a span long enough to reach the COARSENED tier.

Every TEMPO NO2 bundle already on disk spans two days, which is 48 hourly
buckets and stays in tier one. The 60-frame budget is reached at ~2.5 days
(Phase 3 §2), so seeing tier two on real data needs a longer retrieval than
anything materialized so far. This drives the production
``safe_retrieve`` -> ``await_retrieval`` composites against the live MCP, with
no stubs at all.

Reproduce with::

    docker exec tta-backend sh -c 'cd /app && python scripts/probe_t59_phase8_retrieve.py \
        --time-range 2025-06-14T00:00:00/2025-06-19T23:59:59'
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

TEXAS_BBOX = [-106.6458459, 25.83706, -93.5078217, 36.5004529]
VARIABLES = ["product/vertical_column_troposphere", "product/main_data_quality_flag"]


def _parse(raw):
    from tta_backend.earthdata_mcp.results import parse_tool_result
    return parse_tool_result(raw)


async def run(args) -> int:
    from tta_backend.config.settings import get_settings
    from tta_backend.earthdata_mcp.client import load_raw_mcp_tools
    from tta_backend.services.retrieval_composites import await_retrieval, safe_retrieve

    from tta_backend.earthdata_mcp.workspace import bind_workspace

    # Bound to a workspace, exactly as a request does it. Without this the
    # handles land in workspace "default" and every later provenance lookup
    # (``get_lineage``/``get_citations``, which methods.md needs) refuses them
    # as not owned by the caller.
    tools = bind_workspace(
        await load_raw_mcp_tools(get_settings()), lambda: args.user_id,
    )
    print(f"mcp tools: {len(tools)}  workspace=user-{args.user_id}")

    found = _parse(await tools["search_datasets"].ainvoke({"query": args.short_name}))
    rows = found.get("results") or found.get("datasets") or []
    dataset_handle = None
    for row in rows:
        summary = row.get("summary") or row
        if str(summary.get("short_name") or "") == args.short_name:
            dataset_handle = row.get("handle") or summary.get("handle")
            break
    if not dataset_handle:
        print(f"!! no dataset handle for {args.short_name}; got {json.dumps(found)[:800]}")
        return 1
    print(f"dataset_handle={dataset_handle}")

    aoi = _parse(await tools["define_area_of_interest"].ainvoke({"location": args.location}))
    aoi_handle = aoi.get("handle") or aoi.get("aoi_handle")
    print(f"aoi_handle={aoi_handle}  ({json.dumps(aoi)[:300]})")

    t0 = time.perf_counter()
    result = await safe_retrieve(
        dataset_handle, aoi_handle, args.time_range, VARIABLES, tools, confirmed=True,
    )
    print(f"safe_retrieve -> {json.dumps(result)[:900]}")
    job_handle = result.get("job_handle")
    if not job_handle:
        return 1

    terminal = await await_retrieval(job_handle, tools)
    print(f"\nawait_retrieval after {time.perf_counter() - t0:.0f}s -> "
          f"{json.dumps(terminal)[:900]}")

    export = _parse(await tools["export_result"].ainvoke({
        "handle": terminal.get("obs_handle") or job_handle,
    }))
    print(f"\nexport_result -> {json.dumps(export)[:600]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short-name", default="TEMPO_NO2_L3")
    ap.add_argument("--location", default="Texas")
    ap.add_argument("--user-id", required=True)
    ap.add_argument("--time-range", required=True)
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
