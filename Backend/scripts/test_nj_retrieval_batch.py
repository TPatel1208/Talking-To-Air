"""
test_nj_retrieval_batch.py
==========================
Batch validation focused strictly on New Jersey (NJ) to test map and graph/chart
generation across ground sensor and satellite retrievals.

Evaluates:
- Ground sensor queries in NJ (Newark, New Brunswick, Jersey City, Camden, Statewide)
- Generation of Map artifacts vs Table artifacts
- Satellite spatial heatmaps, time-series graphs, and cross-source comparisons
- Failure modes, missing data, and recursion limits
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from typing import Any

os.environ["LANGCHAIN_TRACING_V2"] = "false"

from tta_backend.agents.earthdata_agent import build_earthdata_agent
from tta_backend.agents.ground_sensor_agent import build_ground_agent
from tta_backend.agents.supervisor_agent import build_agent
from tta_backend.config.settings import get_settings
from tta_backend.earthdata_mcp.client import open_earthdata_session
from tta_backend.repositories.session_metadata_repository import (
    ensure_session_metadata_table,
    save_session_metadata_once,
)
from tta_backend.services.chart_service import ChartService
from tta_backend.services.chat_stream_service import ChatStreamService
from tta_backend.services.intent_router import route_intent
from tta_backend.utils.db import close_db_pool, init_db_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nj_retrieval_test")

USER_ID = "cc0d9466-cf96-45fd-8644-2534daf7fc87"

NJ_GROUND_QUERIES = [
    {
        "id": "G_NJ_1",
        "name": "PM2.5 Daily Map in Newark, NJ",
        "query": "Find the nearest EPA PM2.5 monitor in Newark, NJ and retrieve the daily reading for July 15, 2024 so I can view it on the map.",
    },
    {
        "id": "G_NJ_2",
        "name": "Ozone Daily Map in New Brunswick, NJ (Rutgers)",
        "query": "Get ground ozone daily readings from the Rutgers University monitor in New Brunswick, NJ for June 15, 2024 and plot them on a map.",
    },
    {
        "id": "G_NJ_3",
        "name": "NO2 Quarterly Summary Map in Jersey City, NJ",
        "query": "Retrieve the quarterly ground NO2 summary for Jersey City, NJ for Q2 2024 and show the map.",
    },
    {
        "id": "G_NJ_4",
        "name": "SO2 Annual Summary Map in Camden, NJ",
        "query": "Retrieve the annual ground SO2 summary for Camden, NJ for 2023 and display the map.",
    },
    {
        "id": "G_NJ_5",
        "name": "Statewide NJ Ozone Exceedances Map/Graph",
        "query": "Show EPA ground ozone exceedance days across New Jersey for summer 2023 and show the map.",
    },
]

NJ_SATELLITE_QUERIES = [
    {
        "id": "S_NJ_1",
        "name": "TEMPO NO2 Spatial Map over NJ",
        "query": "Plot a spatial heatmap of TEMPO tropospheric NO2 over New Jersey for June 15, 2024.",
    },
    {
        "id": "S_NJ_2",
        "name": "TROPOMI NO2 Map over New Jersey",
        "query": "Map TROPOMI NO2 column density over New Jersey for June 15, 2024.",
    },
    {
        "id": "S_NJ_3",
        "name": "TEMPO Formaldehyde (HCHO) Map over NJ",
        "query": "Visualize TEMPO formaldehyde (HCHO) spatial distribution over New Jersey for July 20, 2024.",
    },
    {
        "id": "S_NJ_4",
        "name": "TEMPO NO2 Time Series Line Graph over NJ",
        "query": "Produce a time series line graph of TEMPO NO2 over New Jersey from June 10, 2024 to June 16, 2024.",
    },
    {
        "id": "S_NJ_5",
        "name": "TEMPO vs Ground EPA Validation Graph in NJ",
        "query": "Compare TEMPO NO2 against EPA ground monitors in New Jersey for June 2024 and plot the validation graph.",
    },
]


def parse_sse_events(raw_chunks: list[str]) -> list[dict[str, Any]]:
    events = []
    buffer = "".join(raw_chunks)
    blocks = buffer.split("\n\n")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        event_name = "message"
        data_lines = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        raw_data = "\n".join(data_lines)
        parsed_data: Any = raw_data
        try:
            parsed_data = json.loads(raw_data)
        except Exception:
            pass
        events.append({"event": event_name, "data": parsed_data})
    return events


async def execute_query(
    item: dict[str, Any],
    supervisor: Any,
    ground_agent: Any,
    satellite_agent: Any,
    chat_service: ChatStreamService,
) -> dict[str, Any]:
    query_id = item["id"]
    name = item["name"]
    query = item["query"]
    thread_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())

    classified_intent = route_intent(query)
    logger.info("=" * 60)
    logger.info(f"RUNNING [{query_id}] {name}")
    logger.info(f"Query: {query}")
    logger.info(f"Thread: {thread_id} | Intent: {classified_intent}")

    await save_session_metadata_once(thread_id, query, USER_ID)

    start_time = time.perf_counter()
    raw_chunks: list[str] = []
    error_msg: str | None = None

    try:
        async for chunk in chat_service.stream_chat_events(
            supervisor,
            ground_agent,
            satellite_agent,
            query,
            thread_id,
            USER_ID,
            request_id,
        ):
            raw_chunks.append(chunk)
    except Exception as exc:
        logger.exception(f"Error executing [{query_id}]: {exc}")
        error_msg = str(exc)

    elapsed = round(time.perf_counter() - start_time, 2)
    parsed_events = parse_sse_events(raw_chunks)

    stages = []
    tool_calls = []
    artifacts = []
    charts = []
    terminal_response = ""
    saw_done = False
    saw_error = False

    for ev in parsed_events:
        ename = ev["event"]
        edata = ev["data"]
        if ename == "stage":
            stages.append(edata)
        elif ename == "status":
            if isinstance(edata, dict) and "message" in edata:
                msg = edata["message"]
                if "Calling" in msg or "tool" in msg.lower():
                    tool_calls.append(msg)
        elif ename == "done":
            saw_done = True
            if isinstance(edata, dict):
                terminal_response = edata.get("response", "")
                artifacts = edata.get("artifacts", [])
                charts = edata.get("charts", edata.get("image_urls", []))
        elif ename == "error":
            saw_error = True
            if not error_msg:
                error_msg = str(edata)

    # Check specifically if map or graph/chart was generated
    has_map = any(a.get("type") == "map" for a in artifacts) or any("map" in str(c).lower() for c in charts)
    has_graph = any("chart" in str(c).lower() or "line" in str(c).lower() for c in charts) or any(a.get("type") == "chart" for a in artifacts)
    success = saw_done and not saw_error and error_msg is None and bool(terminal_response)

    result = {
        "id": query_id,
        "name": name,
        "query": query,
        "thread_id": thread_id,
        "intent": classified_intent,
        "elapsed_sec": elapsed,
        "success": success,
        "has_map": has_map,
        "has_graph": has_graph,
        "error": error_msg,
        "artifacts": artifacts,
        "charts": charts,
        "response_preview": terminal_response[:300] if terminal_response else "",
        "full_response": terminal_response,
    }

    logger.info(
        f"RESULT [{query_id}]: Success={success} | Elapsed={elapsed}s | "
        f"Map={has_map} | Graph={has_graph} | Artifacts={len(artifacts)}"
    )
    if error_msg:
        logger.error(f"Error [{query_id}]: {error_msg}")
    return result


async def main():
    settings = get_settings()
    await init_db_pool()
    await ensure_session_metadata_table()

    logger.info("Building ground agent...")
    ground_agent = build_ground_agent()

    logger.info("Opening Earthdata MCP session...")
    async with open_earthdata_session(settings) as held:
        logger.info(f"Earthdata MCP ready with {len(held.tools)} tools.")
        satellite_agent = build_earthdata_agent(mcp_tools=held.tools)
        logger.info("Building supervisor agent...")
        supervisor = await build_agent(
            settings.llm_model,
            ground_agent=ground_agent,
            satellite_agent=satellite_agent,
        )

        chat_service = ChatStreamService(
            ChartService(),
            settings.long_request_seconds,
            None,
            settings.chat_turn_timeout_seconds,
        )

        all_results = []

        # 1. 5 NJ Ground Queries
        logger.info("\n" + "=" * 60)
        logger.info("STARTING 5 NJ GROUND RETRIEVAL QUERIES")
        logger.info("=" * 60)
        for item in NJ_GROUND_QUERIES:
            res = await execute_query(item, supervisor, ground_agent, satellite_agent, chat_service)
            all_results.append(res)
            await asyncio.sleep(2)

        # 2. 5 NJ Satellite & Graph Queries
        logger.info("\n" + "=" * 60)
        logger.info("STARTING 5 NJ SATELLITE & GRAPH QUERIES")
        logger.info("=" * 60)
        for item in NJ_SATELLITE_QUERIES:
            res = await execute_query(item, supervisor, ground_agent, satellite_agent, chat_service)
            all_results.append(res)
            await asyncio.sleep(2)

        print("\n" + "=" * 80)
        print("NEW JERSEY RETRIEVAL, MAP & GRAPH REPORT")
        print("=" * 80)
        for r in all_results:
            status_str = "PASS" if r["success"] else "FAIL"
            map_str = "MAP:YES" if r["has_map"] else "MAP:NO"
            graph_str = "GRAPH:YES" if r["has_graph"] else "GRAPH:NO"
            print(f"[{status_str}] {r['id']}: {r['name']} ({r['elapsed_sec']}s | {map_str} | {graph_str})")
            if r["error"]:
                print(f"       ERROR: {r['error']}")
            elif r["response_preview"]:
                print(f"       Summary: {r['response_preview'][:130]}...")

        print("=" * 80)

        output_path = "/app/outputs/nj_retrieval_results.json"
        try:
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"Saved NJ results to {output_path}")
        except Exception as exc:
            logger.warning(f"Could not save results: {exc}")

    await close_db_pool()


if __name__ == "__main__":
    asyncio.run(main())
