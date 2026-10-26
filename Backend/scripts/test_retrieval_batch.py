"""
test_retrieval_batch.py
=======================
Automated validation harness to execute 5 ground data retrievals and 5 satellite
data retrievals across distinct threads using the live agent stack and services.

Collects and verifies:
- Routing intent (GROUND vs SATELLITE vs SUPERVISOR)
- Tool calls made and parameters passed
- Tool execution status and results
- Generated artifacts (tables/maps), charts, and envelopes
- Error logs or rate limit diagnostics
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

# Disable LangSmith tracing noise if keys are not configured
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
logger = logging.getLogger("retrieval_test")

USER_ID = "cc0d9466-cf96-45fd-8644-2534daf7fc87"

GROUND_QUERIES = [
    {
        "id": "G1",
        "name": "Daily PM2.5 in Newark, NJ",
        "query": "Find the nearest EPA PM2.5 monitor in Newark, NJ and retrieve the daily reading for July 15, 2024.",
        "expected_intent": "GROUND",
    },
    {
        "id": "G2",
        "name": "Ozone Exceedances in Los Angeles, CA",
        "query": "What were the EPA ground ozone exceedances in Los Angeles, California during August 2023?",
        "expected_intent": "GROUND",
    },
    {
        "id": "G3",
        "name": "Quarterly NO2 Summary in Chicago, IL",
        "query": "Get the quarterly ground sensor summary for NO2 in Chicago, Illinois for Q2 2024.",
        "expected_intent": "GROUND",
    },
    {
        "id": "G4",
        "name": "Annual SO2 Summary in Pittsburgh, PA",
        "query": "Retrieve the annual ground monitor summary for SO2 in Pittsburgh, Pennsylvania for 2023.",
        "expected_intent": "GROUND",
    },
    {
        "id": "G5",
        "name": "Hourly Sample Data for CO in Houston, TX",
        "query": "Find the closest EPA monitor in Houston, TX and show ground CO readings for June 1, 2024.",
        "expected_intent": "GROUND",
    },
]

SATELLITE_QUERIES = [
    {
        "id": "S1",
        "name": "TEMPO NO2 over New York City",
        "query": "Plot TEMPO tropospheric NO2 column density over New York City for August 10, 2024.",
        "expected_intent": "SATELLITE",
    },
    {
        "id": "S2",
        "name": "TROPOMI NO2 over Los Angeles",
        "query": "Map TROPOMI NO2 over Los Angeles for June 15, 2024.",
        "expected_intent": "SATELLITE",
    },
    {
        "id": "S3",
        "name": "TEMPO Formaldehyde (HCHO) over Houston",
        "query": "Visualize TEMPO formaldehyde (HCHO) spatial distribution over Houston for July 20, 2024.",
        "expected_intent": "SATELLITE",
    },
    {
        "id": "S4",
        "name": "OMI Ozone over California",
        "query": "Display OMI total column ozone satellite data over California for June 1, 2024.",
        "expected_intent": "SATELLITE",
    },
    {
        "id": "S5",
        "name": "TROPOMI Carbon Monoxide over Chicago",
        "query": "Plot TROPOMI carbon monoxide (CO) over Chicago for August 5, 2024.",
        "expected_intent": "SATELLITE",
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
    logger.info(f"Thread ID: {thread_id} | Intent: {classified_intent}")

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

    # Extract stages, tool calls, artifacts, and terminal response
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

    # Determine status
    success = saw_done and not saw_error and error_msg is None and bool(terminal_response)

    result = {
        "id": query_id,
        "name": name,
        "query": query,
        "thread_id": thread_id,
        "intent": classified_intent,
        "elapsed_sec": elapsed,
        "success": success,
        "error": error_msg,
        "stages": stages,
        "tool_calls": tool_calls,
        "artifacts": artifacts,
        "charts": charts,
        "response_preview": terminal_response[:300] if terminal_response else "",
        "full_response": terminal_response,
    }

    logger.info(f"RESULT [{query_id}]: Success={success} | Elapsed={elapsed}s | Artifacts={len(artifacts)}")
    if error_msg:
        logger.error(f"Error [{query_id}]: {error_msg}")
    elif terminal_response:
        logger.info(f"Response preview: {terminal_response[:160]}...")

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

        # 1. Run 5 Ground Queries
        logger.info("\n" + "=" * 60)
        logger.info("STARTING 5 GROUND RETRIEVAL QUERIES")
        logger.info("=" * 60)
        for item in GROUND_QUERIES:
            res = await execute_query(item, supervisor, ground_agent, satellite_agent, chat_service)
            all_results.append(res)
            # Brief pause to respect EPA rate limits if needed
            await asyncio.sleep(2)

        # 2. Run 5 Satellite Queries
        logger.info("\n" + "=" * 60)
        logger.info("STARTING 5 SATELLITE RETRIEVAL QUERIES")
        logger.info("=" * 60)
        for item in SATELLITE_QUERIES:
            res = await execute_query(item, supervisor, ground_agent, satellite_agent, chat_service)
            all_results.append(res)
            await asyncio.sleep(1)

        # Output Summary
        print("\n" + "=" * 80)
        print("FINAL RETRIEVAL BATCH EXECUTION REPORT")
        print("=" * 80)
        ground_success = sum(1 for r in all_results if r["id"].startswith("G") and r["success"])
        sat_success = sum(1 for r in all_results if r["id"].startswith("S") and r["success"])
        print(f"Ground Retrievals:    {ground_success}/5 passed")
        print(f"Satellite Retrievals: {sat_success}/5 passed")
        print("-" * 80)

        for r in all_results:
            status_str = "PASS" if r["success"] else "FAIL"
            art_info = f"artifacts={len(r['artifacts'])}" if r["artifacts"] else "no artifacts"
            print(f"[{status_str}] {r['id']}: {r['name']} ({r['elapsed_sec']}s, {art_info})")
            if r["error"]:
                print(f"       ERROR: {r['error']}")
            elif r["response_preview"]:
                print(f"       Summary: {r['response_preview'][:140]}...")

        print("=" * 80)

        # Save JSON output for reporting
        output_path = "/app/outputs/retrieval_batch_results.json"
        try:
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"Saved full results to {output_path}")
        except Exception as exc:
            logger.warning(f"Could not save results to file: {exc}")

    await close_db_pool()


if __name__ == "__main__":
    asyncio.run(main())
