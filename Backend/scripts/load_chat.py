"""
Repeatable async load test for the /chat SSE endpoint.

Example:
    python Backend/scripts/load_chat.py --url http://localhost:8000 --concurrency 20
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx


async def run_one(client: httpx.AsyncClient, index: int) -> float:
    started = time.perf_counter()
    response = await client.post(
        "/chat",
        json={"message": f"Health-check style load test request {index}"},
    )
    response.raise_for_status()
    async for _ in response.aiter_text():
        pass
    return time.perf_counter() - started


async def run_load(url: str, concurrency: int) -> None:
    async with httpx.AsyncClient(base_url=url, timeout=None) as client:
        started = time.perf_counter()
        latencies = await asyncio.gather(
            *(run_one(client, index) for index in range(concurrency))
        )
        elapsed = time.perf_counter() - started

    print(f"requests={concurrency}")
    print(f"elapsed={elapsed:.2f}s")
    print(f"throughput={concurrency / elapsed:.2f} req/s")
    print(f"latency_min={min(latencies):.2f}s")
    print(f"latency_p50={statistics.median(latencies):.2f}s")
    print(f"latency_max={max(latencies):.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(run_load(args.url.rstrip("/"), args.concurrency))


if __name__ == "__main__":
    main()
