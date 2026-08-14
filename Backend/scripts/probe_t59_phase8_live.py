"""T59 Phase 8 -- drive the REAL ``plot_singular`` against a materialized bundle
and persist the chart, so a live frames payload reaches the browser.

Phases 1b-5 were measured on real bundles through the production open + mask
path; Phases 6 and 7 were built entirely against fixtures hand-shaped from
``frame_store._axis_block``. Nothing has ever been rendered against a live
frames payload. This closes that: it calls the actual tool -- the same
``open_handle`` -> mask -> ``aggregate`` -> ``_attach_frames`` ->
``store_frame_stack`` -> ``_save_chart`` chain the agent calls -- and writes the
result through ``chart_repository.save_chart``, the one persistence boundary.

**Exactly one thing is stubbed: the MCP's ``export_result``.** It is replaced by
a function returning the file URI of a bundle already on disk, which is what the
MCP would have returned for a job in state ``ready``. Every other byte of the
path is production code. The MCP is stubbed rather than called because a
crash-restarting MCP is a known recurring failure here and it is not what this
verification is about.

Reproduce with::

    docker exec tta-backend sh -c 'cd /app && python scripts/probe_t59_phase8_live.py \
        --bundle /data/harmony/job_52a95bb4cb79e2ee/result.nc.zip \
        --location Texas --user-id <uid> --thread t59-phase8'
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time


class _FakeExportResult:
    """The MCP's ``export_result`` for a job whose bundle is already on disk.

    Shaped as ``open_handle._export`` consumes it: ``ainvoke`` returning the
    JSON string ``parse_tool_result`` parses, with the ``status``/
    ``storage_uri``/``media_type`` triple ``_open_export`` reads and the
    ``content_digest``/``partial`` pair T54's cube identity is derived from.
    """

    name = "export_result"

    def __init__(self, path: str) -> None:
        self._path = path

    async def ainvoke(self, args: dict) -> str:
        return json.dumps({
            "status": "ready",
            "handle": args.get("handle"),
            "storage_uri": f"file://{self._path}",
            "media_type": "application/x-netcdf-bundle",
            "size_bytes": None,
            # No digest: the cube cache then falls back to mtime+size, which is
            # what it did before T54 and is correct for a file that never moves.
            "content_digest": None,
            "partial": False,
        })


async def run(args) -> int:
    from tta_backend.repositories.chart_repository import save_chart
    from tta_backend.tools.satellite_tools.plot_tools import make_plot_singular
    from tta_backend.utils import streaming

    plot_singular = make_plot_singular({"export_result": _FakeExportResult(args.bundle)})

    captured: list[dict] = []
    token = streaming._chart_emitter.set(captured.append)
    t0 = time.perf_counter()
    try:
        summary = await plot_singular.ainvoke({
            "handle": args.handle,
            "location": args.location,
            "title": args.title or "",
            **({"variable": args.variable} if args.variable else {}),
        })
    finally:
        streaming._chart_emitter.reset(token)
    seconds = time.perf_counter() - t0

    print(f"\nplot_singular returned in {seconds:.1f}s")
    print(f"model-facing summary: {summary[:1200]}")

    if not captured:
        print("\n!! no chart payload was emitted -- the tool refused or errored")
        return 1

    payload = captured[-1]
    stored = await save_chart(args.thread, payload, args.user_id)
    chart_id = stored.get("chart_id")
    print(f"\nsaved chart {chart_id} on thread {args.thread!r} for user {args.user_id!r}")

    # Written before the frames report, not after it: a refusal is exactly the
    # payload worth keeping, and an early return that skips the dump means the
    # one run you wanted the file for is the one that does not produce it.
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, default=str)
        print(f"        payload written to {args.json_out}")

    frames = payload.get("frames")
    if not isinstance(frames, dict):
        print(f"frames: ABSENT.  frames_unavailable={payload.get('frames_unavailable')}")
        return 0

    entries = frames.get("frames") or []
    print(
        f"frames: tier={frames.get('tier')} cadence={frames.get('cadence')} "
        f"n={len(entries)} buckets/frame={frames.get('buckets_per_frame')} "
        f"shape={frames.get('shape')} k={frames.get('coarsen_k')} "
        f"cells/frame={frames.get('cells_per_frame')} period_index={frames.get('period_index')}"
    )
    print(f"        url={frames.get('url')}  value_range={frames.get('value_range')}")
    print(f"        delta={frames.get('delta')}")
    lats, lons = frames.get("lats") or [], frames.get("lons") or []
    if lats and lons:
        print(
            f"        lats[{len(lats)}] {lats[0]} .. {lats[-1]} "
            f"({'ascending' if lats[0] < lats[-1] else 'descending'})  "
            f"lons[{len(lons)}] {lons[0]} .. {lons[-1]} "
            f"({'ascending' if lons[0] < lons[-1] else 'descending'})"
        )
    empty_qa = sum(1 for f in entries if f.get("n_granules") == 0 and f.get("qa_pass_rate") is not None)
    empty_none = sum(1 for f in entries if f.get("n_granules") == 0 and f.get("qa_pass_rate") is None)
    print(f"        empty: {empty_qa} qa-rejected + {empty_none} nothing-retrieved of {len(entries)}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, default=str)
        print(f"        payload written to {args.json_out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--handle", default=None,
                    help="handle string to plot under; defaults to the bundle's job dir")
    ap.add_argument("--location", required=True)
    ap.add_argument("--variable", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--user-id", required=True)
    ap.add_argument("--thread", default="t59-phase8")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    if not args.handle:
        args.handle = args.bundle.rstrip("/").split("/")[-2]
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
