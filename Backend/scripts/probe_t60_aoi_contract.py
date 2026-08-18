"""T60 Phase 1.5 gate probes V5 and V7, run by hand against the live MCP.

V5 -- does ``define_area_of_interest`` actually accept what the dispatcher
plans to send? V3 (Phase 0) read the contract out of the running container's
source; this measures it over the wire. Three calls: a "W,S,E,N" bbox string,
a small GeoJSON polygon string, and the real 15.8 KB OTR polygon serialized.
The third is the one that decides design tension 2 -- a 15.8 KB argument on an
MCP call may be accepted, truncated, or rejected, and those are three different
answers.

V7 -- what does the bbox envelope cost versus the polygon? Equal-area (EPSG:5070)
areas plus a live ``estimate_retrieval_size`` on both footprints for a fixed
dataset and time range, so tension 2 is decided on a ratio.

Network-touching and never a test. Results are transcribed into
docs/prds/prd-t60-phase1.5-gate-verdict.md.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from tta_backend.config.settings import get_settings  # noqa: E402
from tta_backend.earthdata_mcp.client import open_earthdata_session  # noqa: E402
from tta_backend.utils.plotting import load_preset_polygons  # noqa: E402

WORKSPACE = "user-t60-phase15-probe"
DATASET_QUERY = "TEMPO NO2"
TIME_RANGE = "2026-07-01/2026-07-03"


def _unwrap(raw):
    """The MCP returns a langchain content list for some tools and a bare JSON
    string for others; parse_tool_result is the repo's one place that knows the
    difference, so the probe measures exactly what production sees."""
    from tta_backend.earthdata_mcp.results import parse_tool_result

    return parse_tool_result(raw)


def _summarize(label: str, raw) -> dict:
    text = raw if isinstance(raw, str) else json.dumps(raw, default=str)
    print(f"\n--- {label} ---")
    print(f"  raw length: {len(text)} chars")
    try:
        parsed = _unwrap(raw)
    except Exception as exc:  # noqa: BLE001
        print(f"  UNPARSEABLE ({type(exc).__name__}): {text[:400]}")
        return {}
    if isinstance(parsed, dict):
        for key in ("handle", "aoi_handle", "bbox", "geometry_summary", "source", "error", "message", "category"):
            if key in parsed:
                value = parsed[key]
                shown = json.dumps(value, default=str)
                print(f"  {key}: {shown[:300]}")
        return parsed
    print(f"  non-dict result: {text[:300]}")
    return {}


async def main() -> None:
    settings = get_settings()
    polygons = load_preset_polygons()
    otr = polygons.get("otc")
    if otr is None:
        sys.exit("preset_regions.geojson has no 'otc' feature -- run build_preset_regions.py")

    from shapely.geometry import mapping

    otr_geojson = json.dumps(mapping(otr))
    minx, miny, maxx, maxy = otr.bounds
    otr_bbox_str = f"{minx},{miny},{maxx},{maxy}"

    print("=" * 70)
    print("V5 -- define_area_of_interest wire contract")
    print("=" * 70)
    print(f"OTR polygon serialized: {len(otr_geojson)} chars / "
          f"{len(otr_geojson.encode('utf-8'))} utf-8 bytes")
    print(f"OTR bbox string: {otr_bbox_str!r} ({len(otr_bbox_str)} chars)")

    small_poly = json.dumps({
        "type": "Polygon",
        "coordinates": [[[-75.0, 40.0], [-74.0, 40.0], [-74.0, 41.0], [-75.0, 41.0], [-75.0, 40.0]]],
    })

    cases = [
        ("V5a bbox string (small)", "-75,40,-74,41"),
        ("V5b GeoJSON polygon string (small)", small_poly),
        ("V5c OTR bbox string", otr_bbox_str),
        ("V5d OTR GeoJSON polygon string (15.8 KB)", otr_geojson),
    ]

    handles: dict[str, str] = {}
    async with open_earthdata_session(settings) as held:
        aoi_tool = held.tools["define_area_of_interest"]
        for label, location in cases:
            try:
                raw = await aoi_tool.ainvoke({"location": location, "workspace_id": WORKSPACE})
            except Exception as exc:  # noqa: BLE001 -- a probe reports, never raises
                print(f"\n--- {label} ---")
                print(f"  RAISED {type(exc).__name__}: {str(exc)[:400]}")
                continue
            parsed = _summarize(label, raw)
            handle = parsed.get("handle") or parsed.get("aoi_handle")
            if handle:
                handles[label] = handle

        print()
        print("=" * 70)
        print("V7 -- what the envelope costs")
        print("=" * 70)
        import pyproj
        from shapely.geometry import box as shapely_box
        from shapely.ops import transform

        to_albers = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True).transform
        envelope = shapely_box(minx, miny, maxx, maxy)
        poly_km2 = transform(to_albers, otr).area / 1e6
        env_km2 = transform(to_albers, envelope).area / 1e6
        print(f"  OTR polygon area:   {poly_km2:12,.0f} km2")
        print(f"  OTR envelope area:  {env_km2:12,.0f} km2")
        print(f"  envelope / polygon: {env_km2 / poly_km2:.3f}x "
              f"(+{env_km2 - poly_km2:,.0f} km2 of non-region)")

        search = held.tools["search_datasets"]
        raw = await search.ainvoke({"query": DATASET_QUERY, "workspace_id": WORKSPACE})
        results = _unwrap(raw)
        if isinstance(results, list):
            rows = results
        else:
            rows = results.get("datasets") or results.get("results") or []
        if not rows:
            print(f"  !! no datasets for {DATASET_QUERY!r}; skipping size estimate")
            return
        print(f"  first row keys: {sorted(rows[0])}")
        if isinstance(rows[0].get("summary"), dict):
            print(f"  summary keys:   {sorted(rows[0]['summary'])}")
        # Live MCP search puts identity under ``summary``, not at the top level.
        first = rows[0]
        summary = first.get("summary") if isinstance(first.get("summary"), dict) else {}
        dataset_handle = (
            first.get("handle") or first.get("dataset_handle")
            or summary.get("handle") or summary.get("dataset_handle")
        )
        print(f"  dataset: {dataset_handle}  ({DATASET_QUERY}), time {TIME_RANGE}")

        estimate = held.tools["estimate_retrieval_size"]
        for label, key in (("bbox envelope", "V5c OTR bbox string"),
                           ("polygon", "V5d OTR GeoJSON polygon string (15.8 KB)")):
            handle = handles.get(key)
            if handle is None:
                print(f"  {label}: no AOI handle from V5 -- cannot estimate")
                continue
            try:
                raw = await estimate.ainvoke({
                    "dataset_handle": dataset_handle,
                    "aoi_handle": handle,
                    "time_range": TIME_RANGE,
                    "workspace_id": WORKSPACE,
                })
            except Exception as exc:  # noqa: BLE001
                print(f"  {label}: RAISED {type(exc).__name__}: {str(exc)[:200]}")
                continue
            parsed = _summarize(f"V7 estimate_retrieval_size -- {label}", raw)
            print(f"  FULL: {json.dumps(parsed, default=str)}")

        # The envelope-vs-polygon comparison above is partly tautological: the
        # MCP derives the same bbox from both, so CMR runs the same query. This
        # asks the non-tautological question -- does granule count respond to
        # extent at all for this dataset? If a much smaller AOI returns the same
        # count, then "granules" is not the metric the envelope is paid in.
        print()
        print("  V7b -- is granule count even extent-sensitive?")
        for label, bbox_str in (
            ("OTR envelope  ", otr_bbox_str),
            ("New England   ", None),
            ("one 1-deg cell", "-75,40,-74,41"),
        ):
            if bbox_str is None:
                ne = polygons.get("new england")
                if ne is None:
                    continue
                a, b, c, d = ne.bounds
                bbox_str = f"{a},{b},{c},{d}"
            aoi_raw = await aoi_tool.ainvoke({"location": bbox_str, "workspace_id": WORKSPACE})
            handle = _unwrap(aoi_raw).get("handle")
            est = _unwrap(await estimate.ainvoke({
                "dataset_handle": dataset_handle,
                "aoi_handle": handle,
                "time_range": TIME_RANGE,
                "workspace_id": WORKSPACE,
            }))
            print(f"    {label}  bbox={bbox_str[:44]:<44} granules={est.get('total_granules')}")


if __name__ == "__main__":
    asyncio.run(main())
