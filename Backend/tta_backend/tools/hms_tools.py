"""HMS tool returning the existing AgentResult/chart envelope."""
import asyncio
from datetime import datetime, timezone

from langchain.tools import tool
from shapely.geometry import box

from tta_backend.models import AgentResult, ChartPayload, agent_result_to_json
from tta_backend.services.hms_service import CAVEAT, HMSError, download_archive, parse_archive, product_date
from tta_backend.utils.streaming import emit_status


@tool
async def query_hms_smoke(date: str, region: str = "", bounds: list[float] | None = None) -> str:
    """Map NOAA HMS smoke for ONE UTC date (YYYY-MM-DD).

    Supply a named region OR bounds [west,south,east,north]. With neither,
    show the full daily product. Density is categorical, not surface PM2.5.
    NOAA is accessed directly; NASA Earthdata credentials are not required.
    """
    try:
        product_date(date)
        geometry = None
        label = region or "full product"
        if region and bounds is not None:
            raise HMSError("invalid_region", "Supply a named region or bounds, not both.")
        if bounds is not None:
            if len(bounds) != 4 or not (-180 <= bounds[0] < bounds[2] <= 180 and -90 <= bounds[1] < bounds[3] <= 90):
                raise HMSError("invalid_region", "Bounds must be [west,south,east,north] without crossing the dateline.")
            geometry = box(*bounds)
            label = "selected bounds"
        elif region:
            from tta_backend.utils.plotting import RegionResolver
            resolved = await RegionResolver().aresolve_location(region)
            geometry = resolved["geometry"]
            bounds = list(geometry.bounds)
        emit_status(f"Downloading NOAA HMS smoke for {date}")
        content = await download_archive(date)
        retrieved_at = datetime.now(timezone.utc).isoformat()
        result = await asyncio.to_thread(parse_archive, content, date, geometry)
        result["provenance"]["retrieved_at"] = retrieved_at
        result["provenance"]["region"] = label
        text = (f"NOAA HMS {date}: {result['feature_count']} smoke polygon observations over {label}. "
                "Polygon counts are not counts of fires. " + CAVEAT)
        if not result["feature_count"]:
            text = f"NOAA HMS {date}: no usable smoke polygons intersect this region; this does not establish smoke-free conditions. " + CAVEAT
        chart = ChartPayload(type="smoke_map", title=f"NOAA HMS smoke — {date} — {label}",
                             bounds=bounds, date=date, **result)
        return agent_result_to_json(AgentResult(text=text, charts=[chart]))
    except (HMSError, ValueError) as exc:
        return agent_result_to_json(AgentResult(text=str(exc), metadata={"source": "NOAA HMS", "status": getattr(exc, "status", "invalid_region")}))
