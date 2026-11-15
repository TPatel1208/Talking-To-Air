"""NOAA HMS smoke extent for one date, as polygons."""
from __future__ import annotations

from functools import partial
from typing import Annotated, Any

import httpx
from pydantic import Field
from pyproj import Transformer
from shapely import make_valid
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform, unary_union

from tta_backend.domain.smoke.hms import CAVEAT, HMSError, download_archive, parse_archive, product_date
from tta_backend.harness.capabilities import failures
from tta_backend.harness.capabilities.call import Call
from tta_backend.harness.capabilities.definition import Capability, Params, handle_param
from tta_backend.harness.capabilities.registry import Group
from tta_backend.harness.capabilities.result import CapabilityResult
from tta_backend.harness.capabilities.shapes import polygons_collection
from tta_backend.harness.capabilities.sources.areas import area_facts, area_geometry
from tta_backend.harness.handles.model import Descriptor, Period
from tta_backend.harness.trace import FailureCategory

NAME = "sources.smoke.get_extent"
PRODUCT = "NOAA HMS smoke"
FULL_PRODUCT = "full product"

BAD_DATE = (FailureCategory.BAD_INPUT, "{name} needs a date that is not in the future, as YYYY-MM-DD.")
BAD_AREA = (FailureCategory.BAD_INPUT, "{name} could not read the footprint of the area handle.")
MISSING_PRODUCT = "NOAA has not published the HMS smoke product for this date, so smoke presence is unknown."
NO_SMOKE = "No smoke was analysed in the NOAA HMS product for this date and area; this does not establish smoke-free conditions."
POLYGON_NOTE = "Polygon counts are not counts of fires."

GROUP = Group(
    name="sources.smoke",
    description="Satellite-observed smoke plumes from NOAA HMS.",
    notes=(
        "get_extent returns the smoke plumes for one UTC date as a polygons handle; each feature carries "
        "its density (light, medium, heavy) and its observation window (start, end). "
        "Pass an area handle to keep only the plumes over that area, clipped to its boundary. "
        "Density is a category, not a PM2.5 concentration, and clouds can hide smoke. "
        "An empty result means no smoke was analysed there, not that the air was clean. "
        "A date NOAA has not published is reported as insufficient evidence, never as no smoke."
    ),
)


class GetExtentParams(Params):
    date: str = Field(description="UTC date as YYYY-MM-DD; not in the future.")
    area: Annotated[str | None, handle_param("polygons")] = Field(
        default=None, description="Optional area handle; keeps only the smoke over this area."
    )


async def _read_area(call: Call[GetExtentParams]) -> tuple[BaseGeometry, dict[str, Any]]:
    """The area's footprint, to clip to, and the region the result records."""
    facts = area_facts(call.handle("area"), call=call)
    try:
        geometry: BaseGeometry = shape(await area_geometry(call, "area"))
        if not geometry.is_valid:
            geometry = make_valid(geometry)
    except (ValueError, TypeError, AttributeError, KeyError):
        raise failures.failure(BAD_AREA, name=call.capability.name) from None
    if geometry.is_empty:
        raise failures.failure(BAD_AREA, name=call.capability.name)
    return geometry, facts.region()


def _area_km2(collection: dict[str, Any]) -> float | None:
    """Area covered by the plumes, overlaps counted once, on an equal-area projection."""
    geometries = [shape(f["geometry"]) for f in collection["features"]]
    if not geometries:
        return 0.0
    projector = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True).transform
    try:
        union = unary_union([transform(projector, g) for g in geometries])
    except Exception:  # a geometry the projection or union cannot handle leaves the area out
        return None
    return float(round(union.area / 1e6, 1))


def _summarise(collection: dict[str, Any], counts: dict[str, int]) -> dict[str, Any]:
    features = collection["features"]
    summary: dict[str, Any] = {"plumes": len(features), "by_density": counts}
    area = _area_km2(collection)
    if area is not None:
        summary["area_km2"] = area
    if features:
        summary["first_start"] = min(f["properties"]["start"] for f in features)
        summary["last_end"] = max(f["properties"]["end"] for f in features)
    return summary


async def _get_extent(call: Call[GetExtentParams], *, transport: httpx.AsyncBaseTransport | None) -> CapabilityResult:
    name = call.capability.name
    date = call.params.date
    try:
        product_date(date)
    except HMSError:
        raise failures.failure(BAD_DATE, name=name) from None

    geometry: BaseGeometry | None = None
    region: Any = FULL_PRODUCT
    if call.params.area is not None:
        geometry, region = await _read_area(call)

    try:
        content = await download_archive(date, transport=transport)
        parsed = await call.compute(parse_archive, content, date, geometry)
    except HMSError as error:
        if error.status == "unavailable":
            return CapabilityResult(status="insufficient_evidence", reason=MISSING_PRODUCT)
        category = {"download_failed": "provider_unavailable", "too_large": "too_large"}.get(error.status, "contract")
        raise failures.failure(failures.SOURCE_CATEGORIES[category], name=name, connector="NOAA HMS") from None

    collection = polygons_collection(parsed["geojson"])
    summary = _summarise(collection, parsed["counts"])
    provenance = parsed["provenance"]
    label = region if isinstance(region, str) else region.get("name", "the selected area")
    line = f"{PRODUCT} for {date}: {summary['plumes']} plume(s) over {label}."
    record = await call.mint(
        "polygons",
        collection,
        Descriptor(
            summary=line,
            quantity="smoke_extent",
            product=PRODUCT,
            region=region,
            period=Period(start=date, end=date),
        ),
    )
    disclosures = [CAVEAT, POLYGON_NOTE] if summary["plumes"] else [NO_SMOKE, CAVEAT]
    if geometry is not None:
        disclosures.append(f"Plumes were clipped to the boundary of {label}; area_km2 counts only the smoke inside it.")
    if provenance["skipped_records"]:
        disclosures.append(f"{provenance['skipped_records']} unreadable record(s) in the NOAA product were skipped.")
    return call.done(
        record,
        summary=summary,
        quality={"repaired_records": provenance["repaired_records"], "skipped_records": provenance["skipped_records"]},
        values={"source_url": provenance["source_url"], "sha256": provenance["sha256"]},
        disclosures=tuple(disclosures),
    )


def make_get_extent(transport: httpx.AsyncBaseTransport | None = None) -> Capability[GetExtentParams]:
    """The capability, reading the NOAA archive through ``transport`` (tests serve it without a network)."""
    return Capability(
        name=NAME,
        description="NOAA HMS smoke plumes for one date, optionally limited to an area, as polygons with density and time window.",
        params=GetExtentParams,
        output_kind="polygons",
        label="Retrieving NOAA HMS smoke",
        method="NOAA HMS smoke for {date} was retrieved, giving {summary.plumes} plume(s).",
        retrieves=True,
        retry_safe=True,
        run=partial(_get_extent, transport=transport),
    )


get_extent = make_get_extent()

CAPABILITIES = (get_extent,)
