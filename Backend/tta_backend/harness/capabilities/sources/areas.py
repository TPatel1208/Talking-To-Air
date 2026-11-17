"""Areas: a place in the user's words, resolved once and passed on as a handle.

An area handle is a ``polygons`` handle holding one GeoJSON feature, the
area's footprint. Its descriptor's ``region`` holds the resolved ``name``,
its ``bounds`` as [west, south, east, north], the ``wording`` the user gave,
verbatim, and the ``region_type`` of the footprint.

Other modules read an area handle only through ``area_facts`` (the recorded
region) and ``area_geometry`` / ``area_geometry_of`` (the footprint).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from pydantic import Field, JsonValue

from tta_backend.harness.capabilities import failures
from tta_backend.harness.capabilities.call import Call
from tta_backend.harness.capabilities.definition import Capability, Params
from tta_backend.harness.capabilities.registry import Group
from tta_backend.harness.capabilities.result import CapabilityResult, Question
from tta_backend.harness.handles.errors import HandleError
from tta_backend.harness.handles.model import Descriptor, HandleRecord
from tta_backend.harness.run_context import PlaceRefused, Region
from tta_backend.harness.trace import FailureCategory, StepFailure

GROUP = Group(
    name="sources.areas",
    description="Resolve a place in the user's words to an area handle the data sources take.",
    notes=(
        "Pass the place exactly as the user wrote it: do not expand acronyms or add words. "
        "Resolve an area once and pass its handle to every step that needs it. "
        "A place that cannot be found comes back as a question for the user; it is never "
        "replaced by a nearby or broader place."
    ),
)

_QUOTED_CHARS = 120

# The question for each refusal, filled from its facts and ``wording``, the user's own text.
REFUSAL_PROMPTS: dict[str, str] = {
    "member_not_found": (
        "'{member}' in '{wording}' could not be found as a U.S. state or a country, so the combined area "
        "was not built and nothing was substituted. Name each part as a state, such as NY or New York, "
        "or a country, such as France."
    ),
    "member_ambiguous": (
        "'{member}' in '{wording}' names both a U.S. state and a country, so it was not guessed. "
        "Write '{state_form}' for the state or '{country_form}' for the country."
    ),
    "member_too_large": (
        "'{member}' ({name}) covers about {cells} data cells on its own, above the {max_cells}-cell limit, "
        "because its boundary includes distant territory. Name a smaller area, such as the states or "
        "provinces you mean."
    ),
    "combined_too_large": (
        "The parts of '{wording}' are far apart: together they cover about {cells} data cells, above the "
        "{max_cells}-cell limit. Ask for fewer or nearer parts."
    ),
    "empty_member": (
        "'{wording}' has an empty part between its '+' signs. Write each part between the signs, as in NY + NJ."
    ),
    "distance_unreadable": (
        "'{wording}' looks like a distance around a place, but the distance could not be read. "
        "Write it as 'within 50 miles of Boston' or 'within 50 km of Boston'."
    ),
    "distance_unit_missing": (
        "'{wording}' does not say whether {amount} means miles or kilometres, and nothing was assumed. "
        "Write 'within {amount} miles of ...' or 'within {amount} km of ...'."
    ),
    "distance_unit_unknown": (
        "'{unit}' in '{wording}' is not a distance unit that can be used. Use miles or km."
    ),
    "centre_is_region": (
        "'{centre}' names a region, not a point, so an area within {radius} of it cannot be built. "
        "Name a town or city to centre it on, or ask for '{centre}' on its own."
    ),
    "centre_not_found": (
        "'{centre}' could not be found as a place, so there is no point to measure {radius} from, "
        "and nothing was substituted. Check the spelling, or name a nearby town or city."
    ),
    "reaches_pole": (
        "The area within {radius} of {centre} would reach the {pole}, {pole_km} km away, and an area holding a "
        "pole cannot be drawn on a latitude-longitude grid. Ask for a radius under {pole_km} km."
    ),
    "crosses_antimeridian": (
        "The area within {radius} of {centre} crosses the 180th meridian, where it cannot be drawn as one "
        "latitude-longitude area. Ask for a smaller radius or another centre."
    ),
    "distance_too_large": (
        "The area within {radius} of {centre} covers about {cells} data cells, above the {max_cells}-cell limit "
        "that can be processed, so nothing was retrieved. Ask for a smaller radius."
    ),
}
_PROMPT_CHARS = 500


class ResolveParams(Params):
    place: str = Field(min_length=1, max_length=500, description="The place exactly as the user wrote it.")


def _quoted(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _QUOTED_CHARS else text[: _QUOTED_CHARS - 1] + "…"


def area_feature_collection(region: Region) -> dict[str, Any]:
    """The polygons payload of an area handle."""
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": region.geometry,
            "properties": {"name": region.name, "region_type": region.region_type},
        }],
    }


@dataclass(frozen=True)
class AreaFacts:
    """What an area handle records: its id, resolved ``name``, ``bounds`` as
    (west, south, east, north), the user's ``wording`` and the ``region_type``."""

    id: str
    name: str
    bounds: tuple[float, float, float, float]
    wording: str | None
    region_type: str | None

    def region(self) -> dict[str, JsonValue]:
        """The region a handle made over this area records: name, bounds, the area's id and the wording."""
        return {"name": self.name, "bounds": list(self.bounds), "area": self.id, "wording": self.wording}


NOT_AN_AREA = "{name}: {param} is not an area handle; resolve the place with sources.areas.resolve."
NO_GEOMETRY = "{name}: the area handle {param} holds no footprint."


def _not_an_area(call: Call[Any] | None, param: str, wording: str = NOT_AN_AREA) -> StepFailure:
    name = call.capability.name if call is not None else "This step"
    return StepFailure(FailureCategory.BAD_INPUT, wording.format(name=name, param=param))


def area_facts(record: HandleRecord, *, call: Call[Any] | None = None, param: str = "area") -> AreaFacts:
    """The region facts an area handle records.

    Refuses as bad input a handle that is not an area: not ``polygons``, or
    with no name or no valid [west, south, east, north] bounds.
    """
    region = record.descriptor.region
    if record.kind != "polygons" or not isinstance(region, dict):
        raise _not_an_area(call, param)
    name, bounds = region.get("name"), region.get("bounds")
    if not isinstance(name, str) or not name.strip() or not isinstance(bounds, list) or len(bounds) != 4:
        raise _not_an_area(call, param)
    if not all(isinstance(b, int | float) and not isinstance(b, bool) and math.isfinite(b) for b in bounds):
        raise _not_an_area(call, param)
    west, south, east, north = (float(b) for b in bounds)  # type: ignore[arg-type]
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise _not_an_area(call, param)
    wording, region_type = region.get("wording"), region.get("region_type")
    return AreaFacts(
        id=record.id,
        name=name,
        bounds=(west, south, east, north),
        wording=wording if isinstance(wording, str) else None,
        region_type=region_type if isinstance(region_type, str) else None,
    )


def collection_geometry(collection: Any) -> dict[str, Any] | None:
    """The footprint in an area handle's payload, as one GeoJSON geometry, or None when it holds none."""
    features = collection.get("features") if isinstance(collection, dict) else None
    geometries = [
        f["geometry"] for f in features or [] if isinstance(f, dict) and isinstance(f.get("geometry"), dict)
    ]
    if not geometries:
        return None
    if len(geometries) == 1:
        geometry: dict[str, Any] = geometries[0]
        return geometry
    return {"type": "GeometryCollection", "geometries": geometries}


async def area_geometry(call: Call[Any], name: str) -> dict[str, Any]:
    """The GeoJSON footprint of the area handle given as parameter ``name``.

    Refuses as bad input a handle that is not an area or holds no footprint;
    raises InputEvicted when its data is no longer stored.
    """
    record = call.handle(name)
    area_facts(record, call=call, param=name)
    geometry = collection_geometry(await call.open(name))
    if geometry is None:
        raise _not_an_area(call, name, NO_GEOMETRY)
    return geometry


async def area_geometry_of(call: Call[Any], area_id: str) -> tuple[AreaFacts, dict[str, Any]]:
    """The facts and GeoJSON footprint of an area recorded by id on another
    handle, such as a retrieval grid's ``region.area``.

    Refuses as bad input an id that is not this user's area handle or holds no
    footprint; raises InputEvicted when its data is no longer stored.
    """
    resolver, user = call.env.resolver, call.user_id
    try:
        record = await resolver.get(user, area_id)
    except HandleError as error:
        if error.reason != "not_found":
            raise
        raise _not_an_area(call, "the recorded area") from None
    facts = area_facts(record, call=call, param="the recorded area")
    try:
        collection = await resolver.open(user, area_id, "polygons")
    except HandleError as error:
        if error.reason != "evicted":
            raise
        raise failures.InputEvicted(call.capability.name, (area_id,)) from None
    geometry = collection_geometry(collection)
    if geometry is None:
        raise _not_an_area(call, "the recorded area", NO_GEOMETRY)
    return facts, geometry


def refusal_prompt(refused: PlaceRefused, wording: str) -> str | None:
    """The question for a refusal, or None when its code or facts do not fill a prompt."""
    template = REFUSAL_PROMPTS.get(refused.code)
    if template is None:
        return None
    facts = {key: _quoted(value) for key, value in refused.facts.items()}
    try:
        text = template.format(**facts, wording=_quoted(wording))
    except (KeyError, IndexError, ValueError):
        return None
    return text if len(text) <= _PROMPT_CHARS else text[: _PROMPT_CHARS - 1] + "…"


# A latitude,longitude pair, such as "40.735,-74.172 (Newark, NJ)"; a label in
# brackets names the point. Both numbers need decimals, so "Route 1, 2 lanes" is not a point.
_NUMBER = r"[-+]?\d+(?:\.\d+)?"
_COORDINATE = r"[-+]?\d+\.\d+"
_PAIR = re.compile(rf"(?<![\d.])({_COORDINATE})\s*,\s*({_COORDINATE})(?![\d.])")
_LABEL = re.compile(r"\(([^()]{1,80})\)")
# Half the side of the square drawn around a point: about a kilometre, inside one grid cell.
POINT_HALF_SIDE_DEGREES = 0.01


def point_region(wording: str) -> Region | None:
    """The point a wording gives as one latitude,longitude pair, as a small
    square around it; None when the wording holds any other numbers, such as
    a west,south,east,north box or a distance."""
    pairs = _PAIR.findall(wording)
    if len(pairs) != 1 or len(re.findall(_NUMBER, _LABEL.sub("", wording))) != 2:
        return None
    lat, lon = (float(value) for value in pairs[0])
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    half = POINT_HALF_SIDE_DEGREES
    west, south, east, north = lon - half, lat - half, lon + half, lat + half
    label = _LABEL.search(wording)
    where = f"{lat:g}, {lon:g}"
    return Region(
        name=f"{label.group(1).strip()} ({where})" if label else where,
        bounds=(west, south, east, north),
        geometry={
            "type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
        },
        region_type="point",
    )


async def _resolve(call: Call[ResolveParams]) -> CapabilityResult:
    place = call.params.place
    try:
        region = point_region(place) or await call.env.places.resolve(place)
    except PlaceRefused as refused:
        prompt = refusal_prompt(refused, place)
        if prompt is None:
            raise failures.failure(failures.SOURCE_CATEGORIES["user_input"], name=call.capability.name) from None
        return CapabilityResult(
            status="needs_input", question=Question(kind="text", prompt=prompt), values={"refusal": refused.code},
        )
    if region is None:
        return CapabilityResult(status="needs_input", question=Question(
            kind="text",
            prompt=(
                f"'{_quoted(place)}' could not be found as a place. Name the area another way, "
                "such as a state, a city, or a west,south,east,north box."
            ),
        ))
    west, south, east, north = (float(v) for v in region.bounds)
    descriptor = Descriptor(
        summary=f"Area: {_quoted(region.name)}",
        region={
            "name": region.name,
            "bounds": [west, south, east, north],
            "wording": place,
            "region_type": region.region_type,
        },
    )
    record = await call.mint("polygons", area_feature_collection(region), descriptor)
    disclosures: tuple[str, ...] = ()
    if " ".join(place.lower().split()) != " ".join(region.name.lower().split()):
        disclosures = (f"'{_quoted(place)}' was resolved to {_quoted(region.name)}.",)
    return call.done(
        record,
        summary={"name": region.name, "bounds": [west, south, east, north], "region_type": region.region_type},
        disclosures=disclosures,
    )


resolve = Capability(
    name="sources.areas.resolve",
    description="Resolve a place in the user's words to an area handle with its name and bounds.",
    params=ResolveParams,
    output_kind="polygons",
    label="Resolving the area",
    method="The area '{place}' was resolved to {summary.name}.",
    retrieves=False,
    retry_safe=True,
    run=_resolve,
)

CAPABILITIES = (resolve,)
