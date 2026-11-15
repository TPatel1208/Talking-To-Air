"""The app's implementations of the harness's host ports, over existing services.

Nothing in production builds a harness run yet; these are what it will be
handed: the per-user earthdata session, retrieval opening, heavy-compute
admission, size and time limits, and place resolution.
"""
from __future__ import annotations

import contextvars
import functools
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool
from shapely.geometry import mapping

from tta_backend.earthdata_mcp.results import CATEGORY_CONTRACT, MCPToolError, call_tool, parse_tool_result
from tta_backend.harness.handles.errors import HandleError
from tta_backend.harness.handles.model import HandleRecord
from tta_backend.harness.handles.resolver import OpenRetrieval
from tta_backend.harness.run_context import AnsweredRequests, Limits, PlaceRefused, Region
from tta_backend.infrastructure.settings import Settings
from tta_backend.services import admission
from tta_backend.services.open_handle import HandleGoneError, open_handle
from tta_backend.utils import region_buffer, region_composition
from tta_backend.utils.plotting import RegionResolver


class McpEarthdataSession:
    """The earthdata service over one user's workspace-bound MCP tools.

    The service has no idempotency parameter, so a call repeating an
    idempotency key and request this session already answered returns the
    first answer without calling the service again; the same key with a
    different request is refused as a contract failure. The session keeps
    the most recent keys only. A saved job reference covers a resumed run in
    another process; only a crash between the service's reply and the harness
    saving the job can submit twice.
    """

    def __init__(self, tools: dict[str, BaseTool], *, remembered: int = 256) -> None:
        self._tools = tools
        self._answered = AnsweredRequests(remembered)

    async def call(
        self, tool: str, arguments: dict[str, Any], *, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        if idempotency_key is not None:
            answer = self._answered.get(idempotency_key, tool, arguments)
            if answer is not None:
                return answer
        if tool not in self._tools:
            raise MCPToolError(CATEGORY_CONTRACT, f"The data service offers no tool named {tool!r}.")
        result = parse_tool_result(await call_tool(self._tools[tool], arguments))
        if idempotency_key is not None:
            self._answered.put(idempotency_key, tool, arguments, result)
        return result


def open_retrieval_over(tools: dict[str, BaseTool], user_id: str) -> OpenRetrieval:
    """An ``OpenRetrieval`` over one user's MCP tools; another user's handle is
    not found, and one whose data is gone and cannot be rematerialised is evicted."""

    async def open_retrieval(owner: str, record: HandleRecord) -> Any:
        if owner != user_id:
            raise HandleError("not_found", record.id)
        try:
            return await open_handle(record.id, tools)
        except HandleGoneError:
            raise HandleError("evicted", record.id) from None

    return open_retrieval


class AdmissionCompute:
    """Heavy work on the app's bounded pool, admitted like every chart tool's."""

    async def run[**A, T](self, fn: Callable[A, T], /, *args: A.args, **kwargs: A.kwargs) -> T:
        return await admission.run_heavy(functools.partial(fn, *args, **kwargs))


def limits_from_settings(settings: Settings) -> Limits:
    """The harness's retrieval limits, from the same settings the current system uses."""
    return Limits(
        retrieval_soft_bytes=settings.retrieval_soft_cap_bytes,
        retrieval_hard_bytes=settings.retrieval_hard_cap_bytes,
        max_point_timeseries_days=settings.retrieval_max_timeseries_days,
        retrieval_timeout_seconds=float(settings.await_retrieval_timeout_seconds),
        poll_min_seconds=float(settings.await_retrieval_poll_min_seconds),
        poll_max_seconds=float(settings.await_retrieval_poll_max_seconds),
    )


def _bounds_area(bounds: Any) -> float:
    west, south, east, north = (float(v) for v in bounds)
    return (east - west) * (north - south)


def _cells(area: float) -> str:
    return f"{area * region_composition.CELLS_PER_SQUARE_DEGREE:,.0f}"


def _max_cells() -> str:
    from tta_backend.domain.aggregation.frame_stack import MAX_FRAME_NATIVE_CELLS

    return f"{MAX_FRAME_NATIVE_CELLS:,}"


# What the geocoder answered, hits and misses alike, during one ResolverPlaces.resolve call.
_ANSWERS: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("place_answers", default=None)


class _RememberingGeocoder:
    """The resolver's geocoder, answering a repeated lookup within one resolve
    call from that call's memory. Outside such a call it is the geocoder as given."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def geocode(self, location_name: str) -> Any:
        return self._inner.geocode(location_name)

    async def ageocode(self, location_name: str) -> Any:
        answers = _ANSWERS.get()
        key = " ".join(location_name.lower().split())
        if answers is not None and key in answers:
            return answers[key]
        found = await self._inner.ageocode(location_name)
        if answers is not None:
            answers[key] = found
        return found


class ResolverPlaces:
    """Places resolved by the app's ``RegionResolver``: presets, states, composites, buffers, geocoding.

    A refusal the resolver raises for a combined area or a distance around a
    place becomes PlaceRefused. Its facts are worked out again here from the
    region modules' own rules and the user's wording, since the resolver's
    errors carry only text; a refusal not recognised that way is re-raised
    as it came. Working out the facts never asks the geocoder again: its
    geocoder is wrapped so a lookup repeated within one call is answered from
    that call's first answer, a miss included.
    """

    def __init__(self, resolver: RegionResolver | None = None) -> None:
        self._resolver = resolver or RegionResolver()
        if not isinstance(self._resolver.geocoding_service, _RememberingGeocoder):
            self._resolver.geocoding_service = _RememberingGeocoder(self._resolver.geocoding_service)  # type: ignore[assignment]

    async def resolve(self, wording: str) -> Region | None:
        token = _ANSWERS.set({})
        try:
            region = await self._resolver.aresolve_location(wording)
        except MCPToolError:
            refused = await self._refusal(wording)
            if refused is None:
                raise
            raise refused from None
        finally:
            _ANSWERS.reset(token)
        if region is None:
            return None
        west, south, east, north = (float(v) for v in region["bounds"])
        return Region(
            name=str(region.get("display_name") or region.get("name") or wording),
            bounds=(west, south, east, north),
            geometry=dict(mapping(region["geometry"])),
            region_type=region.get("region_type"),
        )

    async def _refusal(self, wording: str) -> PlaceRefused | None:
        if region_composition.is_composite(wording):
            return self._combined_refusal(wording)
        if region_buffer.is_buffer(wording, self._resolver._normalize_location_name):
            return await self._distance_refusal(wording)
        return None

    def _combined_refusal(self, wording: str) -> PlaceRefused | None:
        """The same checks, in the same order, as ``region_composition.dispatch_composite``."""
        from shapely.ops import unary_union

        from tta_backend.datasets.us_states import US_STATES

        resolver = self._resolver
        written = [part.strip() for part in wording.split(region_composition.SEPARATOR)]
        tokens = region_composition.split_tokens(wording, resolver._normalize_location_name)
        if any(not token for token in tokens):
            return PlaceRefused("empty_member")
        limit = region_composition.MAX_COMPOSITE_ENVELOPE_DEG2
        geometries = []
        for member_text, token in zip(written, tokens, strict=True):
            if token in region_composition.ambiguous_member_tokens():
                return PlaceRefused("member_ambiguous", {
                    "member": member_text,
                    "state_form": str(US_STATES[token]["postal"]),
                    "country_form": f"{member_text} {region_composition.COUNTRY_SUFFIX}",
                })
            member = region_composition._member_geometry(token, resolver)
            if member is None:
                return PlaceRefused("member_not_found", {"member": member_text})
            area = _bounds_area(member["bounds"])
            if area > limit:
                return PlaceRefused("member_too_large", {
                    "member": member_text, "name": str(member["name"]), "cells": _cells(area), "max_cells": _max_cells(),
                })
            geometries.append(member["geometry"])
        area = _bounds_area(unary_union(geometries).bounds)
        if area > limit:
            return PlaceRefused("combined_too_large", {"cells": _cells(area), "max_cells": _max_cells()})
        return None

    async def _distance_refusal(self, wording: str) -> PlaceRefused | None:
        """The same checks, in the same order, as ``region_buffer.adispatch_buffer``."""
        from pyproj import Geod

        resolver = self._resolver
        normalize = resolver._normalize_location_name
        match = region_buffer._PARSE.match(normalize(wording))
        if match is None:
            return PlaceRefused("distance_unreadable")
        unit = match.group("unit")
        if not unit:
            return PlaceRefused("distance_unit_missing", {"amount": match.group("amount")})
        if unit not in region_buffer.METRES_PER_UNIT:
            return PlaceRefused("distance_unit_unknown", {"unit": unit})
        request = region_buffer.parse_buffer(wording, normalize)
        label = region_buffer._UNIT_LABEL.get(region_buffer.METRES_PER_UNIT[unit], unit)
        radius = f"{request.amount:g} {label}"
        # The centre as the user wrote it, for the two refusals that quote it.
        written = wording.strip()
        at = written.lower().rfind(" of ")
        centre_text = written[at + 4:].strip() if at >= 0 else request.target
        if normalize(request.target) in region_buffer._named_region_vocabulary(resolver):
            return PlaceRefused("centre_is_region", {"centre": centre_text, "radius": radius})
        # Answered from this call's first lookup, not the geocoder.
        found = await resolver.geocoding_service.ageocode(request.target)
        if found is None:
            return PlaceRefused("centre_not_found", {"centre": centre_text, "radius": radius})
        centre = str(found["display_name"])
        geod = Geod(ellps="WGS84")
        longitude, latitude = found["longitude"], found["latitude"]
        for pole_latitude, pole in ((90.0, "North Pole"), (-90.0, "South Pole")):
            _, _, distance = geod.inv(longitude, latitude, longitude, pole_latitude)
            if distance <= request.metres:
                return PlaceRefused("reaches_pole", {
                    "radius": radius, "centre": centre, "pole": pole, "pole_km": f"{distance / 1000:,.0f}",
                })
        geometry = region_buffer.build_buffer(latitude, longitude, request.metres)
        if region_buffer._crosses_antimeridian(geometry):
            return PlaceRefused("crosses_antimeridian", {"radius": radius, "centre": centre})
        area = _bounds_area(geometry.bounds)
        if area > region_composition.MAX_COMPOSITE_ENVELOPE_DEG2:
            return PlaceRefused("distance_too_large", {
                "radius": radius, "centre": centre, "cells": _cells(area), "max_cells": _max_cells(),
            })
        return None
