"""
earthdata_mcp/workspace.py
===========================
Binds every earthdata-retrieval MCP tool call to a workspace_id the model
never sees or invents. Each wrapped tool's schema drops workspace_id
entirely; the wrapper resolves it at call time via ``user_id_getter`` and
injects ``user-{user_id}``, so a researcher's handles persist across their
threads without the model being able to see or forge workspace identity.
"""
from __future__ import annotations

import copy
import functools
import json
import logging
from typing import Any, Callable, Protocol

from langchain_core.tools import BaseTool, StructuredTool

from tta_backend.config.workflow_stages import STAGE_AOI, STAGE_COVERAGE, STAGE_SEARCH
from tta_backend.earthdata_mcp import tool_cache
from tta_backend.earthdata_mcp.results import (
    CATEGORY_CONTRACT,
    CATEGORY_TOKEN_INVALID,
    CATEGORY_USER_INPUT,
    MCPToolError,
    call_tool,
    parse_tool_result,
)
from tta_backend.utils.streaming import emit_status

logger = logging.getLogger(__name__)

# T31: parameters injected here and hidden from the model-facing schema —
# workspace_id always, edl_token only for a tool whose advertised schema
# includes it (feature-detected per tool at bind time, see _bind_one).
_HIDDEN_PARAMS = ("workspace_id", "edl_token")


class EdlCredentialInjector(Protocol):
    """Duck-typed seam bind_workspace calls into for T31 per-user Earthdata
    credential injection — implemented by
    services.connector_credential_service.EdlCredentialInjector. Kept as a
    Protocol (not an import) so earthdata_mcp stays decoupled from the
    services layer; tests pass a minimal fake with this shape."""

    async def resolve(self, user_id: str) -> str | None: ...

    def mark_used(self, user_id: str) -> None: ...

    async def mark_invalid(self, user_id: str) -> None: ...


# T19: one wrapper covers every curated discovery tool without touching the
# MCP — searching/resolving/checking are exactly the stages the model-facing
# discovery tools correspond to (earthdata_mcp/client.py's CURATED_TOOL_NAMES).
_STAGE_BY_TOOL_NAME: dict[str, tuple[str, str]] = {
    "search_datasets": (STAGE_SEARCH, "Searching datasets..."),
    "describe_dataset": (STAGE_SEARCH, "Inspecting dataset..."),
    "preview_dataset": (STAGE_SEARCH, "Previewing dataset..."),
    "define_area_of_interest": (STAGE_AOI, "Resolving area of interest..."),
    "check_availability": (STAGE_COVERAGE, "Checking availability..."),
    "check_coverage": (STAGE_COVERAGE, "Checking coverage..."),
}


class MissingUserContextError(RuntimeError):
    """A workspace-bound MCP tool was called with no user context bound
    (T26). A context-less workspace is an isolation hole, not a default —
    minting a shared ``"user-None"`` workspace silently pooled every
    caller's retrievals together, so this fails loud instead. Since T37 it
    is returned classified as a T18 ``contract`` error envelope (still
    logged loudly — it indicates a programming error) rather than raised
    bare into an unclassified traceback string."""


def bind_workspace(
    tools: dict[str, BaseTool],
    user_id_getter: Callable[[], str],
    *,
    edl_injector: EdlCredentialInjector | None = None,
) -> dict[str, BaseTool]:
    """Return copies of ``tools`` with workspace_id injected and hidden from
    the schema. ``edl_injector`` (T31) additionally injects the calling
    user's decrypted Earthdata token as ``edl_token`` — but only for a tool
    whose advertised schema includes that parameter, and only when the
    injector resolves one (connected ∧ unexpired); omitting ``edl_injector``
    (the default) reproduces the pre-T31 behavior exactly, token or no
    token, advertised or not."""
    bound = {name: _bind_one(tool, user_id_getter, edl_injector) for name, tool in tools.items()}
    # T60 D13: applied here, not in curated_model_tools, and deliberately so.
    # model_view_describe_dataset's *shape* is the precedent; its application
    # point is not. That one wraps only the model-facing copy because
    # discovery_service needs the untrimmed tool. The requirement here is the
    # opposite -- all three AOI callers must be translated, and all three take
    # the dict this function returns.
    if "define_area_of_interest" in bound:
        bound["define_area_of_interest"] = region_aware_area_of_interest(
            bound["define_area_of_interest"]
        )
    return bound


def _schema_properties(schema) -> dict:
    return schema.get("properties", {}) if isinstance(schema, dict) else schema.schema().get("properties", {})


def _bind_one(
    tool: BaseTool,
    user_id_getter: Callable[[], str],
    edl_injector: EdlCredentialInjector | None,
) -> BaseTool:
    # T31: feature-detected once at bind time, off the MCP's advertised
    # schema for *this* tool — an un-upgraded MCP (or a tool PRD-022 hasn't
    # reached yet) simply never gets edl_token, no matter how live the
    # user's connector is. Never added to the required-parameter contract
    # check (earthdata_mcp/connection.py's REQUIRED_TOOL_PARAMS), so an
    # un-upgraded MCP still reaches ready.
    advertises_edl_token = "edl_token" in _schema_properties(tool.args_schema)
    schema = _schema_without_hidden_params(tool.args_schema)
    stage_info = _STAGE_BY_TOOL_NAME.get(tool.name)

    async def _call(**kwargs):
        user_id = user_id_getter()
        if user_id is None:
            exc = MissingUserContextError(
                f"No user context bound for tool {tool.name!r} — refusing to "
                "mint a shared 'user-None' workspace."
            )
            logger.error(
                "missing_user_context",
                extra={"_event": "missing_user_context", "_tool": tool.name},
            )
            return MCPToolError(CATEGORY_CONTRACT, str(exc)).to_tool_json()
        kwargs["workspace_id"] = f"user-{user_id}"
        if stage_info is not None:
            emit_status(stage_info[1], stage=stage_info[0])

        # T53: the discovery-metadata cache, read here — after workspace_id is
        # known (it is part of the key) and *before* both the credential
        # resolve and the call_tool round-trip, so a hit costs neither a
        # decrypt nor a network call. No credential was used on a hit, so
        # mark_used deliberately never fires for one.
        cacheable = tool_cache.is_cacheable(tool.name)
        if cacheable:
            hit = tool_cache.lookup(tool.name, kwargs["workspace_id"], kwargs)
            if hit is not None:
                return hit

        # T31 injection policy: connected ∧ unexpired ∧ advertising: send
        # nothing otherwise and let the MCP fall back to its shared env
        # credential. edl_injector.resolve() owns the connected/unexpired
        # check and the just-in-time decrypt; it never caches plaintext.
        injected = False
        if advertises_edl_token and edl_injector is not None:
            token = await edl_injector.resolve(user_id)
            if token is not None:
                kwargs["edl_token"] = token
                injected = True

        # T18: bind_workspace is the one place every model-facing MCP tool
        # call passes through — classify here (call_tool catches a raised
        # transport failure, parse_tool_result classifies the returned
        # content) and hand back the structured error envelope instead of a
        # raw exception. On success ``raw`` is returned unchanged, so a
        # backend composite's own parse_tool_result(raw) call downstream
        # behaves exactly as before; on error, that same downstream call
        # recognizes the envelope and re-raises the typed MCPToolError.
        try:
            raw = await call_tool(tool, kwargs)
            result = parse_tool_result(raw)
        except MCPToolError as exc:
            # T31: TOKEN_INVALID attributed to a token *this call actually
            # injected* flips the connector's stored status so the
            # Connectors tab agrees with the failure. TOKEN_EXPIRED relies
            # on T30's derived-expired instead (no stored-status write), and
            # EULA_NOT_ACCEPTED never touches status — the token is fine,
            # the entitlement isn't. Never fires for the shared-credential
            # path (injected is False), so one user's bad token can't flip
            # another's connector.
            if injected and exc.category == CATEGORY_TOKEN_INVALID:
                await edl_injector.mark_invalid(user_id)
            # T46 story #4: a rejected AOI *input* must leave a greppable trace.
            # The live 2026-07-17 incident (define_area_of_interest rejected an
            # inverted bbox, the agent improvised "North America") left nothing
            # to grep for. Log the named event with the offending location so a
            # silent-substitution regression is discoverable from logs; the
            # dispatch-layer guard (services/subagent_dispatch.py) uses the same
            # signal to refuse the substituted answer.
            if tool.name == "define_area_of_interest" and exc.category == CATEGORY_USER_INPUT:
                logger.warning(
                    "aoi_user_input_rejected",
                    extra={
                        "_event": "aoi_user_input_rejected",
                        "_location": kwargs.get("location"),
                        "_detail": exc.message,
                    },
                )
            return exc.to_tool_json()
        if injected:
            # Fire-and-forget and coalesced per agent turn inside mark_used
            # itself — never on this call's critical path, never failing it.
            edl_injector.mark_used(user_id)
        # T19 story #3: surface the granule count once check_coverage's own
        # response is known, so a researcher sees why their request is
        # small or large before the (potentially long) retrieval wait.
        if tool.name == "check_coverage" and isinstance(result, dict) and "granule_count" in result:
            granule_count = result["granule_count"]
            emit_status(f"Checking coverage — {granule_count} granules...", stage=STAGE_COVERAGE, detail=granule_count)
        # Only reached on success: every classified failure returned above, so
        # a transient provider_unavailable is retried on the next attempt
        # rather than replayed for the rest of the TTL. The **raw** result is
        # what is stored, so parse_tool_result and model_view_describe_dataset
        # behave byte-identically on hit and miss.
        if cacheable:
            tool_cache.store(tool.name, kwargs["workspace_id"], kwargs, raw)
        return raw

    return StructuredTool.from_function(
        coroutine=_call,
        name=tool.name,
        description=tool.description,
        args_schema=schema,
    )


def _schema_without_hidden_params(schema):
    schema = copy.deepcopy(schema)
    properties = schema.get("properties", {})
    for name in _HIDDEN_PARAMS:
        properties.pop(name, None)
    schema["required"] = [name for name in schema.get("required", []) if name not in _HIDDEN_PARAMS]
    return schema


@functools.lru_cache(maxsize=1)
def _region_resolver():
    """One shared RegionResolver for the translation, built lazily.

    Lazy because ``utils.plotting`` pulls in cartopy/rasterio, and
    ``earthdata_mcp`` is imported in contexts that have no reason to pay for
    that. Cached because ``global_regions`` is instance state rebuilt on every
    construction, and this is a per-tool-call path.

    Deliberately *not* registered in tests/cache_isolation.py's
    ``clear_process_caches``. That policy covers caches holding data a test
    could leave behind for the next one; this holds only the static preset
    tables, so clearing it could never change an outcome — the same reasoning
    that leaves ``load_preset_polygons``'s own lru_cache unregistered."""
    from tta_backend.utils.plotting import RegionResolver

    return RegionResolver()


def region_aware_area_of_interest(tool: BaseTool) -> BaseTool:
    """Wrap an already workspace-bound ``define_area_of_interest`` so the T60
    region vocabulary reaches the *retrieval* plane (D13).

    Phase 1 shipped a correct mask for the Ozone Transport Region that no user
    could reach from a prompt: the spatial subset that actually pulls data is
    resolved by the MCP, in another repo, by a geocoder that has never heard of
    the word. Measured live in the Phase 1.5 gate (V6), ``"ozone over the OTC"``
    does not politely fail -- the AOI step refuses ``"Ozone Transport
    Commission"``, and the agent then silently substitutes ``"Northeastern
    United States"`` and retrieves *that*, which is not a superset of the OTR.
    The mask would then clip against a cube that never covered the Mid-Atlantic
    third of the region, with nothing to say so (Risk 5).

    **Why a named wrapper and not a branch in ``_bind_one``.** ``_bind_one`` is
    generic MCP plumbing -- workspace binding, credential injection, error
    classification. Region vocabulary is a domain concept and does not belong
    inside a transport layer. Kept here it is greppable, and the whole two-plane
    story lives in one function with one docstring.

    **Why the whole story is in one place.** Three callers reach this tool:
    services/retrieval_composites.py, services/discovery_service.py::_resolve_aoi,
    and -- the majority path -- the agent itself, which calls it directly as a
    curated tool because workflow step 2 of the agent prompt makes the AOI step
    mandatory and first. No composite edit can intercept that third one. This
    wrapper is applied in ``bind_workspace``, which is the single dict all three
    take their tools from.

    **Which side of the T53 cache this sits on, and why it matters.** Outside,
    i.e. *before*. The rewrite happens here, and only then does ``_bind_one``'s
    ``_call`` compute the cache key from ``kwargs``. So "otc", "the OTC", "otr"
    and "ozone transport region" all collapse to a single key, and no key can
    ever hold a raw coalition string. Were this a branch *inside* ``_call``
    placed after the lookup, the cache would answer "otc" with whatever had been
    stored for the raw string.

    A string outside the vocabulary is passed through untouched -- byte-identical
    to pre-T60 behavior, which is the mirror of the mask plane's NOT_CLAIMED
    fall-through and the regression this seam is most likely to cause.
    """
    from tta_backend.utils import region_dispatch

    async def _call(**kwargs):
        location = kwargs.get("location")
        if isinstance(location, str):
            resolver = _region_resolver()
            # D11a: the same normalization the mask plane uses, so neither
            # plane can normalize differently than the other.
            normalized = resolver._normalize_location_name(location)
            dispatched = region_dispatch.dispatch_extent(normalized, resolver.global_regions)
            if dispatched.claimed:
                if dispatched.location is None:
                    # Claimed and failed. Refusing is the point: falling
                    # through would hand the MCP's Nominatim the very string
                    # D3b exists to keep away from it.
                    logger.error(
                        "region_extent_unresolved",
                        extra={
                            "_event": "region_extent_unresolved",
                            "_location": location,
                        },
                    )
                    return MCPToolError(
                        CATEGORY_USER_INPUT,
                        f"'{location}' is a known region, but its boundary asset "
                        "is missing, so the area to retrieve can't be determined.",
                        suggestion="Try a different region, or name the member states directly.",
                    ).to_tool_json()
                kwargs["location"] = dispatched.location
        return await tool.ainvoke(kwargs)

    return StructuredTool.from_function(
        coroutine=_call,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


def model_view_describe_dataset(tool: BaseTool) -> BaseTool:
    """Wrap an already workspace-bound ``describe_dataset`` tool so its
    model-facing result stays proportional to what the model actually uses
    (T13): variable names/units/advisories to subset, never every
    fill-value/valid-range record a many-variable collection carries.

    Applied only to the curated model-facing tool list
    (earthdata_mcp/toolset.py::curated_model_tools) — the original tool in
    the shared workspace-bound dict is left untouched, since discovery-pane
    consumers (services/discovery_service.py) call it directly by name and
    need the full per-variable records.
    """

    async def _call(**kwargs):
        raw = await tool.ainvoke(kwargs)
        try:
            result = parse_tool_result(raw)
        except MCPToolError as exc:
            # ``tool`` is already bind_workspace-wrapped, so ``raw`` here is
            # either real content or bind_workspace's own error envelope
            # (T18) — re-raised by parse_tool_result and passed straight
            # through unchanged rather than re-wrapped.
            return exc.to_tool_json()
        return json.dumps(_compact_describe_dataset_result(result))

    return StructuredTool.from_function(
        coroutine=_call,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


def _compact_describe_dataset_result(result: dict) -> dict:
    variables = result.get("variables")
    if not isinstance(variables, list):
        return result
    compacted = dict(result)
    compacted["variables"] = [
        _compact_variable(var) if isinstance(var, dict) else var for var in variables
    ]
    return compacted


def _compact_variable(var: dict) -> dict:
    """name/long_name/units/advisory_notes plus a one-line mask_note derived
    from fill/range presence — the model needs variable names to subset, not
    every fill-value/valid-range record (T13 story #11)."""
    out: dict[str, Any] = {
        "name": var.get("name"),
        "long_name": var.get("long_name"),
        "units": var.get("units"),
        "advisory_notes": var.get("advisory_notes", []),
    }
    has_fill = bool(var.get("fill_values"))
    has_range = bool(var.get("valid_ranges"))
    if has_fill and has_range:
        out["mask_note"] = "fill values and a valid range are defined"
    elif has_fill:
        out["mask_note"] = "fill values are defined, no valid range"
    elif has_range:
        out["mask_note"] = "a valid range is defined, no fill values"
    else:
        out["mask_note"] = "no fill/range metadata"
    if "mask_metadata_note" in var:
        out["mask_metadata_note"] = var["mask_metadata_note"]
    return {k: v for k, v in out.items() if v is not None}
