"""
earthdata_mcp/tool_cache.py
============================
T53 — a process-lifetime cache of *discovery metadata* MCP results, read and
written by the one seam every model-facing tool call already passes through
(``earthdata_mcp/workspace.py``'s ``_bind_one``).

The governing discipline is ``jobs_service``'s: cache what is provably
immutable, do not cache on a TTL guess. Collection metadata and deterministic
AOI resolution are admitted as *effectively* immutable with a bounded TTL as
the safety net; the allowlist below is what keeps that honest.
"""
from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from typing import Any

from tta_backend.config.settings import get_settings
from tta_backend.utils.metrics import record_cache_hit, record_cache_miss

logger = logging.getLogger(__name__)

# An ALLOWLIST, never a denylist. A new MCP tool is uncached until someone
# deliberately adds it here — a denylist would silently cache whatever the MCP
# grows next.
#
# What is deliberately absent is the load-bearing part of T53:
#   * check_coverage / check_availability look cacheable and are not. New
#     granules land continuously, and the NRT/current-date work exists
#     precisely because the agent got availability wrong near the present.
#     _inject_satellite_context passes prior handles as UNVERIFIED on purpose
#     to force a re-check; caching coverage would quietly defeat that rule.
#   * Every retrieval tool (retrieve_subset, estimate_retrieval_size,
#     export_result, rematerialize, convert_format, align, retrieve_timeseries,
#     cancel_retrieval) has side effects or returns state that changes.
#   * get_retrieval_status is already correctly handled by jobs_service, which
#     caches only provably-terminal outcomes. A second, weaker cache over the
#     same calls would be strictly worse than the one that exists.
CACHEABLE_TOOL_NAMES = frozenset(
    {
        "describe_dataset",
        "search_datasets",
        "define_area_of_interest",
        "preview_dataset",
    }
)

# Never part of a key and never stored: it is a credential, and it is already
# implied by workspace_id.
_CREDENTIAL_PARAMS = ("edl_token",)

# key -> (stored_at_monotonic, raw). Insertion-ordered and re-ordered on read,
# so the eviction victim under the entry cap is genuinely the least recently
# *used* one, not merely the oldest write.
#
# ``raw`` is stored exactly as ``call_tool`` returned it — a string for most
# tools, a content list for some — never a parsed or re-serialized form, so
# downstream parse_tool_result and model_view_describe_dataset behave
# identically on a hit and on a miss.
_CACHE: OrderedDict[str, tuple[float, Any]] = OrderedDict()

# tool name -> {"hits": n, "misses": n}. Only tools that actually consult the
# cache appear: an excluded tool is not a miss, it never asks.
_STATS: dict[str, dict[str, int]] = {}


def is_cacheable(tool_name: str) -> bool:
    return tool_name in CACHEABLE_TOOL_NAMES


def _cache_key(tool_name: str, workspace_id: str, args: dict) -> str:
    """``(tool_name, workspace_id, canonical_json(args))``.

    ``workspace_id`` participates even though CMR metadata is public: it costs
    nothing at this volume and means the entitlement question (T31 per-user EDL
    tokens, ``CATEGORY_EULA_NOT_ACCEPTED``) never has to be answered at all.

    ``edl_token`` is dropped rather than keyed on — it is a credential, and it
    is already implied by ``workspace_id``.
    """
    payload = {k: v for k, v in args.items() if k not in _CREDENTIAL_PARAMS}
    return json.dumps([tool_name, workspace_id, payload], sort_keys=True, default=str)


def lookup(tool_name: str, workspace_id: str, args: dict) -> Any | None:
    """The cached raw result for this exact call, or None if there is none or
    it has aged past the TTL. Records the hit/miss."""
    key = _cache_key(tool_name, workspace_id, args)
    entry = _CACHE.get(key)
    if entry is not None:
        stored_at, raw = entry
        if time.monotonic() - stored_at <= get_settings().mcp_metadata_cache_ttl_seconds:
            _CACHE.move_to_end(key)
            _count(tool_name, "hits")
            record_cache_hit("memory")
            # Greppable, and named per tool: the live check for this cache is
            # "repeated describe_dataset hits, check_coverage never appears".
            logger.info(
                "mcp_tool_cache_hit",
                extra={"_event": "mcp_tool_cache_hit", "_tool": tool_name},
            )
            return raw
        del _CACHE[key]
    _count(tool_name, "misses")
    record_cache_miss()
    return None


def store(tool_name: str, workspace_id: str, args: dict, raw: Any) -> None:
    """Record a *successful* result. Callers must never store a failure: a
    provider_unavailable replayed for the rest of the TTL turns an MCP restart
    window into a sticky wall."""
    key = _cache_key(tool_name, workspace_id, args)
    _CACHE[key] = (time.monotonic(), raw)
    _CACHE.move_to_end(key)
    max_entries = get_settings().mcp_metadata_cache_max_entries
    while len(_CACHE) > max_entries:
        _CACHE.popitem(last=False)


def tool_cache_stats() -> dict[str, dict[str, int]]:
    """Hits and misses per tool name. Per-tool rather than a single pair, so
    it is visible *which* metadata call is repeating and therefore whether the
    allowlist is earning its place."""
    return {name: dict(counts) for name, counts in _STATS.items()}


def _count(tool_name: str, outcome: str) -> None:
    _STATS.setdefault(tool_name, {"hits": 0, "misses": 0})[outcome] += 1


def clear_tool_cache() -> None:
    """Drop every cached result and its counters. A test seam (the cache is
    process-global), matching ``jobs_service.clear_terminal_status_cache()``."""
    _CACHE.clear()
    _STATS.clear()
