"""The instrument a satellite request names.

Every ``run_workflow`` call to a workflow with a ``named`` input must name
an instrument or product, so the answer can never describe data from a
different instrument than the one retrieved. It is the one the user named,
or, when they named none, one the supervisor picks from those that serve
the quantity. Code checks both: the name must match a registered product,
and an instrument the user wrote in their message must be the one passed.

Exploration is held to the instrument the user named too: when the message
names one, a specialist's satellite data call must name it, or a product of
it, instead of letting the registry choose or switching to another.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

from tta_backend.datasets.registry import CollectionConfig, load_registry
from tta_backend.harness.capabilities.products import match_named

NEEDS_INSTRUMENT = (
    "Pass named: the instrument for this request. Use the one the user named, as they wrote it; "
    "if they named none, choose one that serves {quantity}: {instruments}."
)
NOT_REGISTERED = "'{named}' names no registered product or instrument; pass one of: {instruments}."
USER_NAMED = "The user named {mentioned}; pass named as they wrote it, not '{named}'."
EXPLORE_UNNAMED = (
    "The user named {mentioned}; pass named: '{first}' so this call uses it. If it cannot serve the request, "
    "say so and offer another instrument instead of using one."
)
EXPLORE_OTHER = (
    "The user named {mentioned}, not '{named}'. Use {mentioned}; if it cannot serve the request, say so and "
    "offer another instrument instead of using one."
)
# The satellite calls that choose a product to read data from.
DATA_CALLS = frozenset({
    "sources.satellite.get_data", "sources.satellite.get_point_series", "sources.satellite.check_availability",
})
_QUOTED_CHARS = 60


def _short(instrument: str) -> str:
    """An instrument as the supervisor names it: the part before the platform, as in "OMI" for "OMI/Aura"."""
    return instrument.split("/")[0].strip()


def instruments(registry: Mapping[str, CollectionConfig] | None = None) -> dict[str, tuple[str, ...]]:
    """Each quantity's instruments in the registry, in rank order."""
    entries = registry if registry is not None else load_registry()
    ranked = sorted(entries.values(), key=lambda c: (c.quantity or "", c.rank if c.rank is not None else 999))
    found: dict[str, list[str]] = {}
    for cfg in ranked:
        if cfg.quantity and cfg.instrument:
            names = found.setdefault(cfg.quantity, [])
            if _short(cfg.instrument) not in names:
                names.append(_short(cfg.instrument))
    return {quantity: tuple(names) for quantity, names in sorted(found.items())}


def mentioned(message: str, registry: Mapping[str, CollectionConfig] | None = None) -> tuple[str, ...]:
    """The registered instruments the user's message names as whole words, in the order they appear.

    Only instrument names count, such as TEMPO or MODIS, never platforms such
    as Aqua or Terra, which are also ordinary words."""
    entries = registry if registry is not None else load_registry()
    words = {part.strip() for c in entries.values() if c.instrument for part in _short(c.instrument).split("+")}
    hits: list[tuple[int, str]] = []
    for word in words:
        if not word:
            continue
        found = re.search(rf"(?<![A-Za-z0-9]){re.escape(word)}(?![A-Za-z0-9])", message, re.IGNORECASE)
        if found:
            hits.append((found.start(), word))
    return tuple(word for _, word in sorted(hits))


def instrument_refusal(
    named: object, quantity: object, message: str, registry: Mapping[str, CollectionConfig] | None = None,
) -> str | None:
    """Why ``named`` cannot go into a satellite workflow call, or None when it can."""
    entries = registry if registry is not None else load_registry()
    serving = instruments(entries).get(quantity, ()) if isinstance(quantity, str) else ()
    every = sorted({name for names in instruments(entries).values() for name in names})
    offered = ", ".join(serving or every)
    if not isinstance(named, str) or not named.strip():
        return NEEDS_INSTRUMENT.format(quantity=quantity if isinstance(quantity, str) else "the quantity", instruments=offered)
    quoted = " ".join(named.split())[:_QUOTED_CHARS]
    keys, _ = match_named(named, entries)
    if not keys:
        return NOT_REGISTERED.format(named=quoted, instruments=offered)
    user = mentioned(message, entries)
    if not user:
        return None
    if any(word.lower() in _instrument_words(keys, entries) for word in user):
        return None
    return USER_NAMED.format(mentioned=" or ".join(user), named=quoted)


def _instrument_words(keys: tuple[str, ...], entries: Mapping[str, CollectionConfig]) -> set[str]:
    return {part.strip().lower() for key in keys for part in re.split(r"[/+]", entries[key].instrument or "")}


def explore_refusal(
    capability: str, params: Mapping[str, object], user: tuple[str, ...],
    registry: Mapping[str, CollectionConfig] | None = None,
) -> str | None:
    """Why a specialist's call ``capability`` with ``params`` may not run while the user named the
    instruments ``user``, or None when it may. A name matching no registered product is left to
    the capability, which refuses it with its own wording."""
    if not user or capability not in DATA_CALLS:
        return None
    entries = registry if registry is not None else load_registry()
    chosen = params.get("product") or params.get("named")
    said = " or ".join(user)
    if not isinstance(chosen, str) or not chosen.strip():
        return EXPLORE_UNNAMED.format(mentioned=said, first=user[0])
    keys, _ = match_named(chosen, entries)
    if not keys or any(word.lower() in _instrument_words(keys, entries) for word in user):
        return None
    return EXPLORE_OTHER.format(mentioned=said, named=" ".join(chosen.split())[:_QUOTED_CHARS])
