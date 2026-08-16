"""T60 Phase 3b: the ``+`` composition grammar (D5).

``region_dispatch`` is a table lookup -- a string is in the vocabulary or it is
not. This is a different kind of thing: a grammar that *constructs* a region
per request out of tokens that each have to resolve, with a failure mode
(``"NY + NJ + Wakanda"``) that must name which token failed. Keeping the two
apart leaves the PRESET tier readable as the one-sitting table it is.

**Why this is not called from ``region_dispatch.dispatch``**, which the phase
prompt recommended: ``dispatch`` is defined over an *already normalized* string,
and D11a fixes the order as **split on ``+`` first, then normalize each token**.
Normalizing first strips only the *leading* ``"the "``, so ``"the ny + nj"``
would resolve and ``"ny + the nj"`` would hard-fail on ``"the nj"`` -- two
spellings of one request, two outcomes. Routing composition through ``dispatch``
would mean handing it both the raw and the normalized form, muddying the
one-input contract that keeps ``dispatch`` simple. So this is its own gate,
ahead of normalization, on both resolvers and on the retrieval-plane wrapper.

Network-free, like ``region_dispatch``: every member is a checked-in Natural
Earth boundary and the geocoder is never consulted (D8).
"""
from dataclasses import dataclass
from typing import Any

from tta_backend.datasets.us_states import US_STATES

# D5: symbol-only. No natural-language "and"/"or" -- no legitimate place name
# contains a literal "+", so the split can never collide with a real place,
# and no disambiguation heuristic is needed that could itself misfire
# ("New York and New Jersey" versus "Trinidad and Tobago").
SEPARATOR = "+"

# Tension 2, resolved to option (b) by gate V17: a postal code is a composite
# *member*, never a bare place name. Measured live, five of seven ambiguous
# codes resolve to a foreign country today -- DE Germany, CA Canada, IN India,
# LA Laos, ME Montenegro -- and each is a correct answer to a different
# question. Claiming them bare would swap a right region for a confident wrong
# one under a name the user typed, which is the D12b regression exactly.
#
# The ``+`` is what supplies the missing context: nobody writes "DE + NJ"
# meaning Germany plus New Jersey.
#
# This table deliberately never enters ``global_regions``, so it cannot shadow
# a preset and ``AliasCollisionError`` stays green by construction -- which
# matters, because "in", "or", "me", "de" and "la" as bare region names would
# be exactly what that guard was built to watch for.
POSTAL_TO_STATE: dict[str, str] = {
    state["postal"].lower(): key for key, state in US_STATES.items()
}


# D16's extent gate, and the number is deliberately NOT a new one.
#
# Risk 6 is what it bounds: a sparse composite's envelope is not its footprint.
# "CA + NY" unions two small polygons into a continent-wide box and
# _crop_to_mask_footprint crops to that envelope, not per-part -- the render
# path OOM T50/T59 already met, reachable in one typed string.
#
# The resolver runs BEFORE any cube exists, so it cannot count real grid cells;
# it has an envelope in degrees. The conversion is anchored to this codebase's
# own full-TEMPO-domain figure (frame_stack.py: 2950 x 5771 = 17.0 M cells)
# over the TEMPO field of regard, lat 14-73 N and lon 168-13 W:
#
#   17,024,450 cells / (59 deg x 155 deg) = 1,861.6 cells/deg^2
#
# Gate V15 measured every candidate against the two ceilings already decided
# here, and MAX_FRAME_NATIVE_CELLS is the only one that fits:
#
#   NY + NJ              88,731    hawaii (bare)      389,240
#   CA + NY           1,218,220    alaska (bare)    1,776,288
#   CA + ME           1,594,902    alaska + florida 8,563,388   <- refused
#   all 51 unioned   10,857,189                                 <- refused
#
# MAX_PLANE_NATIVE_CELLS (1,000,000) is excluded BY MEASUREMENT: it would
# refuse bare "alaska", a token Phase 3a shipped as working. And at 4,000,000
# no single 3a token trips the gate (Alaska, the worst, sits at 44% of it), so
# this is composite-only with no hole -- while still biting on 62 of the 1,275
# two-state pairs (4.9%), all of them genuinely continent-spanning.
#
# Deriving rather than restating is the point of tension 4: two constants both
# meaning "too big" can drift apart, and then a refusal names a limit that is
# not the one that fired. If a cheaper reduction ever raises the frame ceiling,
# this ceiling should move with it -- they answer the same question about the
# same container.
CELLS_PER_SQUARE_DEGREE = (2950 * 5771) / (59.0 * 155.0)


def _envelope_ceiling_deg2() -> float:
    from tta_backend.preprocessing.frame_stack import MAX_FRAME_NATIVE_CELLS

    return MAX_FRAME_NATIVE_CELLS / CELLS_PER_SQUARE_DEGREE


MAX_COMPOSITE_ENVELOPE_DEG2 = _envelope_ceiling_deg2()


@dataclass(frozen=True)
class CompositeResult:
    """Claimed or not, in the shape ``region_dispatch.DispatchResult`` uses.

    There is no third "claimed and failed" state here, and that is the whole
    of D14: a composition that fails *raises*, because the caller has to be
    told **which token** failed and a ``None`` cannot carry that.
    """

    claimed: bool
    region: dict[str, Any] | None = None


NOT_CLAIMED = CompositeResult(claimed=False)


def is_composite(raw_name: str) -> bool:
    """Is this string syntactically a composition (D8)?

    Deliberately answered on the **raw** string and on the separator alone.
    Once this returns ``True`` the geocoder is off the table for good, even on
    partial failure -- so the question has to be decided before any token is
    known to resolve, or a bad token would silently reopen the fallback.
    """
    return isinstance(raw_name, str) and SEPARATOR in raw_name


def split_tokens(raw_name: str, normalize) -> list[str]:
    """D11a's order, and it is the whole trap: **split first, normalize each**.

    ``normalize`` is ``RegionResolver._normalize_location_name`` passed in, not
    re-implemented -- T42's known bug was the sync and async paths normalizing
    differently, and a dispatcher with its own casing/whitespace/``"the "``
    handling reintroduces it one layer up.
    """
    return [normalize(token) for token in raw_name.split(SEPARATOR)]


def _member_geometry(token: str, resolver) -> Any:
    """Resolve one token to a member boundary, or ``None`` if it is not one.

    D15's order, minus the country tier that is Phase 4's: **U.S. state by
    postal code or full name**. A postal hit resolves *through* the canonical
    state key, so ``"NY + NJ"`` and ``"new york + new jersey"`` build from
    byte-identical member objects -- the same rule ``ALIASES`` follows.
    """
    key = POSTAL_TO_STATE.get(token, token)
    if key not in US_STATES:
        return None
    preset = resolver.global_regions.get(key)
    if preset is None:
        return None
    return resolver._finalize_preset(preset, key)


def _malformed(raw_name: str):
    """D15: an empty member is a *syntax* mistake, not a vocabulary one.

    ``"NY +"``, ``"+ NJ"`` and ``"NY + + NJ"`` all produce an empty token, and
    reporting that as ``"'' is not a U.S. state"`` would send someone hunting
    for a state they never typed. Kept separate from ``_unresolved_token`` so
    the two mistakes get two answers.
    """
    from tta_backend.earthdata_mcp.results import CATEGORY_USER_INPUT, MCPToolError

    return MCPToolError(
        CATEGORY_USER_INPUT,
        f"'{raw_name}' has an empty member, so it is not a valid composite "
        "region.",
        suggestion="Write each member between the '+' signs, as in 'NY + NJ'.",
    )


def _unresolved_token(token: str, raw_name: str):
    """D8's refusal, and D14's channel for it.

    Naming the token is the entire point. Over the old ``dict | None`` contract
    every call site collapsed the failure to
    ``"Could not resolve location: 'NY + NJ + Wakanda'"`` -- which tells the
    researcher (and the model, which could otherwise retry with a fix) nothing
    about *which* of the three words was the problem.

    The message also names what the token was tried against, which D15 makes
    possible by closing the vocabulary: presets and coalitions are **not**
    members, so there is a finite, statable list.
    """
    from tta_backend.earthdata_mcp.results import CATEGORY_USER_INPUT, MCPToolError

    return MCPToolError(
        CATEGORY_USER_INPUT,
        f"'{token}' in '{raw_name}' is not a U.S. state, so the composite "
        "region could not be built.",
        suggestion=(
            "Composite members are U.S. states, named by postal code ('NY') or "
            "in full ('new york'). Presets and coalitions such as 'conus' or "
            "'otc' cannot be combined with '+'."
        ),
    )


def _too_large(raw_name: str, bounds) -> "MCPToolError":
    """D16's refusal. Names the envelope, the estimate, **and** the limit that
    fired -- a message quoting a number nothing compared against is how a
    refusal ends up misleading the person reading it."""
    from tta_backend.earthdata_mcp.results import CATEGORY_TOO_LARGE, MCPToolError
    from tta_backend.preprocessing.frame_stack import MAX_FRAME_NATIVE_CELLS

    deg2 = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    return MCPToolError(
        CATEGORY_TOO_LARGE,
        f"'{raw_name}' spans a {deg2:,.0f} deg^2 bounding box "
        f"(~{deg2 * CELLS_PER_SQUARE_DEGREE:,.0f} native cells at TEMPO L3 "
        f"resolution), above the {MAX_FRAME_NATIVE_CELLS:,}-cell limit this "
        "deployment can process.",
        suggestion=(
            "The members are far apart, so their combined bounding box is much "
            "larger than the region itself. Ask for fewer, or nearer, members."
        ),
    )


def _check_extent(raw_name: str, bounds) -> None:
    if (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) > MAX_COMPOSITE_ENVELOPE_DEG2:
        raise _too_large(raw_name, bounds)


def dispatch_composite_extent(raw_name: str, resolver):
    """The retrieval plane's answer for a ``+`` string (tension 3, D13).

    Not optional, and for a stronger reason than the coalitions had. A
    coalition at least has a *name* the MCP's geocoder might resolve (badly --
    "OTC" is an aerodrome in Chad). A composite has **no key at all**: it is
    constructed per request, and D8 forbids the geocoder from ever seeing the
    string. So there is nothing behind this. Unclaimed here means the AOI step
    either fails outright or resolves the raw string to something unrelated,
    after which the mask clips a cube that never covered the region and
    nothing says so (Risk 5).

    Returns the union's **envelope** as a bbox, by the same argument Phase 1.5
    pinned and 3a re-proved over 51 states: retrieval must *contain* the mask,
    not equal it. The extent gate has already run inside ``dispatch_composite``,
    so an envelope too large to process is refused here too rather than
    retrieved and then refused downstream.
    """
    from tta_backend.utils.region_dispatch import (
        EXTENT_NOT_CLAIMED, ExtentDispatch, _bbox_string,
    )

    composed = dispatch_composite(raw_name, resolver)
    if not composed.claimed:
        return EXTENT_NOT_CLAIMED
    if composed.region is None:
        # Not reachable today -- ``dispatch_composite`` raises rather than
        # returning claimed-with-None, which is the whole of D14. Guarded
        # anyway because the alternative here is fail-OPEN: returning
        # EXTENT_NOT_CLAIMED would send the raw "+" string to the MCP's own
        # Nominatim, which is precisely what D8 forbids.
        raise _malformed(raw_name)
    # Formatted by region_dispatch's own helper, so the two planes cannot
    # disagree about the wire format of a bbox.
    return ExtentDispatch(claimed=True, location=_bbox_string(composed.region["bounds"]))


def dispatch_composite(raw_name: str, resolver) -> CompositeResult:
    """Build the union for a ``+`` string, or decline to claim it."""
    from shapely.ops import unary_union

    if not is_composite(raw_name):
        return NOT_CLAIMED

    tokens = split_tokens(raw_name, resolver._normalize_location_name)
    if any(not token for token in tokens):
        raise _malformed(raw_name)

    members = []
    for token in tokens:
        member = _member_geometry(token, resolver)
        if member is None:
            raise _unresolved_token(token, raw_name)
        members.append(member)

    geometry = unary_union([member["geometry"] for member in members])
    _check_extent(raw_name, geometry.bounds)
    names = [member["name"] for member in members]
    return CompositeResult(claimed=True, region={
        "geometry": geometry,
        "bounds": geometry.bounds,
        "name": " + ".join(names),
        "display_name": f"{' + '.join(names)} (composite of {len(names)} U.S. states)",
        # D10: a construction, not a named place. D10a: the same fact is
        # written to ``region_origin`` as well, because ``region_type`` is the
        # *rasterization* slot and ``apply_mask_region_type`` overwrites it.
        "region_type": "composite_union",
        "region_origin": "composite_union",
    })
