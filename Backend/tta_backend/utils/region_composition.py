"""The ``+`` composition grammar ("NY + NJ").

``region_dispatch`` is a table lookup -- a string is in the vocabulary or it is
not. This is a different kind of thing: a grammar that *constructs* a region
per request out of tokens that each have to resolve, with a failure mode
(``"NY + NJ + Wakanda"``) that must name which token failed. Keeping the two
apart leaves the PRESET tier readable as the one-sitting table it is.

**Why this is not called from ``region_dispatch.dispatch``**: ``dispatch``
takes an *already normalized* string, but composition must **split on ``+``
first, then normalize each token**. Normalizing first strips only the *leading* ``"the "``, so ``"the ny + nj"``
would resolve and ``"ny + the nj"`` would hard-fail on ``"the nj"`` -- two
spellings of one request, two outcomes. Routing composition through ``dispatch``
would mean handing it both the raw and the normalized form, muddying the
one-input contract that keeps ``dispatch`` simple. So this is its own gate,
ahead of normalization, on both resolvers and on the retrieval-plane wrapper.

Network-free, like ``region_dispatch``: every member is a checked-in Natural
Earth boundary and the geocoder is never consulted.
"""
from dataclasses import dataclass
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from tta_backend.datasets.us_states import US_STATES

if TYPE_CHECKING:  # the runtime imports stay function-local (circularity);
    # this one only feeds the annotations on the refusal factories below.
    from tta_backend.earthdata_mcp.results import MCPToolError
    from tta_backend.utils.plotting import RegionResolver
    from tta_backend.utils.region_dispatch import ExtentDispatch

# Symbol-only. No natural-language "and"/"or" -- no legitimate place name
# contains a literal "+", so the split can never collide with a real place,
# and no disambiguation heuristic is needed that could itself misfire
# ("New York and New Jersey" versus "Trinidad and Tobago").
SEPARATOR = "+"

# A postal code is only a composite *member*, never a bare place name: bare
# codes like DE, CA, IN, LA and ME correctly geocode to countries (Germany,
# Canada, ...), and claiming them would swap a right answer for a wrong one.
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


# Country spelling -> the normalized ``ADMIN`` it means. Hand-curated and never
# fuzzy-matched (no edit distance).
#
# Needed because ``ADMIN`` names often aren't what people type: of 61
# plausible spellings, 20 missed. Some
# are Natural Earth being formal where nobody else is (``United Republic of
# Tanzania``, ``Hong Kong S.A.R.``, ``Republic of Serbia``); some are ordinary
# English usage the source simply does not carry (``UK``, ``Britain``,
# ``Holland``, ``UAE``); some are the previous name of a country that renamed
# (``Burma``, ``Swaziland``, ``Czech Republic``, ``Macedonia``, ``Cape Verde``).
#
# Deliberately small, and every entry is a *synonym for one country*, never a
# guess between two. ``"congo"`` is absent on purpose: it is genuinely ambiguous
# between ``Republic of the Congo`` and ``Democratic Republic of the Congo``, so
# it stays unresolvable and hard-fails naming the token rather than picking one.
#
# ``ivory coast`` and ``south korea`` are absent because they are already
# ``ADMIN`` values -- an alias that shadows the thing it points at is the
# collision ``assert_no_country_collisions`` exists to catch.
COUNTRY_ALIASES: dict[str, str] = {
    "uk": "united kingdom",
    "britain": "united kingdom",
    "great britain": "united kingdom",
    "holland": "netherlands",
    "uae": "united arab emirates",
    "burma": "myanmar",
    "czech republic": "czechia",
    "swaziland": "eswatini",
    "cape verde": "cabo verde",
    "macedonia": "north macedonia",
    "turkiye": "turkey",
    "timor-leste": "east timor",
    "tanzania": "united republic of tanzania",
    "hong kong": "hong kong s.a.r.",
    "macau": "macao s.a.r",
    "macao": "macao s.a.r",
    "serbia": "republic of serbia",
    "vatican city": "vatican",
    "holy see": "vatican",
}

# Escape hatch for the *country* side of a state/country collision
# ("georgia (country)"). The state side already has one (the postal code);
# without this, a disambiguation error could offer no way to say the country.
#
# A suffix rather than an alias-table row, because ``ambiguous_member_tokens``
# derives its collisions from the two vocabularies -- so the hatch has to work
# for a collision that does not exist yet and therefore cannot have a row.
COUNTRY_SUFFIX = "(country)"


# Extent gate, derived from an existing limit rather than a new number.
#
# A sparse composite's envelope isn't its footprint: "CA + NY" unions two small
# polygons into a continent-wide box, and the crop uses that envelope, so one
# typed string could reach the render OOM.
#
# The resolver runs BEFORE any cube exists, so it cannot count real grid cells;
# it has an envelope in degrees. The conversion is anchored to this codebase's
# own full-TEMPO-domain figure (frame_stack.py: 2950 x 5771 = 17.0 M cells)
# over the TEMPO field of regard, lat 14-73 N and lon 168-13 W:
#
#   17,024,450 cells / (59 deg x 155 deg) = 1,861.6 cells/deg^2
#
# Measured against the existing ceilings, only MAX_FRAME_NATIVE_CELLS fits:
#
#   NY + NJ              88,731    hawaii (bare)      389,240
#   CA + NY           1,218,220    alaska (bare)    1,776,288
#   CA + ME           1,594,902    alaska + florida 8,563,388   <- refused
#   all 51 unioned   10,857,189                                 <- refused
#
# MAX_PLANE_NATIVE_CELLS (1,000,000) would refuse bare "alaska". At 4,000,000
# no single state trips it, but 62 of 1,275 continent-spanning state pairs do.
#
# Derived, not restated, so the two "too big" limits can't drift apart. If the
# frame ceiling rises, this should rise with it.
CELLS_PER_SQUARE_DEGREE = (2950 * 5771) / (59.0 * 155.0)


def _envelope_ceiling_deg2() -> float:
    from tta_backend.preprocessing.frame_stack import MAX_FRAME_NATIVE_CELLS

    return MAX_FRAME_NATIVE_CELLS / CELLS_PER_SQUARE_DEGREE


MAX_COMPOSITE_ENVELOPE_DEG2 = _envelope_ceiling_deg2()


@dataclass(frozen=True)
class CompositeResult:
    """Claimed or not, in the shape ``region_dispatch.DispatchResult`` uses.

    There is no third "claimed and failed" state: a composition that fails
    *raises*, because the caller has to be told **which token** failed and a
    ``None`` cannot carry that.
    """

    claimed: bool
    region: dict[str, Any] | None = None


NOT_CLAIMED = CompositeResult(claimed=False)


def is_composite(raw_name: str) -> bool:
    """Is this string syntactically a composition?

    Deliberately answered on the **raw** string and on the separator alone.
    Once this returns ``True`` the geocoder is off the table for good, even on
    partial failure -- so the question has to be decided before any token is
    known to resolve, or a bad token would silently reopen the fallback.
    """
    return isinstance(raw_name, str) and SEPARATOR in raw_name


def split_tokens(raw_name: str, normalize: Callable[[str], str]) -> list[str]:
    """**Split first, then normalize each token** (the order matters).

    ``normalize`` is ``RegionResolver._normalize_location_name`` passed in, not
    re-implemented, so there's only one normalization to keep in sync.
    """
    return [normalize(token) for token in raw_name.split(SEPARATOR)]


def _member_geometry(token: str, resolver: "RegionResolver") -> Any:
    """Resolve one token to a member boundary, or ``None`` if it is not one.

    Order: **U.S. state by postal code or full name, then country by
    ``ADMIN``**. A postal hit resolves *through* the canonical state
    key, so ``"NY + NJ"`` and ``"new york + new jersey"`` build from
    byte-identical member objects -- the same rule ``ALIASES`` follows.

    The state tier is first and that is what makes ``"GA + France"`` resolve
    ``GA`` as Georgia rather than Gabon -- but the ordering is a tiebreak of
    last resort here, not the mechanism. The mechanism: **countries never
    match on an ISO code at all**, only on a name. 26 of 51 postal codes
    collide with real ISO alpha-2 codes, so a grammar
    that admitted codes would be ambiguous for over half the states; banning
    them removes the collision at the root instead of resolving it by order.
    """
    _check_unambiguous(token)
    key = POSTAL_TO_STATE.get(token, token)
    if key in US_STATES:
        preset = resolver.global_regions.get(key)
        if preset is None:
            return None
        # ``kind`` travels with the member so ``_composition_label`` can say
        # what the composite is made of without re-deriving it from the token,
        # which is the only place that knowledge is unambiguous.
        return dict(resolver._finalize_preset(preset, key), kind="state")
    return _country_geometry(token)


def assert_no_country_collisions(global_regions: dict[str, Any]) -> None:
    """The alias-collision guard, extended to ``COUNTRY_ALIASES``.

    Deliberately checks only what is knowable **without opening the asset**.
    ``RegionResolver.__init__`` calls this, and reading
    ``admin0_countries.geojson`` here would spend the 165 ms cold parse on
    every process that constructs a resolver -- including every request for
    ``"paris"`` -- which is the cost the second asset exists to avoid.

    The asset-dependent invariants are asserted where they belong instead: the
    builder refuses a duplicate feature id at build time, and tests pin that
    every alias target is a real ``ADMIN`` value, that no alias shadows one,
    and that no country name is a postal code. A checked-in asset cannot change
    between builds, so a build-time check and a test are a stronger guarantee
    than a startup check, not a weaker one.

    The country *names* are exempt from the preset check on purpose: ``georgia``
    and ``antarctica`` legitimately exist in both vocabularies, which is why
    they live in separate assets and why a country is member-only. The aliases
    are not exempt -- those are ours to choose.
    """
    from tta_backend.utils.region_dispatch import (
        ALIASES, COALITIONS, AliasCollisionError,
    )

    for alias, target in COUNTRY_ALIASES.items():
        for table, what in (
            (global_regions, "global_regions preset"),
            (ALIASES, "T60 alias"),
            (COALITIONS, "coalition"),
            (POSTAL_TO_STATE, "U.S. postal code"),
            (US_STATES, "U.S. state"),
        ):
            if alias in table:
                raise AliasCollisionError(
                    f"country alias {alias!r} (-> {target!r}) shadows the "
                    f"{what} of the same name; a token cannot mean two places"
                )


def ambiguous_member_tokens() -> set[str]:
    """Tokens that name both a U.S. state and a country -- **derived**.

    Computed from the two member vocabularies rather than hand-maintained, so
    a new collision fails closed as soon as the asset is rebuilt.

    Today this is exactly ``{"georgia"}``. ``"antarctica"`` is deliberately
    *not* here: it collides with a preset, and presets aren't composite
    members.
    """
    from tta_backend.utils.plotting import load_admin0_polygons

    return set(US_STATES) & set(load_admin0_polygons())


def _check_unambiguous(token: str) -> None:
    """Whether a token is an ambiguous state/country name.

    Asked of the **token**, never of the key it resolves through, so the
    escape hatch ``"GA"`` can't trip the guard.
    """
    if token not in ambiguous_member_tokens():
        return

    from tta_backend.earthdata_mcp.results import CATEGORY_USER_INPUT, MCPToolError

    # Both hatches are *derived*, not written out. A message that named "GA"
    # literally would be right for the only collision that exists today and
    # wrong for the next one, which is the failure mode Risk 2 describes.
    postal = US_STATES[token]["postal"]
    raise MCPToolError(
        CATEGORY_USER_INPUT,
        f"'{token}' names both a U.S. state and a country, so it cannot be "
        "resolved as a composite member without guessing which one you meant.",
        suggestion=(
            f"Write '{postal}' for the U.S. state, or "
            f"'{token} {COUNTRY_SUFFIX}' for the country."
        ),
    )


def _country_geometry(token: str) -> Any:
    """The country tier: a country, by name only.

    Countries are **member-only** vocabulary -- no ``global_regions`` entry, no
    ``_POLYGON_PRESET_IDS`` entry, no bounds table. Unlike states, many have no
    honest bounding box (Russia, Fiji, the USA span 360 degrees), so a bare
    ``"russia"`` still goes to the geocoder as before.

    So the member dict is built here rather than through ``_finalize_preset``:
    there is no preset to finalize, and routing a country through it would
    label it ``bounding_box`` (it is not in ``_POLYGON_PRESET_IDS``) for a
    geometry that is a real boundary.
    """
    from tta_backend.utils.plotting import load_admin0_polygons

    # The "(country)" escape hatch is a grammar form, not a table entry, so it
    # works for collisions nobody has written down yet. No ADMIN value ends in
    # "(country)", so it can't shadow a real name.
    if token.endswith(f" {COUNTRY_SUFFIX}"):
        token = token[: -len(COUNTRY_SUFFIX) - 1].strip()

    country = load_admin0_polygons().get(COUNTRY_ALIASES.get(token, token))
    if country is None:
        return None
    polygon = country["geometry"]
    return {"geometry": polygon, "bounds": polygon.bounds,
            "name": country["name"], "kind": "country"}


def _composition_label(kinds: list[str]) -> str:
    """What the composite is a composite *of*, for the region disclosure.

    Follows the actual members, so ``"Spain + Portugal"`` isn't described as
    a composite of U.S. states.
    """
    distinct = set(kinds)
    if distinct == {"state"}:
        noun = "U.S. states" if len(kinds) > 1 else "U.S. state"
    elif distinct == {"country"}:
        noun = "countries" if len(kinds) > 1 else "country"
    else:
        noun = "members"
    return f"composite of {len(kinds)} {noun}"


def _malformed(raw_name: str) -> "MCPToolError":
    """An empty member is a *syntax* mistake, not a vocabulary one.

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


def _unresolved_token(token: str, raw_name: str) -> "MCPToolError":
    """The refusal for a token that isn't a known member.

    Names the token, so the researcher (or the model, retrying) knows which
    word in ``'NY + NJ + Wakanda'`` was the problem.

    Also names what it was tried against: the member vocabulary is closed
    (presets and coalitions aren't members), so the list is finite.
    """
    from tta_backend.earthdata_mcp.results import CATEGORY_USER_INPUT, MCPToolError

    return MCPToolError(
        CATEGORY_USER_INPUT,
        f"'{token}' in '{raw_name}' is not a U.S. state or a country, so the "
        "composite region could not be built.",
        suggestion=(
            "Composite members are U.S. states -- by postal code ('NY') or in "
            "full ('new york') -- and countries by name ('France'), never by "
            "country code. Presets and coalitions such as 'conus' or 'otc' "
            "cannot be combined with '+'."
        ),
    )


def _too_large(raw_name: str, bounds: tuple[float, float, float, float]) -> "MCPToolError":
    """The too-large refusal. Names the envelope, the estimate, **and** the limit that
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


def _member_too_large(token: str, name: str, bounds: tuple[float, float, float, float]) -> "MCPToolError":
    """The too-large refusal for a *single* member that cannot fit on its own.

    Separate from ``_too_large``, which says the
    members are far apart and to ask for nearer ones -- true and useful for
    ``"Brazil + Japan"``, and **actively wrong** for ``"France + Germany"``,
    which is the case that made this branch necessary. France and Germany share
    a border. There are no nearer members. The user has done nothing to fix.

    The 8,524 deg^2 is French Guiana (-61.8) to Reunion (+55.8): ``ADMIN``
    France includes the overseas departments, which is correct and which no
    reader guesses from the word "France". So the message says the span out
    loud rather than leaving someone to wonder what they typed wrong.

    Ten of the 242 countries land here -- the USA, New Zealand, Russia,
    Antarctica, France, Kiribati, Canada, Fiji, the Netherlands and China. No
    U.S. state does: Alaska, the worst, is 954 deg^2 against a 2,148.7 ceiling.
    """
    from tta_backend.earthdata_mcp.results import CATEGORY_TOO_LARGE, MCPToolError
    from tta_backend.preprocessing.frame_stack import MAX_FRAME_NATIVE_CELLS

    west, south, east, north = bounds
    deg2 = (east - west) * (north - south)
    return MCPToolError(
        CATEGORY_TOO_LARGE,
        f"'{token}' ({name}) spans a {deg2:,.0f} deg^2 bounding box on its own "
        f"-- {west:.1f} to {east:.1f} longitude, {south:.1f} to {north:.1f} "
        f"latitude -- which is ~{deg2 * CELLS_PER_SQUARE_DEGREE:,.0f} native "
        f"cells at TEMPO L3 resolution, above the {MAX_FRAME_NATIVE_CELLS:,}-"
        "cell limit this deployment can process.",
        suggestion=(
            f"{name} covers territory far outside the part you probably mean "
            "-- overseas or antimeridian-crossing land included in the "
            "national boundary -- so its bounding box is much larger than the "
            "region itself. Name a smaller region, such as the individual "
            "states or provinces you are interested in."
        ),
    )


def _check_member_extent(token: str, member: dict[str, Any]) -> None:
    """Refuse a member whose *own* envelope cannot fit, before the union.

    Ordering is deliberate: checked per member as each resolves, so the refusal
    can name the token responsible. Checked after the union instead, all that
    is knowable is that the total was too big -- which is what 3b did, and is
    why ``"France + Germany"`` blamed the phrasing.
    """
    bounds = member["bounds"]
    if (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) > MAX_COMPOSITE_ENVELOPE_DEG2:
        raise _member_too_large(token, member["name"], bounds)


def _check_extent(raw_name: str, bounds: tuple[float, float, float, float]) -> None:
    if (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) > MAX_COMPOSITE_ENVELOPE_DEG2:
        raise _too_large(raw_name, bounds)


def dispatch_composite_extent(raw_name: str, resolver: "RegionResolver") -> "ExtentDispatch":
    """The retrieval plane's answer for a ``+`` string.

    Required: a composite has no name the MCP's geocoder could resolve, so
    without this the AOI step fails or resolves something unrelated, and the
    mask then clips a cube that never covered the region.

    Returns the union's **envelope** as a bbox: retrieval must *contain* the
    mask, not equal it. The extent gate has already run inside ``dispatch_composite``,
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
        # Unreachable (``dispatch_composite`` raises instead), but guarded:
        # falling through would send the raw "+" string to the MCP's geocoder.
        raise _malformed(raw_name)
    # Formatted by region_dispatch's own helper, so the two planes cannot
    # disagree about the wire format of a bbox.
    return ExtentDispatch(claimed=True, location=_bbox_string(composed.region["bounds"]))


def dispatch_composite(raw_name: str, resolver: "RegionResolver") -> CompositeResult:
    """Build the union for a ``+`` string, or decline to claim it."""
    from shapely.ops import unary_union

    if not is_composite(raw_name):
        return NOT_CLAIMED

    tokens = split_tokens(raw_name, resolver._normalize_location_name)
    if any(not token for token in tokens):
        raise _malformed(raw_name)

    members, kinds = [], []
    for token in tokens:
        member = _member_geometry(token, resolver)
        if member is None:
            raise _unresolved_token(token, raw_name)
        _check_member_extent(token, member)
        members.append(member)
        kinds.append(member["kind"])

    geometry = unary_union([member["geometry"] for member in members])
    _check_extent(raw_name, geometry.bounds)
    names = [member["name"] for member in members]
    return CompositeResult(claimed=True, region={
        "geometry": geometry,
        "bounds": geometry.bounds,
        "name": " + ".join(names),
        "display_name": f"{' + '.join(names)} ({_composition_label(kinds)})",
        # A construction, not a named place. Also written to ``region_origin``,
        # because ``apply_mask_region_type`` can overwrite ``region_type``.
        "region_type": "composite_union",
        "region_origin": "composite_union",
    })
