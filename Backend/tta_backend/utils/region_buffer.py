"""T60 Phase 5: the ``"within N miles|km of X"`` grammar (D9).

**This module is the one region module that is NOT network-free**, and that is
why it is a separate file rather than a third grammar inside
``region_composition``. That module's docstring claims network-freedom in its
first paragraph and the claim is load-bearing -- every composite member is a
checked-in Natural Earth boundary and the geocoder is never consulted (D8).
Here ``X`` resolves through ``geocoding_service`` (D9), so the claim would stop
being true for the file that makes it. Keeping the I/O in its own module lets
``region_composition`` keep its contract honestly, and gives D11b's sync/async
fork exactly one place to live.

**The fork is one line wide.** Parsing, unit conversion, the AEQD projection,
the antimeridian and pole checks and the extent gate are all pure and shared;
only the geocode call differs between :func:`dispatch_buffer` (sync, reached by
``export_service``) and :func:`adispatch_buffer` (async, reached by every
analysis tool and by the retrieval-plane wrapper). Gate V24 verified all three
surfaces against the live code.
"""
import math
import re
from dataclasses import dataclass
from typing import Any

# D16, and V22 chose the spellings deliberately rather than accepting whatever
# a regex happened to admit. Both systems, because for an atmospheric-science
# tool "within 50 km of X" is at least as likely as miles -- and a bare number
# is **refused** rather than defaulted (see ``_missing_units``).
METRES_PER_UNIT: dict[str, float] = {
    "km": 1000.0,
    "kilometre": 1000.0, "kilometres": 1000.0,
    "kilometer": 1000.0, "kilometers": 1000.0,
    "mi": 1609.344,
    "mile": 1609.344, "miles": 1609.344,
}

# Which word the disclosure says back, keyed by what the metres mean. Echoing
# the user's own spelling would make "within 50 kilometres of X" and
# "within 50 km of X" -- one physical region -- cite themselves two ways.
_UNIT_LABEL = {1000.0: "km", 1609.344: "miles"}

# Syntactically a buffer: "within <something> of <something>". Deliberately
# loose, and asked *before* the strict parse, for the same reason
# ``is_composite`` answers on the separator alone (D8): once a string is
# recognisably a buffer request the geocoder is off the table, so a malformed
# radius has to reach a named refusal rather than falling through to Nominatim
# as a literal place name. Gate V23 measured what that fall-through costs --
# ``"within 50 miles of NYC"`` returns **zero** Nominatim hits, after which the
# agent silently substitutes a different region (T46 Phase 2, V6).
_SHAPE = re.compile(r"^within\s+.+\s+of\s+.+$")

# The strict parse. ``unit`` is optional in the *pattern* so that a bare number
# reaches ``_missing_units`` and gets told both units, instead of failing as an
# unparseable phrase that names neither.
_PARSE = re.compile(
    r"^within\s+(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>[a-z]*)\s+of\s+(?P<target>.+)$"
)

# V22, design tension 4: measured at r = 100 km, 40 N. 64 segments per quadrant
# is a 257-vertex ring retaining **99.988%** of pi r^2; 128 buys 0.007
# percentage points for double the vertices, and 32 costs 0.03. The 0.012%
# shortfall is roughly four orders of magnitude below one TEMPO L3 cell, and
# since ``dispatch_buffer_extent`` sends a bbox rather than geometry (V23,
# following Phase 1.5's V5/V7) the vertex count never reaches the wire.
QUAD_SEGS = 64


@dataclass(frozen=True)
class BufferResult:
    """Claimed or not, in the shape ``CompositeResult`` uses.

    There is no third "claimed and failed" state, for the same reason D14 gives
    the composite grammar: a buffer that fails *raises*, because the caller has
    to be told which part of the request was the problem -- the units, the
    target, the antimeridian, the size -- and a ``None`` carries none of that.
    """

    claimed: bool
    region: dict[str, Any] | None = None


NOT_CLAIMED = BufferResult(claimed=False)


@dataclass(frozen=True)
class BufferRequest:
    """A parsed request, before anything has been resolved or built. Pure, and
    shared by both twins -- the sync/async fork happens strictly after this.

    ``amount`` and ``unit`` are carried alongside ``metres`` rather than being
    reconstructed from it. Recovering "was this miles or km?" by testing
    ``metres % 1609.344`` is a float-equality question with no right answer:
    50 km is 50,000 m and 31.07 miles, and both readings are arithmetically
    defensible. The parser knows which the user wrote; nothing downstream
    should have to guess.
    """

    metres: float
    amount: float
    unit: str
    target: str
    raw_name: str

    @property
    def label(self) -> str:
        return f"{self.amount:g} {self.unit}"


def is_buffer(raw_name: str, normalize) -> bool:
    """Is this string syntactically a buffer request?

    Answered on the normalized string, but **after** ``is_composite`` has had
    it (V25): ``"within 50 km of NY + NJ"`` contains a ``+``, so the composition
    grammar claims it first and hard-fails naming the token -- which is the
    right answer, because D9 forbids ``X`` from resolving recursively through
    COMPOSITE and the Non-Goals list recursive CUSTOM composition. Reversing
    the order would silently turn that refusal into a buffer around a region
    D9 does not allow.
    """
    return isinstance(raw_name, str) and bool(_SHAPE.match(normalize(raw_name)))


def parse_buffer(raw_name: str, normalize) -> BufferRequest:
    """Parse a string ``is_buffer`` has already claimed, or raise naming why."""
    match = _PARSE.match(normalize(raw_name))
    if match is None:
        raise _malformed(raw_name)

    unit = match.group("unit")
    if not unit:
        raise _missing_units(raw_name, match.group("amount"))
    if unit not in METRES_PER_UNIT:
        raise _unknown_units(raw_name, unit)

    amount = float(match.group("amount"))
    metres = amount * METRES_PER_UNIT[unit]
    if metres <= 0:
        raise _malformed(raw_name)
    return BufferRequest(
        metres=metres,
        amount=amount,
        unit=_UNIT_LABEL[METRES_PER_UNIT[unit]],
        target=match.group("target"),
        raw_name=raw_name,
    )


def build_buffer(latitude: float, longitude: float, metres: float):
    """D9's geometry: project to a local azimuthal-equidistant CRS centred on
    the point, buffer in metres, project back.

    Pure, and shared by both twins. The alternative D9 rejects -- a degree
    buffer with a fixed miles-per-degree constant -- was measured in the gate
    and retains 76.5% of the intended area at 40 N, 50.2% at 60 N and 17.5% at
    80 N, because a longitude degree shrinks by ``cos(latitude)``. This
    approach holds 99.99% at every latitude tested including 89 N.
    """
    from pyproj import CRS, Transformer
    from shapely.geometry import Point
    from shapely.ops import transform

    wgs84 = CRS.from_epsg(4326)
    aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={latitude} +lon_0={longitude} "
        "+datum=WGS84 +units=m +no_defs"
    )
    to_aeqd = Transformer.from_crs(wgs84, aeqd, always_xy=True).transform
    to_wgs84 = Transformer.from_crs(aeqd, wgs84, always_xy=True).transform
    disc = transform(to_aeqd, Point(longitude, latitude)).buffer(
        metres, quad_segs=QUAD_SEGS
    )
    return transform(to_wgs84, disc)


def _region(request: BufferRequest, geo_result: dict, geometry) -> dict:
    """The resolved region, in ``dispatch_composite``'s shape.

    ``display_name`` cites the **geocoder's own label** and says the shape is a
    construction. Gate V25 measured ``"Springfield"`` resolving confidently to
    Springfield, Illinois out of some thirty candidates -- the existing hazard
    of every bare place name, not one this grammar introduces, and the existing
    mitigation is exactly this: put the geocoder's answer where a reader can
    catch "Paris, Texas".
    """
    name = f"{request.label} around {geo_result['display_name']}"
    return {
        "geometry": geometry,
        "bounds": geometry.bounds,
        "name": name,
        # The radius is already in ``name``; what ``display_name`` adds is the
        # T42 disclosure that this is a *constructed* shape and how it was
        # measured -- "geodesic" being the load-bearing word, since the same
        # phrase computed in degrees would be half this size at 60 N.
        "display_name": f"{name} (geodesic buffer)",
        # D10: a request-time construction, not a named place, and explicitly
        # NOT ``point_buffer`` -- which already means "we could not find a
        # boundary, so here is a 0.1 deg box". Labelling a deliberate,
        # precisely-sized buffer with it would invert the disclosure.
        "region_type": "buffer",
        # D10a: the same fact in the slot ``apply_mask_region_type`` does not
        # overwrite. A small buffer on a coarse grid self-heals to
        # ``boundary_cells``, which is the *likely* path, not a corner.
        "region_origin": "buffer",
    }


def _check_pole(request: BufferRequest, geo_result: dict) -> None:
    """V22, decided separately from the antimeridian, and checked **first**.

    A buffer containing a pole is not a distorted polygon, it is an
    unrepresentable one. Measured at 89.5 N with r = 200 km: the pole is 55.8
    km away, so it is physically inside -- and the AEQD polygon's maximum
    latitude is **88.7094**, because going 200 km north passes over the pole
    and comes back down the far meridian at that latitude. The shape does not
    cover the pole and **does not cover its own centre**. The region wanted is
    a polar cap with a hole at the top; no lat/lon polygon expresses it, and
    splitting at +/-180 (option (b)) does not help -- run on this case it
    returns a ``GeometryCollection`` with the same two holes.

    Ordered ahead of the antimeridian check because such a ring wraps 360
    degrees of longitude as well, so that check would also fire, truthfully,
    and send the reader to a fix that cannot work: no smaller radius helps if
    the centre is near a pole, and moving off the antimeridian does nothing.

    Decided on distance to the pole rather than on the geometry, so the
    refusal states the real reason and is independent of how the ring came
    out. At 85 N the pole is 558.5 km away and a 200 km buffer is ordinary.
    """
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    longitude, latitude = geo_result["longitude"], geo_result["latitude"]
    for pole, name in ((90.0, "North Pole"), (-90.0, "South Pole")):
        _, _, distance = geod.inv(longitude, latitude, longitude, pole)
        if distance > request.metres:
            continue

        from tta_backend.earthdata_mcp.results import CATEGORY_USER_INPUT, MCPToolError

        raise MCPToolError(
            CATEGORY_USER_INPUT,
            f"a {request.label} buffer around {geo_result['display_name']} "
            f"reaches the {name}, which is {distance / 1000:,.0f} km away. A "
            "region containing a pole cannot be expressed as a "
            "latitude/longitude polygon at all -- the shape would silently "
            "become a ring around the pole with the pole itself left out.",
            suggestion=(
                "Ask for a radius smaller than the distance to the pole, or "
                "name a bounding box directly."
            ),
        )


def _crosses_antimeridian(geometry) -> bool:
    """Does the ring wrap the long way round in a lat/lon frame?

    Measured against adjacent-vertex longitude jumps, and the two populations
    are not close: an ordinary buffer's largest jump is **0.024 deg** (NYC,
    50 mi) or **0.023 deg** (Suva, 100 km), while a crossing one reads
    **359.980 deg**. A factor of ~15,000 with nothing in between, so the 180
    threshold needs no tuning and cannot drift into a real request.

    Note this is not the same question as "is the bbox wide": a buffer at 89 N
    legitimately spans 127 deg of longitude (V22) without crossing anything.
    """
    for ring in _rings(geometry):
        lons = [x for x, _ in ring.coords]
        if any(abs(a - b) > 180 for a, b in zip(lons, lons[1:])):
            return True
    return False


def _rings(geometry):
    parts = getattr(geometry, "geoms", [geometry])
    return [part.exterior for part in parts if part.exterior is not None]


def _check_antimeridian(request: BufferRequest, geo_result: dict, geometry) -> None:
    """V22, option (a): refuse, rather than split.

    The shape being refused is not merely distorted -- it is inverted. Measured
    at 20 N, 179.5 E with r = 100 km: a point 10 km east reads ``False``, a
    point 10 km **west** reads ``False``, a point 150 km east (genuinely
    outside) reads ``True``, and so does (0 E, 20 N) off West Africa. There is
    no direction in which the polygon is right, and its area is 99.99% of
    pi r^2 the whole time.

    Splitting it at +/-180 (option (b)) fixes the mask and leaves the bounding
    box at ``-180 .. 180``: 1,210,757 cells retrieved for a 6,052-cell region,
    which D16's gate passes at 650.4 deg^2. Phase 4 already refuses the USA,
    Russia, Fiji, New Zealand, Kiribati and Antarctica as composite members for
    that exact property, and a request-time buffer does not get an exemption
    the checked-in countries are denied.

    What this costs was measured over nine real places near the antimeridian:
    **nothing at all at 50 km or 100 km**; a 250 km buffer at Suva or Anadyr,
    and 500 km at four Pacific cities. TEMPO's field of regard (lon 168 W-13 W)
    does not reach the antimeridian at all.
    """
    if not _crosses_antimeridian(geometry):
        return

    from tta_backend.earthdata_mcp.results import CATEGORY_USER_INPUT, MCPToolError

    raise MCPToolError(
        CATEGORY_USER_INPUT,
        f"a {request.label} buffer around {geo_result['display_name']} crosses "
        "the antimeridian (180 degrees longitude), where it cannot be "
        "represented as a single latitude/longitude polygon -- the shape would "
        "silently become a band wrapped the wrong way around the globe, "
        "excluding the area you asked for and including most of the planet.",
        suggestion=(
            "Ask for a smaller radius, or centre the request on a place that "
            "does not sit within that distance of the 180th meridian."
        ),
    )


def _resolved(request: BufferRequest, geo_result) -> dict:
    """Everything after the geocode -- shared, so the two twins differ in one
    line and cannot drift apart on the checks."""
    if geo_result is None:
        raise _unresolved_target(request)
    _check_pole(request, geo_result)
    geometry = build_buffer(
        geo_result["latitude"], geo_result["longitude"], request.metres
    )
    _check_antimeridian(request, geo_result, geometry)
    _check_extent(request, geo_result, geometry)
    return _region(request, geo_result, geometry)


def dispatch_buffer(raw_name: str, resolver) -> BufferResult:
    """The sync twin (D11b). Reached by ``export_service``, which calls
    ``resolve_location`` -- verified against the live code in gate V24."""
    request = _claim(raw_name, resolver)
    if request is None:
        return NOT_CLAIMED
    return BufferResult(
        claimed=True,
        region=_resolved(request, resolver.geocoding_service.geocode(request.target)),
    )


async def adispatch_buffer(raw_name: str, resolver) -> BufferResult:
    """The async twin (D11b). Reached by every analysis tool and by the
    retrieval-plane wrapper, both of which are ``async`` -- a single blocking
    twin would put ``requests.get(timeout=15)`` on the event loop, the hazard
    the sync/async split exists to prevent."""
    request = _claim(raw_name, resolver)
    if request is None:
        return NOT_CLAIMED
    geo_result = await resolver.geocoding_service.ageocode(request.target)
    return BufferResult(claimed=True, region=_resolved(request, geo_result))


async def adispatch_buffer_extent(raw_name: str, resolver):
    """The retrieval plane's answer for a buffer phrase (D13, gate V23).

    Not optional, and the argument is stronger than the composite's. A
    coalition has a name the MCP's geocoder might resolve, badly -- ``"OTC"``
    is an aerodrome in Chad. A composite has no key but does contain real place
    names. A buffer phrase has **nothing**: measured live in the gate,
    ``"within 50 miles of NYC"`` returns *zero* Nominatim hits. Unclaimed, the
    AOI step fails outright, and T46 Phase 2's V6 measured what the agent does
    next -- it silently substitutes a different region and retrieves that,
    after which the mask clips a cube that never covered the buffer (Risk 5).

    Returns the buffer's **envelope** as a bbox, by the invariant Phase 1.5
    pinned and 3a/3b re-proved: retrieval must *contain* the mask, not equal
    it. Sends a bbox rather than geometry because V5/V7 measured a 519-vertex
    OTR polygon buying **zero** granule reduction over its bbox while costing
    12,951 bytes on every call and in the T53 cache key.

    Async only, and that is not an omission: the retrieval-plane wrapper's
    ``_call`` is itself ``async def`` (gate V24), so there is no caller for a
    sync twin here and adding one would put a blocking geocode within reach of
    the event loop for no purpose.

    Because V22 refuses the antimeridian and polar cases before this point,
    every buffer that arrives has an honest, non-wrapping bounding box -- the
    V22 refusal is what makes this function a one-liner. Had the split option
    been taken, this would have had to send ``-180,...,180``.
    """
    from tta_backend.utils.region_dispatch import EXTENT_NOT_CLAIMED, ExtentDispatch

    buffered = await adispatch_buffer(raw_name, resolver)
    if not buffered.claimed:
        return EXTENT_NOT_CLAIMED
    if buffered.region is None:
        # Not reachable today -- ``adispatch_buffer`` raises rather than
        # returning claimed-with-None, which is the whole of D14. Guarded
        # anyway, and guarded by *raising*, because the alternative here is
        # fail-OPEN: returning EXTENT_NOT_CLAIMED would send the raw phrase to
        # the MCP's own Nominatim, which is what D8 forbids. The same guard
        # ``dispatch_composite_extent`` carries, for the same reason.
        raise _malformed(raw_name)
    return ExtentDispatch(claimed=True, location=_outward_bbox(buffered.region["bounds"]))


def _outward_bbox(bounds) -> str:
    """``"W,S,E,N"`` at ``_bbox_string``'s six decimals, rounded **outward**.

    Found by the containment test rather than reasoned about in advance: a
    200 km buffer at 85 N has ``maxx = 21.009686000335122``, and
    ``_bbox_string``'s ``f"{v:.6f}"`` renders that as ``21.009686`` -- 3.4e-10
    degrees *inside* the mask. Physically 0.04 mm, and completely irrelevant to
    what is retrieved; but "retrieval contains the mask" is the invariant Phase
    1.5 pinned and 3a/3b re-proved by mutation, and an invariant that holds
    except by a rounding error is not one. Flooring the minima and ceiling the
    maxima makes it exact, at a cost of at most 1.1e-6 degrees (~11 cm) of
    over-retrieval per edge.

    **Why this is not pushed down into ``_bbox_string``.** That helper is
    shared with the coalition, state and composite planes, whose geometries
    come from checked-in assets and happen to round the other way; changing it
    would alter four shipped extents (outward, so still containing -- but
    silently) and contradict 3b's assertion that the composite bbox equals
    ``round(bounds, 6)`` exactly. A buffer is the one region built at request
    time out of full-precision floats, so it is the one that needs this. The
    general case is worth revisiting deliberately, not as a side effect here.
    """
    west, south, east, north = bounds
    return ",".join([
        f"{math.floor(west * 1e6) / 1e6:.6f}",
        f"{math.floor(south * 1e6) / 1e6:.6f}",
        f"{math.ceil(east * 1e6) / 1e6:.6f}",
        f"{math.ceil(north * 1e6) / 1e6:.6f}",
    ])


def _claim(raw_name: str, resolver) -> BufferRequest | None:
    """Claim, parse and vet the target, or decline. Network-free, and the whole
    of what the twins share before they diverge."""
    if not is_buffer(raw_name, resolver._normalize_location_name):
        return None
    request = parse_buffer(raw_name, resolver._normalize_location_name)
    _check_not_a_named_region(request, resolver)
    return request


def _named_region_vocabulary(resolver) -> set[str]:
    """Every token that already names a *region* rather than a point.

    Assembled from the live tables rather than written out, so a coalition or
    alias added later is covered the day it lands -- the same reason
    ``ambiguous_member_tokens`` derives its collisions instead of listing them.
    """
    from tta_backend.utils.plotting import load_admin0_polygons
    from tta_backend.utils.region_composition import POSTAL_TO_STATE
    from tta_backend.utils.region_dispatch import ALIASES, COALITIONS

    return (
        set(resolver.global_regions) | set(COALITIONS) | set(ALIASES)
        | set(POSTAL_TO_STATE) | set(load_admin0_polygons())
    )


def _check_not_a_named_region(request: BufferRequest, resolver) -> None:
    """D9: ``X`` resolves **only** as a single geocoded point, never
    recursively through PRESET or COMPOSITE (a Non-Goal).

    Without this, ``"within 50 km of otc"`` geocodes the bare string ``"otc"``,
    which the Phase 0 gate measured resolving live to **an aerodrome in Chad**
    -- shipped as ``region_type: buffer``, faithfully disclosed, and wrong by a
    continent. That is D12b's failure mode with a new label on it.

    Checked before the geocode, so a refused request costs no network call and
    no Nominatim rate-limit slot.

    This costs nothing a researcher would type: all 90 ``global_regions`` keys
    are regions -- continents, sub-continents, countries, U.S. states -- with
    **not one city among them** (gate V25), so ``"within 50 km of paris"``,
    ``"...of nyc"`` and ``"...of chicago"`` are untouched.
    """
    target = resolver._normalize_location_name(request.target)
    if target not in _named_region_vocabulary(resolver):
        return

    from tta_backend.earthdata_mcp.results import CATEGORY_USER_INPUT, MCPToolError

    raise MCPToolError(
        CATEGORY_USER_INPUT,
        f"'{request.target}' names a region, not a point, so a "
        f"{request.label} buffer around it is not something this tool can "
        "build -- buffering a whole boundary is a different operation from "
        "buffering a location.",
        suggestion=(
            f"Name a place to centre the buffer on, as in "
            f"'within {request.label} of Boston' -- or ask for "
            f"'{request.target}' on its own."
        ),
    )


def _unresolved_target(request: BufferRequest):
    """D8's rule, carried to this grammar: once a string is syntactically a
    buffer, the geocoder never sees the *phrase*.

    Gate V23 measured ``"within 50 miles of NYC"`` returning **zero** Nominatim
    hits, so falling through would not even produce a wrong answer -- it would
    produce a failed AOI step, after which T46 Phase 2's V6 measured the agent
    silently substituting a different region and retrieving that instead.
    """
    from tta_backend.earthdata_mcp.results import CATEGORY_USER_INPUT, MCPToolError

    return MCPToolError(
        CATEGORY_USER_INPUT,
        f"'{request.target}' in '{request.raw_name}' could not be found, so "
        "there is no point to build the buffer around.",
        suggestion=(
            "Check the spelling, or name a nearby town or city instead."
        ),
    )


def _check_extent(request: BufferRequest, geo_result: dict, geometry) -> None:
    """D16's gate, reusing ``MAX_COMPOSITE_ENVELOPE_DEG2`` rather than minting a
    third constant -- two numbers both meaning "too big" drift apart, and then
    a refusal names a limit that is not the one that fired.

    Measured against the 2,148.7 deg^2 ceiling: 50 km of NYC is 1.1, 50 miles
    2.8, 500 miles 276.5, 1000 miles 1,113.1 -- and 3000 miles is **11,270.2
    deg^2 = 20,980,751 cells**, the PRD's own example.

    The message is new even though the number is not: ``_too_large`` says the
    members are far apart and to ask for nearer ones, which is meaningless
    advice for a buffer, whose only knob is the radius.
    """
    from tta_backend.earthdata_mcp.results import CATEGORY_TOO_LARGE, MCPToolError
    from tta_backend.preprocessing.frame_stack import MAX_FRAME_NATIVE_CELLS
    from tta_backend.utils.region_composition import (
        CELLS_PER_SQUARE_DEGREE, MAX_COMPOSITE_ENVELOPE_DEG2,
    )

    west, south, east, north = geometry.bounds
    deg2 = (east - west) * (north - south)
    if deg2 <= MAX_COMPOSITE_ENVELOPE_DEG2:
        return

    raise MCPToolError(
        CATEGORY_TOO_LARGE,
        f"a {request.label} buffer around {geo_result['display_name']} spans a "
        f"{deg2:,.0f} deg^2 bounding box (~{deg2 * CELLS_PER_SQUARE_DEGREE:,.0f} "
        f"native cells at TEMPO L3 resolution), above the "
        f"{MAX_FRAME_NATIVE_CELLS:,}-cell limit this deployment can process.",
        suggestion="Ask for a smaller radius.",
    )


def _malformed(raw_name: str):
    from tta_backend.earthdata_mcp.results import CATEGORY_USER_INPUT, MCPToolError

    return MCPToolError(
        CATEGORY_USER_INPUT,
        f"'{raw_name}' looks like a distance request but the radius could not "
        "be read.",
        suggestion="Write it as 'within 50 miles of Boston' or 'within 50 km of Boston'.",
    )


def _missing_units(raw_name: str, amount: str):
    """D16, and the gate's tension-3 decision: a bare number **refuses**.

    Defaulting silently is a 50%-wrong answer for half of users -- 50 miles is
    80.5 km, not a rounding error -- and D12b is the standing precedent against
    replacing an unknown with a confident guess. So the message names both
    units rather than picking one.
    """
    from tta_backend.earthdata_mcp.results import CATEGORY_USER_INPUT, MCPToolError

    return MCPToolError(
        CATEGORY_USER_INPUT,
        f"'{raw_name}' does not say whether {amount} means miles or "
        "kilometres, and the two differ by 61%.",
        suggestion=(
            f"Write 'within {amount} miles of ...' or "
            f"'within {amount} km of ...'."
        ),
    )


def _unknown_units(raw_name: str, unit: str):
    from tta_backend.earthdata_mcp.results import CATEGORY_USER_INPUT, MCPToolError

    return MCPToolError(
        CATEGORY_USER_INPUT,
        f"'{unit}' in '{raw_name}' is not a distance unit this tool accepts.",
        suggestion=(
            "Use miles ('mi', 'mile', 'miles') or kilometres "
            "('km', 'kilometre', 'kilometres')."
        ),
    )
