"""T60 region dispatch: the door a polygon-only preset can get through.

``RegionResolver`` has historically had exactly one gate --
``if location_lower in self.global_regions`` -- with everything else falling
through to Nominatim. ``_POLYGON_PRESET_IDS`` is consulted *inside*
``_finalize_preset``, i.e. only after that gate has already passed: it can
upgrade a preset to a real polygon, but it can never reach one. So a
checked-in coalition polygon with no ``global_regions`` key -- which is
exactly what D3 requires, since a coalition's bounding box would silently
swallow West Virginia, most of Ohio and part of Ontario -- was unreachable.

This module is that missing door. It runs *ahead* of the existing exact-match
lookup and answers in three states, not two (see ``DispatchResult``).

Scope: PRESET tier only. The ``+`` composition grammar (D5) and the CUSTOM
buffer grammar (D9) are later phases; this module is deliberately a pure,
network-free lookup so both resolvers can share one copy of it (D11b's
sync/async fork only becomes necessary when CUSTOM lands).
"""
from dataclasses import dataclass
from typing import Any

from tta_backend.datasets.us_states import US_STATES


@dataclass(frozen=True)
class DispatchResult:
    """Three outcomes, which an ``Optional`` cannot express.

    - **claimed, with a region** -- here is the answer.
    - **not claimed** (``NOT_CLAIMED``) -- "paris", "north america", anything
      outside this module's vocabulary. The caller falls through to exactly
      its previous behavior.
    - **claimed, region is None** -- the string matched a known coalition or
      alias and then failed to produce a polygon (missing or corrupt asset).

    The third state is the reason this is not a ``dict | None``. Collapsing it
    into the second is the specific bug D3b exists to prevent: a missing asset
    would become a Nominatim query for the string ``"OTC"``, which resolves
    live to an *airport in Chad* and is handed back labelled
    ``region_type: polygon`` (Phase 0 gate, V2). Fail-closed here is not a
    degraded answer versus a good one -- it is a not-found versus a mask over
    the wrong continent.
    """

    claimed: bool
    region: dict[str, Any] | None = None


NOT_CLAIMED = DispatchResult(claimed=False)


@dataclass(frozen=True)
class _Coalition:
    """A named multi-jurisdiction region that is not an OSM place.

    ``name`` is the short label a plot title should carry; ``display_name`` is
    the string the *answer* should cite, and for an approximated coalition it
    is where the approximation is disclosed (D3a) -- not in a code comment,
    where no researcher will read it.
    """

    name: str
    display_name: str


# Coalition ids are feature ids in datasets/preset_regions.geojson, emitted by
# scripts/build_preset_regions.py. By D3 none of these may have a
# ``global_regions`` entry; ``assert_no_alias_collisions`` enforces that.
COALITIONS: dict[str, _Coalition] = {
    "otc": _Coalition(
        name="Ozone Transport Region",
        # The V1 verdict, disclosed at the point of use. The polygon is the
        # eleven whole States CAA 184(a) names, plus DC; the statute's twelfth
        # component is the DC-area CMSA, which 50m admin-1 cannot express.
        display_name=(
            "Ozone Transport Region (CAA 184(a): 11 whole states + DC; "
            "excludes the Northern Virginia portion of the region)"
        ),
    ),
    "new england": _Coalition(
        name="New England",
        # Six whole states -- exact, so nothing to disclose.
        display_name="New England (CT, ME, MA, NH, RI, VT)",
    ),
}

# Alias -> canonical target, which is either a coalition id above or a
# ``global_regions`` preset key. Hand-curated, never fuzzy-matched (D12):
# token overlap on a closed vocabulary risks a confident wrong preset, which
# is worse than a geocode miss.
#
# Normalization happens *before* lookup and is ``RegionResolver``'s own
# (D11a), so "the OTC", "  OTC  " and "otc" all arrive here as "otc" and
# neither resolver can normalize differently than the other.
ALIASES: dict[str, str] = {
    # The Commission is the body; the Region is the boundary. Both spellings
    # land on the region, because the region is the only one of the two that
    # has a footprint.
    "otr": "otc",
    "ozone transport commission": "otc",
    "ozone transport region": "otc",
    # D12: pure phrasing variance. Today "northeastern us" misses the preset
    # dict entirely and geocodes to a *railway platform* at Northeastern
    # University in Boston, ~11 m across (Phase 0 gate, V2).
    "northeastern us": "northeast us",
}


@dataclass(frozen=True)
class ExtentDispatch:
    """The retrieval plane's answer, in the same three states as
    ``DispatchResult`` and for the same reason.

    - **claimed, with a location** -- send this to ``define_area_of_interest``
      instead of what the caller passed.
    - **not claimed** (``EXTENT_NOT_CLAIMED``) -- outside this module's
      vocabulary. The caller sends its original string, byte-identical.
    - **claimed, location is None** -- a known coalition or alias that failed
      to produce a footprint.

    The third state exists here for exactly the reason D3b gives on the mask
    plane, one plane over. Falling through on a missing asset hands ``"OTC"``
    to the MCP's own geocoder, which is Nominatim, which returns an aerodrome
    in **Chad** (Phase 0 gate, V2). Phase 1 fails closed on the mask; the
    retrieval plane must not fail open behind it, or the mask clips a cube that
    never covered the region and nothing says so.
    """

    claimed: bool
    location: str | None = None


EXTENT_NOT_CLAIMED = ExtentDispatch(claimed=False)


def _bbox_string(bounds) -> str:
    """``"W,S,E,N"`` decimal degrees -- the channel the Phase 1.5 gate chose.

    V5 measured both channels live: the MCP accepts the 12,951-byte OTR
    GeoJSON whole (``vertex_count: 519``, untruncated) *and* derives from it
    the identical ``bbox`` this string carries. V7 then measured what the
    difference buys -- ``estimate_retrieval_size`` returned **81 granules for
    both**, so the polygon costs 12,951 bytes on every call (and in the T53
    cache key) for a measured zero reduction in what is fetched.

    The envelope is 2.453x the polygon's area, and that over-inclusion is
    *correct*: the mask clips it. Retrieval must contain the mask, not equal
    it. Revisit only if someone measures Harmony-side polygon subsetting
    reducing materialized bytes; the AOI response already reports ``source``
    (``bbox`` vs ``geojson``), so the switch would be visible.
    """
    return ",".join(f"{v:.6f}" for v in bounds)


def dispatch_extent(normalized_name: str, global_regions: dict) -> ExtentDispatch:
    """Translate a T60 vocabulary word into an extent the MCP can consume.

    ``normalized_name`` must already have been through
    ``RegionResolver._normalize_location_name`` (D11a), so both planes
    normalize identically and cannot disagree about what string they were
    handed.

    Scope is ``COALITIONS`` plus ``ALIASES`` plus the 51 U.S. admin-1 units,
    and nothing else. Any *other* ``global_regions`` key typed directly
    ("northeast us", "north america", "paris") is *not* claimed and reaches the
    MCP byte-identical to today; those already work, and translating them would
    be a blast radius no phase has measured.

    The *alias* forms are claimed even when they point at a plain preset, and
    that is not an inconsistency -- it is Risk 5. Phase 1 taught the mask plane
    that "northeastern us" means the ``northeast us`` box; the MCP still
    geocodes that string to an ~11 m railway platform in Boston (Phase 0, V2).
    Leaving the alias untranslated is precisely the two-plane divergence where
    the mask clips against a cube that never covered it.

    The **states** are claimed for the same reason, on a measurement rather
    than on symmetry (Phase 3a gate, V13). Two of five probed tokens retrieved
    a different place outright -- "new york" as New York City, "washington" as
    Washington DC, 0.00% overlap. But the deciding finding was the three that
    geocoded *correctly* and still failed containment: Georgia covers 99.99% of
    the shipped mask and its retrieved extent still stops 0.018 deg short on
    the west edge, because the mask is Natural Earth 50m and the extent was
    OSM. Two providers, two sets of edges. No geocoder improvement closes that
    -- only both planes deriving from one polygon, which is what claiming the
    state here does. A state reaches the ``global_regions`` branch below, so
    the extent it sends is the same envelope ``_finalize_preset`` masks with.
    """
    from tta_backend.utils.plotting import load_preset_polygons

    target = ALIASES.get(normalized_name, normalized_name)
    if (target not in COALITIONS
            and normalized_name not in ALIASES
            and normalized_name not in US_STATES):
        return EXTENT_NOT_CLAIMED  # not our vocabulary -- caller sends its own string

    if target in COALITIONS:
        polygon = load_preset_polygons().get(target)
        if polygon is None:
            return ExtentDispatch(claimed=True)  # claimed and failed -- do NOT geocode
        return ExtentDispatch(claimed=True, location=_bbox_string(polygon.bounds))

    preset = global_regions.get(target)
    bounds = preset.get("bounds") if preset else None
    if not bounds:
        return ExtentDispatch(claimed=True)
    return ExtentDispatch(claimed=True, location=_bbox_string(bounds))


class AliasCollisionError(RuntimeError):
    """An alias or coalition id shadows something that already resolves."""


def assert_no_alias_collisions(global_regions: dict) -> None:
    """Fail loudly if the hand-maintained tables shadow anything (D12a).

    A table that silently shadows ``"georgia"`` or ``"us"`` is the cheapest
    possible way to reintroduce a confident wrong region, and review is not a
    reliable check for it. Called from ``RegionResolver.__init__`` -- the
    tables are only meaningful relative to ``global_regions``, which is
    instance state, so this is the earliest honest moment to run it.
    """
    for coalition_id in COALITIONS:
        if coalition_id in global_regions:
            raise AliasCollisionError(
                f"coalition id {coalition_id!r} shadows a global_regions preset; "
                "D3 requires coalitions to have no bounding-box fallback"
            )
    for alias, target in ALIASES.items():
        if alias in global_regions:
            raise AliasCollisionError(
                f"alias {alias!r} shadows the global_regions preset of the same name"
            )
        if alias in COALITIONS:
            raise AliasCollisionError(
                f"alias {alias!r} shadows the coalition of the same name"
            )
        if target not in COALITIONS and target not in global_regions:
            raise AliasCollisionError(
                f"alias {alias!r} points at {target!r}, which resolves to nothing"
            )


def dispatch(normalized_name: str, resolver) -> DispatchResult:
    """Resolve ``normalized_name`` if it is in this module's vocabulary.

    ``normalized_name`` must already have been through
    ``RegionResolver._normalize_location_name`` (D11a). ``resolver`` supplies
    ``global_regions`` and ``_finalize_preset`` -- every region this module
    returns is built by ``_finalize_preset``, never assembled here, so an
    alias is indistinguishable from its canonical key (same ``region_type``,
    same copy semantics) and there is only ever one definition of what a
    region is.
    """
    from tta_backend.utils.plotting import load_preset_polygons

    target = ALIASES.get(normalized_name, normalized_name)
    if target not in COALITIONS and normalized_name not in ALIASES:
        return NOT_CLAIMED  # not our vocabulary -- caller proceeds as before

    if target in COALITIONS:
        coalition = COALITIONS[target]
        polygon = load_preset_polygons().get(target)
        if polygon is None:
            # D3b: claimed and failed. The caller must NOT geocode.
            return DispatchResult(claimed=True, region=None)
        preset = {
            "geometry": polygon,
            "bounds": polygon.bounds,
            "name": coalition.name,
            "display_name": coalition.display_name,
        }
        return DispatchResult(claimed=True, region=resolver._finalize_preset(preset, target))

    # An alias onto an existing preset: resolve *through* the canonical key so
    # the object is identical to the one that key produces.
    preset = resolver.global_regions.get(target)
    if preset is None:
        return DispatchResult(claimed=True, region=None)
    return DispatchResult(claimed=True, region=resolver._finalize_preset(preset, target))
