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
