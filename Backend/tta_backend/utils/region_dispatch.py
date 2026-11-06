"""Region dispatch: how polygon-only presets (coalitions, aliases, states)
get resolved.

``RegionResolver``'s exact-match gate only covers ``global_regions`` keys,
and coalitions deliberately have none (a coalition's bounding box would
swallow West Virginia, most of Ohio and part of Ontario). This runs *ahead*
of that lookup and answers in three states, not two (see ``DispatchResult``).

Preset tier only; the ``+`` and buffer grammars live in region_composition and
region_buffer. A pure, network-free lookup so both resolvers share one copy.
"""
from dataclasses import dataclass
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tta_backend.utils.plotting import RegionResolver

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
    into the second would turn a missing asset into a Nominatim query for
    ``"OTC"``, which resolves to an *airport in Chad* labelled
    ``region_type: polygon``. Fail-closed here is not a
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
    is where the approximation is disclosed -- not in a code comment,
    where no researcher will read it.
    """

    name: str
    display_name: str


# Coalition ids are feature ids in datasets/preset_regions.geojson, emitted by
# scripts/build_preset_regions.py. None of these may have a
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

# Normalization happens *before* lookup and is ``RegionResolver``'s own,
# so "the OTC", "  OTC  " and "otc" all arrive here as "otc" and
# neither resolver can normalize differently than the other.
ALIASES: dict[str, str] = {
    "otr": "otc",
    "ozone transport commission": "otc",
    "ozone transport region": "otc",
    "ozone transport commission region": "otc",
    "ozone transport commission (otc) region": "otc",
    "ozone transport commission (otc)": "otc",
    "otc region": "otc",
    "otr region": "otc",
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

    The third state exists for the same reason as on the mask side: falling
    through on a missing asset hands ``"OTC"`` to the MCP's geocoder, which
    returns an aerodrome in **Chad**, and the mask then clips a cube that never
    covered the region.
    """

    claimed: bool
    location: str | None = None


EXTENT_NOT_CLAIMED = ExtentDispatch(claimed=False)


def _bbox_string(bounds: Sequence[float]) -> str:
    """``"W,S,E,N"`` decimal degrees, not GeoJSON.

    Sending the polygon fetches the same granules as its bbox, while adding
    ~13 KB to every call and cache key.

    The envelope is 2.453x the polygon's area, and that over-inclusion is
    *correct*: the mask clips it. Retrieval must contain the mask, not equal
    it. Revisit only if someone measures Harmony-side polygon subsetting
    reducing materialized bytes; the AOI response already reports ``source``
    (``bbox`` vs ``geojson``), so the switch would be visible.
    """
    return ",".join(f"{v:.6f}" for v in bounds)


def dispatch_extent(normalized_name: str, global_regions: dict[str, Any]) -> ExtentDispatch:
    """Translate a dispatch vocabulary word into an extent the MCP can consume.

    ``normalized_name`` must already have been through
    ``RegionResolver._normalize_location_name``, so both planes
    normalize identically and cannot disagree about what string they were
    handed.

    Scope is ``COALITIONS`` plus ``ALIASES`` plus the 51 U.S. admin-1 units,
    and nothing else. Any *other* ``global_regions`` key typed directly
    ("northeast us", "north america", "paris") is *not* claimed and reaches the
    MCP unchanged; those already work.

    *Aliases* are claimed even when they point at a plain preset: the mask
    knows "northeastern us" means the ``northeast us`` box, but the MCP
    geocodes it to a railway platform in Boston, so the cube wouldn't cover
    the mask.

    **States** are claimed too: "new york" and "washington" geocode to the
    cities, and even correct geocodes use OSM edges that don't contain the
    Natural Earth mask. Deriving both from one polygon fixes that. A state reaches the ``global_regions`` branch below, so
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


def assert_no_alias_collisions(global_regions: dict[str, Any]) -> None:
    """Fail loudly if the hand-maintained tables shadow anything.

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


def dispatch(normalized_name: str, resolver: "RegionResolver") -> DispatchResult:
    """Resolve ``normalized_name`` if it is in this module's vocabulary.

    ``normalized_name`` must already have been through
    ``RegionResolver._normalize_location_name``. ``resolver`` supplies
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
            # Claimed and failed. The caller must NOT geocode.
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
    canonical = resolver.global_regions.get(target)
    if canonical is None:
        return DispatchResult(claimed=True, region=None)
    return DispatchResult(claimed=True, region=resolver._finalize_preset(canonical, target))
