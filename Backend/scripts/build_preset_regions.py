"""Generate Backend/data/preset_regions.geojson (T42 region fidelity, T60 coalitions).

One-off build step, not run at request time. Turns Natural Earth 110m
admin-0 countries into the handful of real preset boundaries the
RegionResolver ships: the United States (full, and clipped to CONUS to drop
Alaska/Hawaii), and each continent dissolved from its member countries by
the ``CONTINENT`` field. Geometries are simplified (preserve_topology) so
the checked-in file stays small; the source is fetched once here, never at
runtime.

T60 additionally dissolves *coalitions* -- named multi-jurisdiction regions
that are not OSM places and cannot be geocoded -- from the Natural Earth
**50m admin-1** layer, at a finer tolerance (see ``ADMIN1_TOLERANCE``).

Re-run with:  python scripts/build_preset_regions.py
Source: Natural Earth 110m + 50m (public domain), via the natural-earth-vector
GeoJSON mirror.
"""
import json
import os
import urllib.request

from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

SOURCE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_110m_admin_0_countries.geojson"
)
# T60/D2: coalitions dissolve *standalone jurisdictions*, so they need 50m,
# not the 110m the continent dissolves are built from.
ADMIN1_SOURCE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_50m_admin_1_states_provinces.geojson"
)
# T60/D2a: measured in the Phase 0 gate (docs/prds/prd-t60-phase0-gate-verdict.md
# V4). The builder's existing 0.1 constant retains only **58.71%** of DC's area
# -- and with Virginia excluded, DC's Potomac edge is on the OTR's outer
# boundary, so that distortion would land on the boundary itself. 0.01 (~1.1 km)
# holds DC to within 0.2% and costs 15.8 KB for the OTR polygon. The 110m
# continent/US features keep their existing tolerances, unchanged.
ADMIN1_TOLERANCE = 0.01
CONUS_BOX = box(-125, 24, -66, 50)
# Continents we ship as presets, in the CONTINENT spelling Natural Earth uses.
CONTINENTS = [
    "North America", "South America", "Europe",
    "Africa", "Asia", "Oceania", "Antarctica",
]

# T60: named coalitions, dissolved from whole admin-1 units by postal code.
# The ``properties`` travel into the checked-in asset so the approximation is
# legible in the data itself, not only in a code comment.
COALITIONS = {
    "otc": {
        "name": "Ozone Transport Region",
        # CAA 184(a) names these eleven States as whole units, plus DC.
        "members": ["CT", "DE", "ME", "MD", "MA", "NH", "NJ", "NY", "PA", "RI", "VT", "DC"],
        "properties": {
            "authority": "Clean Air Act 184(a) (42 U.S.C. 7511c(a))",
            "object": (
                "The Ozone Transport REGION. The Ozone Transport Commission is the "
                "body that governs it; the region is the only one of the two with a "
                "boundary. Virginia holds a seat on the Commission but the statute "
                "does not name it as a whole-State member of the Region."
            ),
            "coverage": (
                "The eleven whole States named by CAA 184(a) -- CT, DE, ME, MD, MA, "
                "NH, NJ, NY, PA, RI, VT -- plus the District of Columbia."
            ),
            "approximation": (
                "EXCLUDES the Virginia portion of the OTR: the Consolidated "
                "Metropolitan Statistical Area that includes the District of "
                "Columbia, i.e. Arlington, Fairfax, Loudoun, Prince William and "
                "Stafford Counties and the Cities of Alexandria, Fairfax, Falls "
                "Church, Manassas and Manassas Park (4,127 km2, ~2.5M people). "
                "50m admin-1 data cannot express a partial State. Including whole "
                "Virginia instead would over-include 98,689 km2 -- 96.0% of the "
                "State, +20.6% of the region, and more land than CT+DE+RI+DC+NH+"
                "VT+MA combined. See docs/prds/prd-t60-phase0-gate-verdict.md, V1."
            ),
        },
    },
    "new england": {
        "name": "New England",
        "members": ["CT", "ME", "MA", "NH", "RI", "VT"],
        "properties": {
            "authority": "Conventional definition: six whole States.",
            "coverage": "CT, ME, MA, NH, RI, VT -- exact, no approximation.",
        },
    },
}

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tta_backend", "datasets", "preset_regions.geojson",
)


def _load_features(url: str) -> list[dict]:
    with urllib.request.urlopen(url, timeout=300) as resp:
        return json.load(resp)["features"]


def _load_countries() -> list[dict]:
    return _load_features(SOURCE_URL)


def _feature(preset_id: str, name: str, geom, tolerance: float,
             extra_properties: dict | None = None) -> dict:
    simplified = geom.simplify(tolerance, preserve_topology=True)
    properties = {"preset_id": preset_id, "name": name}
    properties.update(extra_properties or {})
    return {
        "type": "Feature",
        "id": preset_id,
        "properties": properties,
        "geometry": mapping(simplified),
    }


def _coalition_features() -> list[dict]:
    """Dissolve each T60 coalition from whole 50m admin-1 US jurisdictions.

    Fails loudly on a missing postal code: a coalition silently short a member
    is precisely the confident-wrong-shape this PRD exists to prevent, and it
    would be invisible in the output."""
    features = _load_features(ADMIN1_SOURCE_URL)
    by_postal = {
        f["properties"].get("postal"): shape(f["geometry"])
        for f in features
        if f["properties"].get("admin") == "United States of America"
    }

    out = []
    for preset_id, spec in COALITIONS.items():
        missing = [code for code in spec["members"] if code not in by_postal]
        if missing:
            raise KeyError(f"{preset_id}: admin-1 source is missing {missing}")
        dissolved = unary_union([by_postal[code] for code in spec["members"]])
        out.append(_feature(
            preset_id, spec["name"], dissolved, ADMIN1_TOLERANCE,
            extra_properties={"members": spec["members"], **spec["properties"]},
        ))
    return out


def main() -> None:
    features = _load_countries()
    by_admin = {f["properties"].get("ADMIN"): shape(f["geometry"]) for f in features}

    us = by_admin["United States of America"]
    out_features = [
        _feature("united states", "United States", us, 0.1),
        _feature("conus", "Continental US", us.intersection(CONUS_BOX), 0.1),
    ]

    for continent in CONTINENTS:
        members = [
            shape(f["geometry"])
            for f in features
            if f["properties"].get("CONTINENT") == continent
        ]
        dissolved = unary_union(members)
        out_features.append(_feature(continent.lower(), continent, dissolved, 0.2))

    out_features.extend(_coalition_features())

    fc = {"type": "FeatureCollection", "features": out_features}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(fc, fh)
    print(f"wrote {OUT_PATH} ({os.path.getsize(OUT_PATH)} bytes, "
          f"{len(out_features)} features)")


if __name__ == "__main__":
    main()
