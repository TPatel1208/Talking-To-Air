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

_DATASETS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tta_backend", "datasets",
)
OUT_PATH = os.path.join(_DATASETS, "preset_regions.geojson")
# T60/Phase 3a: the states need a bounding box that does *not* require reading
# the GeoJSON, because a ``global_regions`` entry has to exist even when the
# asset does not -- otherwise a missing asset drops 51 keys silently instead of
# degrading them to honest rectangles the way every other preset does. Emitted
# from the same fetch, in the same run, so the two artifacts cannot drift; a
# test asserts they agree on all 51 keys and bounds rather than trusting it.
STATES_OUT_PATH = os.path.join(_DATASETS, "us_states.py")

# Natural Earth's own ``type`` for a US admin-1 unit, mapped to the
# parenthetical the resolved region cites. Phase 3a gate V12 measured why this
# is not cosmetic: ``"washington"`` geocodes live to Washington DC with 0.00%
# overlap with Washington State, and ``"new york"`` to New York City with
# 0.47% of the state. D15 gives both tokens to the state, so the answer has to
# say which "Washington" it meant -- in ``display_name``, which is what an
# answer cites (T42), not in a code comment.
ADMIN1_KINDS = {
    "State": ("state", "U.S. state"),
    "Federal District": ("federal_district", "U.S. federal district"),
}


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


EXPECTED_US_ADMIN1 = 51  # 50 states + DC, measured in the Phase 0 gate (V4)


def _us_admin1(features: list[dict]) -> list[dict]:
    """The 51 US admin-1 units -- 50 states plus DC -- in name order.

    Sorted so the emitted asset and table are byte-stable across rebuilds; the
    source's own ordering is not guaranteed, and an unstable diff on a 293 KB
    checked-in file makes every boundary revision unreviewable.

    Fails loudly on anything unexpected, for the same reason
    ``_coalition_features`` does: a mirror update that adds a territory or
    renames a state would otherwise ship a silently different vocabulary, and
    ``global_regions`` keys are derived from these names."""
    units = sorted(
        (f for f in features
         if f["properties"].get("admin") == "United States of America"),
        key=lambda f: f["properties"]["name"].lower(),
    )
    if len(units) != EXPECTED_US_ADMIN1:
        raise ValueError(
            f"admin-1 source has {len(units)} US units, expected "
            f"{EXPECTED_US_ADMIN1}; the mirror changed -- re-run the Phase 0 V4 "
            "measurement before moving this constant"
        )
    for field in ("name", "postal"):
        values = [f["properties"].get(field) for f in units]
        if len(set(values)) != EXPECTED_US_ADMIN1 or not all(values):
            raise ValueError(f"US admin-1 {field!r} values are not 51 distinct non-empty strings")
    unknown = {f["properties"]["type"] for f in units} - set(ADMIN1_KINDS)
    if unknown:
        raise KeyError(f"admin-1 source has unmapped type(s) {sorted(unknown)}; "
                       "add them to ADMIN1_KINDS with an explicit display label")
    return units


def _state_features(features: list[dict]) -> list[dict]:
    """One feature per US admin-1 unit, at the admin-1 tolerance (D2a).

    Undissolved -- these are the same units ``_coalition_features`` merges,
    kept individually. Phase 0's V4 settled 0.01 deg on DC, which retains
    100.17% of its area there and only **58.71%** at the builder's 0.1
    constant."""
    out = []
    for f in _us_admin1(features):
        props = f["properties"]
        name, postal = props["name"], props["postal"]
        kind, label = ADMIN1_KINDS[props["type"]]
        out.append(_feature(
            name.lower(), name, shape(f["geometry"]), ADMIN1_TOLERANCE,
            extra_properties={
                "postal": postal,
                "kind": kind,
                # Carried into the asset so the disambiguation survives a
                # rebuild and is legible to anyone who opens the file.
                "display_name": f"{name} ({label})",
            },
        ))
    return out


def _write_states_table(state_features: list[dict]) -> None:
    """Emit ``datasets/us_states.py`` from the features just built.

    Bounds come from the *simplified* geometry -- the one that ships -- and are
    written as exact ``repr`` floats, not rounded, so the table and the polygon
    agree bit for bit. A table whose box were a hair inside the polygon would
    put the retrieval extent inside the mask, which is the Risk 5 this phase
    exists to close."""
    lines = [
        '"""Generated by scripts/build_preset_regions.py -- do not edit by hand.',
        "",
        "The 51 U.S. admin-1 units (50 states + DC) as RegionResolver presets.",
        "``bounds`` is the envelope of the simplified polygon in",
        "``preset_regions.geojson``, emitted in the same run from the same",
        "source, so the two cannot drift; tests assert they still agree.",
        '"""',
        "",
        "US_STATES: dict[str, dict] = {",
    ]
    for feature in state_features:
        props = feature["properties"]
        bounds = shape(feature["geometry"]).bounds
        lines.append(
            f'    {feature["id"]!r}: {{'
            f'"name": {props["name"]!r}, '
            f'"postal": {props["postal"]!r}, '
            f'"kind": {props["kind"]!r},'
        )
        lines.append(
            f'        "display_name": {props["display_name"]!r}, '
            f'"bounds": {tuple(bounds)!r}}},'
        )
    lines.append("}")
    lines.append("")
    # encoding is explicit: the default on this platform truncates on the first
    # character it cannot encode, silently.
    with open(STATES_OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {STATES_OUT_PATH} ({os.path.getsize(STATES_OUT_PATH)} bytes, "
          f"{len(state_features)} states)")


def _coalition_features(features: list[dict]) -> list[dict]:
    """Dissolve each T60 coalition from whole 50m admin-1 US jurisdictions.

    Fails loudly on a missing postal code: a coalition silently short a member
    is precisely the confident-wrong-shape this PRD exists to prevent, and it
    would be invisible in the output."""
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

    # One fetch of the 2.3 MB admin-1 layer, shared by the coalitions (which
    # dissolve it) and the states (which keep it undissolved).
    admin1 = _load_features(ADMIN1_SOURCE_URL)
    out_features.extend(_coalition_features(admin1))

    state_features = _state_features(admin1)
    out_features.extend(state_features)
    _write_states_table(state_features)

    fc = {"type": "FeatureCollection", "features": out_features}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(fc, fh)
    print(f"wrote {OUT_PATH} ({os.path.getsize(OUT_PATH)} bytes, "
          f"{len(out_features)} features)")


if __name__ == "__main__":
    main()
