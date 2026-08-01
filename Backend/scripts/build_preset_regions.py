"""Generate Backend/data/preset_regions.geojson (T42 region fidelity).

One-off build step, not run at request time. Turns Natural Earth 110m
admin-0 countries into the handful of real preset boundaries the
RegionResolver ships: the United States (full, and clipped to CONUS to drop
Alaska/Hawaii), and each continent dissolved from its member countries by
the ``CONTINENT`` field. Geometries are simplified (preserve_topology) so
the checked-in file stays small; the source is fetched once here, never at
runtime.

Re-run with:  python scripts/build_preset_regions.py
Source: Natural Earth 110m (public domain), via the natural-earth-vector
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
CONUS_BOX = box(-125, 24, -66, 50)
# Continents we ship as presets, in the CONTINENT spelling Natural Earth uses.
CONTINENTS = [
    "North America", "South America", "Europe",
    "Africa", "Asia", "Oceania", "Antarctica",
]
OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tta_backend", "datasets", "preset_regions.geojson",
)


def _load_countries() -> list[dict]:
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as resp:
        return json.load(resp)["features"]


def _feature(preset_id: str, name: str, geom, tolerance: float) -> dict:
    simplified = geom.simplify(tolerance, preserve_topology=True)
    return {
        "type": "Feature",
        "id": preset_id,
        "properties": {"preset_id": preset_id, "name": name},
        "geometry": mapping(simplified),
    }


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

    fc = {"type": "FeatureCollection", "features": out_features}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(fc, fh)
    print(f"wrote {OUT_PATH} ({os.path.getsize(OUT_PATH)} bytes, "
          f"{len(out_features)} features)")


if __name__ == "__main__":
    main()
