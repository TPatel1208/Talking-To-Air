"""
datasets/preset_collections.py
================================
A small, hand-picked list of well-known datasets for prompt guidance only —
suggestions to help the model pick a reasonable starting point, never a
ceiling on what can be discovered. Any dataset findable via search_datasets
is fair game; this list exists so the model doesn't have to search from a
blank slate for the handful of collections that come up constantly.

Each preset's identifiers (concept_id, short_name) are pulled from the
dataset registry (collections.yaml) by key, so the table the agent sees can
never drift from the registered collection it stands for. The agent is told
to search by **concept_id**: a CMR concept_id resolves to exactly one
collection (live-verified 2026-07-11 — every registry concept_id returns a
single rank-0 search hit). The previous human-facing labels
("MODIS_AOD_TERRA", "OMI_NO2", ...) were not real CMR short_names and
returned *zero* search results, so the agent free-ranged and — for AOD —
landed on unsupported products (HDF4 MCD19A2CMG, MERRA-2) instead of the
registered L3 grid. Even the real short_names are ambiguous (OMHCHOd,
OMI_MINDS_NO2d resolve to the wrong or a non-top collection), which is why
the query key is the concept_id, not the short_name.
"""
from __future__ import annotations

from datasets.registry import load_registry

# (registry key, human-facing description). The concept_id and short_name are
# read from the registry entry under each key, never hand-copied here.
_PRESETS: list[tuple[str, str]] = [
    ("OMI_NO2", "Tropospheric NO2 column, daily, global"),
    ("TROPOMI_NO2", "NO2 column, monthly, global"),
    ("TEMPO_NO2", "Tropospheric NO2 vertical column, hourly, North America"),
    ("TEMPO_O3TOT", "Total ozone column, hourly, North America"),
    ("OMI_O3", "Total ozone column, daily, global"),
    ("TEMPO_HCHO", "HCHO vertical column, hourly, North America"),
    ("OMI_HCHO", "HCHO vertical column, daily, global"),
    ("MODIS_AOD_TERRA", "Aerosol optical depth at 550nm, daily, global"),
]


def get_preset_collections() -> list[dict[str, str]]:
    """Build the preset suggestion list from the registry, so every row's
    ``concept_id``/``short_name`` is exactly the registered collection's — a
    preset can never again point at a label that resolves to nothing (or to
    the wrong product).

    collections.yaml is the file whose header promises "no code changes
    needed", so a typo in a preset's key must not fail as a bare ``KeyError``
    that names nothing useful. Validate every key up front and raise one clear
    error naming *all* the missing keys, so dataset onboarding is as safe as
    the header claims (T44)."""
    reg = load_registry()
    missing = [key for key, _ in _PRESETS if key not in reg]
    if missing:
        raise ValueError(
            "preset_collections references registry key(s) that collections.yaml "
            f"does not define: {', '.join(missing)}. Fix the key(s) in "
            "datasets/preset_collections.py or add the collection(s) to collections.yaml."
        )
    return [
        {
            "concept_id": reg[key].collection_id,
            "short_name": reg[key].short_name,
            "description": description,
        }
        for key, description in _PRESETS
    ]


def __getattr__(name: str):
    """Resolve ``PRESET_COLLECTIONS`` lazily (PEP 562), so importing this
    module can never turn a collections.yaml data error into an import-time
    cascade — the named validation error above surfaces on first *access*
    instead, where a caller can catch and report it."""
    if name == "PRESET_COLLECTIONS":
        return get_preset_collections()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
