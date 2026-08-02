"""
datasets/registry.py
====================
Loads and validates the dataset registry from collections.yaml.

Usage
-----
    from tta_backend.datasets.registry import load_registry, CollectionConfig

    registry = load_registry()          # cached after first call
    col: CollectionConfig = registry["TEMPO_NO2"]
    print(col.collection_id, col.primary_var)

    # Check all available keys:
    print(list(registry.keys()))
"""

from __future__ import annotations

import math
import pathlib
import logging
from functools import lru_cache
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, field_validator, model_validator

logger = logging.getLogger(__name__)

_REGISTRY_PATH = pathlib.Path(__file__).parent / "collections.yaml"


class CollectionConfig(BaseModel):
    # ── Identity ──────────────────────────────────────────────────────────
    collection_id: str
    short_name:    str = ""
    version:       str = ""
    description:   str = ""
    cadence:       Literal["hourly", "daily", "monthly"] = "daily"
    # ── Provenance display (T32) ─────────────────────────────────────────────
    # Free-text, not parsed by any masking/aggregation logic -- surfaced as
    # "<provider> — <instrument>" in the Metadata tab's Source dataset block.
    provider:      str = ""
    instrument:    str = ""

    # ── Variable selection ────────────────────────────────────────────────
    primary_var:                 str
    quality_flag_var:            Optional[str] = None
    # T25 Phase 3: the Tier-1 pinned QA rule for quality_flag_var -- which
    # flag values count as good (or, equivalently, bad). At most one is
    # normally set; qa_good_values takes precedence if both are (datasets/
    # qa_flags.py::resolve_qa_info).
    qa_good_values:               Optional[list[int]] = None
    qa_bad_values:                Optional[list[int]] = None
    variables:                   list[str]     = []
    supports_variable_subsetting: bool         = False
    groups:                      list[str]     = []

    # ── Physical metadata ─────────────────────────────────────────────────
    units:     str
    fill_value: float
    valid_min:  float
    valid_max:  float

    @field_validator("fill_value", "valid_min", "valid_max", mode="before")
    @classmethod
    def _allow_inf(cls, v):
        """Accept YAML '.inf' / '-.inf' which PyYAML parses as float('inf')."""
        if isinstance(v, float):
            return v
        if isinstance(v, str):
            v = v.strip()
            if v in (".inf", "inf", "Inf"):
                return math.inf
            if v in ("-.inf", "-inf", "-Inf"):
                return -math.inf
        return float(v)

    @model_validator(mode="after")
    def _valid_range_makes_sense(self) -> "CollectionConfig":
        if self.valid_min > self.valid_max:
            raise ValueError(
                f"valid_min ({self.valid_min}) must be <= valid_max ({self.valid_max})"
            )
        return self


@lru_cache(maxsize=1)
def load_registry(path: str | None = None) -> dict[str, CollectionConfig]:
    """
    Load, validate, and cache the dataset registry.

    Parameters
    ----------
    path : optional override for the YAML file location (useful in tests).

    Returns
    -------
    dict mapping registry key (e.g. 'TEMPO_NO2') → CollectionConfig.

    Raises
    ------
    FileNotFoundError  if the YAML file is missing.
    ValidationError    if any entry fails Pydantic validation — caught at
                       startup rather than mid-request.
    """
    yaml_path = pathlib.Path(path) if path else _REGISTRY_PATH

    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset registry not found: {yaml_path}")

    raw: dict = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

    registry: dict[str, CollectionConfig] = {}
    errors: list[str] = []

    for key, values in raw.items():
        try:
            registry[key] = CollectionConfig(**values)
        except Exception as exc:
            errors.append(f"  [{key}] {exc}")

    if errors:
        raise ValueError(
            "Dataset registry validation failed:\n" + "\n".join(errors)
        )

    logger.info("Dataset registry loaded: %d collections", len(registry))
    return registry


# Collections established to publish no per-pixel quality flag variable at all,
# with how that was established. A collection missing from BOTH this record and
# collections.yaml's ``quality_flag_var`` masks nothing and says only "not
# applied -- semantics unknown", which reads identically to a product whose flag
# we simply forgot to pin. Keeping the two cases distinct is the difference
# between "there is nothing to apply" and "we never looked".
#
# This matters more than it looks: measured 2026-08-02, the Tier-2 CF path has
# never fired on a real NASA product (see tests/test_collection_registry_qa_
# coverage.py), so an unpinned collection is an unmasked collection.
COLLECTIONS_WITHOUT_QUALITY_FLAG: dict[str, str] = {
    "TEMPO_O3TOT": (
        "granule-verified 2026-08-02 (TEMPO_O3TOT_L3_V04_20260802T011507Z_S017): "
        "16 merged bands, none carrying CF flag_values/flag_meanings"
    ),
    "MODIS_AOD_AQUA": (
        "inventory-verified (describe_dataset): L3 AOD is quality-screened at L2 "
        "before gridding, so the grid publishes no per-pixel flag band"
    ),
    "MODIS_AOD_TERRA": (
        "inventory-verified (describe_dataset): L3 AOD is quality-screened at L2 "
        "before gridding, so the grid publishes no per-pixel flag band"
    ),
    "OMI_NO2": "inventory-verified (describe_dataset): no flag band in the L3 grid",
    "OMI_O3": "inventory-verified (describe_dataset): no flag band in the L3 grid",
    "TROPOMI_NO2": (
        "inventory-verified (describe_dataset): no flag band. The registered product is "
        "HAQ_TROPOMI_NO2_GLOBAL_M_L3, a MONTHLY global composite -- not the S5P_L2 swath, "
        "whose continuous 0-1 `qa_value` the enumerated doctrine could not express anyway. "
        "Being monthly-only is its real limitation for air-quality work, not the masking"
    ),
    "VIIRS_AOD_SNPP": (
        "inventory-verified (describe_dataset, 2026-08-02): 13 bands, none a flag. Same "
        "AER_DBDT_D10KM_L3 algorithm as the MODIS entries, quality-screened at L2 before "
        "gridding, and publishes an identical variable list to them"
    ),
    "VIIRS_AOD_NOAA20": (
        "inventory-verified (describe_dataset, 2026-08-02): 13 bands, none a flag. Same "
        "AER_DBDT_D10KM_L3 algorithm as the MODIS entries, quality-screened at L2 before "
        "gridding, and publishes an identical variable list to them"
    ),
    "AERDA_AOD_NRT": (
        "granule-verified 2026-08-02: all 432 bands are one of Mean / Pixel_Counts / "
        "Standard_Deviation / Histogram_Counts across 121 product groups -- no flag band "
        "anywhere. The product filters on pre-defined QA thresholds before L3 aggregation "
        "(per its own abstract), so screening happens upstream and the QF3 groups are "
        "already-filtered variants rather than a per-pixel flag to apply"
    ),
    "AERDA_AOD_NRT_3H": (
        "granule-verified 2026-08-02: the 3-hourly cut of AERDA_AOD_NRT, confirmed "
        "structurally identical (432 bands, 121 groups, same four leaf names), so the "
        "same upstream-QA-threshold reasoning applies unchanged"
    ),
}


def known_quality_flag_vars() -> frozenset[str]:
    """Leaf names of every ``quality_flag_var`` pinned in the registry.

    Used to exclude QA-flag variables from science-variable choice (T25): a
    flag riding along in a retrieval request or an opened multi-variable file
    is never a science-variable candidate. Registry ``variables`` lists are
    HDF group-qualified (e.g. ``product/main_data_quality_flag``) while
    open_handle merges those groups down to bare leaf names, so the leaf is
    the only stable key both sides agree on. Reads through ``load_registry``'s
    cache, so it stays consistent with ``reload_registry``."""
    return frozenset(
        cfg.quality_flag_var.rsplit("/", 1)[-1]
        for cfg in load_registry().values()
        if cfg.quality_flag_var
    )


def reload_registry() -> dict[str, CollectionConfig]:
    """
    Clear the cache and reload from disk.
    Useful when collections.yaml is updated at runtime without a restart.
    """
    load_registry.cache_clear()
    return load_registry()
