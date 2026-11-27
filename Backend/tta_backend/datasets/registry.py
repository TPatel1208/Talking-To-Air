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
import re
from datetime import date
from functools import lru_cache
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tta_backend.datasets.cadence import Cadence

logger = logging.getLogger(__name__)

_REGISTRY_PATH = pathlib.Path(__file__).parent / "collections.yaml"

_QUANTITY_PATTERN = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")


class CoverageBox(BaseModel):
    """A regional product's spatial coverage, in degrees (lon -180..180, lat -90..90).

    west < east and south < north: a box never crosses the antimeridian. A
    product that covers the whole globe uses ``coverage: global`` instead."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    west:  float = Field(ge=-180, le=180)
    south: float = Field(ge=-90, le=90)
    east:  float = Field(ge=-180, le=180)
    north: float = Field(ge=-90, le=90)

    @model_validator(mode="after")
    def _ordered(self) -> "CoverageBox":
        if self.west >= self.east:
            raise ValueError(f"coverage west ({self.west}) must be < east ({self.east})")
        if self.south >= self.north:
            raise ValueError(f"coverage south ({self.south}) must be < north ({self.north})")
        return self


class CollectionConfig(BaseModel):
    # ── Identity ──────────────────────────────────────────────────────────
    collection_id: str
    short_name:    str = ""
    version:       str = ""
    description:   str = ""
    # seasonal and annual granules are one mean over a calendar period; see
    # datasets/cadence.py for how seasons are bounded and named.
    cadence:       Cadence = "daily"
    # ── Provenance display ──────────────────────────────────────────────────
    # Free-text, not parsed by any masking/aggregation logic -- surfaced as
    # "<provider> — <instrument>" in the Metadata tab's Source dataset block.
    provider:      str = ""
    instrument:    str = ""
    # ── Scientific maturity ─────────────────────────────────────────────────
    # Where the product sits in its provider's validation lifecycle, and the
    # caveat that comes with that. This is scientific provenance, not
    # descriptive metadata: a Beta product's own user guide can say publication
    # is "not recommended and highly discouraged", and that sentence has to
    # reach the artifacts someone pulls while writing a paper -- methods.md and
    # the chart's disclosure -- not just a tooltip they saw once.
    #
    # "unknown" is the default and stays meaningfully different from
    # "validated": it says nobody checked, which is not the same as a checked
    # clean bill of health. maturity_note carries the provider's own wording so
    # the caveat is quotable rather than paraphrased.
    maturity:      Literal["beta", "provisional", "validated", "unknown"] = "unknown"
    maturity_note: str = ""

    # ── Variable selection ────────────────────────────────────────────────
    primary_var:                 str
    quality_flag_var:            Optional[str] = None
    # The Tier-1 pinned QA rule for quality_flag_var -- which
    # flag values count as good (or, equivalently, bad). At most one is
    # normally set; qa_good_values takes precedence if both are (datasets/
    # qa_flags.py::resolve_qa_info).
    qa_good_values:               Optional[list[int]] = None
    qa_bad_values:                Optional[list[int]] = None
    # The per-pixel physical vertical axes a LAYERED product's science variable
    # is uninterpretable without -- pressure and/or altitude. Same standing as
    # quality_flag_var and enforced by the same guard
    # (retrieval_composites._pinned_companion_variables): a subset that drops
    # these comes back with values and nothing to plot them against, which
    # renders as a bare layer index and, since layer 0 is the top of the
    # atmosphere, upside down. Empty for every non-layered product.
    vertical_axis_vars:          list[str]     = []
    variables:                   list[str]     = []
    supports_variable_subsetting: bool         = False
    groups:                      list[str]     = []

    # ── Physical metadata ─────────────────────────────────────────────────
    units:     str
    fill_value: float
    valid_min:  float
    valid_max:  float

    # ── Product choice ────────────────────────────────────────────────────
    # Facts the harness product resolver uses to choose a product for a
    # quantity. All optional: an entry without quantity and coverage can still
    # be used when a user names it, a workflow approves it or a run pins it,
    # but never chosen from registry facts alone.
    #
    # quantity: the canonical quantity the product measures, lower_snake,
    #   e.g. "no2_tropospheric_column". Entries measuring the same thing share
    #   the same string exactly.
    quantity:     Optional[str] = None
    # coverage: "global", or a {west, south, east, north} box in degrees.
    #   A request is served only when its whole area lies inside.
    coverage:     Literal["global"] | CoverageBox | None = None
    # record_start / record_end: first and last day (UTC) with data.
    #   record_end null means the record is ongoing.
    record_start: Optional[date] = None
    record_end:   Optional[date] = None
    # stream: "standard" (final processing) or "near_real_time" (rapid
    #   processing, published sooner).
    stream:       Optional[Literal["standard", "near_real_time"]] = None
    # latency_days: whole days from observation to publication. The newest
    #   day treated as published is today minus this (0 = today's data is
    #   already appearing, 1 = yesterday is complete). For a seasonal or
    #   annual product, whole days from the end of the period, since one
    #   file holds the whole period.
    latency_days: Optional[int] = Field(default=None, ge=0)
    # retention_days: whole days of recent data a rolling archive keeps. The
    #   oldest day still held is today minus this. Null means everything
    #   since record_start is kept.
    retention_days: Optional[int] = Field(default=None, ge=1)
    # rank: order among entries with the same quantity when no workflow list
    #   decides; 1 is preferred over 2. Unranked entries come after ranked.
    rank:         Optional[int] = Field(default=None, ge=1)

    @field_validator("quantity")
    @classmethod
    def _quantity_is_lower_snake(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not _QUANTITY_PATTERN.match(v):
            raise ValueError(f"quantity {v!r} must be lower_snake, e.g. 'no2_tropospheric_column'")
        return v

    @model_validator(mode="after")
    def _record_dates_ordered(self) -> "CollectionConfig":
        if (
            self.record_start is not None
            and self.record_end is not None
            and self.record_end < self.record_start
        ):
            raise ValueError(
                f"record_end ({self.record_end}) must be on or after "
                f"record_start ({self.record_start})"
            )
        return self

    @field_validator("fill_value", "valid_min", "valid_max", mode="before")
    @classmethod
    def _allow_inf(cls, v: Any) -> float:
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

    raw: dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

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
    "TEMPO_O3PROF": (
        "granule-verified 2026-08-08 (TEMPO_O3PROF_L3_V04_20251001T120743Z_S002): the "
        "file's only groups are geolocation / product / support_data -- there is no "
        "qa_statistics group and no band carrying CF flag_values/flag_meanings. Absence "
        "is by design: the user guide states the L3 producer already screened on "
        "fit_RMS, avg_residual and retrieval_exit_status BEFORE gridding. It recommends "
        "further screening on eff_cloud_fraction and the zenith angles, which are "
        "continuous thresholds rather than an enumerated flag, so they are a declarative "
        "threshold mask and not something the good/bad-values doctrine can "
        "express"
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
    "TROPOMI_NO2_ANNUAL": (
        "CMR-checked 2026-11-20, not granule-verified: HAQ_TROPOMI_NO2_GLOBAL_A_L3, the annual "
        "cut of the same HAQAST family as TROPOMI_NO2, lists Tropospheric_NO2, Number_obs, "
        "Latitude and Longitude and no flag band. Its file names say QA75: pixels with "
        "qa_value of 0.75 or more were averaged before gridding, so screening happened upstream"
    ),
    "TROPOMI_NO2_CONUS_ANNUAL": (
        "CMR-checked 2026-11-20, not granule-verified: HAQ_TROPOMI_NO2_CONUS_A_L3, the 0.01-degree "
        "CONUS annual cut of the TROPOMI_NO2 family, lists the same four variables and no flag "
        "band. QA75 file names: pixels with qa_value of 0.75 or more were averaged before gridding"
    ),
    "TROPOMI_NO2_CONUS_SEASONAL": (
        "CMR-checked 2026-11-20, not granule-verified: HAQ_TROPOMI_NO2_CONUS_S_L3, the 0.01-degree "
        "CONUS seasonal cut of the TROPOMI_NO2 family, lists the same four variables and no flag "
        "band. QA75 file names: pixels with qa_value of 0.75 or more were averaged before gridding"
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

    Used to exclude QA-flag variables from science-variable choice: a
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
