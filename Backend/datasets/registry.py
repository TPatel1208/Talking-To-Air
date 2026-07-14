"""
datasets/registry.py
====================
Loads and validates the dataset registry from collections.yaml.

Usage
-----
    from datasets.registry import load_registry, CollectionConfig

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
