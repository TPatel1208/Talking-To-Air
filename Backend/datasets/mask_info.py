"""
datasets/mask_info.py
======================
The retrieval MCP embeds each variable's fill value and valid range as
CF-convention attrs (_FillValue/valid_min/valid_max) on the Zarr/NetCDF it
materializes — AggregationService.apply_quality_mask already reads those
directly off the opened DataArray. MASK_OVERRIDES corrects the rare
known-wrong UMM-Var/CF record, keyed by short_name, taking precedence over
whatever the file itself says. Populated from the live-matrix quirk ledger
as entries arrive.

resolve_mask_info (T25 Phase 1) layers a third tier — describe_dataset's
per-variable UMM-Var facts — between the override/collections.yaml tier
above and the CF-attrs fallback, and records which tier won so a caller
never has to guess whether masking ran silently.
"""
from __future__ import annotations

from typing import Any

MASK_OVERRIDES: dict[str, dict[str, float]] = {}

SOURCE_YAML = "collections_yaml"
SOURCE_UMM_VAR = "umm_var"
SOURCE_CF_ATTRS = "cf_attrs"
SOURCE_NONE = "none"


def override_for(short_name: str | None, overrides: dict[str, dict[str, float]] | None = None) -> dict[str, float]:
    """Return the col_info override for short_name, or {} if none is recorded."""
    overrides = MASK_OVERRIDES if overrides is None else overrides
    if not short_name:
        return {}
    return dict(overrides.get(short_name, {}))


def col_info_for_short_name(short_name: str | None) -> dict[str, Any]:
    """The pinned collections.yaml registry entry for ``short_name`` -- the
    identity marker a real opened NASA granule carries as a dataset-level
    global attribute, not a registry dict key -- merged with MASK_OVERRIDES
    on top (T25 masking-execution fix).

    Before this, a tool's col_info was built from MASK_OVERRIDES alone
    (always {} today, since it only holds hand-verified quirk corrections),
    so collections.yaml's pinned qa_good_values/quality_flag_var for
    TEMPO_NO2 etc. never reached apply_quality_mask: the tool layer had no
    collection_id, and the science variable name (e.g.
    ``vertical_column_troposphere``) is not a registry key. Matching is
    case-insensitive against ``CollectionConfig.short_name``.
    """
    if not short_name:
        return {}
    from datasets.registry import load_registry  # local import: registry.py never imports this module back

    normalized = short_name.upper()
    registry_info: dict[str, Any] = {}
    for cfg in load_registry().values():
        if cfg.short_name and cfg.short_name.upper() == normalized:
            registry_info = cfg.model_dump()
            break
    return {**registry_info, **override_for(normalized)}


def resolve_mask_info(
    yaml_info: dict[str, Any] | None = None,
    umm_var_variable: dict[str, Any] | None = None,
    cf_attrs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Layer fill_value/valid_min/valid_max/units per the PRD T25 precedence:
    collections.yaml/MASK_OVERRIDES (``yaml_info``) -> describe_dataset's
    per-variable UMM-Var facts (``umm_var_variable``) -> the file's own CF
    attrs (``cf_attrs``) -> mask nothing. The first tier that supplies a
    fact wins it.

    Returns ``(resolved_col_info, provenance)``. ``provenance`` records
    which tier won each fact (SOURCE_YAML/SOURCE_UMM_VAR/SOURCE_CF_ATTRS/
    SOURCE_NONE) plus ``applied`` — whether any fill/valid fact was found
    at all — so a caller never has to guess whether masking ran silently.
    """
    yaml_info = yaml_info or {}
    umm_var_variable = umm_var_variable or {}
    cf_attrs = cf_attrs or {}

    resolved: dict[str, Any] = {}
    provenance: dict[str, Any] = {
        "fill_value_source": SOURCE_NONE,
        "valid_range_source": SOURCE_NONE,
    }

    fill_value = yaml_info.get("fill_value")
    if fill_value is not None:
        provenance["fill_value_source"] = SOURCE_YAML
    else:
        fill_value = _first_fill_value(umm_var_variable.get("fill_values"))
        if fill_value is not None:
            provenance["fill_value_source"] = SOURCE_UMM_VAR
        else:
            fill_value = cf_attrs.get("_FillValue")
            if fill_value is not None:
                provenance["fill_value_source"] = SOURCE_CF_ATTRS
    if fill_value is not None:
        resolved["fill_value"] = fill_value

    valid_min, valid_max = yaml_info.get("valid_min"), yaml_info.get("valid_max")
    if valid_min is not None or valid_max is not None:
        provenance["valid_range_source"] = SOURCE_YAML
    else:
        valid_min, valid_max = _first_valid_range(umm_var_variable.get("valid_ranges"))
        if valid_min is not None or valid_max is not None:
            provenance["valid_range_source"] = SOURCE_UMM_VAR
        else:
            valid_min = cf_attrs.get("valid_min")
            valid_max = cf_attrs.get("valid_max")
            if valid_min is not None or valid_max is not None:
                provenance["valid_range_source"] = SOURCE_CF_ATTRS
    if valid_min is not None:
        resolved["valid_min"] = valid_min
    if valid_max is not None:
        resolved["valid_max"] = valid_max

    units = yaml_info.get("units") or umm_var_variable.get("units") or cf_attrs.get("units")
    if units is not None:
        resolved["units"] = units

    provenance["applied"] = "fill_value" in resolved or "valid_min" in resolved or "valid_max" in resolved
    return resolved, provenance


def _first_fill_value(fill_values: list[dict[str, Any]] | None) -> float | None:
    for entry in fill_values or []:
        value = entry.get("value") if isinstance(entry, dict) else entry
        if value is not None:
            return value
    return None


def _first_valid_range(valid_ranges: list[dict[str, Any]] | None) -> tuple[float | None, float | None]:
    for entry in valid_ranges or []:
        if isinstance(entry, dict) and (entry.get("min") is not None or entry.get("max") is not None):
            return entry.get("min"), entry.get("max")
    return None, None


def match_umm_var_variable(umm_var_facts: Any, name: str | None) -> dict[str, Any] | None:
    """Find the describe_dataset variable record matching ``name``.

    Accepts describe_dataset's raw ``variables`` list (matched by each
    entry's ``name`` field), a name-keyed mapping, or an already-selected
    single variable record (passed straight through) — so a caller doesn't
    need to unwrap describe_dataset's shape itself.
    """
    if not umm_var_facts:
        return None
    if isinstance(umm_var_facts, dict):
        if "fill_values" in umm_var_facts or "valid_ranges" in umm_var_facts:
            return umm_var_facts
        return umm_var_facts.get(name) if name else None
    if isinstance(umm_var_facts, list):
        for var in umm_var_facts:
            if isinstance(var, dict) and var.get("name") == name:
                return var
    return None
