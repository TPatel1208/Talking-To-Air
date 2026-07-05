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
"""
from __future__ import annotations

MASK_OVERRIDES: dict[str, dict[str, float]] = {}


def override_for(short_name: str | None, overrides: dict[str, dict[str, float]] | None = None) -> dict[str, float]:
    """Return the col_info override for short_name, or {} if none is recorded."""
    overrides = MASK_OVERRIDES if overrides is None else overrides
    if not short_name:
        return {}
    return dict(overrides.get(short_name, {}))
