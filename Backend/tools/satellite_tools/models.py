"""
models.py
---------
Shared Pydantic models for satellite tool inputs/outputs.

Kept in a separate module so plot_tools, stat_tools, and harmony_api can all
import DataDict without creating circular dependencies.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DataDict(BaseModel):
    """
    Structured return value of fetch_environmental_data.

    Pass this object directly to plot_singular, plot_multiple,
    conduct_temporal_statistic, compute_statistic_tool, and find_daily_peak.
    Do not construct it manually — always use the output of fetch_environmental_data.
    """
    variable:     str                       = Field(description="Dataset key e.g. 'TROPOMI_NO2'")
    units:        str                       = Field(description="Physical units e.g. 'molecules/cm^2'")
    bbox:         str                       = Field(description="Bounding box 'min_lon,min_lat,max_lon,max_lat'")
    times:        List[str]                 = Field(default_factory=list,
                                                    description="ISO timestamps of available granules")
    n_granules:   int                       = Field(default=0,
                                                    description="Number of granules found")
    cadence:      str                       = Field(default="daily",
                                                    description="Temporal cadence: hourly, daily, or monthly")
    source:       str                       = Field(default="",
                                                    description="Human-readable data source label")
    fetch_params: Optional[Dict[str, Any]]  = Field(default=None,
                                                    description="Internal reload params for _load_data. "
                                                                 "Do not construct manually.")
