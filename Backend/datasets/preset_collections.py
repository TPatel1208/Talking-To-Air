"""
datasets/preset_collections.py
================================
A small, hand-picked list of well-known datasets for prompt guidance only —
suggestions to help the model pick a reasonable starting point, never a
ceiling on what can be discovered. Any dataset findable via search_datasets
is fair game; this list exists so the model doesn't have to search from a
blank slate for the handful of collections that come up constantly.
"""
from __future__ import annotations

PRESET_COLLECTIONS: list[dict[str, str]] = [
    {"short_name": "OMI_NO2", "description": "Tropospheric NO2 column, daily, global"},
    {"short_name": "TROPOMI_NO2", "description": "NO2 column, monthly, global"},
    {"short_name": "TEMPO_NO2", "description": "Tropospheric NO2 vertical column, hourly, North America"},
    {"short_name": "TEMPO_O3TOT", "description": "Total ozone column, hourly, North America"},
    {"short_name": "OMI_O3", "description": "Total ozone column, daily, global"},
    {"short_name": "TEMPO_HCHO", "description": "HCHO vertical column, hourly, North America"},
    {"short_name": "OMI_HCHO", "description": "HCHO vertical column, daily, global"},
    {"short_name": "MODIS_AOD_TERRA", "description": "Aerosol Optical Depth, daily, global"},
]
