import json
import sys
import os
import numpy as np
from langchain.tools import tool
import pandas as pd
from typing import Optional
from tools.satellite_tools.models import DataDict
import re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.data_utils import _load_data
from utils.plotting import _normalize_to_2d, mask_data_by_geometry, RegionResolver
from tools.satellite_tools.harmony_api import COLLECTIONS
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
_resolver = RegionResolver()

VALID_STATS = {"mean", "median", "max", "min", "std"}



def _get(data_dict, key, default=None):
    """Access a field from a DataDict object or plain dict interchangeably."""
    if isinstance(data_dict, dict):
        return data_dict.get(key, default)
    return getattr(data_dict, key, default)

@tool
def compute_statistic_tool(
    data_dict: dict,
    location: str,
    stats: list[str] = ["mean", "median", "max", "min"]
) -> str:
    """
    Compute basic statistics (mean, median, max, min, std) over a region
    for a single fetched dataset.

    Use when the user asks questions like:
      - 'What is the average NO2 in Texas?'
      - 'What was the max pollution in California on April 8?'
      - 'Give me summary statistics for NO2 over New York'

    Args:
        data_dict: dict directly from fetch_environmental_data
        location:  place name to spatially mask before computing e.g. 'Texas'
        stats:     list of statistics to compute.
                   Any of: 'mean', 'median', 'max', 'min', 'std'

    Returns:
        JSON string with each requested statistic and its value.
    """
    # --- 1. Load and normalize ---
    try:
        da = _load_data(data_dict)
    except Exception as e:
        return json.dumps({"error": f"Failed to load data: {e}"})

    da = _normalize_to_2d(da)

    # --- 2. Mask to region ---
    region = _resolver.resolve_location(location)
    if region is None:
        return json.dumps({"error": f"Could not resolve location: '{location}'"})

    da = mask_data_by_geometry(da, region['geometry'])

    # --- 3. Extract valid pixels ---'
    var = _get(data_dict, "variable", "")
    col_info = COLLECTIONS.get(var, {})
    values = da.values
    fill_value = col_info.get("fill_value", -1.267651e+30)
    max_valid  = col_info.get("valid_max",   1e18)
    min_valid  = col_info.get("valid_min",  -1e15)
    valid = values[
        (~np.isnan(values)) &
        (values > fill_value * 0.5) &  
        (values > min_valid) &              
        (values < max_valid)              
    ]
    if len(valid) == 0:
        return json.dumps({
            "error": f"No valid data found for '{location}'. "
                     "The region may be outside the data bbox."
        })

    # --- 4. Compute requested stats ---
    stat_fns = {
        "mean":   lambda v: float(np.mean(v)),
        "median": lambda v: float(np.median(v)),
        "max":    lambda v: float(np.max(v)),
        "min":    lambda v: float(np.min(v)),
        "std":    lambda v: float(np.std(v)),
    }

    invalid_stats = [s for s in stats if s not in VALID_STATS]
    if invalid_stats:
        return json.dumps({"error": f"Unknown stats: {invalid_stats}. Valid: {sorted(VALID_STATS)}"})

    result = {
        "location": location,
        "variable": _get(data_dict, "variable", ""),
        "units":    _get(data_dict, "units", ""),
        "n_pixels": int(len(valid)),
        "times":    list(_get(data_dict, "times", [])),
    }
    for s in stats:
        result[s] = stat_fns[s](valid)

    return json.dumps(result)


@tool
def find_daily_peak(
    data_dict: dict,
    location: str,
) -> str:
    """
    Find the peak (maximum) value and its lat/lon location within a region.

    Use when the user asks questions like:
      - 'Where was NO2 highest in Texas on April 8?'
      - 'What was the worst pollution point in California?'
      - 'Find the peak NO2 location in New York'

    Args:
        data_dict: dict directly from fetch_environmental_data
        location:  place name to spatially mask before searching e.g. 'Texas'

    Returns:
        JSON string with peak value, lat, lon, and metadata.
    """
  
    # Load and normalize
    try:
        da = _load_data(data_dict)
    except Exception as e:
        return json.dumps({"error": f"Failed to load data: {e}"})

    da = _normalize_to_2d(da)

    # Mask to region
    region = _resolver.resolve_location(location)
    if region is None:
        return json.dumps({"error": f"Could not resolve location: '{location}'"})

    geom   = region['geometry']
    bounds = geom.bounds

    da_before   = da.copy()
    da          = mask_data_by_geometry(da, geom)
    before_valid = int(np.sum(np.isfinite(da_before.values)))
    after_valid  = int(np.sum(np.isfinite(da.values)))

    # Resolve dim names and positions early
    lat_dim = next((d for d in da.dims if d.lower() in ['lat', 'latitude']), None)
    lon_dim = next((d for d in da.dims if d.lower() in ['lon', 'longitude']), None)

    if lat_dim is None or lon_dim is None:
        msg = f"Could not find lat/lon dimensions. Available dims: {list(da.dims)}"
        return json.dumps({"error": msg})

    lat_array = da[lat_dim].values
    lon_array = da[lon_dim].values

    # Filter
    var        = _get(data_dict, "variable", "")
    col_info   = COLLECTIONS.get(var, {})
    fill_value = col_info.get("fill_value", -1.267651e+30)
    max_valid  = col_info.get("valid_max",   1e18)
    min_valid  = col_info.get("valid_min",  -1e15)
    values     = da.values
    valid_mask = (
        np.isfinite(values) &
        (values != fill_value) &
        (values > min_valid) &
        (values < max_valid)
    )
    valid_count = int(np.sum(valid_mask))

    if not np.any(valid_mask):
        msg = f"No valid data found for '{location}'. The region may be outside the data bbox."
        return json.dumps({"error": msg})

    # Find peak
    masked_values = np.where(valid_mask, values, np.nan)
    flat_idx      = np.nanargmax(masked_values)
    dim0_idx, dim1_idx = np.unravel_index(flat_idx, masked_values.shape)

    # Determine which axis corresponds to lat and lon
    dims    = list(da.dims)
    lat_pos = dims.index(lat_dim)
    lon_pos = dims.index(lon_dim)
    indices = [dim0_idx, dim1_idx]
    lat_idx = indices[lat_pos]
    lon_idx = indices[lon_pos]

 

    try:
        peak_lat = float(lat_array[lat_idx] if lat_array.ndim == 1 else lat_array[lat_idx, lon_idx])
        peak_lon = float(lon_array[lon_idx] if lon_array.ndim == 1 else lon_array[lat_idx, lon_idx])
    except (IndexError, TypeError) as e:
        return json.dumps({"error": f"Failed to extract peak coordinates: {e}"})

    peak_val = float(masked_values[dim0_idx, dim1_idx])


    result = json.dumps({
        "location":   location,
        "variable":   var,
        "units":      _get(data_dict, "units", ""),
        "times":      list(_get(data_dict, "times", [])),
        "peak_value": peak_val,
        "peak_lat":   peak_lat,
        "peak_lon":   peak_lon,
    })
    return result


def main():
    import logging

    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    from Backend.tools.satellite_tools.harmony_api import fetch_environmental_data
    # Fetch some data
    data = fetch_environmental_data.invoke({
        "variable": "TEMPO_NO2",
        "bbox": "-106.6458,25.8371,-93.5078,36.5005",
        "start_date": "2026-04-06T00:00:00Z",
        "end_date": "2026-04-06T23:59:59Z"
    })

    # Run stats
    """
    result = compute_statistic_tool.invoke({
        "data_dict": data,
        "location": "Texas",
        "stats": ["mean","median", "max", "min", "std"]
    })

    print(result)
    """

    result = find_daily_peak.invoke({
    "data_dict": data,   # reuse the fetch from before
    "location": "Texas"
    })

    print(result)

if __name__ == "__main__":
    main()