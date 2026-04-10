import json
import sys
import os
import numpy as np
from langchain.tools import tool
import pandas as pd
from typing import Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.data_utils import _load_data
from utils.plotting import _normalize_to_2d, mask_data_by_geometry, RegionResolver
from tools.harmony_api import COLLECTIONS
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
_resolver = RegionResolver()

VALID_STATS = {"mean", "median", "max", "min", "std"}


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
    var = data_dict.get("variable")
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
        "variable": data_dict.get("variable"),
        "units":    data_dict.get("units"),
        "n_pixels": int(len(valid)),
        "times":    data_dict.get("times", []),
    }
    for s in stats:
        result[s] = stat_fns[s](valid)

    return json.dumps(result)


@tool
def conduct_temporal_statistic(
    data_dict: dict,
    location: str,
    stat: str = "mean",
) -> str:
    """
    Compute a statistic over each time step in a dataset and plot the trend.

    Use when the user asks questions like:
      - 'How did NO2 change over time in Texas?'
      - 'Show me the trend of NO2 in California over the past 3 months'
      - 'Plot the daily mean NO2 in New York for January 2024'

    Args:
        data_dict: dict directly from fetch_environmental_data
                   (should cover a multi-day or multi-month range)
        location:  place name to spatially mask before computing e.g. 'Texas'
        stat:      statistic to compute at each time step.
                   One of: 'mean', 'median', 'max', 'min', 'std'

    Returns:
        File path of the saved trend plot PNG, or error JSON string.
    """
    import matplotlib.dates as mdates
    from datetime import datetime

    # --- 1. Load --- 
    try:
        da = _load_data(data_dict)
    except Exception as e:
        return json.dumps({"error": f"Failed to load data: {e}"})

    # Do NOT normalize to 2D here — we need the time dimension
    if "time" not in da.dims:
        return json.dumps({"error": f"No time dimension found. dims={list(da.dims)}"})

    # --- 2. Mask to region ---
    region = _resolver.resolve_location(location)
    if region is None:
        return json.dumps({"error": f"Could not resolve location: '{location}'"})

    # Mask the full 3D array — mask_data_by_geometry handles 3D
    da = mask_data_by_geometry(da, region['geometry'])

    # --- 3. Filter setup ---
    var        = data_dict.get("variable", "")
    col_info   = COLLECTIONS.get(var, {})
    fill_value = col_info.get("fill_value", -1.267651e+30)
    max_valid  = col_info.get("valid_max",   1e18)
    min_valid  = col_info.get("valid_min",  -1e15)

    stat_fn = {
        "mean":   np.nanmean,
        "median": np.nanmedian,
        "max":    np.nanmax,
        "min":    np.nanmin,
        "std":    np.nanstd,
    }.get(stat)

    if stat_fn is None:
        return json.dumps({"error": f"Unknown stat '{stat}'. Use: mean, median, max, min, std"})

    # --- 4. Iterate over time steps ---
    times  = []
    values = []

    for i in range(da.sizes["time"]):
        slice_2d = da.isel(time=i).values

        # Apply validity filter
        valid = slice_2d[
            np.isfinite(slice_2d) &
            (slice_2d != fill_value) &
            (slice_2d > min_valid) &
            (slice_2d < max_valid)
        ]

        if len(valid) == 0:
            continue

        # Parse timestamp
        raw_time = da["time"].values[i]
        timestamp = pd.Timestamp(raw_time).to_pydatetime()

        times.append(timestamp)
        values.append(float(stat_fn(valid)))

    if not times:
        return json.dumps({"error": f"No valid data found for '{location}' across any time step."})

    # --- 5. Sort by time (granules may be unordered) ---
    paired = sorted(zip(times, values), key=lambda x: x[0])
    times, values = zip(*paired)

    # --- 6. Plot ---
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)

    ax.plot(times, values, marker='o', linewidth=1.5, markersize=4, color='steelblue')
    ax.fill_between(times, values, alpha=0.1, color='steelblue')

   
    time_range_days = (times[-1] - times[0]).days
    time_range_hours = (times[-1] - times[0]).total_seconds() / 3600

    if time_range_hours <= 24:
        # Same day — show hours
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_xlabel("Time (UTC)")
    elif time_range_days <= 31:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.set_xlabel("Date")
    elif time_range_days <= 366:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.set_xlabel("Date")
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_xlabel("Year")

    plt.xticks(rotation=45)
    date_str = times[0].strftime('%Y-%m-%d')
    if time_range_hours <= 24:
        ax.set_title(f"{var} {stat} over {location} ({date_str})", fontsize=12, fontweight='bold')
    else:
        ax.set_title(f"{var} {stat} over {location}", fontsize=12, fontweight='bold')
    ax.set_ylabel(f"{stat} ({data_dict.get('units', '')})")
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # --- 7. Save ---
    safe_location = re.sub(r'[^\w\-]', '_', location)
    output_path = os.path.join(OUTPUT_DIR, f"{var}_{stat}_trend_{safe_location}.png")
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


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
    #Load and normalize
    try:
        da = _load_data(data_dict)
    except Exception as e:
        return json.dumps({"error": f"Failed to load data: {e}"})

    da = _normalize_to_2d(da)

    #Mask to region
    region = _resolver.resolve_location(location)
    if region is None:
        return json.dumps({"error": f"Could not resolve location: '{location}'"})

    da = mask_data_by_geometry(da, region['geometry'])

    #Filter
    var        = data_dict.get("variable", "")
    col_info   = COLLECTIONS.get(var, {})
    fill_value = col_info.get("fill_value", -1.267651e+30)
    max_valid  = col_info.get("valid_max",   1e18)
    min_valid  = col_info.get("valid_min",  -1e15)

    values = da.values
    valid_mask = (
        np.isfinite(values) &
        (values != fill_value) &
        (values > min_valid) &
        (values < max_valid)
    )

    if not np.any(valid_mask):
        return json.dumps({
            "error": f"No valid data found for '{location}'. "
                     "The region may be outside the data bbox."
        })

    #Find peak
    # Mask invalid pixels to NaN
    masked_values = np.where(valid_mask, values, np.nan)
    flat_idx = np.nanargmax(masked_values)
    lat_idx, lon_idx = np.unravel_index(flat_idx, masked_values.shape)

    
    lat_names = ['lat', 'latitude', 'Latitude', 'LAT']
    lon_names = ['lon', 'longitude', 'Longitude', 'LON', 'long']

    lat_coord = next((c for c in lat_names if c in da.coords), None)
    lon_coord = next((c for c in lon_names if c in da.coords), None)

    if lat_coord is None or lon_coord is None:
        return json.dumps({
            "error": f"Could not find lat/lon coordinates. "
                     f"Available: {list(da.coords.keys())}"
        })

    peak_lat = float(da[lat_coord].values[lat_idx])
    peak_lon = float(da[lon_coord].values[lon_idx])
    peak_val = float(masked_values[lat_idx, lon_idx])

    return json.dumps({
        "location":    location,
        "variable":    var,
        "units":       data_dict.get("units"),
        "times":       data_dict.get("times", []),
        "peak_value":  peak_val,
        "peak_lat":    peak_lat,
        "peak_lon":    peak_lon,
    })





def main():
    import logging

    logging.basicConfig(
        level=logging.INFO,  # change to DEBUG for more detail
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    from tools.harmony_api import fetch_environmental_data
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