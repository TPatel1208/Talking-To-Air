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
import logging
logger = logging.getLogger(__name__)

from utils.data_utils import _load_data
from utils.plotting import _normalize_to_2d, mask_data_by_geometry, RegionResolver
from tools.satellite_tools.harmony_api import COLLECTIONS
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

    # No 2D normilization, we need the time dimension
    if "time" not in da.dims:
        return json.dumps({"error": f"No time dimension found. dims={list(da.dims)}"})

    # --- 2. Mask to region ---
    region = _resolver.resolve_location(location)
    if region is None:
        return json.dumps({"error": f"Could not resolve location: '{location}'"})

    # Mask the full 3D array, mask_data_by_geometry handles 3D
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
    logger = logging.getLogger(__name__)
    logger.info(f"[find_daily_peak] Called with location='{location}', variable='{data_dict.get('variable')}'")

    # Load and normalize
    try:
        da = _load_data(data_dict)
        logger.info(f"[find_daily_peak] Data loaded. Shape: {da.shape}, Dims: {list(da.dims)}")
    except Exception as e:
        logger.error(f"[find_daily_peak] Failed to load data: {e}", exc_info=True)
        return json.dumps({"error": f"Failed to load data: {e}"})

    da = _normalize_to_2d(da)
    logger.info(f"[find_daily_peak] After normalize_to_2d. Shape: {da.shape}, Dims: {list(da.dims)}, Coords: {list(da.coords.keys())}")

    # Mask to region
    region = _resolver.resolve_location(location)
    if region is None:
        logger.error(f"[find_daily_peak] Could not resolve location: '{location}'")
        return json.dumps({"error": f"Could not resolve location: '{location}'"})

    geom   = region['geometry']
    bounds = geom.bounds
    logger.info(f"[find_daily_peak] Region bounds: minx={bounds[0]:.2f}, miny={bounds[1]:.2f}, maxx={bounds[2]:.2f}, maxy={bounds[3]:.2f}")

    da_before   = da.copy()
    da          = mask_data_by_geometry(da, geom)
    before_valid = int(np.sum(np.isfinite(da_before.values)))
    after_valid  = int(np.sum(np.isfinite(da.values)))
    logger.info(f"[find_daily_peak] Valid pixels before mask: {before_valid}, after mask: {after_valid}")

    # Resolve dim names and positions early
    lat_dim = next((d for d in da.dims if d.lower() in ['lat', 'latitude']), None)
    lon_dim = next((d for d in da.dims if d.lower() in ['lon', 'longitude']), None)
    logger.info(f"[find_daily_peak] Resolved lat_dim='{lat_dim}', lon_dim='{lon_dim}'")

    if lat_dim is None or lon_dim is None:
        msg = f"Could not find lat/lon dimensions. Available dims: {list(da.dims)}"
        logger.error(f"[find_daily_peak] {msg}")
        return json.dumps({"error": msg})

    lat_array = da[lat_dim].values
    lon_array = da[lon_dim].values
    logger.info(f"[find_daily_peak] lat_array shape={lat_array.shape}, lon_array shape={lon_array.shape}")
    logger.info(f"[find_daily_peak] Lat range: {float(lat_array.min()):.4f} to {float(lat_array.max()):.4f}")
    logger.info(f"[find_daily_peak] Lon range: {float(lon_array.min()):.4f} to {float(lon_array.max()):.4f}")

    # Filter
    var        = data_dict.get("variable", "")
    col_info   = COLLECTIONS.get(var, {})
    fill_value = col_info.get("fill_value", -1.267651e+30)
    max_valid  = col_info.get("valid_max",   1e18)
    min_valid  = col_info.get("valid_min",  -1e15)
    logger.info(f"[find_daily_peak] Filter params — fill_value={fill_value}, min_valid={min_valid}, max_valid={max_valid}")

    values     = da.values
    valid_mask = (
        np.isfinite(values) &
        (values != fill_value) &
        (values > min_valid) &
        (values < max_valid)
    )
    valid_count = int(np.sum(valid_mask))
    logger.info(f"[find_daily_peak] Valid pixel count after filtering: {valid_count}")

    if not np.any(valid_mask):
        msg = f"No valid data found for '{location}'. The region may be outside the data bbox."
        logger.warning(f"[find_daily_peak] {msg}")
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

    logger.info(f"[find_daily_peak] dims={dims}, lat_pos={lat_pos}, lon_pos={lon_pos}")
    logger.info(f"[find_daily_peak] Peak array indices — lat_idx={lat_idx}, lon_idx={lon_idx}")
    logger.info(f"[find_daily_peak] Peak raw value at index: {masked_values[dim0_idx, dim1_idx]:.6f}")

    try:
        peak_lat = float(lat_array[lat_idx] if lat_array.ndim == 1 else lat_array[lat_idx, lon_idx])
        peak_lon = float(lon_array[lon_idx] if lon_array.ndim == 1 else lon_array[lat_idx, lon_idx])
    except (IndexError, TypeError) as e:
        logger.error(f"[find_daily_peak] Failed to extract peak coordinates: {e}", exc_info=True)
        return json.dumps({"error": f"Failed to extract peak coordinates: {e}"})

    peak_val = float(masked_values[dim0_idx, dim1_idx])

    # Sanity check
    if not (bounds[1] <= peak_lat <= bounds[3] and bounds[0] <= peak_lon <= bounds[2]):
        logger.warning(
            f"[find_daily_peak] COORDINATE MISMATCH — peak lat={peak_lat}, lon={peak_lon} "
            f"is OUTSIDE region bounds {bounds}. mask_data_by_geometry may not be working correctly."
        )
    else:
        logger.info(f"[find_daily_peak] Coordinate check passed — peak is inside region bounds.")

    result = json.dumps({
        "location":   location,
        "variable":   var,
        "units":      data_dict.get("units"),
        "times":      data_dict.get("times", []),
        "peak_value": peak_val,
        "peak_lat":   peak_lat,
        "peak_lon":   peak_lon,
    })
    logger.info(f"[find_daily_peak] Returning: {result}")
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