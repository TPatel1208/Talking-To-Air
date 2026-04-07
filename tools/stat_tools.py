import json
import sys
import os
import numpy as np
from langchain.tools import tool
from typing import Optional
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.data_utils import _load_data
from utils.plotting import _normalize_to_2d, mask_data_by_geometry, RegionResolver
from tools.harmony_api import COLLECTIONS

_resolver = RegionResolver()

VALID_STATS = {"mean", "median", "max", "min", "std"}


@tool
def compute_statistic_tool(
    data_dict: dict,
    location: str,
    stats: list = ["mean", "median", "max", "min"]
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
    fill_value = col_info.get("fill_value")
    max_valid = col_info.get("valid_max")
    min_valid = col_info.get("valid_min")
    valid = values[
        (~np.isnan(values)) &
        (values > fill_value * 0.85) &  
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
def conduct_temporal_statistic(data_json: str, frequency: str = "monthly") -> str:
    """
    Docstring for conduct_temporal_statistic
    :param data_json: Description
    :type data_json: str
    :param frequency: Description
    :type frequency: str
    :return: Description
    :rtype: str
    """
    return "0"


@tool
def find_daily_peak(data_json: str) -> str:
    """
    Docstring for find_daily_peak
    :param data_json: Description
    :type data_json: str
    :return: Description
    :rtype: str
    """
    return "0"


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
        "start_date": "2026-04-06T22:00:00Z",
        "end_date": "2026-04-06T23:30:59Z"
    })

    # Run stats
    result = compute_statistic_tool.invoke({
        "data_dict": data,
        "location": "Texas",
        "stats": ["mean","median", "max", "min", "std"]
    })

    print(result)

if __name__ == "__main__":
    main()