from langchain.tools import tool
import json
import os
import sys
import xarray as xr
from typing import List, Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
from utils.data_utils import _load_data




sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.plotting import mask_data_by_geometry, RegionResolver
from utils.plotting import plot_diff_maps

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_resolver = RegionResolver()





@tool
def plot_singular(data_dict: dict, variable: str, location: str, title: str="",cmap: Optional[str] = "Spectral_r"):
    """
    Plot a single environmental variable over a single location.
    Use when the user asks for a map of one variable in one place,
    e.g. 'Plot the NO2 levels in New York City on February 10th 2026 at 6pm'.

    Args:
        data_dict : dict from fetch_environmental_data.
        variable  : Variable name e.g. 'NO2','CO', 'CO2', etc.
        location  : Place name e.g. 'New York City', 'California'.
        title     : Plot title. Auto-generated from variable + location if omitted.
        cmap      : Matplotlib colormap (default 'Spectral_r').

    Returns:
        File path of the saved PNG, or an error message string.
    """
    try:
        data_array = _load_data(data_dict)
    except Exception as e:
        return f"Failed to load data: {e}"
    
    resolved_title = title if title else f"{data_array.name or 'Variable'} over {location}"

    try:
        fig, ax = _resolver.plot_singular(
            data_array=data_array,
            location_name=location,
            title=resolved_title,
            time_slice=0,
            cmap=cmap,
        )
    except Exception as e:
        return f"Plotting failed: {e}"
     # Sanitize title for use as a filename
    safe_name = re.sub(r'[^\w\-]', '_', resolved_title)
    output_path = os.path.join(OUTPUT_DIR, f"{safe_name}.png")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


@tool
def plot_multiple(
    data_dicts: List[dict],
    variable: str,
    locations: List[str],
    title: str = "",
    cmap: Optional[str] = "Spectral_r",
) -> str:
    """
    Plot the same environmental variable across multiple locations side by side.

    IMPORTANT — call fetch_environmental_data separately for each location first,
    collecting each result into a list. If a fetch fails for one dataset, try a
    fallback dataset before adding to the list. Only call this tool once you have
    a successful data_dict for every location.

    Example agent flow for 'Compare NO2 in NYC and LA':
        1. fetch_environmental_data(TEMPO_NO2, bbox_nyc, ...) → data_dict_nyc
        2. fetch_environmental_data(TEMPO_NO2, bbox_la, ...)  → data_dict_la
        3. plot_multiple([data_dict_nyc, data_dict_la], 'NO2', ['New York City', 'Los Angeles'])

    If datasets differ across locations (e.g. TROPOMI for one, TEMPO for another),
    that is fine, you can plot_multiple handles mixed sources.
    ...
    """
    if len(data_dicts) != len(locations):
        return f"Error: number of data_dicts ({len(data_dicts)}) must match number of locations ({len(locations)})"

    # Load and select primary variable for each location
    data_arrays = []
    for i, (data_dict, location) in enumerate(zip(data_dicts, locations)):
        try:
            da = _load_data(data_dict)
            data_arrays.append(da)
        except Exception as e:
            return f"Failed to load data for '{location}': {e}"

    # Resolve geometry for each location
    regions = []
    for location in locations:
        region = _resolver.resolve_location(location)
        if region is None:
            return f"Could not geocode location: '{location}'"
        regions.append(region)

    # Mask each data array to its region and select time slice
    masked_arrays = []
    for da, region in zip(data_arrays, regions):
        # Handle time dimension — take first slice
        time_dims = ['time', 'Time', 'TIME', 't']
        for td in time_dims:
            if td in da.dims and da.sizes[td] > 1:
                da = da.isel({td: 0})
                break
            elif td in da.dims:
                da = da.isel({td: 0})
                break
        try:
            masked = mask_data_by_geometry(da, region['geometry'])
        except Exception as e:
            return f"Failed to mask data for '{region['name']}': {e}"
        masked_arrays.append(masked)

    resolved_title = title if title else f"{variable} Comparison"
    titles = [r['name'] for r in regions]
    extents = [r['bounds'] for r in regions]
    geometries = [r['geometry'] for r in regions]

    try:
        fig, axes = plot_diff_maps(
            data_arrays=masked_arrays,
            titles=titles,
            extent=extents,
            mask_geometries=geometries,
            cmap=cmap,
        )
    except Exception as e:
        return f"Plotting failed: {e}"

    safe_name = re.sub(r'[^\w\-]', '_', resolved_title)
    output_path = os.path.join(OUTPUT_DIR, f"{safe_name}.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path