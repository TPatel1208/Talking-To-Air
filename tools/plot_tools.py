from langchain.tools import tool
import json
import os
import sys
import xarray as xr
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import re
from utils.data_utils import _load_data



sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.plotting import mask_data_by_geometry, RegionResolver

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
def plot_multiple(data_json: str, variables: list, output_name: str):
    """
    Docstring for plot_multiple
    
    :param data_json: Description
    :type data_json: str
    :param variables: Description
    :type variables: list
    :param output_name: Description
    :type output_name: str
    """
    return "0"