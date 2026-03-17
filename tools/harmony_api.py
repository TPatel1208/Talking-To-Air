import json
import sys
import os
from langchain.tools import tool
from typing import Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.plotting import GeocodingService
from preprocessing.data_loader import DataLoader

_geocoder = GeocodingService()
_data_loader = DataLoader()

COLLECTIONS = {
    # OMI NO2 (Default for 'NO2' variable)
    "OMI_NO2": {
        'collection_id': "C2215175232-GES_DISC",
        'variables': ["ColumnAmountNO2","ColumnAmountNO2CloudScreened","ColumnAmountNO2TropCloudScreened","Weight"],
        'primary_var': "ColumnAmountNO2TropCloudScreened",
        'short_name': "OMI_MINDS_NO2d",
        'version': "1.1",
        'groups': [],
        'units': "molecules/cm^2",
        'description': "OMI NO2 tropospheric column",
    },
    # Tropomi NO2 Monthly
    "TROPOMI_NO2": {
        'collection_id': "C3087325222-GES_DISC",
        'variables': ["Tropospheric_NO2",'Number_obs'],
        'primary_var': "Tropospheric_NO2",
        'short_name': "HAQ_TROPOMI_NO2_GLOBAL_M_L3",
        'version': "2.4",
        'groups': [],
        'units': "molecules/cm^2",
        'description': "Tropomi NO2 monthly mean"
    },
    # Tempo NO2 used for specific time series queries
    "TEMPO_NO2": {
        "collection_id": "C3685896708-LARC_CLOUD",
        "variables":     ["product/vertical_column_troposphere"],
        "primary_var":   "vertical_column_troposphere",
        "short_name":    "TEMPO_NO2_L3",
        "version":       "V04",
        "groups":        ["product"],
        "units":         "molecules/cm^2",
        "description":   "TEMPO tropospheric NO2 vertical column",
    }


}
@tool
def geocode_location(location_name: str) -> dict:
    """
    Convert a place name into a bounding box string 'min_lon,min_lat,max_lon,max_lat'.
    Always call this before fetch_environmental_data to get the bbox argument.

    Args:
        location_name : Place name e.g. 'New York City', 'California', 'Paris'.

    Returns:
        dict with keys: location, bbox, center_lat, center_lon.
        On failure: JSON string with key 'error'.
    """
    result = _geocoder.geocode(location_name)
    if result is None:
        return {"error": f"Could not geocode '{location_name}'"}

    # Nominatim bbox: [south, north, west, east]
    if result["bbox"] and len(result["bbox"]) == 4:
        south, north, west, east = result["bbox"]
    else:
        lat, lon = result["latitude"], result["longitude"]
        south, north, west, east = lat - 1, lat + 1, lon - 1, lon + 1

    bbox_str = f"{west:.4f},{south:.4f},{east:.4f},{north:.4f}"
    return {
        "location":   location_name,
        "bbox":       bbox_str,
        "center_lat": result["latitude"],
        "center_lon": result["longitude"],
    }





@tool
def fetch_environmental_data(
    variable: str,
    bbox: str,
    start_date: str,
    end_date: str,
    max_results: int = 10,
) -> dict:
    """
    Fetch environmental / atmospheric data from NASA Harmony (TEMPO satellite).
    Uses a local Zarr cache — repeated queries for the same parameters are instant.

    Args:
        variable    : Pollutant key e.g. 'OMI_NO2' or 'TEMPO_NO2'.
                      Available: OMI_NO2, TROPOMI_NO2, TEMPO_NO2
        bbox        : Bounding box 'min_lon,min_lat,max_lon,max_lat'
                      — always get this from geocode_location first.
        start_date  : ISO 8601 start datetime e.g. '2026-02-10T18:00:00Z'.
        end_date    : ISO 8601 end datetime   e.g. '2026-02-10T19:00:00Z'.
        max_results : Max granules to download (default 10).

    Returns:
        dict with keys:
            variable      : Variable name e.g. 'NO2'
            units         : Physical units e.g. 'molecules/cm²'
            bbox          : Bounding box string
            times         : List of ISO timestamp strings in the dataset
            n_granules    : Number of granules retrieved
            source        : Human-readable source description
            _fetch_params : Parameters for downstream plot/stat tools to reload
                            the dataset without re-downloading.
    """
    variable  = variable.upper()
    available = ", ".join(COLLECTIONS.keys())

    if variable not in COLLECTIONS:
        return {"error": f"Unknown variable '{variable}'. Available: {available}"}

    try:
        bbox_list = [float(x) for x in bbox.split(",")]
        if len(bbox_list) != 4:
            raise ValueError()
        min_lon, min_lat, max_lon, max_lat = bbox_list
    except Exception:
        return {"error": f"bbox must be 'min_lon,min_lat,max_lon,max_lat', got: '{bbox}'"}

    col = COLLECTIONS[variable]

    try:
        ds = _data_loader.download_dataset_harmony(
            collection_id = col["collection_id"],
            temporal      = (start_date, end_date),
            bounding_box  = tuple(bbox_list),
            variables     = col["variables"],
            max_results   = max_results,
        )
    except ValueError as e:
        return {"error": str(e)}
    except RuntimeError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

    try:
        times = [str(t) for t in ds.time.values] if "time" in ds.coords else []
    except Exception:
        times = []

    return {
        "variable":      variable,
        "units":         col["units"],
        "bbox":          bbox,
        "times":         times,
        "n_granules":    len(times) or 1,
        "source":        f"NASA Harmony — {col['description']}",
        "_fetch_params": {
            "variable":   variable,
            "bbox":       bbox_list,
            "start_date": start_date,
            "end_date":   end_date,
        },
    }
