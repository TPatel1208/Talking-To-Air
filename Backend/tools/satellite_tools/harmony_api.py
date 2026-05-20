import sys
import os
from langchain.tools import tool
from typing import Tuple
import numpy as np
import requests
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.plotting import GeocodingService
from preprocessing.data_loader import DataLoader

_geocoder = GeocodingService()
_data_loader = DataLoader()

COLLECTIONS = {
    # OMI NO2 (Default for 'NO2' variable)
    "OMI_NO2": {
        'collection_id': "C1266136111-GES_DISC",
        'variables': ["ColumnAmountNO2","ColumnAmountNO2CloudScreened","ColumnAmountNO2TropCloudScreened","Weight"],
        'primary_var': "ColumnAmountNO2TropCloudScreened",
        'quality_flag_var': None,
        'short_name': "OMI_MINDS_NO2d",
        'version': "1.1",
        'groups': [],
        'units': "molecules/cm^2",
        'description': "OMI NO2 tropospheric column",
        'supports_variable_subsetting': False,
        'fill_value': -1.267651e+30,       # from _FillValue attribute
        'valid_min': -1e15,                # from valid_min attribute
        'valid_max': 1e18,                 # from valid_max attribute
    },
    # Tropomi NO2 Monthly
    "TROPOMI_NO2": {
        'collection_id': "C3087325222-GES_DISC",
        'variables': ["Tropospheric_NO2",'Number_obs'],
        'primary_var': "Tropospheric_NO2",
        'quality_flag_var': None,
        'short_name': "HAQ_TROPOMI_NO2_GLOBAL_M_L3",
        'version': "2.4",
        'groups': [],
        'units': "molecules/cm^2",
        'description': "Tropomi NO2 monthly mean",
        'supports_variable_subsetting': False,
        'fill_value': -999.0,                # from _FillValue attribute
        'valid_min': -1e15,             
        'valid_max': 1e18,                 
    },
    # Tempo NO2 used for specific time series queries
    "TEMPO_NO2": {
        "collection_id": "C3685896708-LARC_CLOUD",
        "variables":     ["product/vertical_column_troposphere",
                          "product/main_data_quality_flag"],
        "primary_var":   "vertical_column_troposphere",
        "quality_flag_var": "main_data_quality_flag", 
        "short_name":    "TEMPO_NO2_L3",
        "version":       "V04",
        "groups":        ["product"],
        "units":         "molecules/cm^2",
        "description":   "TEMPO tropospheric NO2 vertical column",
        'supports_variable_subsetting': True,
        'fill_value': np.float32(-1e30),         # from _FillValue attribute
        "valid_min": -1e15,               
        "valid_max": 1e18,
    },

    #--------------
    #Ozone datasets
    #--------------

    # TEMPO Total Ozone
    "TEMPO_O3TOT": {
        "collection_id": "C3685896625-LARC_CLOUD",
        "variables": [
            "product/column_amount_o3",
            "product/radiative_cloud_frac",
            "product/fc",
            "product/o3_below_cloud",
            "product/so2_index",
            "product/uv_aerosol_index",
        ],
        "primary_var": "column_amount_o3",
        "quality_flag_var": None,
        "short_name": "TEMPO_O3TOT_L3",
        "version": "V04",
        "groups": ["product"],
        "units": "DU",
        "description": "TEMPO Level 3 total ozone column",
        "supports_variable_subsetting": True,
        "fill_value": np.float32(-1e30),
        "valid_min": 50.0,        # valid_min on column_amount_o3
        "valid_max": 700.0,       # valid_max on column_amount_o3
    },

    # OMI Total Ozone
    "OMI_O3": {
        "collection_id":  "C1266136037-GES_DISC",
        "variables":      [],
        "primary_var":    "ColumnAmountO3",
        "quality_flag_var": None,
        "short_name":     "OMDOAO3e",
        "version":        "003",
        "groups":         [],
        "units":          "DU",
        "description":    "OMI daily total ozone column",
        "supports_variable_subsetting": False,
        "fill_value":     -1.267651e+30,
        "valid_min":      50.0,
        "valid_max":      700.0,
    },
    #---------------------
    #Formaldehyde datasets
    #---------------------
    "TEMPO_HCHO": {
        "collection_id": "C3685897141-LARC_CLOUD",
        "variables": [
            "product/vertical_column",
            "product/vertical_column_uncertainty",
            "product/main_data_quality_flag",
        ],
        "primary_var":        "vertical_column",
        "quality_flag_var":   "main_data_quality_flag",
        "short_name":         "TEMPO_HCHO_L3",
        "version":            "V04",
        "groups":             ["product"],
        "units":              "molecules/cm^2",
        "description":        "TEMPO formaldehyde (HCHO) vertical column",
        "supports_variable_subsetting": True,
        "fill_value":  -1e30,
        "valid_min":   np.float32(0.0),
        "valid_max":   np.inf,
    },

    "TEMPO_HCHO_V03": {
        "collection_id": "C2930761273-LARC_CLOUD",
        "variables": [
            "product/vertical_column",
            "product/vertical_column_uncertainty",
            "product/main_data_quality_flag",
        ],
        "primary_var":        "vertical_column",
        "quality_flag_var":   "main_data_quality_flag",
        "short_name":         "TEMPO_HCHO_L3",
        "version":            "V03",
        "groups":             ["product"],
        "units":              "molecules/cm^2",
        "description":        "TEMPO formaldehyde (HCHO) vertical column (V03)",
        "supports_variable_subsetting": True,
        "fill_value":  -1e30,
        "valid_min":   np.float32(0.0),
        "valid_max":   np.inf,
    },
    "OMI_HCHO": {
        "collection_id":  "C1626121562-GES_DISC",
        "variables":      [],
        "primary_var":    "column_amount",
        "quality_flag_var": "data_quality_flag",
        "short_name":     "OMHCHOd",
        "version":        "003",
        "groups":         ["key_science_data", "qa_statistics"],
        "units":          "molecules/cm^2",
        "description":    "OMI HCHO total column daily",
        "supports_variable_subsetting": False,
        "fill_value":     -1e30,
        "valid_min":      np.float32(0.0),
        "valid_max":      np.inf,
    },
    #Modis Aerosol datasets
    "MODIS_AOD_TERRA": {
        "collection_id":  "C3618500076-GES_DISC",  # verify with CMR search
        "variables":      [],
        "primary_var":    "COMBINE_AOD_550_AVG",
        "quality_flag_var": None,
        "short_name":     "AER_DBDT_D10KM_L3_MODIS_TERRA",
        "version":        "001",
        "groups":         [],
        "units":          "AOD (550nm)",
        "description":    "MODIS Terra AOD at 550nm combined Dark Target + Deep Blue",
        "supports_variable_subsetting": False,
        "fill_value":     -999.0,
        "valid_min":      -0.05,
        "valid_max":      5.0,
    },
    "MODIS_AOD_AQUA": {
        "collection_id":  "C3618504061-GES_DISC",  
        "variables":      [],
        "primary_var":    "COMBINE_AOD_550_AVG",
        "quality_flag_var": None,
        "short_name":     "AER_DBDT_D10KM_L3_MODIS_AQUA",
        "version":        "001",
        "groups":         [],
        "units":          "AOD (550nm)",
        "description":    "MODIS Aqua AOD at 550nm combined Dark Target + Deep Blue",
        "supports_variable_subsetting": False,
        "fill_value":     -999.0,
        "valid_min":      -0.05,
        "valid_max":      5.0,
    },
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
        max_results : Change this to suite the estimated amount for a query (default 10).

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
    fetch_params = {
        "collection_id":  col["collection_id"],
        "bounding_box":   tuple(bbox_list),
        'temporal':    (start_date, end_date),
        "max_results": max_results,
        "cache_path": "./data/cache.zarr"

    }
    if col.get("supports_variable_subsetting", False):
        fetch_params["variables"] = col["variables"]
    try:
        ds = _data_loader.download_dataset_harmony(**fetch_params)
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


@tool
def check_data_availability(
    variable: str,
    bbox: str,
    start_date: str,
    end_date: str,
)-> dict:
    """
    Check if granules exist for a variable over a location and time range BEFORE fetching. Returns a list of available dates so the agent can inform the user exactly which days have data.

    Always call this before fetch_environmental_data when:
    - Unsure if data exists for a location or time period
    - A previous fetch returned no granules
    - User asks about data availability
    - You want to suggest alternatives or broader ranges    

    Args:
        variable    : Pollutant key e.g. 'OMI_NO2' or 'TEMPO_NO2'.
                      Available: OMI_NO2, TROPOMI_NO2, TEMPO_NO2, MODIS_AOD_TERRA, MODIS_AOD_AQUA, OMI_O3, TEMPO_O3TOT, TEMPO_HCHO, TEMPO_HCHO_V03, OMI_HCHO
        bbox        : Bounding box 'min_lon,min_lat,max_lon,max_lat'
                      — always get this from geocode_location first.
        start_date  : ISO 8601 start datetime e.g. '2026-02-10T18:00:00Z'.
        end_date    : ISO 8601 end datetime   e.g. '2026-02-10T19:00:00Z'.  

    Returns:
        dict with keys:
            variable      : Variable name e.g. 'NO2'
            num_granules    : Number of granules available for the query
            dates_available: List of ISO date strings for which data is available

    """
    variable = variable.upper()
    try:
        col = COLLECTIONS[variable]
    except KeyError:
        return {"error": f"Unknown variable '{variable}'. Available: {', '.join(COLLECTIONS.keys())}"}

    try:
        bbox_list = [float(x) for x in bbox.split(",")]
        if len(bbox_list) != 4:
            raise ValueError()
    except Exception:
        return {"error": f"bbox must be 'min_lon,min_lat,max_lon,max_lat', got: '{bbox}'"}

    # Ensure correct format for CMR temporal parameter
    def _fmt(dt: str) -> str:
        dt = dt.strip()
        return dt if dt.endswith('Z') else dt + 'Z'

    params = {
        'concept_id':   col["collection_id"],
        'temporal[]':   f"{_fmt(start_date)},{_fmt(end_date)}",
        'bounding_box': bbox,
        'page_size':    100,
        'sort_key':     'start_date',
    }

    try:
        resp = requests.get(
            "https://cmr.earthdata.nasa.gov/search/granules.json",
            params=params,
            timeout=15
        )
        resp.raise_for_status()

        total   = int(resp.headers.get('CMR-Hits', 0))
        entries = resp.json().get('feed', {}).get('entry', [])

        # Build dates list
        dates_available = [
            {
                'start': g.get('time_start'),
                'end':   g.get('time_end'),
            }
            for g in entries if g.get('time_start')
        ]

        return {
            'variable':       variable,
            'num_granules':   total,
            'dates_available': dates_available,
        }

    except Exception as e:
        return {'error': str(e)}


def main():
    # Step 1: Geocode
    location_info = geocode_location.invoke({
        "location_name": "New York City"
    })
    print("Geocoding result:", location_info)

    if "error" in location_info:
        print("Geocoding failed, cannot proceed with data fetch.")
        return

    bbox = location_info["bbox"]

    # Step 2: Check availability (FIXED ARGUMENT NAMES)
    availability = check_data_availability.invoke({
        "variable": "OMI_NO2",
        "bbox": bbox,
        "start_date": "2026-01-01T00:00:00Z",
        "end_date": "2026-01-02T23:59:59Z"
    })

    print("Data availability:", availability)


if __name__ == "__main__":
    main()