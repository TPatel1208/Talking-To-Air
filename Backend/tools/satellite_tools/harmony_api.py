import sys
import os
import asyncio
from langchain.tools import tool
from typing import Tuple
import httpx
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.plotting import get_geocoding_service
from utils.streaming import emit_status
from preprocessing.data_loader import DataLoader, _bounded_max_results
from tools.satellite_tools.models import DataDict
from tools.satellite_tools.query_parser import is_valid_location_candidate
from datasets.registry import load_registry

_data_loader = None


def _get_geocoder():
    return get_geocoding_service()


def _get_data_loader():
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader()
    return _data_loader

def _load_collections() -> dict:
    """Expose collections.yaml as the legacy dict shape used by tool modules."""
    return {
        key: config.model_dump()
        for key, config in load_registry().items()
    }


COLLECTIONS = _load_collections()
@tool
async def geocode_location(location_name: str) -> dict:
    """
    Convert a place name into a bounding box string 'min_lon,min_lat,max_lon,max_lat'.
    Always call this before fetch_environmental_data to get the bbox argument.

    Args:
        location_name : Place name e.g. 'New York City', 'California', 'Paris'.

    Returns:
        dict with keys: location, bbox, center_lat, center_lon.
    """
    emit_status("Resolving requested location...")
    if not is_valid_location_candidate(location_name):
        emit_status("Location lookup failed.")
        return {"error": f"Invalid location candidate '{location_name}'"}

    result = await _get_geocoder().ageocode(location_name)
    if result is None:
        emit_status("Location lookup failed.")
        return {"error": f"Could not geocode '{location_name}'"}

    # Nominatim bbox: [south, north, west, east]
    if result["bbox"] and len(result["bbox"]) == 4:
        south, north, west, east = result["bbox"]
    else:
        lat, lon = result["latitude"], result["longitude"]
        south, north, west, east = lat - 1, lat + 1, lon - 1, lon + 1

    bbox_str = f"{west:.4f},{south:.4f},{east:.4f},{north:.4f}"
    emit_status("Location identified.")
    return {
        "location":   location_name,
        "bbox":       bbox_str,
        "center_lat": result["latitude"],
        "center_lon": result["longitude"],
    }





@tool
async def fetch_environmental_data(
    variable: str,
    bbox: str,
    start_date: str,
    end_date: str,
    max_results: int = 10,
) -> dict:
    """
    Fetch environmental / atmospheric data from NASA Harmony.

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
    emit_status("Preparing satellite data request...")
    variable  = variable.upper()
    max_results = _bounded_max_results(max_results)
    available = ", ".join(COLLECTIONS.keys())

    if variable not in COLLECTIONS:
        emit_status("Satellite data request failed.")
        return {"error": f"Unknown variable '{variable}'. Available: {available}"}

    try:
        bbox_list = [float(x) for x in bbox.split(",")]
        if len(bbox_list) != 4:
            raise ValueError()
        min_lon, min_lat, max_lon, max_lat = bbox_list
    except Exception:
        emit_status("Satellite data request failed.")
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
        emit_status("Downloading satellite granules...")
        ds = await asyncio.to_thread(_get_data_loader().download_dataset_harmony, **fetch_params)
        emit_status("Processing downloaded data...")
    except ValueError as e:
        emit_status("Download failed while retrieving NASA Harmony output.")
        return {"error": str(e)}
    except RuntimeError as e:
        emit_status("Download failed while retrieving NASA Harmony output.")
        return {"error": str(e)}
    except Exception as e:
        emit_status("Download failed while retrieving NASA Harmony output.")
        return {"error": f"Unexpected {type(e).__name__}: {str(e)}"}

    try:
        times = [str(t) for t in ds.time.values] if "time" in ds.coords else []
    except Exception:
        times = []
    n_granules = int(ds.attrs.get("n_granules", len(times) or 1))
    cadence = str(ds.attrs.get("cadence", col.get("cadence", "daily")))

    # Build a clean, JSON-safe fetch_params for storage on DataDict.
    # The internal fetch_params (used above for download) contains tuples
    # which are not JSON-serialisable — don't reuse it here.
    serialisable_fetch_params = {
        "variable":     variable,
        "start_date":   start_date,
        "end_date":     end_date,
        "bbox":         bbox_list,
        "bounding_box": bbox_list,
        "cache_path":   "./data/cache.zarr",
        "max_results":  max_results,
    }
    if col.get("supports_variable_subsetting", False):
        serialisable_fetch_params["variables"] = list(col["variables"])

    emit_status("Satellite data ready.")
    return DataDict(
        variable=variable,
        units=col["units"],
        bbox=bbox,
        times=times,
        n_granules=n_granules,
        cadence=cadence,
        source=f"NASA Harmony — {col['description']}",
        fetch_params=serialisable_fetch_params,
    )


@tool
async def check_data_availability(
    variable: str,
    bbox: str,
    start_date: str,
    end_date: str,
)-> dict:
    """
    Check if granules exist for a variable over a location and time range BEFORE fetching. Returns a list of available dates so the agent can inform the user exactly which days have data.  

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
    emit_status("Checking satellite data availability...")
    variable = variable.upper()
    try:
        col = COLLECTIONS[variable]
    except KeyError:
        emit_status("Satellite data availability check failed.")
        return {"error": f"Unknown variable '{variable}'. Available: {', '.join(COLLECTIONS.keys())}"}

    try:
        bbox_list = [float(x) for x in bbox.split(",")]
        if len(bbox_list) != 4:
            raise ValueError()
    except Exception:
        emit_status("Satellite data availability check failed.")
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
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://cmr.earthdata.nasa.gov/search/granules.json",
                params=params,
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

        emit_status("Satellite data availability checked.")
        return {
            'variable':       variable,
            'num_granules':   total,
            'dates_available': dates_available,
        }

    except Exception as e:
        emit_status("Satellite data availability check failed.")
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
