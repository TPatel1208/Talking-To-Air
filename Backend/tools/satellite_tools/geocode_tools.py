from langchain.tools import tool

from utils.plotting import get_geocoding_service
from utils.streaming import emit_status
from tools.satellite_tools.query_parser import is_valid_location_candidate


def _get_geocoder():
    return get_geocoding_service()


@tool
async def geocode_location(location_name: str) -> dict:
    """
    Convert a place name into a bounding box string 'min_lon,min_lat,max_lon,max_lat'.

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
