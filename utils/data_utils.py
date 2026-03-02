import json
import logging
import xarray as xr
from typing import Optional

logger = logging.getLogger(__name__)
_loader_instance = None


def get_loader():
    """
    Return a shared DataLoader instance, initializing on first call.
    """
    global _loader_instance
    if _loader_instance is None:
        from preprocessing.data_loader import DataLoader
        _loader_instance = DataLoader()
    return _loader_instance


def _load_data(data_json: str) -> xr.DataArray:
    """
    Parse the JSON output of fetch_environmental_data and reload the primary
    DataArray from the Zarr cache via DataLoader.

    Parameters
    ----------
    data_json : str
        JSON string returned by fetch_environmental_data.
        Must contain '_fetch_params' with keys:
            variable   : e.g. 'NO2'
            bbox       : [min_lon, min_lat, max_lon, max_lat] as floats
            start_date : ISO 8601 string
            end_date   : ISO 8601 string

    Returns
    -------
    xr.DataArray
        Primary variable array with 'units' and 'long_name' attrs set.

    Raises
    ------
    ValueError   : Bad JSON, error key present, missing keys, unknown variable.
    RuntimeError : Dataset reloads with no data variables.
    """
    # Imported inside function to avoid circular import
    from tools.harmony_api import COLLECTIONS

    # --- Parse ---
    try:
        data = json.loads(data_json) if isinstance(data_json, str) else data_json
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in data_json: {e}")

    if "error" in data:
        raise ValueError(f"Data fetch previously failed: {data['error']}")

    if "_fetch_params" not in data:
        raise ValueError(
            "Missing '_fetch_params' — pass the direct output of fetch_environmental_data."
        )

    params  = data["_fetch_params"]
    missing = [k for k in ("variable", "bbox", "start_date", "end_date") if k not in params]
    if missing:
        raise ValueError(f"'_fetch_params' is missing required keys: {missing}")

    variable  = params["variable"].upper()
    bbox_list = params["bbox"]
    start     = params["start_date"]
    end       = params["end_date"]

    if variable not in COLLECTIONS:
        raise ValueError(
            f"Unknown variable '{variable}'. Available: {', '.join(COLLECTIONS)}"
        )

    col = COLLECTIONS[variable]

    logger.info(f"Reloading {variable} from cache: {start} → {end}")
    try:
        ds = get_loader().download_dataset_harmony(
            collection_id = col["collection_id"],
            temporal      = (start, end),
            bounding_box  = tuple(bbox_list),
            variables     = col["variables"],
        )
    except Exception as e:
        raise RuntimeError(f"Failed to reload dataset for '{variable}': {e}") from e

    # --- Select primary DataArray ---
    data_vars = list(ds.data_vars)
    if not data_vars:
        raise RuntimeError(
            f"Dataset for '{variable}' has no data variables. "
            f"Check COLLECTIONS config."
        )

    primary_var = col.get("primary_var")

    preferred = next(
        (v for v in data_vars if v == primary_var),  # exact match first
        next(
            (v for v in data_vars if variable.lower() in v.lower()),  # fallback
            data_vars[0]  # last resort
        )
    )
    logger.debug(f"Selected '{preferred}' from {data_vars}")

    da = ds[preferred]
    da.attrs.setdefault("units",     col["units"])
    da.attrs.setdefault("long_name", col["description"])
    da.name = variable

    return da