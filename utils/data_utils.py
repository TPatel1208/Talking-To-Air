import json
import logging
import xarray as xr

logger = logging.getLogger(__name__)
_loader_instance = None



def get_loader():
    global _loader_instance
    if _loader_instance is None:
        from preprocessing.data_loader import DataLoader
        _loader_instance = DataLoader()
    return _loader_instance


def _load_data(data_json: dict, apply_quality_flag: bool = True) -> xr.DataArray:
    from tools.harmony_api import COLLECTIONS

    data = data_json if isinstance(data_json, dict) else json.loads(data_json)

    if "_fetch_params" not in data:
        raise ValueError("Missing '_fetch_params' — pass the direct output of fetch_environmental_data.")

    params  = data["_fetch_params"]
    missing = [k for k in ("variable", "bbox", "start_date", "end_date") if k not in params]
    if missing:
        raise ValueError(f"'_fetch_params' is missing required keys: {missing}")

    variable  = params["variable"].upper()
    bbox_list = params["bbox"]
    start     = params["start_date"]
    end       = params["end_date"]

    if variable not in COLLECTIONS:
        raise ValueError(f"Unknown variable '{variable}'. Available: {', '.join(COLLECTIONS)}")

    col    = COLLECTIONS[variable]
    qf_var = col.get("quality_flag_var")  # None for OMI/TROPOMI, string for TEMPO

    logger.info(f"Reloading {variable} from cache: {start} → {end}")

    fetch_params = {
        "collection_id": col["collection_id"],
        "temporal":      (start, end),
        "bounding_box":  tuple(bbox_list),
        "cache_path":    "./data/cache.zarr",
    }

    if col.get("supports_variable_subsetting", False):
        base_vars = list(col.get("variables", []))
        if apply_quality_flag and qf_var and qf_var not in base_vars:
            base_vars = base_vars + [qf_var]
        fetch_params["variables"] = base_vars

    try:
        ds = get_loader().download_dataset_harmony(**fetch_params)
    except Exception as e:
        raise RuntimeError(f"Failed to reload dataset for '{variable}': {e}") from e

    data_vars = list(ds.data_vars)
    if not data_vars:
        raise RuntimeError(f"Dataset for '{variable}' has no data variables. Check COLLECTIONS config.")

    primary_var = col.get("primary_var")
    preferred = next(
        (v for v in data_vars if v == primary_var),
        next(
            (v for v in data_vars if variable.lower() in v.lower()),
            data_vars[0]
        )
    )
    logger.debug(f"Selected '{preferred}' from {data_vars}")

    da = ds[preferred]
    da.attrs.setdefault("units",     col["units"])
    da.attrs.setdefault("long_name", col["description"])
    da.name = variable

    # --- Apply quality flag mask ---
    if apply_quality_flag and qf_var and qf_var in ds.data_vars:
        qf = ds[qf_var]
        n_bad = int((qf != 0).sum())
        logger.debug(f"Quality flag masking removed {n_bad} bad pixels")
        da = da.where(qf == 0)
    elif apply_quality_flag and qf_var:
        logger.warning(
            f"Quality flag variable '{qf_var}' not found in dataset. "
            f"Available vars: {data_vars}. Proceeding without quality masking."
        )

    return da