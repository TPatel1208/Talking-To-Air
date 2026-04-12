import json
import logging
import xarray as xr
import numpy as np

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
        # Use the variables from COLLECTIONS config (already includes quality flags)
        fetch_params["variables"] = list(col.get("variables", []))
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
    fill_value = col.get("fill_value")
    valid_min  = col.get("valid_min")
    valid_max  = col.get("valid_max")

    # Float32 fill values don't compare exactly in float64.
    # Use the _FillValue stored in the DataArray's own attrs first,
    # then fall back to the COLLECTIONS config.
    actual_fill = da.attrs.get("_FillValue", fill_value)

    if actual_fill is not None:
        da = da.where(~np.isclose(da, actual_fill, rtol=0, atol=abs(actual_fill) * 1e-3))

    if valid_min is not None and valid_max is not None:
        da = da.where((da >= valid_min) & (da <= valid_max))

    da.attrs.setdefault("units",     col["units"])
    da.attrs.setdefault("long_name", col["description"])
    da.name = variable

    # --- Apply quality flag mask ---
    if apply_quality_flag and qf_var and qf_var in ds.data_vars:
        qf = ds[qf_var]
        if variable == "OMI_HCHO":
            bad_mask = (qf == 2)  # 0 and 1 are both good, 2 is bad
        else:
            bad_mask = (qf != 0)  # TEMPO: only 0 is good
        n_bad = int(bad_mask.sum())
        logger.debug(f"Quality flag masking removed {n_bad} bad pixels")
        da = da.where(~bad_mask)
    elif apply_quality_flag and qf_var:
        logger.warning(
            f"Quality flag variable '{qf_var}' not found in dataset. "
            f"Available vars: {data_vars}. Proceeding without quality masking."
        )

    return da