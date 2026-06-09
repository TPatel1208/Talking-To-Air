import json
import logging
import xarray as xr

from preprocessing.aggregation_service import AggregationService

logger = logging.getLogger(__name__)
_loader_instance = None
_aggregation_service = AggregationService()



def get_loader():
    global _loader_instance
    if _loader_instance is None:
        from tools.satellite_tools.harmony_api import _get_data_loader
        _loader_instance = _get_data_loader()
    return _loader_instance


def _load_data(data_json, apply_quality_flag: bool = True) -> xr.DataArray:
    from tools.satellite_tools.harmony_api import COLLECTIONS

    # Coerce input to a plain dict regardless of how LangChain serialised it:
    #   • DataDict Pydantic object  → .model_dump()
    #   • JSON string (LLM output)  → json.loads()
    #   • plain dict                → use as-is
    if hasattr(data_json, "model_dump"):
        data = data_json.model_dump()
    elif isinstance(data_json, str):
        data = json.loads(data_json)
    else:
        data = data_json

    if "fetch_params" not in data:
        raise ValueError("Missing 'fetch_params' — pass the direct output of fetch_environmental_data.")

    params  = data["fetch_params"]
    missing = [k for k in ("variable", "bbox", "start_date", "end_date") if k not in params]
    if missing:
        raise ValueError(f"'fetch_params' is missing required keys: {missing}")

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
        "max_results":   params.get("max_results", 10),
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
    da = _aggregation_service.apply_quality_mask(
        da,
        ds,
        col,
        apply_quality_flag=apply_quality_flag,
        variable=variable,
    )

    da.attrs.setdefault("units",     col["units"])
    da.attrs.setdefault("long_name", col["description"])
    da.name = variable

    if apply_quality_flag and qf_var and qf_var not in ds.data_vars:
        logger.warning(
            f"Quality flag variable '{qf_var}' not found in dataset. "
            f"Available vars: {data_vars}. Proceeding without quality masking."
        )

    return da
