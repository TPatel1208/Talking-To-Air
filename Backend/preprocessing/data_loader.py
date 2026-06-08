"""
preprocessing/data_loader.py
=============================
Thin orchestrator for the fetch pipeline.

Fetch order (default)
---------------------
1. Cache lookup (PostGIS → Zarr)
2. Provider routing:
   a. LARC_CLOUD + supports_variable_subsetting → Harmony (variable sub-
      setting — fast, server-side; skips S3 unless S3_FORCE_FETCH=1 and
      running in us-west-2)
   b. LARC_CLOUD (no variable subsetting) → S3FetchService
      (raises S3OutsideRegionError outside us-west-2 → falls back to c)
   c. GES_DISC → OPeNDAPFetchService (direct CE, not pydap)
   d. Any provider → Harmony as last resort
3. CacheManager.store

Routing override
----------------
Set ``DATA_FETCH_MODE`` in the environment to lock the fetch path:

  DATA_FETCH_MODE=auto      — default: Harmony primary, provider fallback
  DATA_FETCH_MODE=harmony   — force Harmony only; no provider fallback
  DATA_FETCH_MODE=opendap   — force OPeNDAP CE (GES_DISC only)
  DATA_FETCH_MODE=s3        — force S3 (requires S3_FORCE_FETCH=1
                              outside us-west-2 or you'll get an error)

No file-parsing logic lives here. All normalisation is in DatasetParser.
All cache logic is in CacheManager.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from collections import OrderedDict
from typing import List, Optional, Tuple

import xarray as xr

from preprocessing.cache_manager import CacheManager, make_group_key, _normalise_bbox
from preprocessing.dataset_parser import DatasetParser
from repositories.cache_index_repository import CacheIndexRepository
from repositories.zarr_repository import ZarrRepository
from config.settings import get_settings
from datasets.registry import load_registry
from utils.earthaccess_client import get_earthaccess_auth

logger = logging.getLogger(__name__)

DOWNLOAD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "downloads"
)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Collections that have no embedded time coordinate in their files.
_NEEDS_GRANULE_TIMES: set[str] = {
    "C3087325222-GES_DISC",  # TROPOMI_NO2
}

_LARC_CLOUD_SUFFIX = "-LARC_CLOUD"
_GES_DISC_SUFFIX = "-GES_DISC"

# Valid values for DATA_FETCH_MODE
_VALID_MODES = {"auto", "harmony", "opendap", "s3"}
_DEFAULT_MAX_RESULTS_CAP = 20
_MEMORY_CACHE_MAX_ITEMS = 8


def _provider(collection_id: str) -> str:
    if collection_id.endswith(_LARC_CLOUD_SUFFIX):
        return "LARC_CLOUD"
    if collection_id.endswith(_GES_DISC_SUFFIX):
        return "GES_DISC"
    return "UNKNOWN"


def _fetch_mode() -> str:
    """Read DATA_FETCH_MODE from centralized settings."""
    mode = get_settings().data_fetch_mode
    if mode not in _VALID_MODES:
        logger.warning(
            "Unknown DATA_FETCH_MODE=%r — falling back to 'auto'. "
            "Valid values: %s",
            mode, ", ".join(sorted(_VALID_MODES)),
        )
        return "auto"
    return mode


def _max_results_cap() -> int:
    """Read the safety cap for provider result counts."""
    return get_settings().satellite_max_results_cap


def _bounded_max_results(max_results: int) -> int:
    """Keep LLM/tool input from requesting an unexpectedly large fetch."""
    try:
        requested = int(max_results)
    except (TypeError, ValueError):
        logger.warning("Invalid max_results=%r; using 10", max_results)
        return 10

    if requested < 1:
        logger.warning("Invalid max_results=%r; using 1", max_results)
        return 1

    cap = _max_results_cap()
    if requested > cap:
        logger.warning("Capping max_results from %d to %d", requested, cap)
        return cap
    return requested


class DataLoader:

    def __init__(self, cache_path: str = "./data/cache.zarr"):
        # Auth is lazy: only satellite paths that need EarthAccess log in.
        self.auth = None

        self._default_cache_path = cache_path
        self._parser = DatasetParser()

        # Lazy fetch services — instantiated on first use
        self._s3_service = None
        self._opendap_service = None
        self._harmony_service = None

        # Cache stack
        zarr_repo = ZarrRepository(cache_path)
        try:
            index_repo = CacheIndexRepository()
        except Exception as exc:
            logger.warning("PostGIS index unavailable — Zarr-only mode (%s)", exc)
            index_repo = None
        self._cache = CacheManager(zarr_repo, index_repository=index_repo)
        self._memory_cache: OrderedDict[str, xr.Dataset] = OrderedDict()

        # Registry (lru_cached after first load)
        self._registry_by_id: dict = {
            cfg.collection_id: cfg
            for cfg in load_registry().values()
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def download_dataset_harmony(
        self,
        collection_id: str,
        temporal: Tuple[str, str],
        bounding_box: Optional[Tuple[float, float, float, float]] = None,
        variables: Optional[List[str]] = None,
        max_results: int = 10,
        output_format: str = "application/x-netcdf4",
        cache_path: str = "./data/cache.zarr",
    ) -> xr.Dataset:
        """
        Return a dataset for the requested collection / time range / bbox.

        See module docstring for the fetch order and override options.
        The method name is kept as-is so nothing in the agent/tool layer changes.
        """
        # ── Cache lookup ──────────────────────────────────────────────────
        max_results = _bounded_max_results(max_results)
        bounding_box = _normalise_bbox(bounding_box) if bounding_box else None
        memory_key = make_group_key(
            collection_id,
            temporal[0],
            temporal[1],
            bounding_box if bounding_box else (),
        )

        cached_in_memory = self._memory_cache.get(memory_key)
        if cached_in_memory is not None:
            self._memory_cache.move_to_end(memory_key)
            logger.info(
                "cache_hit",
                extra={"_tier": "memory", "_collection_id": collection_id, "_group_key": memory_key},
            )
            return cached_in_memory

        cached = self._cache.lookup(
            collection_id=collection_id,
            temporal=temporal,
            bbox=bounding_box,
        )
        if cached is not None:
            logger.info("cache_hit", extra={"_collection_id": collection_id})
            self._remember_dataset(memory_key, cached)
            return cached

        # ── Fetch ─────────────────────────────────────────────────────────
        mode = _fetch_mode()
        provider = _provider(collection_id)
        col = self._registry_by_id.get(collection_id)

        logger.info(
            "cache_miss",
            extra={"_collection_id": collection_id, "_provider": provider, "_mode": mode},
        )
        t0 = time.time()

        ds = self._route(
            mode=mode,
            provider=provider,
            col=col,
            collection_id=collection_id,
            temporal=temporal,
            bounding_box=bounding_box,
            variables=variables,
            max_results=max_results,
            output_format=output_format,
        )

        logger.info("Fetch complete in %.2fs", time.time() - t0)

        # ── Cache write ───────────────────────────────────────────────────
        self._cache.store(ds, collection_id, temporal, bounding_box, cache_path)
        self._remember_dataset(memory_key, ds)
        return ds

    def _remember_dataset(self, key: str, ds: xr.Dataset) -> None:
        self._memory_cache[key] = ds
        self._memory_cache.move_to_end(key)
        while len(self._memory_cache) > _MEMORY_CACHE_MAX_ITEMS:
            self._memory_cache.popitem(last=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Routing
    # ─────────────────────────────────────────────────────────────────────────

    def _route(
        self,
        mode: str,
        provider: str,
        col,
        collection_id: str,
        temporal,
        bounding_box,
        variables,
        max_results,
        output_format,
    ) -> xr.Dataset:
        """
        Select and execute the appropriate fetch strategy, with Harmony
        as the final fallback for any strategy that raises.
        """
        from services.s3_fetch_service import S3OutsideRegionError

        # ── Explicit overrides ────────────────────────────────────────────
        if mode == "harmony":
            return self._fetch_harmony_fallback(
                collection_id, temporal, bounding_box, variables, max_results, output_format
            )

        if mode == "opendap":
            return self._fetch_opendap(
                collection_id, temporal, bounding_box, variables, max_results
            )

        if mode == "s3":
            group = col.groups[0] if col and col.groups else None
            return self._fetch_s3(
                collection_id, temporal, bounding_box, group, max_results
            )

        # ── Auto routing ──────────────────────────────────────────────────
        # Strategy: Harmony is the primary fetch path for every collection.
        # Provider-native paths are fallbacks only: S3 for LARC_CLOUD and
        # OPeNDAP for GES_DISC.

        supports_var_sub = col.supports_variable_subsetting if col else False
        col_variables = col.variables if col else None
        harmony_variables = variables
        if supports_var_sub:
            harmony_variables = variables or col_variables or None
        elif variables:
            logger.info(
                "Ignoring variable subset for %s because collection metadata "
                "does not advertise Harmony variable subsetting",
                collection_id,
            )
            harmony_variables = None

        harmony_error = None
        try:
            return self._fetch_harmony_fallback(
                collection_id, temporal, bounding_box,
                harmony_variables, max_results, output_format
            )
        except Exception as exc:
            harmony_error = exc
            logger.warning(
                "Harmony primary fetch failed for %s (%s) — trying %s fallback",
                collection_id, harmony_error, provider,
            )

        if provider == "LARC_CLOUD":
            group = col.groups[0] if col and col.groups else None
            try:
                return self._fetch_s3(
                    collection_id, temporal, bounding_box, group, max_results
                )
            except S3OutsideRegionError as exc:
                raise RuntimeError(
                    "Harmony primary fetch failed, and S3 fallback is unavailable "
                    f"outside us-west-2: Harmony={harmony_error}; S3={exc}"
                ) from harmony_error
            except Exception as exc:
                raise RuntimeError(
                    "Harmony primary fetch and S3 fallback both failed: "
                    f"Harmony={harmony_error}; S3={exc}"
                ) from exc

        elif provider == "GES_DISC":
            try:
                return self._fetch_opendap(
                    collection_id, temporal, bounding_box, variables, max_results
                )
            except Exception as exc:
                raise RuntimeError(
                    "Harmony primary fetch and OPeNDAP fallback both failed: "
                    f"Harmony={harmony_error}; OPeNDAP={exc}"
                ) from exc

        else:
            raise RuntimeError(
                f"Harmony primary fetch failed for {collection_id}, and no "
                f"fallback is configured for provider={provider}: {harmony_error}"
            ) from harmony_error

    # ─────────────────────────────────────────────────────────────────────────
    # Fetch strategies
    # ─────────────────────────────────────────────────────────────────────────

    def _fetch_s3(self, collection_id, temporal, bbox, group, max_results):
        self.auth = get_earthaccess_auth()
        if self._s3_service is None:
            from services.s3_fetch_service import S3FetchService
            self._s3_service = S3FetchService()
        return self._s3_service.fetch(
            collection_id=collection_id,
            temporal=temporal,
            bbox=bbox,
            group=group,
            max_results=max_results,
        )

    def _fetch_opendap(self, collection_id, temporal, bbox, variables, max_results):
        if self._opendap_service is None:
            from services.opendap_fetch_service import OPeNDAPFetchService
            self._opendap_service = OPeNDAPFetchService()
        return self._opendap_service.fetch(
            collection_id=collection_id,
            temporal=temporal,
            bbox=bbox,
            variables=variables,
            max_results=max_results,
        )

    def _fetch_harmony_fallback(
        self, collection_id, temporal, bbox, variables, max_results, output_format
    ) -> xr.Dataset:
        """
        Harmony path — fast for tight bounding boxes (server-side subsetting,
        small pre-clipped NetCDF4 response).  Also serves as the universal
        fallback for any provider.
        """
        if self._harmony_service is None:
            from services.async_harmony_service import AsyncHarmonyService
            self._harmony_service = AsyncHarmonyService()

        files = self._harmony_service.submit_and_download_sync(
            collection_id=collection_id,
            temporal=temporal,
            bbox=bbox,
            variables=variables,
            max_results=max_results,
            output_format=output_format,
            download_dir=DOWNLOAD_DIR,
        )

        if collection_id in _NEEDS_GRANULE_TIMES:
            granule_times = self._get_granule_times(collection_id, temporal, bbox)
        else:
            granule_times = {}

        datasets = []
        for path in files:
            try:
                datasets.append(self._parser.parse_granule(str(path), granule_times))
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", path, exc)
            finally:
                path.unlink(missing_ok=True)

        if not datasets:
            raise RuntimeError("Harmony returned no parseable datasets")

        if len(datasets) == 1:
            combined = datasets[0]
        else:
            valid = [d for d in datasets if "time" in d.dims or "time" in d.coords]
            if not valid:
                raise RuntimeError("No Harmony granules had a time coordinate")
            combined = xr.concat(valid, dim="time")

        for coord in combined.coords:
            combined[coord].attrs.pop("units", None)
            combined[coord].attrs.pop("calendar", None)

        return combined

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_granule_times(collection_id, temporal, bbox) -> dict:
        """CMR time lookup — only called for collections without embedded time."""
        import earthaccess
        import pandas as pd
        get_earthaccess_auth()
        try:
            params = {"concept_id": collection_id, "temporal": temporal}
            if bbox:
                params["bounding_box"] = bbox
            results = earthaccess.search_data(**params)
            lookup = {}
            for granule in results:
                meta = granule.get("umm", {})
                identifiers = meta.get("DataGranule", {}).get("Identifiers", [])
                granule_id = next(
                    (i["Identifier"] for i in identifiers
                     if i.get("IdentifierType") == "ProducerGranuleId"),
                    None,
                )
                time_str = (
                    meta.get("TemporalExtent", {})
                        .get("RangeDateTime", {})
                        .get("BeginningDateTime")
                )
                if granule_id and time_str:
                    lookup[Path(granule_id).stem] = pd.Timestamp(time_str, tz="UTC")
            return lookup
        except Exception as exc:
            logger.warning("Granule time lookup failed: %s", exc)
            return {}


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    logging.basicConfig(level=logging.DEBUG)

    data_loader = DataLoader()
    ds = data_loader.download_dataset_harmony(
        collection_id="C1266136037-GES_DISC",  # OMI Ozone
        temporal=("2024-07-08T00:00:00Z", "2024-08-08T23:59:59Z"),
        bounding_box=(-106.6458, 25.8371, -93.5078, 36.5005),
        output_format="application/x-netcdf4",
        max_results=1,
        cache_path="cache_test.zarr",
    )

    logger.info("Dataset structure")
    logger.info("Data vars: %s", list(ds.data_vars))
    logger.info("Coords: %s", list(ds.coords))
    logger.info("Dims: %s", dict(ds.sizes))
    for var in ds.data_vars:
        logger.info("%s: %s %s", var, ds[var].dims, ds[var].shape)


if __name__ == "__main__":
    main()
