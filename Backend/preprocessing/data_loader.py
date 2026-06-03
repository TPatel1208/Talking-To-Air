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

  DATA_FETCH_MODE=harmony   — always use Harmony (fast for tight bbox)
  DATA_FETCH_MODE=opendap   — always use OPeNDAP CE (GES_DISC only)
  DATA_FETCH_MODE=s3        — always use S3 (requires S3_FORCE_FETCH=1
                              outside us-west-2 or you'll get an error)
  DATA_FETCH_MODE=auto      — default provider routing described above

No file-parsing logic lives here. All normalisation is in DatasetParser.
All cache logic is in CacheManager.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import earthaccess
import xarray as xr

from preprocessing.cache_manager import CacheManager, make_group_key
from preprocessing.dataset_parser import DatasetParser
from repositories.cache_index_repository import CacheIndexRepository
from repositories.zarr_repository import ZarrRepository
from datasets.registry import load_registry

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


def _provider(collection_id: str) -> str:
    if collection_id.endswith(_LARC_CLOUD_SUFFIX):
        return "LARC_CLOUD"
    if collection_id.endswith(_GES_DISC_SUFFIX):
        return "GES_DISC"
    return "UNKNOWN"


def _fetch_mode() -> str:
    """Read DATA_FETCH_MODE from environment, defaulting to 'auto'."""
    mode = os.getenv("DATA_FETCH_MODE", "auto").strip().lower()
    if mode not in _VALID_MODES:
        logger.warning(
            "Unknown DATA_FETCH_MODE=%r — falling back to 'auto'. "
            "Valid values: %s",
            mode, ", ".join(sorted(_VALID_MODES)),
        )
        return "auto"
    return mode


class DataLoader:

    def __init__(self, cache_path: str = "cache.zarr"):
        # Auth — must happen before S3FetchService is created
        try:
            self.auth = earthaccess.login(strategy="environment")
            if not self.auth:
                raise RuntimeError("earthaccess login returned no credentials")
        except Exception as exc:
            logger.error("earthaccess auth failed: %s", exc)
            raise

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
        cache_path: str = "cache.zarr",
    ) -> xr.Dataset:
        """
        Return a dataset for the requested collection / time range / bbox.

        See module docstring for the fetch order and override options.
        The method name is kept as-is so nothing in the agent/tool layer changes.
        """
        # ── Cache lookup ──────────────────────────────────────────────────
        cached = self._cache.lookup(
            collection_id=collection_id,
            temporal=temporal,
            bbox=bounding_box,
        )
        if cached is not None:
            logger.info("Cache hit for %s", collection_id)
            return cached

        # ── Fetch ─────────────────────────────────────────────────────────
        mode = _fetch_mode()
        provider = _provider(collection_id)
        col = self._registry_by_id.get(collection_id)

        logger.info(
            "Cache miss — fetching %s (provider=%s, mode=%s)",
            collection_id, provider, mode,
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
        return ds

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
        # Strategy: Harmony is preferred for any collection that supports
        # server-side variable subsetting (tight, pre-clipped NetCDF4
        # responses are reliably faster than protocol or HDF5 overhead).
        # Fall through to provider-native paths only when Harmony is not
        # available or the collection doesn't support subsetting.

        supports_var_sub = col.supports_variable_subsetting if col else False
        col_variables = col.variables if col else None

        if provider == "LARC_CLOUD":
            if supports_var_sub:
                # Harmony variable subsetting — fast server-side clip
                effective_vars = variables or col_variables or None
                try:
                    return self._fetch_harmony_fallback(
                        collection_id, temporal, bounding_box,
                        effective_vars, max_results, output_format
                    )
                except Exception as exc:
                    logger.warning(
                        "Harmony variable subsetting failed for %s (%s) "
                        "— falling back to S3",
                        collection_id, exc,
                    )

            # S3 direct — only worthwhile inside us-west-2
            group = col.groups[0] if col and col.groups else None
            try:
                return self._fetch_s3(
                    collection_id, temporal, bounding_box, group, max_results
                )
            except S3OutsideRegionError as exc:
                logger.info("%s — routing to Harmony", exc)
            except Exception as exc:
                logger.warning(
                    "S3 fetch failed for %s (%s) — falling back to Harmony",
                    collection_id, exc,
                )

            # Final fallback for LARC_CLOUD
            return self._fetch_harmony_fallback(
                collection_id, temporal, bounding_box,
                variables, max_results, output_format
            )

        elif provider == "GES_DISC":
            try:
                return self._fetch_opendap(
                    collection_id, temporal, bounding_box, variables, max_results
                )
            except Exception as exc:
                logger.warning(
                    "OPeNDAP fetch failed for %s (%s) — falling back to Harmony",
                    collection_id, exc,
                )
                return self._fetch_harmony_fallback(
                    collection_id, temporal, bounding_box,
                    variables, max_results, output_format
                )

        else:
            # Unknown provider — go straight to Harmony
            logger.warning("Unknown provider for %s — using Harmony", collection_id)
            return self._fetch_harmony_fallback(
                collection_id, temporal, bounding_box,
                variables, max_results, output_format
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Fetch strategies
    # ─────────────────────────────────────────────────────────────────────────

    def _fetch_s3(self, collection_id, temporal, bbox, group, max_results):
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
            from services.harmony_service import HarmonyService
            self._harmony_service = HarmonyService()

        files = self._harmony_service.submit_and_download(
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
        import pandas as pd
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

    print("\n=== Dataset structure ===")
    print("Data vars:", list(ds.data_vars))
    print("Coords:   ", list(ds.coords))
    print("Dims:     ", dict(ds.sizes))
    for var in ds.data_vars:
        print(f"  {var}: {ds[var].dims} {ds[var].shape}")


if __name__ == "__main__":
    main()