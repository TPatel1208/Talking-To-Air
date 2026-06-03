"""
services/s3_fetch_service.py
============================
Direct S3 fetch for NASA Earthdata Cloud (LARC_CLOUD) collections.

IMPORTANT: Caller must have called earthaccess.login() before instantiating
this class. DataLoader.__init__ does this automatically.

Performance note
----------------
HDF5 is not a cloud-native format.  Reading it from S3 requires many small
random range requests per chunk.  This is fast (~1 ms per request) only
inside AWS us-west-2 (same region as NASA's buckets).  From any other
location, Harmony typically outperforms direct S3 access.

This service automatically detects whether it is running inside us-west-2.
If not, it raises ``S3OutsideRegionError`` so the caller can route to
Harmony instead.  Set ``S3_FORCE_FETCH=1`` to bypass the region check
(useful for debugging or if you have measured acceptable latency from your
location).

TEMPO double-open fix
---------------------
The previous implementation opened each HDF5 file twice (once for root
coordinates, once for the product group).  This version opens the file once
using h5py and merges the coordinate and product datasets from a single
file handle.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import earthaccess
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# ── Region detection ────────────────────────────────────────────────────────

_AWS_METADATA_URL = "http://169.254.169.254/latest/meta-data/placement/region"
_PREFERRED_REGION = "us-west-2"
_region_cache: Optional[str] = None


def _detect_aws_region(timeout: float = 0.5) -> Optional[str]:
    """
    Probe the EC2 instance metadata endpoint (non-blocking, fast timeout).
    Returns the region string on EC2/ECS, None everywhere else.
    """
    global _region_cache
    if _region_cache is not None:
        return _region_cache
    try:
        import requests as _requests
        resp = _requests.get(_AWS_METADATA_URL, timeout=timeout)
        if resp.ok:
            _region_cache = resp.text.strip()
            return _region_cache
    except Exception:
        pass
    _region_cache = ""  # cache the miss so we only probe once
    return None


def _in_preferred_region() -> bool:
    """Return True when running inside AWS us-west-2."""
    return _detect_aws_region() == _PREFERRED_REGION


class S3OutsideRegionError(RuntimeError):
    """Raised when S3 fetch is attempted outside us-west-2 and not forced."""


class S3FetchService:
    """
    Fetch LARC_CLOUD granules directly from S3.

    IMPORTANT: Caller must have called earthaccess.login() before
    instantiating this class. DataLoader.__init__ does this.

    Parameters
    ----------
    force : bool
        Skip the region check and always attempt S3 fetches.
        Equivalent to setting the ``S3_FORCE_FETCH=1`` env variable.
    """

    def __init__(self, force: bool = False) -> None:
        self._force = force or os.getenv("S3_FORCE_FETCH", "").strip() == "1"

    def fetch(
        self,
        collection_id: str,
        temporal: Tuple[str, str],
        bbox: Optional[Tuple[float, float, float, float]] = None,
        group: Optional[str] = None,
        max_results: int = 10,
    ) -> xr.Dataset:
        """
        Search CMR, open matching granules from S3, clip to bbox.

        Raises S3OutsideRegionError if not running in us-west-2 and
        S3_FORCE_FETCH is not set — caller should fall back to Harmony.

        Parameters
        ----------
        collection_id : NASA CMR concept-id (must be a LARC_CLOUD concept-id)
        temporal      : (start_iso, end_iso)
        bbox          : (min_lon, min_lat, max_lon, max_lat) or None
        group         : HDF5 group to open (e.g. 'product' for TEMPO), or None
        max_results   : cap on number of granules
        """
        if not self._force and not _in_preferred_region():
            raise S3OutsideRegionError(
                f"S3 fetch skipped: not running in {_PREFERRED_REGION}. "
                "Set S3_FORCE_FETCH=1 to override. Routing to Harmony instead."
            )

        try:
            return self._fetch_inner(collection_id, temporal, bbox, group, max_results)
        except S3OutsideRegionError:
            raise
        except Exception as exc:
            # One retry on credential expiry — earthaccess tokens last ~1 hour
            if "credential" in str(exc).lower() or "token" in str(exc).lower():
                logger.warning(
                    "S3 credential may have expired, re-authenticating and retrying"
                )
                earthaccess.login(strategy="environment", force=True)
                return self._fetch_inner(collection_id, temporal, bbox, group, max_results)
            raise

    def _fetch_inner(
        self,
        collection_id: str,
        temporal: Tuple[str, str],
        bbox: Optional[Tuple[float, float, float, float]],
        group: Optional[str],
        max_results: int,
    ) -> xr.Dataset:
        search_params: dict = {
            "concept_id": collection_id,
            "temporal": temporal,
            "count": max_results,
        }
        if bbox:
            search_params["bounding_box"] = bbox

        logger.info(
            "S3 search: collection=%s temporal=%s bbox=%s", collection_id, temporal, bbox
        )
        results = earthaccess.search_data(**search_params)

        if not results:
            raise RuntimeError(
                f"No granules found for {collection_id} temporal={temporal} bbox={bbox}"
            )

        logger.info("Found %d granule(s), opening from S3", len(results))

        # earthaccess.open() returns fsspec file objects — no disk writes
        files = earthaccess.open(results)

        datasets = []
        for f in files:
            try:
                ds = self._open_granule(f, group=group)
                datasets.append(ds)
            except Exception as exc:
                logger.warning("Could not open granule %s: %s", f, exc)

        if not datasets:
            raise RuntimeError("All S3 granules failed to open")

        if len(datasets) == 1:
            combined = datasets[0]
        else:
            valid = [d for d in datasets if "time" in d.dims or "time" in d.coords]
            if not valid:
                raise RuntimeError("No granules had a time coordinate")
            combined = xr.concat(valid, dim="time")

        if bbox:
            combined = self._clip_bbox(combined, bbox)

        return combined.load()

    def _open_granule(self, fileobj, group: Optional[str]) -> xr.Dataset:
        """
        Open one fsspec file object as an xr.Dataset.

        TEMPO files store coords at root and science data in the 'product'
        group.  Previously this opened the file *twice* (once for root coords,
        once for the product group), doubling HDF5 metadata round-trips.

        This version opens the file once with h5py via a single fsspec handle
        and constructs both datasets from the same open file object, halving
        the random-access overhead on S3.
        """
        from datetime import datetime
        from preprocessing.dataset_parser import DatasetParser

        _TEMPO_EPOCH = datetime(1980, 1, 6)

        if group:
            # Open the underlying bytes once, build both views from the same
            # in-memory buffer to avoid a second S3 round-trip.
            try:
                import h5py
                with h5py.File(fileobj, "r") as h5f:
                    root_ds = xr.open_dataset(
                        xr.backends.H5NetCDFStore(h5f),
                        decode_times=False,
                    )
                    product_ds = xr.open_dataset(
                        xr.backends.H5NetCDFStore(h5f[group]),
                        decode_times=False,
                    )
                    # Materialise before closing h5f
                    root_ds = root_ds.load()
                    product_ds = product_ds.load()
            except Exception:
                # h5py path failed — fall back to sequential xr.open_dataset
                # (original behaviour, two opens)
                logger.debug(
                    "_open_granule: h5py single-open failed, falling back to two-open path"
                )
                root_ds = xr.open_dataset(fileobj, engine="h5netcdf", decode_times=False).load()
                product_ds = xr.open_dataset(
                    fileobj, engine="h5netcdf", group=group, decode_times=False
                ).load()

            coords = {}
            for coord in ("latitude", "longitude"):
                if coord in root_ds:
                    coords[coord] = root_ds[coord]
            if "time" in root_ds:
                coords["time"] = DatasetParser._decode_time(
                    root_ds["time"], _TEMPO_EPOCH, unit="s"
                )
            return product_ds.assign_coords(**coords)
        else:
            # Non-grouped file: open once, no chunks= to avoid Dask overhead
            return xr.open_dataset(
                fileobj, engine="h5netcdf", decode_times=False
            ).load()

    @staticmethod
    def _clip_bbox(
        ds: xr.Dataset,
        bbox: Tuple[float, float, float, float],
    ) -> xr.Dataset:
        """Label-based bbox clip."""
        min_lon, min_lat, max_lon, max_lat = bbox
        sel: dict = {}

        for dim, lo, hi in (
            ("latitude", min_lat, max_lat),
            ("longitude", min_lon, max_lon),
        ):
            if dim not in ds.dims:
                continue
            vals = ds[dim].values
            diffs = np.diff(vals)
            if np.all(diffs > 0):
                sel[dim] = slice(lo, hi)
            elif np.all(diffs < 0):
                sel[dim] = slice(hi, lo)
            else:
                logger.warning(
                    "_clip_bbox: '%s' is non-monotonic — clip skipped for this dim", dim
                )

        if sel:
            try:
                ds = ds.sel(**sel)
            except Exception as exc:
                logger.warning("bbox clip failed (%s) — returning full dataset", exc)

        return ds