"""
services/harmony_service.py
===========================
Handles NASA Harmony API interactions: submit, wait, download.

Knows nothing about caching, file parsing, or PostGIS. Pure Harmony orchestration.

Example
-------
    from services.harmony_service import HarmonyService
    from preprocessing.dataset_parser import DatasetParser

    svc = HarmonyService()
    files = svc.submit_and_download(
        collection_id="C1266136111-GES_DISC",
        temporal=("2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"),
        bbox=(-74, 40, -73, 41),
        variables=None,
    )
    # files is a list[Path]

    parser = DatasetParser()
    for f in files:
        ds = parser.parse_granule(f)
"""

import concurrent.futures
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import requests
from config.settings import get_settings
from harmony import BBox, Client, Collection, Environment, Request
from utils.earthaccess_client import ensure_earthdata_environment_from_edl
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient errors that should be retried."""
    if isinstance(exc, requests.exceptions.Timeout):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        return exc.response.status_code >= 500
    return False


_RETRY = dict(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class HarmonyService:
    """Harmony API client — submit requests, download results."""

    def __init__(self, client: Optional[Client] = None):
        """
        Parameters
        ----------
        client : harmony.Client
            If None, creates one authenticated via environment variables
            (EDL_USERNAME, EDL_PASSWORD).
        """
        if client is None:
            settings = get_settings()
            ensure_earthdata_environment_from_edl()
            username = settings.edl_username
            password = settings.edl_password
            if not username or not password:
                raise RuntimeError(
                    "EDL credentials required: set EDL_USERNAME and EDL_PASSWORD"
                )
            client = Client(
                env=Environment.PROD,
                auth=(username, password),
            )
        self.client = client

    @retry(**_RETRY)
    def submit_and_download(
        self,
        collection_id: str,
        temporal: Tuple[str, str],
        download_dir: str = ".",
        bbox: Optional[Tuple[float, float, float, float]] = None,
        variables: Optional[List[str]] = None,
        max_results: int = 10,
        output_format: str = "application/x-netcdf4",
    ) -> List[Path]:
        """
        Submit a request to Harmony, wait for processing, download results.

        Parameters
        ----------
        collection_id : str
            NASA CMR concept ID.
        temporal : tuple
            (start_iso, end_iso) in 'YYYY-MM-DDTHH:MM:SSZ' format.
        download_dir : str
            Directory to download files into.
        bbox : tuple, optional
            (min_lon, min_lat, max_lon, max_lat). None = global.
        variables : list, optional
            Variable paths for subsetting.
        max_results : int
            Max granules to request.
        output_format : str
            MIME type (default: application/x-netcdf4).

        Returns
        -------
        list[Path]
            Downloaded file paths.

        Raises
        ------
        ValueError
            If temporal format is invalid or Harmony request is malformed.
        RuntimeError
            If no results are downloaded.
        """
        # Parse temporal
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        try:
            start_dt = datetime.strptime(temporal[0], fmt)
            end_dt = datetime.strptime(temporal[1], fmt)
        except ValueError as exc:
            raise ValueError(
                f"Temporal format must be 'YYYY-MM-DDTHH:MM:SSZ': {exc}"
            )

        # Build request
        collection = Collection(id=collection_id)
        request_params = {
            "collection": collection,
            "temporal": {"start": start_dt, "stop": end_dt},
            "max_results": max_results,
            "format": output_format,
        }
        if variables:
            request_params["variables"] = variables
        if bbox:
            request_params["spatial"] = BBox(*bbox)

        request = Request(**request_params)
        if not request.is_valid():
            raise ValueError("Harmony request validation failed")

        logger.info(
            "Submitting Harmony request — collection=%s temporal=%s bbox=%s",
            collection_id,
            temporal,
            bbox,
        )

        # Submit and wait
        job_id = self.client.submit(request)
        logger.info("Job submitted: %s", job_id)

        fetch_start = time.time()
        self.client.wait_for_processing(job_id, show_progress=True)
        logger.info("Processing completed in %.2fs", time.time() - fetch_start)

        # Download all
        logger.info("Downloading results to %s …", download_dir)
        futures = self.client.download_all(
            job_id, directory=download_dir, overwrite=True
        )

        files = []
        for future in concurrent.futures.as_completed(futures):
            try:
                filepath = future.result()
                files.append(Path(filepath))
                logger.info("Downloaded: %s", filepath)
            except Exception as exc:
                logger.error("Download failed: %s", exc)

        if not files:
            raise RuntimeError(f"No files downloaded for job {job_id}")

        logger.info("Download complete: %d file(s)", len(files))
        return files
