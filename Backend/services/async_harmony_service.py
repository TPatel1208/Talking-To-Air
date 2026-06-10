"""
services/async_harmony_service.py
==================================
Async Harmony API client — submit, poll, download without blocking the
FastAPI event loop.

Addresses Phase 4 of the refactor plan:
  - All network I/O runs in the asyncio event loop via httpx.AsyncClient.
  - Polling replaces harmony.Client.wait_for_processing (which is a
    blocking loop) with asyncio.sleep.
  - Tenacity provides retry / exponential back-off on transient 5xx errors
    and timeouts; 400 / 401 are not retried (caller's fault).
  - A sync shim (submit_and_download_sync) wraps the async path for
    backward-compatible call sites that cannot yet use await.

Retry strategy (per plan):
  - 4 attempts total (initial + 3 retries)
  - Exponential back-off: 2 s → 4 s → 8 s, capped at 60 s
  - Retry on: httpx.TimeoutException, httpx.HTTPStatusError with 5xx status
  - No retry on: 400 Bad Request, 401 Unauthorized

Example
-------
    import asyncio
    from services.async_harmony_service import AsyncHarmonyService

    async def main():
        svc = AsyncHarmonyService()
        files = await svc.submit_and_download(
            collection_id="C1266136111-GES_DISC",
            temporal=("2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"),
            bbox=(-74, 40, -73, 41),
        )

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import httpx
import requests
from config.settings import get_settings
from harmony import BBox, Client, Collection, Environment, Request
from utils.earthaccess_client import ensure_earthdata_environment_from_edl
from utils.metrics import increment_metric, observe_harmony_fetch
from utils.streaming import current_thread_id, emit_status
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tenacity retry predicate
# ---------------------------------------------------------------------------

def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient errors that should be retried."""
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        # Retry server errors; do not retry client errors (4xx)
        return exc.response.status_code >= 500
    if isinstance(exc, requests.exceptions.HTTPError):
        response = exc.response
        return response is not None and response.status_code >= 500
    message = str(exc).lower()
    if "service unavailable" in message or "temporarily unavailable" in message:
        return True
    return False


_RETRY = dict(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

# Poll interval in seconds for job-status checks
_POLL_INTERVAL = 5
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_EARTHDATA_LOGIN_HOST = "urs.earthdata.nasa.gov"


class HarmonyError(RuntimeError):
    """Base class for Harmony orchestration failures."""


class HarmonyTimeoutError(TimeoutError, HarmonyError):
    """Raised when a Harmony job exceeds the configured processing timeout."""

    def __init__(
        self,
        message: str,
        *,
        job_url: str | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.job_url = job_url
        self.elapsed_seconds = elapsed_seconds


class HarmonyAuthenticationError(HarmonyError):
    """Raised when Harmony redirects polling to Earthdata Login."""


class HarmonyProtocolError(HarmonyError):
    """Raised when Harmony returns an unexpected protocol response."""


class HarmonyJobFailedError(HarmonyError):
    """Raised when Harmony reports a terminal non-success job status."""


class AsyncHarmonyService:
    """
    Async Harmony API client.

    Uses httpx for non-blocking HTTP and asyncio.sleep for status polling.
    The underlying harmony.Client is only used for request construction and
    job submission (both are fast synchronous calls); the long-running
    wait-and-download path is fully async.

    Parameters
    ----------
    client : harmony.Client, optional
        Pre-built client (useful for tests). If None, one is created from
        EDL_USERNAME / EDL_PASSWORD environment variables.
    poll_interval : int
        Seconds between status-poll requests (default 5).
    download_dir : str
        Default directory for downloaded files.
    """

    def __init__(
        self,
        client: Optional[Client] = None,
        poll_interval: int = _POLL_INTERVAL,
        download_dir: str = ".",
    ) -> None:
        settings = get_settings()
        ensure_earthdata_environment_from_edl()
        if client is None:
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
        self._client = client
        self._auth = (settings.edl_username, settings.edl_password)
        self._poll_interval = poll_interval
        self._download_dir = download_dir
        self._processing_timeout_seconds = settings.harmony_processing_timeout_seconds

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def submit_and_download(
        self,
        collection_id: str,
        temporal: Tuple[str, str],
        download_dir: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        variables: Optional[List[str]] = None,
        max_results: int = 10,
        output_format: str = "application/x-netcdf4",
    ) -> List[Path]:
        """
        Submit a Harmony job, poll until complete, download all results.

        Parameters
        ----------
        collection_id : str
            NASA CMR concept ID.
        temporal : tuple
            (start_iso, end_iso) in 'YYYY-MM-DDTHH:MM:SSZ' format.
        download_dir : str, optional
            Override for the instance-level download directory.
        bbox : tuple, optional
            (min_lon, min_lat, max_lon, max_lat).
        variables : list, optional
            Variable paths for subsetting.
        max_results : int
            Max granules to request.
        output_format : str
            MIME type (default application/x-netcdf4).

        Returns
        -------
        list[Path]
            Paths of downloaded files.

        Raises
        ------
        ValueError
            Bad temporal format or invalid Harmony request.
        RuntimeError
            Job failed on the Harmony side, or no files downloaded.
        httpx.HTTPStatusError
            Non-retryable HTTP error from Harmony (e.g. 400, 401).
        """
        dest = download_dir or self._download_dir
        Path(dest).mkdir(parents=True, exist_ok=True)

        # ── 1. Build and submit (sync, fast) ─────────────────────────────
        request = self._build_request(
            collection_id, temporal, bbox, variables, max_results, output_format
        )
        logger.info(
            "Submitting Harmony request: collection=%s temporal=%s bbox=%s variables=%s max_results=%s",
            collection_id,
            temporal,
            bbox,
            variables,
            max_results,
        )
        emit_status("Submitting request to NASA Harmony...")
        job_id = await self._submit_request(request)
        increment_metric("harmony_jobs_submitted")
        logger.info("Harmony job submitted: %s", job_id)

        # ── 2. Async poll until done ─────────────────────────────────────
        status_url = self._status_url(job_id)
        emit_status("NASA Harmony is preparing data...")
        started = asyncio.get_running_loop().time()
        thread_id = current_thread_id()
        client_downloaded_files: Optional[List[Path]] = None
        try:
            async with asyncio.timeout(self._processing_timeout_seconds):
                try:
                    await self._wait_for_processing(
                        status_url,
                        max_poll_seconds=self._processing_timeout_seconds,
                    )
                except HarmonyAuthenticationError:
                    logger.warning(
                        "Harmony httpx polling was redirected to Earthdata Login "
                        "for job %s; falling back to harmony-py authenticated "
                        "wait/download",
                        job_id,
                    )
                    client_downloaded_files = await self._wait_and_download_with_client_timeout(
                        self._wait_and_download_with_client,
                        job_id,
                        dest,
                        max_poll_seconds=self._processing_timeout_seconds,
                    )
        except TimeoutError as exc:
            elapsed = int(asyncio.get_running_loop().time() - started)
            increment_metric("harmony_jobs_timed_out")
            logger.warning(
                "harmony_job_timeout",
                extra={
                    "_event": "harmony_job_timeout",
                    "_job_id": job_id,
                    "_job_url": status_url,
                    "_elapsed_seconds": elapsed,
                    "_timeout_seconds": self._processing_timeout_seconds,
                    "_thread_id": thread_id,
                },
            )
            emit_status("Satellite data processing timed out. Please try again later.")
            raise HarmonyTimeoutError(
                f"Harmony job {job_id} exceeded "
                f"{self._processing_timeout_seconds}s processing limit",
                job_url=status_url,
                elapsed_seconds=elapsed,
            ) from exc

        if client_downloaded_files is not None:
            if not client_downloaded_files:
                emit_status("Download failed while retrieving NASA Harmony output.")
                raise RuntimeError(f"No files downloaded for job {job_id}")
            increment_metric("harmony_jobs_succeeded")
            observe_harmony_fetch(asyncio.get_running_loop().time() - started)
            emit_status("NASA Harmony finished preparing data.")
            emit_status("Processing downloaded data...")
            logger.info(
                "Download complete via harmony-py client: %d file(s) for job %s",
                len(client_downloaded_files),
                job_id,
            )
            return client_downloaded_files

        # ── 3. Fetch links then download concurrently ────────────────────
        emit_status("Downloading satellite granules...")
        links = await self._fetch_download_links(status_url)
        files = await self._download_all(links, dest)
        if not files:
            emit_status("Download failed while retrieving NASA Harmony output.")
            raise RuntimeError(f"No files downloaded for job {job_id}")

        emit_status("Processing downloaded data...")
        logger.info("Download complete: %d file(s) for job %s", len(files), job_id)
        observe_harmony_fetch(asyncio.get_running_loop().time() - started)
        return files

    def submit_and_download_sync(self, **kwargs) -> List[Path]:
        """
        Synchronous shim — runs the async path on a new event loop.

        Use this from sync call sites (e.g. the @tool fallback path)
        that cannot use ``await``.  FastAPI request handlers should call
        ``await submit_and_download(...)`` directly instead.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.submit_and_download(**kwargs))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                lambda: asyncio.run(self.submit_and_download(**kwargs))
            )
            return future.result()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_request(
        self,
        collection_id: str,
        temporal: Tuple[str, str],
        bbox: Optional[Tuple[float, float, float, float]],
        variables: Optional[List[str]],
        max_results: int,
        output_format: str,
    ) -> Request:
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        try:
            start_dt = datetime.strptime(temporal[0], fmt)
            end_dt   = datetime.strptime(temporal[1], fmt)
        except ValueError as exc:
            raise ValueError(
                f"Temporal format must be 'YYYY-MM-DDTHH:MM:SSZ': {exc}"
            ) from exc

        params: dict = {
            "collection": Collection(id=collection_id),
            "temporal":   {"start": start_dt, "stop": end_dt},
            "max_results": max_results,
            "format":      output_format,
        }
        if variables:
            params["variables"] = variables
        if bbox:
            params["spatial"] = BBox(*bbox)

        req = Request(**params)
        if not req.is_valid():
            raise ValueError("Harmony request validation failed")
        return req

    def _status_url(self, job_id: str) -> str:
        """Build the Harmony job-status URL."""
        base = "https://harmony.earthdata.nasa.gov"
        return f"{base}/jobs/{job_id}"

    @retry(**_RETRY)
    async def _submit_request(self, request: Request) -> str:
        """Submit a Harmony request, retrying only pre-job transient failures."""
        return await asyncio.to_thread(self._client.submit, request)

    def _wait_and_download_with_client(self, job_id: str, dest: str) -> List[Path]:
        """Use harmony.Client for authenticated wait/download orchestration."""
        self._client.wait_for_processing(job_id, show_progress=True)
        futures = self._client.download_all(job_id, directory=dest, overwrite=True)

        files: List[Path] = []
        for future in concurrent.futures.as_completed(futures):
            try:
                filepath = future.result()
                files.append(Path(filepath))
                logger.info("Downloaded: %s", filepath)
            except Exception as exc:
                logger.error("Download failed: %s", exc)
        return files

    async def _wait_and_download_with_client_timeout(
        self,
        fn,
        job_id: str,
        dest: str,
        *,
        max_poll_seconds: float,
    ) -> List[Path]:
        """
        Run harmony-py's blocking wait/download in an isolated worker.

        ``wait_for`` cannot stop a C extension or blocking HTTP call already in
        progress, but using a dedicated executor prevents a stalled Harmony job
        from consuming the event loop's shared default executor indefinitely.
        """
        loop = asyncio.get_running_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = loop.run_in_executor(executor, fn, job_id, dest)
        timed_out = False
        try:
            return await asyncio.wait_for(future, timeout=max_poll_seconds)
        except asyncio.TimeoutError as exc:
            timed_out = True
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise HarmonyTimeoutError(
                f"Harmony job {job_id} exceeded {max_poll_seconds}s processing limit",
                job_url=self._status_url(job_id),
                elapsed_seconds=max_poll_seconds,
            ) from exc
        finally:
            if not timed_out:
                executor.shutdown(wait=True)

    @retry(**_RETRY)
    async def _poll_status(
        self, client: httpx.AsyncClient, url: str
    ) -> dict:
        """Single status poll — retried on transient errors."""
        resp = await client.get(url, follow_redirects=False)
        self._validate_json_response(resp, url)
        resp.raise_for_status()
        return resp.json()

    def _validate_json_response(self, resp: httpx.Response, url: str) -> None:
        """Classify redirects and non-JSON Harmony responses before parsing."""
        location = resp.headers.get("location", "")
        if resp.status_code in _REDIRECT_STATUSES:
            host = urlparse(location).netloc.lower()
            if _EARTHDATA_LOGIN_HOST in host:
                increment_metric("harmony_auth_failures")
                raise HarmonyAuthenticationError(
                    "Harmony authentication failed: job status request was "
                    f"redirected to Earthdata Login ({location})"
                )
            increment_metric("harmony_protocol_failures")
            raise HarmonyProtocolError(
                "Harmony returned an unexpected redirect while polling "
                f"{url}: status={resp.status_code} location={location or '<missing>'}"
            )

        if resp.status_code in {401, 403}:
            increment_metric("harmony_auth_failures")
            raise HarmonyAuthenticationError(
                f"Harmony authentication failed while polling {url}: "
                f"status={resp.status_code}"
            )

        if resp.is_error:
            return

        content_type = resp.headers.get("content-type", "")
        if "application/json" not in content_type.lower():
            increment_metric("harmony_protocol_failures")
            raise HarmonyProtocolError(
                "Harmony returned a non-JSON response: "
                f"url={url} status={resp.status_code} content_type={content_type or '<missing>'}"
            )

    async def _wait_for_processing(
        self,
        status_url: str,
        max_poll_seconds: float = 600,
    ) -> None:
        """Poll the job-status endpoint until the job reaches a terminal state."""
        terminal = {"successful", "failed", "canceled"}
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + max_poll_seconds
        async with httpx.AsyncClient(
            auth=self._auth,
            timeout=httpx.Timeout(30.0),
            follow_redirects=False,
        ) as client:
            while True:
                data   = await self._poll_status(client, status_url)
                status = data.get("status", "").lower()
                pct    = data.get("progress", 0)
                logger.info("Harmony job status: %s (%s%%)", status, pct)

                if status in terminal:
                    if status != "successful":
                        msg = data.get("message", "No details provided")
                        increment_metric("harmony_jobs_failed")
                        emit_status("Download failed while retrieving NASA Harmony output.")
                        logger.error(
                            "harmony_job_failed",
                            extra={
                                "_event": "harmony_job_failed",
                                "_job_url": status_url,
                                "_error_message": str(msg),
                                "_thread_id": current_thread_id(),
                            },
                        )
                        raise HarmonyJobFailedError(
                            f"Harmony job ended with status '{status}': {msg}"
                        )
                    increment_metric("harmony_jobs_succeeded")
                    emit_status("NASA Harmony finished preparing data.")
                    return

                elapsed = int(loop.time() - started)
                if pct:
                    emit_status(f"Waiting for NASA Harmony processing ({pct}% complete, {elapsed}s elapsed)...")
                else:
                    emit_status(f"Waiting for NASA Harmony processing ({elapsed}s elapsed)...")
                await asyncio.sleep(self._poll_interval)
                now = loop.time()
                if now > deadline:
                    elapsed = now - started
                    raise HarmonyTimeoutError(
                        f"Harmony job polling timed out for {status_url} after "
                        f"{elapsed:.2f}s",
                        job_url=status_url,
                        elapsed_seconds=elapsed,
                    )

    @retry(**_RETRY)
    async def _fetch_download_links(
        self, status_url: str
    ) -> List[str]:
        """Retrieve the list of downloadable file URLs from the completed job."""
        async with httpx.AsyncClient(
            auth=self._auth,
            timeout=httpx.Timeout(30.0),
            follow_redirects=False,
        ) as client:
            resp = await client.get(status_url, follow_redirects=False)
            self._validate_json_response(resp, status_url)
            resp.raise_for_status()
            data  = resp.json()
            links = data.get("links", [])
            return [
                lnk["href"]
                for lnk in links
                if lnk.get("rel") == "data" and lnk.get("href")
            ]

    @retry(**_RETRY)
    async def _download_one(
        self,
        client: httpx.AsyncClient,
        url: str,
        dest: str,
    ) -> Path:
        """Download a single file, streaming to disk."""
        filename = Path(url.split("?")[0]).name or "granule.nc"
        out_path = Path(dest) / filename
        emit_status("Downloading satellite granules...")

        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(out_path, "wb") as fh:
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    await asyncio.to_thread(fh.write, chunk)

        logger.info("Downloaded: %s", out_path)
        return out_path

    async def _download_all(self, links: List[str], dest: str) -> List[Path]:
        """Download all links concurrently."""
        if not links:
            return []

        async with httpx.AsyncClient(
            auth=self._auth,
            timeout=httpx.Timeout(300.0),  # large files need longer timeout
            follow_redirects=True,
        ) as client:
            tasks = [self._download_one(client, url, dest) for url in links]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        files: List[Path] = []
        for r in results:
            if isinstance(r, Exception):
                logger.error("Download error: %s", r)
                emit_status("Download failed while retrieving NASA Harmony output.")
            else:
                files.append(r)
        if files:
            emit_status(f"Downloaded {len(files)} satellite granule{'s' if len(files) != 1 else ''}.")
        return files

    def __repr__(self) -> str:
        return (
            "AsyncHarmonyService("
            f"poll_interval={self._poll_interval}s, "
            f"processing_timeout={self._processing_timeout_seconds}s)"
        )
