import importlib.util
import os
import sys
import unittest
import asyncio
from unittest.mock import AsyncMock, patch

import httpx

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

REQUIRED_MODULES = ["harmony"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "Harmony dependencies are not installed",
)
class AsyncHarmonyServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from utils.metrics import reset_metrics

        reset_metrics()

    def _service(self):
        from services.async_harmony_service import AsyncHarmonyService

        svc = object.__new__(AsyncHarmonyService)
        svc._poll_interval = 0
        svc._processing_timeout_seconds = 1
        svc._auth = ("user", "pass")
        svc._download_dir = "."
        return svc

    def test_validate_json_response_detects_earthdata_login_redirect(self):
        from services.async_harmony_service import HarmonyAuthenticationError
        from utils.metrics import get_metric

        svc = self._service()
        response = httpx.Response(
            303,
            headers={"location": "https://urs.earthdata.nasa.gov/oauth/authorize"},
            request=httpx.Request("GET", "https://harmony.earthdata.nasa.gov/jobs/abc"),
        )

        with self.assertRaises(HarmonyAuthenticationError):
            svc._validate_json_response(response, "https://harmony.earthdata.nasa.gov/jobs/abc")

        self.assertEqual(get_metric("harmony_auth_failures"), 1)

    def test_validate_json_response_rejects_non_json_success(self):
        from services.async_harmony_service import HarmonyProtocolError
        from utils.metrics import get_metric

        svc = self._service()
        response = httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<html>login</html>",
            request=httpx.Request("GET", "https://harmony.earthdata.nasa.gov/jobs/abc"),
        )

        with self.assertRaises(HarmonyProtocolError):
            svc._validate_json_response(response, "https://harmony.earthdata.nasa.gov/jobs/abc")

        self.assertEqual(get_metric("harmony_protocol_failures"), 1)

    async def test_wait_for_processing_raises_classified_job_failure(self):
        from services.async_harmony_service import HarmonyJobFailedError
        from utils.metrics import get_metric

        svc = self._service()
        svc._poll_status = AsyncMock(return_value={"status": "failed", "message": "bad input"})

        with self.assertRaisesRegex(HarmonyJobFailedError, "bad input"):
            await svc._wait_for_processing("https://harmony.earthdata.nasa.gov/jobs/abc")

        self.assertEqual(get_metric("harmony_jobs_failed"), 1)

    async def test_submit_and_download_times_out_stuck_processing(self):
        from services.async_harmony_service import HarmonyTimeoutError
        from utils.metrics import get_metric

        async def long_wait(status_url, *args, **kwargs):
            await asyncio.sleep(1)

        svc = self._service()
        svc._processing_timeout_seconds = 0.01
        svc._build_request = lambda *args, **kwargs: object()
        svc._submit_request = AsyncMock(return_value="abc")
        svc._status_url = lambda job_id: f"https://harmony.earthdata.nasa.gov/jobs/{job_id}"
        svc._wait_for_processing = AsyncMock(side_effect=long_wait)

        with self.assertRaisesRegex(HarmonyTimeoutError, "abc exceeded 0.01s"):
            await svc.submit_and_download(
                collection_id="C1",
                temporal=("2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"),
            )

        self.assertEqual(get_metric("harmony_jobs_submitted"), 1)
        self.assertEqual(get_metric("harmony_jobs_timed_out"), 1)

    async def test_wait_for_processing_raises_timeout_for_stuck_status(self):
        from services.async_harmony_service import HarmonyTimeoutError

        svc = self._service()
        svc._poll_interval = 0
        svc._poll_status = AsyncMock(return_value={"status": "running", "progress": 50})
        status_url = "https://harmony.earthdata.nasa.gov/jobs/stuck"

        with self.assertRaisesRegex(HarmonyTimeoutError, "timed out"):
            await svc._wait_for_processing(status_url, max_poll_seconds=0.01)


if __name__ == "__main__":
    unittest.main()
