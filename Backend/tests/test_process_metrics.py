"""T45: backend memory was invisible during the 2026-07-17 QA session's
jump-and-plateau (steady ~400MB -> ~811MB -> plateaued 780-813MB) -- nothing
in metrics tracked process RSS or open-dataset-adjacent state, so "plateau
vs leak" was unfalsifiable from dashboards. refresh_process_gauges() reads
process RSS, the matplotlib open-figure count, and the bundle extract-cache
size into Prometheus gauges on each /metrics scrape.
"""
import importlib.util
import os
import sys
import unittest
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


def _gauge_value(gauge) -> float:
    return gauge.collect()[0].samples[0].value


class ProcessMetricsTests(unittest.TestCase):
    def test_refresh_process_gauges_sets_a_positive_process_rss(self):
        from utils.metrics import PROCESS_RSS_BYTES, refresh_process_gauges

        refresh_process_gauges()

        # A live Linux process (Docker) always has a nonzero RSS -- this
        # gauge existing and being positive is the whole point (item 6):
        # the dashboard could never see this value before.
        self.assertGreater(_gauge_value(PROCESS_RSS_BYTES), 0)

    @unittest.skipIf(importlib.util.find_spec("matplotlib") is None, "matplotlib is not installed")
    def test_refresh_process_gauges_counts_open_matplotlib_figures(self):
        import matplotlib.pyplot as plt

        from utils.metrics import MATPLOTLIB_OPEN_FIGURES, refresh_process_gauges

        plt.close("all")
        try:
            # Opened for the side effect only — the gauge counts what pyplot
            # is holding, not what this test holds a name for.
            plt.figure()
            plt.figure()
            refresh_process_gauges()
            self.assertEqual(_gauge_value(MATPLOTLIB_OPEN_FIGURES), 2)
        finally:
            plt.close("all")

        refresh_process_gauges()
        self.assertEqual(_gauge_value(MATPLOTLIB_OPEN_FIGURES), 0)

    def test_refresh_process_gauges_reports_the_bundle_extract_cache_size(self):
        from utils.metrics import BUNDLE_EXTRACT_CACHE_BYTES, refresh_process_gauges

        with patch("services.open_handle.extract_cache_size_bytes", return_value=54321):
            refresh_process_gauges()

        self.assertEqual(_gauge_value(BUNDLE_EXTRACT_CACHE_BYTES), 54321)


class MetricsEndpointRefreshesProcessGaugesTests(unittest.IsolatedAsyncioTestCase):
    """The gauges are only meaningful if something actually refreshes them
    before each scrape -- Prometheus pulls, it doesn't push, so a gauge that
    is only ever set once at import time would report a permanently stale
    value."""

    async def asyncSetUp(self):
        _REQUIRED = ["fastapi", "httpx"]
        if any(importlib.util.find_spec(m) is None for m in _REQUIRED):
            self.skipTest("metrics endpoint test dependencies are not installed")
        import httpx
        import api

        self.httpx = httpx
        self.api = api

    async def test_metrics_endpoint_reports_a_positive_process_rss(self):
        from utils.metrics import PROCESS_RSS_BYTES

        # Zero it first: gauges are process-wide state, so without this the
        # test could pass on a stale value some earlier test happened to
        # set, rather than because the endpoint itself refreshes it.
        PROCESS_RSS_BYTES.set(0)

        transport = self.httpx.ASGITransport(app=self.api.app)
        async with self.httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertIn("process_rss_bytes", response.text)
        self.assertNotIn("process_rss_bytes 0.0", response.text)


if __name__ == "__main__":
    unittest.main()
