"""T45: backend memory was invisible during the 2026-07-17 QA session's
jump-and-plateau (steady ~400MB -> ~811MB -> plateaued 780-813MB) -- nothing
in metrics tracked process RSS or open-dataset-adjacent state, so "plateau
vs leak" was unfalsifiable from dashboards. refresh_process_gauges() reads
process RSS, the matplotlib open-figure count, and the bundle extract-cache
size into Prometheus gauges on each /metrics scrape.
"""
import ctypes
import importlib.util
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

# The platforms this project supports, and the reader that must work on each:
# Linux is the deployment (Docker), Windows is the local development host. A
# platform is listed here only if the gauge is expected to work there, so
# `_PLATFORM_READER[sys.platform]` failing is a real defect rather than an
# unsupported environment.
_PLATFORM_READER = {
    "linux": "_linux_rss_bytes",
    "win32": "_windows_rss_bytes",
}


def _gauge_value(gauge) -> float:
    return gauge.collect()[0].samples[0].value


class ProcessMetricsTests(unittest.TestCase):
    def test_refresh_process_gauges_sets_a_positive_process_rss(self):
        from tta_backend.utils.metrics import PROCESS_RSS_BYTES, refresh_process_gauges

        PROCESS_RSS_BYTES.set(0)
        refresh_process_gauges()

        # Any live process has a nonzero RSS -- this gauge existing and being
        # positive is the whole point (item 6): the dashboard could never see
        # this value before.
        self.assertGreater(_gauge_value(PROCESS_RSS_BYTES), 0)

    @unittest.skipIf(importlib.util.find_spec("matplotlib") is None, "matplotlib is not installed")
    def test_refresh_process_gauges_counts_open_matplotlib_figures(self):
        import matplotlib.pyplot as plt

        from tta_backend.utils.metrics import MATPLOTLIB_OPEN_FIGURES, refresh_process_gauges

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
        from tta_backend.utils.metrics import BUNDLE_EXTRACT_CACHE_BYTES, refresh_process_gauges

        with patch("tta_backend.services.open_handle.extract_cache_size_bytes", return_value=54321):
            refresh_process_gauges()

        self.assertEqual(_gauge_value(BUNDLE_EXTRACT_CACHE_BYTES), 54321)


class ProcessRssReaderTests(unittest.TestCase):
    """The RSS read itself, which is where this gauge silently died.

    ``_current_process_rss_bytes`` originally read only /proc/self/status. That
    is correct on the deployment target and absent on the Windows development
    host, where the open raised FileNotFoundError, the helper returned None, and
    ``refresh_process_gauges`` skipped the ``.set()`` — leaving the gauge at its
    default 0.0. The failure surfaced as a test assertion, but the real cost was
    that the metric T45 added to make "plateau vs leak" falsifiable was dead for
    anyone not running in Docker, and nothing said so.
    """

    def test_this_platforms_reader_returns_a_positive_rss(self):
        """Assert against the reader this platform is *supposed* to use.

        Deliberately not just ``_current_process_rss_bytes() > 0``: that passes
        as long as *some* reader works, so the Linux path could rot unnoticed on
        a Windows host (and vice versa). Naming the expected reader per platform
        is what makes this fail on the machine whose support actually broke.
        """
        import tta_backend.utils.metrics as metrics

        reader_name = _PLATFORM_READER.get(sys.platform)
        self.assertIsNotNone(
            reader_name,
            f"{sys.platform!r} is not in the supported-platform table. If this "
            "project now supports it, add a reader and list it here rather than "
            "letting the gauge fall through to None.",
        )

        rss = getattr(metrics, reader_name)()

        self.assertIsNotNone(rss, f"{reader_name} returned None on {sys.platform}")
        self.assertGreater(rss, 0)

    def test_readers_for_other_platforms_return_none_rather_than_raising(self):
        """The fallback chain walks every reader, so the ones that do not apply
        here must decline quietly. A reader that raised on the wrong OS would
        take down the whole /metrics scrape, not just its own gauge."""
        import tta_backend.utils.metrics as metrics

        for platform, reader_name in _PLATFORM_READER.items():
            if platform == sys.platform:
                continue
            with self.subTest(reader=reader_name):
                self.assertIsNone(getattr(metrics, reader_name)())

    def test_rss_reflects_current_footprint_not_the_high_water_mark(self):
        """The property that makes this metric worth having.

        T45 added it because a jump-and-plateau (~400MB -> ~811MB, then flat)
        could not be told apart from a leak. A peak-only reading —
        ``ru_maxrss`` on Linux, ``PeakWorkingSetSize`` on Windows — never comes
        back down, so it cannot express "plateaued" at all. This asserts the
        reading actually falls when the memory is released, which is what rules
        those out.

        Empirical, and therefore the one test here that can fail while the
        metric is fine: it needs the allocator to hand the pages back to the OS.
        128 MiB clears glibc's mmap threshold and Windows' ``VirtualAlloc``
        cutoff, so both supported platforms do release it — but that is a
        property of the allocator, not of this code. The two tests below pin the
        same current-not-peak property at the field each reader names, with no
        allocation involved, so a flake here is never the only thing standing
        between a swap to a peak counter and a green suite.
        """
        from tta_backend.utils.metrics import _current_process_rss_bytes

        blob_bytes = 128 * 1024 * 1024
        baseline = _current_process_rss_bytes()
        self.assertIsNotNone(baseline)

        blob = bytearray(blob_bytes)  # touched by construction: zero-filled
        peak = _current_process_rss_bytes()
        del blob

        released = _current_process_rss_bytes()

        # Half the blob is a deliberately loose margin: the point is direction,
        # not precision, and the allocator is free to retain some of it.
        margin = blob_bytes // 2
        self.assertGreater(
            peak, baseline + margin, "RSS did not rise when 128 MiB was allocated"
        )
        self.assertLess(
            released,
            peak - margin,
            "RSS stayed at its high-water mark after the allocation was freed — "
            "this is reading a peak counter, which cannot distinguish a plateau "
            "from a leak",
        )

    def test_the_linux_reader_takes_vmrss_and_not_the_vmhwm_above_it(self):
        """The Linux half of current-not-peak, asserted without /proc.

        ``/proc/self/status`` lists ``VmHWM`` (the high-water mark) immediately
        before ``VmRSS``, so the failure mode is a one-word edit that reads
        plausibly and produces a gauge that only ever climbs. Pointed at a
        synthetic status file, this runs on the Windows development host too —
        the point being that the deployment platform's reader cannot rot
        unnoticed on a machine that never exercises it.
        """
        import tta_backend.utils.metrics as metrics

        status = (
            "Name:\tpython\n"
            "VmPeak:\t 4194304 kB\n"
            "VmSize:\t 2097152 kB\n"
            "VmHWM:\t  999999 kB\n"  # the high-water mark, listed first
            "VmRSS:\t  123456 kB\n"  # the current reading, which is the one
            "RssAnon:\t 100000 kB\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "status")
            with open(path, "w", encoding="utf-8") as f:
                f.write(status)

            with patch.object(metrics, "_LINUX_STATUS_PATH", path):
                rss = metrics._linux_rss_bytes()

        self.assertEqual(
            rss,
            123456 * 1024,
            "the Linux reader did not return VmRSS — 999999 kB means it is "
            "reading VmHWM, a high-water mark that cannot fall",
        )

    @unittest.skipUnless(sys.platform == "win32", "the Win32 reader only loads on Windows")
    def test_the_windows_reader_takes_the_working_set_and_not_its_peak(self):
        """The Windows half, same property, same technique.

        ``WorkingSetSize`` and ``PeakWorkingSetSize`` are adjacent fields of one
        struct and differ by a word at the call site, so the swap is as easy to
        make here as ``VmHWM`` is on Linux. Sentinel values through a stubbed
        ``kernel32`` make the answer exact: no allocation, no allocator
        behaviour, no tolerance.
        """
        import tta_backend.utils.metrics as metrics

        working_set = 111 * 1024 * 1024
        peak = 999 * 1024 * 1024

        class _StubExport:
            """Stands in for a ctypes foreign function: the reader assigns
            ``argtypes``/``restype`` on it before calling."""

            def __init__(self, fn):
                self._fn = fn
                self.argtypes = None
                self.restype = None

            def __call__(self, *args):
                return self._fn(*args)

        def _fill(_handle, counters_ref, _cb):
            counters = counters_ref._obj  # unwrap ctypes.byref
            counters.WorkingSetSize = working_set
            counters.PeakWorkingSetSize = peak
            return 1

        class _StubKernel32:
            GetCurrentProcess = _StubExport(lambda: 0)
            K32GetProcessMemoryInfo = _StubExport(_fill)

        with patch.object(ctypes, "WinDLL", lambda *a, **kw: _StubKernel32()):
            rss = metrics._windows_rss_bytes()

        self.assertEqual(
            rss,
            working_set,
            "the Windows reader did not return WorkingSetSize — the peak "
            "sentinel means it is reading PeakWorkingSetSize, which never falls",
        )

    def test_the_chain_falls_through_to_the_next_reader(self):
        """Ordering contract, asserted without depending on the host OS."""
        import tta_backend.utils.metrics as metrics

        with patch.object(metrics, "_linux_rss_bytes", return_value=None), patch.object(
            metrics, "_windows_rss_bytes", return_value=None
        ), patch.object(metrics, "_psutil_rss_bytes", return_value=4242):
            self.assertEqual(metrics._current_process_rss_bytes(), 4242)

    def test_a_reader_returning_zero_is_not_treated_as_a_measurement(self):
        """Zero RSS is not a real reading for a live process, so it must not
        short-circuit the chain and publish a value that looks like the very
        bug this fixes."""
        import tta_backend.utils.metrics as metrics

        with patch.object(metrics, "_linux_rss_bytes", return_value=0), patch.object(
            metrics, "_windows_rss_bytes", return_value=None
        ), patch.object(metrics, "_psutil_rss_bytes", return_value=777):
            self.assertEqual(metrics._current_process_rss_bytes(), 777)

    def test_an_unmeasurable_platform_leaves_the_gauge_alone(self):
        """No reader can measure => no series update. Publishing a zero would be
        indistinguishable from a process that really is using no memory, which is
        exactly how this bug hid; leaving the gauge stale-but-honest is the
        documented behaviour and Prometheus can spot a non-advancing series."""
        import tta_backend.utils.metrics as metrics

        metrics.PROCESS_RSS_BYTES.set(123456)
        with patch.object(metrics, "_current_process_rss_bytes", return_value=None):
            metrics.refresh_process_gauges()

        self.assertEqual(_gauge_value(metrics.PROCESS_RSS_BYTES), 123456)


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
        import tta_backend.api as api

        self.httpx = httpx
        self.api = api

    async def test_metrics_endpoint_reports_a_positive_process_rss(self):
        from tta_backend.utils.metrics import PROCESS_RSS_BYTES

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
