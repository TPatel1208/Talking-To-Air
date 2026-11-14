"""Process-health gauges, refreshed on each /metrics scrape."""
from __future__ import annotations

from tta_backend.infrastructure.metrics import (
    BUNDLE_EXTRACT_CACHE_BYTES,
    CUBE_STORE_BYTES,
    PROCESS_RSS_BYTES,
    current_process_rss_bytes,
)


def refresh_process_gauges() -> None:
    """Refresh the process-health gauges (RSS, bundle extract-cache size,
    cube-store size) -- called on each /metrics scrape rather than a timer,
    so idle time costs nothing and every scrape is current as of itself.
    Each source is best-effort: a missing/uninstalled dependency skips that
    one gauge rather than failing the whole scrape.

    There is deliberately no open-figure gauge: figures are built with the
    object-oriented API (``Figure`` + ``FigureCanvasAgg``), never pyplot, so
    it would always read 0, and reading it would import pyplot here. That's
    enforced in CI by
    ``test_export_event_loop_offload.NoProductionModuleImportsPyplotTests``.
    """
    rss = current_process_rss_bytes()
    if rss is not None:
        PROCESS_RSS_BYTES.set(rss)

    try:
        from tta_backend.services.open_handle import extract_cache_size_bytes

        BUNDLE_EXTRACT_CACHE_BYTES.set(extract_cache_size_bytes())
    except ImportError:
        pass

    try:
        from tta_backend.services.cube_cache import store_size_bytes

        CUBE_STORE_BYTES.set(store_size_bytes())
    except ImportError:
        pass
