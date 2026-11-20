"""The named-token refusal reaches the researcher.

``"NY + NJ + Wakanda"`` hard-fails naming the unresolved token, which needs a
raised ``MCPToolError``: over ``dict | None`` every call site collapsed it to
``"Could not resolve location: 'NY + NJ + Wakanda'"``, which names nothing.

There are eleven call sites, nine outside any ``try``. ``export_service`` is
the dangerous one: a bare ``except Exception: region = None`` would swallow a
refusal into **a chart over the entire globe, labelled with the region the
researcher asked for**.

"""
import importlib.util
import os
import sys
import unittest


TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = [
    "langchain", "langchain_mcp_adapters", "fastmcp", "uvicorn",
    "numpy", "xarray", "zarr", "pandas", "shapely", "rasterio", "cartopy", "affine",
]

BAD = "NY + NJ + Wakanda"


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region composition error-channel test dependencies are not installed",
)
class ExportServiceMustNotSwallowTheRefusalTests(unittest.IsolatedAsyncioTestCase):
    """The export path must not swallow a refusal.

    ``except Exception: region = None`` is a reasonable guard against a
    geocoder hiccup and a catastrophe against a refusal: the export proceeds
    with no mask and no extent, producing a **global** chart under the
    region's name. An export cannot honestly degrade "I refuse to guess what
    this region is" into "no region".
    """

    async def test_the_export_paths_let_a_region_refusal_propagate(self):
        """Asserted on the source's own behaviour via the shared helper both
        call sites use, rather than on the text of an `except` clause -- a
        test that greps for a clause name passes on a rewrite that keeps the
        clause and reintroduces the swallow.

        Awaited because the helper moved to the async resolver to get the
        blocking geocoder off the event loop. The claim is untouched by that:
        which resolver is reached decides nothing about whether a refusal is
        swallowed, and this is the assertion that says it must not be."""
        from tta_backend.earthdata_mcp.results import MCPToolError
        from tta_backend.services.export_service import _resolve_export_region

        with self.assertRaises(MCPToolError) as caught:
            await _resolve_export_region(BAD)
        self.assertIn("wakanda", caught.exception.message.lower())

    async def test_an_ordinary_geocode_failure_is_still_tolerated(self):
        """The other half. The guard exists for a reason -- a Nominatim
        timeout during an export should still produce an unmasked chart rather
        than failing the download -- and narrowing it must not remove that."""
        from unittest.mock import AsyncMock, patch

        from tta_backend.services.export_service import _resolve_export_region
        from tta_backend.utils.plotting import RegionResolver

        with patch.object(RegionResolver, "aresolve_location",
                          AsyncMock(side_effect=OSError("boom"))):
            self.assertIsNone(await _resolve_export_region("paris"))
