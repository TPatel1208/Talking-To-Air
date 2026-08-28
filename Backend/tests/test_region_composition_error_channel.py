"""T60 Phase 3b / D14: the named-token refusal reaches the researcher.

D8 says ``"NY + NJ + Wakanda"`` hard-fails naming the unresolved token. D14 is
why that was not implementable before this phase: both resolvers returned
``dict | None``, and every call site collapsed ``None`` into one generic
string -- ``"Could not resolve location: 'NY + NJ + Wakanda'"`` -- which names
the whole composition and therefore names nothing.

The Phase 3b gate (V14) counted the blast radius, and the PRD's "six call
sites" was wrong: there are **eleven**, and nine of them sit outside any
``try``. Two more are worse than untouched. ``export_service`` wraps its
resolve in ``except Exception: region = None`` and then renders with
``extent=None, mask_geometry=None`` -- so a raised refusal would have been
swallowed into **a chart over the entire globe, labelled with the region the
researcher asked for**. That is the T46 silent-scope-substitution failure
arriving through the export path, created by D14 rather than found by it.

One representative per file plus both export sites, mirroring how
test_satellite_tools_mcp_errors.py covers this seam.
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest


TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = [
    "langchain", "langchain_mcp_adapters", "fastmcp", "uvicorn",
    "numpy", "xarray", "zarr", "pandas", "shapely", "rasterio", "cartopy", "affine",
]

BAD = "NY + NJ + Wakanda"
HUGE = "alaska + florida"


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region composition error-channel test dependencies are not installed",
)
class ToolsSurfaceTheNamedTokenTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from fake_earthdata_mcp import HandleVolume, build_fake_mcp, FakeEarthdataMCPServer
        from test_satellite_tools_masking_execution import _tempo_no2_dataset
        from tta_backend.earthdata_mcp.client import load_raw_mcp_tools
        from tta_backend.config.settings import Settings
        import xarray as xr

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.volume = HandleVolume(self._tmpdir.name)

        server = FakeEarthdataMCPServer(build_fake_mcp({
            "export_result": self.volume.export_result,
            "rematerialize": self.volume.rematerialize,
            "get_retrieval_status": self.volume.get_retrieval_status,
        }))
        server.start()
        self.addCleanup(server.stop)
        settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
        self.mcp_tools = await load_raw_mcp_tools(settings)

        # A real handle that opens cleanly, so the *only* thing that can fail
        # is the location -- otherwise the tool would refuse at open_handle and
        # the test would pass without ever reaching the resolver.
        self.volume.add_zarr("obs_1", lambda: _tempo_no2_dataset(
            xr, values=[[[1.0, 2.0], [3.0, 4.0]]], flags=[[[0, 0], [0, 0]]],
            lat=(40.0, 41.0), lon=(-75.0, -74.0),
            time=["2024-01-01T00:00:00"],
        ))

    def _assert_names_the_token(self, raw):
        """The tool answered, in the taxonomy's structured shape, naming the
        token. Any of the three could regress independently: a crash gives no
        JSON, a bare-string catch gives no ``category``, and the pre-D14
        collapse gives a message that never says "Wakanda"."""
        payload = json.loads(raw)
        self.assertIn("error", payload, f"tool did not answer structurally: {raw[:200]}")
        self.assertEqual(payload["error"]["category"], "user_input")
        self.assertIn("wakanda", payload["error"]["message"].lower())

    async def test_compute_statistic_tool_names_the_unresolved_token(self):
        from tta_backend.tools.satellite_tools.stat_tools import make_compute_statistic_tool

        tool = make_compute_statistic_tool(self.mcp_tools)
        self._assert_names_the_token(
            await tool.ainvoke({"handle": "obs_1", "location": BAD, "stats": ["mean"]})
        )

    async def test_find_daily_peak_names_the_unresolved_token(self):
        from tta_backend.tools.satellite_tools.stat_tools import make_find_daily_peak

        tool = make_find_daily_peak(self.mcp_tools)
        self._assert_names_the_token(
            await tool.ainvoke({"handle": "obs_1", "location": BAD})
        )

    async def test_plot_singular_names_the_unresolved_token(self):
        from tta_backend.tools.satellite_tools.plot_tools import make_plot_singular

        tool = make_plot_singular(self.mcp_tools)
        self._assert_names_the_token(
            await tool.ainvoke({"handle": "obs_1", "location": BAD})
        )

    async def test_conduct_temporal_statistic_names_the_unresolved_token(self):
        from tta_backend.tools.satellite_tools.plot_tools import make_conduct_temporal_statistic

        tool = make_conduct_temporal_statistic(self.mcp_tools)
        self._assert_names_the_token(
            await tool.ainvoke({"handle": "obs_1", "location": BAD})
        )

    async def test_plot_vertical_profile_names_the_unresolved_token(self):
        """Needs its own handle: the flat TEMPO fixture has no vertical axis,
        so this tool refuses before it ever reaches the resolver -- and a test
        that never reaches the call site is a test that pins nothing."""
        from test_vertical_profile import _o3prof_dataset
        from tta_backend.tools.satellite_tools.plot_tools import make_plot_vertical_profile
        import numpy as np
        import xarray as xr

        self.volume.add_zarr("obs_prof", lambda: _o3prof_dataset(
            xr, np, layer_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ))

        tool = make_plot_vertical_profile(self.mcp_tools)
        self._assert_names_the_token(
            await tool.ainvoke({"handle": "obs_prof", "location": BAD})
        )

    async def test_validate_against_ground_names_the_unresolved_token(self):
        from tta_backend.tools.satellite_tools.validation_tools import make_validate_against_ground

        tool = make_validate_against_ground(self.mcp_tools)
        self._assert_names_the_token(
            await tool.ainvoke({"handle": "obs_1", "location": BAD})
        )

    async def test_exceedance_overlay_names_the_unresolved_token(self):
        from tta_backend.tools.satellite_tools.validation_tools import make_exceedance_overlay

        tool = make_exceedance_overlay(self.mcp_tools)
        self._assert_names_the_token(
            await tool.ainvoke({"handle": "obs_1", "location": BAD})
        )

    async def test_the_extent_refusal_travels_the_same_channel(self):
        """D16's refusal is the same contract as D8's, so it must not need a
        second mechanism. Different category, same structured answer."""
        from tta_backend.tools.satellite_tools.stat_tools import make_compute_statistic_tool

        tool = make_compute_statistic_tool(self.mcp_tools)
        payload = json.loads(
            await tool.ainvoke({"handle": "obs_1", "location": HUGE, "stats": ["mean"]})
        )
        self.assertEqual(payload["error"]["category"], "too_large")
        self.assertIn("4,000,000", payload["error"]["message"])

    async def test_a_composite_chart_puts_both_disclosure_fields_on_the_wire(self):
        """D10a/D10b's precondition, which gate V16 traced: ``region_type``
        does reach the frontend on ``chart.provenance``, so the new
        ``region_origin`` has to travel the same way or the frontend cannot
        tell a self-healed composite from a self-healed rectangle."""
        from tta_backend.tools.satellite_tools.stat_tools import make_compute_statistic_tool

        tool = make_compute_statistic_tool(self.mcp_tools)
        payload = json.loads(await tool.ainvoke({
            "handle": "obs_1", "location": "NY + NJ", "stats": ["mean"],
        }))

        self.assertNotIn("error", payload)
        self.assertEqual(payload["region_origin"], "composite_union")
        self.assertIn(payload["region_type"], ("composite_union", "boundary_cells"))
        self.assertIn("New York", payload["display_name"])
        self.assertIn("New Jersey", payload["display_name"])

    async def test_an_ordinary_geocode_miss_still_answers_as_it_always_did(self):
        """The regression net. D14 changes how a *composite* fails; a plain
        unresolvable place must still produce the same plain message, or this
        phase has quietly rewritten an error surface it was not asked to."""
        from unittest.mock import patch

        from tta_backend.tools.satellite_tools.stat_tools import make_compute_statistic_tool
        from tta_backend.utils.plotting import RegionResolver

        tool = make_compute_statistic_tool(self.mcp_tools)
        with patch.object(RegionResolver, "aresolve_location", return_value=None):
            payload = json.loads(
                await tool.ainvoke({"handle": "obs_1", "location": "zzz", "stats": ["mean"]})
            )
        self.assertEqual(payload["error"], "Could not resolve location: 'zzz'")


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region composition error-channel test dependencies are not installed",
)
class ExportServiceMustNotSwallowTheRefusalTests(unittest.IsolatedAsyncioTestCase):
    """V14's load-bearing finding, and the only one of the eleven that is a
    bug rather than a chore.

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
