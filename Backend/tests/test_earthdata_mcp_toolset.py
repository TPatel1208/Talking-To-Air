import importlib.util
import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = ["langchain_mcp_adapters", "fastmcp", "uvicorn"]


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "MCP client test dependencies are not installed",
)
class EarthdataToolsetTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer

        self.server = FakeEarthdataMCPServer(build_fake_mcp())
        self.server.start()
        self.addCleanup(self.server.stop)

        from config.settings import Settings

        self.settings = Settings(earthdata_mcp_url=self.server.url, earthdata_mcp_token=None)

    async def test_curated_model_tools_matches_the_curated_list_exactly(self):
        from earthdata_mcp.client import CURATED_TOOL_NAMES
        from earthdata_mcp.toolset import curated_model_tools, load_earthdata_tools

        tools = await load_earthdata_tools(self.settings, lambda: "1")
        model_tools = curated_model_tools(tools)

        self.assertEqual({t.name for t in model_tools}, set(CURATED_TOOL_NAMES))

    async def test_curated_model_tools_excludes_internal_and_hidden_tools(self):
        from earthdata_mcp.toolset import curated_model_tools, load_earthdata_tools

        tools = await load_earthdata_tools(self.settings, lambda: "1")
        model_tool_names = {t.name for t in curated_model_tools(tools)}

        # align is internal as of T08 (the compare tool's period mode calls
        # it directly) but still never model-facing, same as the other
        # internal/hidden composite plumbing. get_retrieval_status,
        # retrieve_timeseries, cite_dataset, and get_provenance are demoted
        # to internal-only as of T11 (MCP-first minimal toolset).
        # inspect_granules joins the internal set as of T21 (discovery
        # pane's granule-inspection endpoint) — the agent keeps coverage +
        # the size gate, so this stays off the model surface too.
        for hidden in (
            "retrieve_subset",
            "estimate_retrieval_size",
            "retrieve_data",
            "align",
            "cancel_retrieval",
            "convert_format",
            "get_retrieval_status",
            "retrieve_timeseries",
            "cite_dataset",
            "get_provenance",
            "inspect_granules",
        ):
            self.assertNotIn(hidden, model_tool_names)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "MCP client test dependencies are not installed",
)
class DescribeDatasetModelViewTests(unittest.IsolatedAsyncioTestCase):
    """T13: describe_dataset's model-facing result stays proportional to what
    the model actually uses (variable names to subset), never every
    fill-value/valid-range record a many-variable TEMPO-style collection
    carries — while the discovery pane's direct calls through the same raw
    tools dict stay full-detail (services/discovery_service.py)."""

    async def asyncSetUp(self):
        from fake_earthdata_mcp import build_fake_mcp, FakeEarthdataMCPServer

        # Mirrors the real earthdata-retrieval MCP's describe_dataset shape
        # (harmony-retrieval-mcp/src/earthdata_mcp/tools/understanding.py):
        # even its own "compact" per-variable record still carries the full
        # fill_values/valid_ranges lists.
        self._full_variables = [
            {
                "name": "NO2_column",
                "long_name": "NO2 tropospheric column",
                "units": "mol/m^2",
                "fill_values": [{"value": -9999.0, "context": "FillValue"}],
                "valid_ranges": [{"min": 0.0, "max": 1.0, "context": "valid_range"}],
                "advisory_notes": ["QA-flagged advisory note"],
            },
            {
                "name": "cloud_fraction",
                "long_name": "Effective cloud fraction",
                "units": "1",
                "fill_values": [],
                "valid_ranges": [],
                "advisory_notes": [],
                "mask_metadata_note": "No fill/range metadata in UMM-Var for this variable.",
            },
        ]

        async def describe_dataset(dataset_handle, detail, workspace_id):
            return {
                "handle": dataset_handle,
                "concept_id": "C123-TEST",
                "metadata": {"short_name": "TEMPO_NO2"},
                "variables": self._full_variables,
                "variable_count": len(self._full_variables),
                "advisory_notes": [],
                "variable_source": "umm-var",
            }

        self.server = FakeEarthdataMCPServer(build_fake_mcp({"describe_dataset": describe_dataset}))
        self.server.start()
        self.addCleanup(self.server.stop)

        from config.settings import Settings

        self.settings = Settings(earthdata_mcp_url=self.server.url, earthdata_mcp_token=None)

    async def test_model_facing_describe_dataset_drops_fill_and_range_record_lists(self):
        from earthdata_mcp.results import parse_tool_result
        from earthdata_mcp.toolset import curated_model_tools, load_earthdata_tools

        tools = await load_earthdata_tools(self.settings, lambda: "1")
        model_tools = {t.name: t for t in curated_model_tools(tools)}

        raw = await model_tools["describe_dataset"].ainvoke({"dataset_handle": "dataset_1"})
        result = parse_tool_result(raw)

        self.assertEqual(result["variable_count"], 2)
        no2 = result["variables"][0]
        self.assertEqual(no2["name"], "NO2_column")
        self.assertEqual(no2["long_name"], "NO2 tropospheric column")
        self.assertEqual(no2["units"], "mol/m^2")
        self.assertEqual(no2["advisory_notes"], ["QA-flagged advisory note"])
        self.assertNotIn("fill_values", no2)
        self.assertNotIn("valid_ranges", no2)
        self.assertIn("mask_note", no2)

        cloud = result["variables"][1]
        self.assertEqual(cloud["mask_metadata_note"], "No fill/range metadata in UMM-Var for this variable.")
        self.assertIn("mask_note", cloud)

    async def test_discovery_pane_still_gets_the_full_per_variable_records(self):
        """discovery_service.py calls tools["describe_dataset"] directly on
        the same workspace-bound dict curated_model_tools reads from — the
        model-view wrapper must not weaken that shared dict."""
        from services.discovery_service import describe_dataset
        from earthdata_mcp.toolset import curated_model_tools, load_earthdata_tools

        tools = await load_earthdata_tools(self.settings, lambda: "1")
        # Build the model-facing list too, to prove it doesn't mutate `tools`.
        curated_model_tools(tools)

        result = await describe_dataset("dataset_1", tools)

        no2 = result["variables"][0]
        self.assertEqual(no2["fill_values"], [{"value": -9999.0, "context": "FillValue"}])
        self.assertEqual(no2["valid_ranges"], [{"min": 0.0, "max": 1.0, "context": "valid_range"}])


if __name__ == "__main__":
    unittest.main()
