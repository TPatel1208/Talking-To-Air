"""T60: a refused region must never become a retrieval of a *different* one.

**Why this is an eval and not a unit test.** Everything below the model is
already unit-tested: `test_region_retrieval_extent.py` proves the seam refuses,
`test_region_buffer.py` and `test_region_composition.py` prove each refusal
names its cause. What none of them can prove is what the *model* does when it
receives one — and the failure this guards against is a model behaviour, not a
backend one. T46 Phase 2's V6 measured it live: the AOI step refused "Ozone
Transport Commission", and the agent silently substituted "Northeastern United
States" and retrieved that, after which the mask clipped a cube that never
covered the region and nothing said so.

The Phase 5 agent gate re-confirmed how cheap that failure is to reach. Before
the alias fix, "the OTC region" resolved to a **0.2-degree box in Connecticut**
standing in for an eleven-state region — disclosed honestly as a geocoded
point, and completely wrong. Silent geographic substitution does not look like
an error; it looks like an answer.

**Why the assertion is what it is.** For a region T60 refuses,
``region_aware_area_of_interest`` returns the error envelope *without* calling
the MCP tool. So the fake MCP's ``define_area_of_interest`` should never run at
all. If the model relays the refusal, it stays unrun. If the model quietly
swaps in some other region, the substitute arrives there and the handler
records it. ``aoi_calls == []`` is therefore exactly "no silent substitution",
with no natural-language matching in the loop.

**Why the geocoder is mocked.** The buffer grammar resolves ``X`` through
``geocoding_service`` (D9), so an un-mocked run would hit live Nominatim once
per iteration, at 1 req/sec, with answers that can change under us. The
investigation probe (``tests/probe_region_grammar.py``) keeps the live path for
end-to-end work; this file is the deterministic, network-free half.

Opt-in (``pytest -m eval``) because it calls a real model and spends real
tokens — two agent workflows.
"""
import importlib.util
import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

import pytest

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

REQUIRED_MODULES = [
    "langchain_mcp_adapters", "fastmcp", "uvicorn", "shapely", "cartopy",
    "rasterio", "pyproj",
]

# Newark, so "within 3000 miles of NYC" and "NY + Wakanda" both centre on a
# real, unambiguous point without asking anyone. Nominatim's own response
# shape, minus the boundary: what a buffer needs from X is a point (D9).
GEOCODER_HIT = {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "display_name": "New York, United States",
    "polygon": None,
    "bbox": [40.4774, 40.9176, -74.2591, -73.7002],
}

# Each case is a region the backend refuses, and the specific reason it names.
# Deliberately NOT included: "within 50 of Newark". The Phase 5 gate measured
# the model supplying "miles" for a bare number, twice, against two different
# prompt wordings — so that refusal never reaches the model at all. It is a
# documented limitation of this model, not a regression to guard, and pinning
# it here would make the eval red for something no code change can fix.
REFUSED_REGIONS = [
    (
        "unresolvable-member",
        "map NO2 over NY + Wakanda for June 1-7 2024",
        "wakanda",
    ),
    (
        "buffer-too-large",
        "map NO2 within 3000 miles of NYC for June 1-7 2024",
        None,  # a size limit; the number it names is not worth pinning verbatim
    ),
]


def _real_google_key_available() -> bool:
    from tta_backend.config.settings import get_settings

    key = get_settings().google_api_key
    return bool(key) and key not in ("test", "your_google_key")


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "region-refusal eval dependencies are not installed",
)
class RegionRefusalEvalTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, prompt: str):
        """Drive the real agent over the **workspace-bound** tool set.

        ``load_earthdata_tools`` rather than ``load_raw_mcp_tools`` is the
        whole point: ``bind_workspace`` is what applies
        ``region_aware_area_of_interest``, so T60's refusals are in the path.
        The scripted eval harness builds its agent the other way, which is why
        no eval task has ever exercised this seam.
        """
        from fake_earthdata_mcp import FakeEarthdataMCPServer, build_fake_mcp
        from eval_harness import _standard_handlers

        from tta_backend.agents.earthdata_agent import build_earthdata_agent
        from tta_backend.config.settings import Settings
        from tta_backend.earthdata_mcp.toolset import load_earthdata_tools
        from tta_backend.utils.streaming import stream_response

        aoi_calls: list[str] = []
        handlers = _standard_handlers()

        async def define_area_of_interest(location, workspace_id):
            aoi_calls.append(location)
            return {"aoi_handle": "aoi_1", "location": location}

        handlers["define_area_of_interest"] = define_area_of_interest

        server = FakeEarthdataMCPServer(build_fake_mcp(handlers))
        server.start()
        try:
            settings = Settings(earthdata_mcp_url=server.url, earthdata_mcp_token=None)
            tools = await load_earthdata_tools(settings, lambda: "eval-user")
            agent = build_earthdata_agent(mcp_tools=tools)

            tool_calls: list[str] = []
            text: list[str] = []
            # D9's geocode, faked at the GeocodingService seam so the buffer
            # grammar resolves X without a live Nominatim call.
            with patch(
                "tta_backend.utils.plotting.GeocodingService.ageocode",
                AsyncMock(return_value=GEOCODER_HIT),
            ):
                async for event_type, data in stream_response(
                    agent, prompt, thread_id=f"refusal-{abs(hash(prompt))}"
                ):
                    if event_type == "tool_call":
                        tool_calls.append(data["name"])
                    elif event_type == "text":
                        text.append(
                            data if isinstance(data, str)
                            else data.get("response", "")
                        )
            return aoi_calls, tool_calls, "".join(text)
        finally:
            server.stop()

    @pytest.mark.eval
    @unittest.skipUnless(_real_google_key_available(), "requires a real GOOGLE_API_KEY")
    async def test_a_refused_region_never_becomes_a_retrieval_of_another_one(self):
        for label, prompt, names in REFUSED_REGIONS:
            with self.subTest(case=label):
                aoi_calls, tool_calls, answer = await self._run(prompt)

                # THE assertion. The wrapper refuses before invoking the tool,
                # so a non-empty list is a region the model chose itself after
                # being told the requested one could not be used.
                self.assertEqual(
                    aoi_calls, [],
                    f"{label}: the refused region was replaced by "
                    f"{aoi_calls!r} and retrieved instead of relayed",
                )
                # ...and nothing downstream ran on a substitute either.
                self.assertNotIn("safe_retrieve", tool_calls)

                # The refusal has to reach the researcher, not be swallowed
                # into a generic "no data" — the whole reason D14 makes these
                # errors name their cause.
                if names:
                    self.assertIn(
                        names, answer.lower(),
                        f"{label}: the answer never names {names!r}",
                    )
