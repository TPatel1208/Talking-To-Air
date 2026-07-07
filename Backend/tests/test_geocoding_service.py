"""
Geocoder unification tests (T11): one shared GeocodingService instance
(one cache, one rate limiter, one policy-compliant User-Agent) should back
every consumer — the satellite plot/stat/validation tools' RegionResolver
and the ground sensor tools' EPA AQS module — instead of each constructing
its own.
"""
import importlib.util
import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install

REQUIRED_MODULES = ["httpx", "cartopy", "shapely", "rasterio"]


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeAsyncClient:
    calls = 0
    payload = [{
        "lat": "40.7128",
        "lon": "-74.0060",
        "display_name": "New York City",
        "geojson": None,
        "boundingbox": ["40.4", "40.9", "-74.3", "-73.7"],
    }]

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None, headers=None):
        type(self).calls += 1
        type(self).last_headers = headers
        return _FakeResponse(type(self).payload)


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED_MODULES),
    "geocoding-service test dependencies are not installed",
)
class GeocodingServiceUnificationTests(unittest.IsolatedAsyncioTestCase):
    """Deliberately does not reset utils.plotting's module-level singleton:
    epa_aqs_tools (and every RegionResolver across the test process) already
    captured a reference to it at import time, so resetting it here would
    desync those references from a freshly-minted instance. Tests that need
    an uncached lookup use a location name unique to this file instead.
    """

    def setUp(self):
        _FakeAsyncClient.calls = 0

    async def test_get_geocoding_service_returns_one_shared_singleton(self):
        from utils.plotting import get_geocoding_service

        self.assertIs(get_geocoding_service(), get_geocoding_service())

    async def test_region_resolver_defaults_to_the_shared_singleton(self):
        from utils.plotting import RegionResolver, get_geocoding_service

        resolver = RegionResolver()

        self.assertIs(resolver.geocoding_service, get_geocoding_service())

    async def test_epa_aqs_tools_geocoding_service_is_the_shared_singleton(self):
        from tools.ground_sensor_tools import epa_aqs_tools
        from utils.plotting import get_geocoding_service

        self.assertIs(epa_aqs_tools.geocoding_service, get_geocoding_service())

    async def test_two_resolutions_of_the_same_place_share_one_network_call(self):
        from tools.ground_sensor_tools import epa_aqs_tools
        from utils.plotting import RegionResolver

        location = "T11 Unification Test Locale"
        with patch("utils.plotting.httpx.AsyncClient", _FakeAsyncClient):
            resolver = RegionResolver()
            await resolver.aresolve_location(location)

            # A second consumer (the EPA tools' module-level geocoding_service,
            # unified onto the same singleton) resolving the same place name
            # must hit the shared cache, not the network again.
            await epa_aqs_tools.geocoding_service.ageocode(location)

        self.assertEqual(_FakeAsyncClient.calls, 1)

    async def test_ageocode_sends_a_policy_compliant_user_agent(self):
        from utils.plotting import NOMINATIM_USER_AGENT, get_geocoding_service

        with patch("utils.plotting.httpx.AsyncClient", _FakeAsyncClient):
            await get_geocoding_service().ageocode("T11 User-Agent Test Locale")

        self.assertEqual(_FakeAsyncClient.last_headers["User-Agent"], NOMINATIM_USER_AGENT)
        self.assertNotIn("Educational project", NOMINATIM_USER_AGENT)


if __name__ == "__main__":
    unittest.main()
