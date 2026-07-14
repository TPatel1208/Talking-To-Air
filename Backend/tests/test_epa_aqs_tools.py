"""
Regression tests for EPA AQS tool normalisation and monitor-selection logic.

These tests cover the class of bugs where leading zeros are stripped from
EPA site/county/state codes, causing bySite queries to return
"No data matched your selection" even for valid monitors.
"""
import importlib.util
import os
import sys
import unittest
from unittest.mock import MagicMock

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


def _load_epa_module():
    """Load epa_aqs_tools with all external dependencies stubbed."""
    fake_settings = MagicMock(aqs_api_email="test@test.com", aqs_api_key="testkey")
    fake_artifact_ref = MagicMock(id="ref-id", type="table")
    fake_artifact_ref.model_dump.return_value = {"id": "ref-id", "type": "table"}
    fake_artifact_store = MagicMock()
    fake_artifact_store.put_table.return_value = fake_artifact_ref

    stubs = {
        "langchain": MagicMock(),
        "langchain.tools": MagicMock(tool=lambda f: f),
        "config": MagicMock(),
        "config.settings": MagicMock(get_settings=MagicMock(return_value=fake_settings)),
        "services": MagicMock(),
        "services.artifact_store": MagicMock(artifact_store=fake_artifact_store),
        "utils": MagicMock(),
        "utils.plotting": MagicMock(get_geocoding_service=MagicMock()),
    }

    prev = {k: sys.modules.pop(k, None) for k in stubs}
    # Also evict any previously-cached copy of the module under test.
    cached_key = next(
        (k for k in list(sys.modules) if "epa_aqs_tools" in k), None
    )
    cached_mod = sys.modules.pop(cached_key, None) if cached_key else None

    sys.modules.update(stubs)
    try:
        path = os.path.join(
            BACKEND_DIR, "tools", "ground_sensor_tools", "epa_aqs_tools.py"
        )
        spec = importlib.util.spec_from_file_location("epa_aqs_tools_isolated", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in prev.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        if cached_key and cached_mod is not None:
            sys.modules[cached_key] = cached_mod

    return mod


_epa = _load_epa_module()
_normalise_numeric_filter = _epa._normalise_numeric_filter
_normalise_site_filter = _epa._normalise_site_filter
_resolve_filter = _epa._resolve_filter


class NormaliseNumericFilterTests(unittest.TestCase):
    """Unit tests for _normalise_numeric_filter."""

    def test_valid_string_returned_unchanged_without_min_width(self):
        self.assertEqual(_normalise_numeric_filter("x", "42"), "42")

    def test_integer_input_converted_to_string(self):
        self.assertEqual(_normalise_numeric_filter("x", 42), "42")

    def test_zero_padding_applied_when_min_width_set(self):
        self.assertEqual(_normalise_numeric_filter("site", "52", min_width=4), "0052")

    def test_already_padded_value_unchanged(self):
        self.assertEqual(_normalise_numeric_filter("site", "0052", min_width=4), "0052")

    def test_integer_zero_padded_correctly(self):
        # LLM may emit site_number=52 as a JSON integer.
        self.assertEqual(_normalise_numeric_filter("site", 52, min_width=4), "0052")

    def test_placeholder_raises_value_error(self):
        for placeholder in ("site", "site_id", "n/a", "unknown", "??", ""):
            with self.subTest(placeholder=placeholder):
                with self.assertRaises(ValueError):
                    _normalise_numeric_filter("x", placeholder)

    def test_non_digit_raises_value_error(self):
        with self.assertRaises(ValueError):
            _normalise_numeric_filter("x", "abc")

    def test_whitespace_stripped_before_validation(self):
        self.assertEqual(_normalise_numeric_filter("x", "  42  ", min_width=4), "0042")


class NormaliseSiteFilterTests(unittest.TestCase):
    """
    Regression tests for the leading-zero bug in _normalise_site_filter.

    The EPA AQS API requires zero-padded codes:
      state  → 2 digits   ("5"  → "05")
      county → 3 digits   ("1"  → "001")
      site   → 4 digits   ("52" → "0052")

    The LLM may pass any of these as bare integers or short strings,
    so the normaliser must always pad to the required width.
    """

    def test_integer_site_number_zero_padded(self):
        _, _, site = _normalise_site_filter("35", "001", 52)
        self.assertEqual(site, "0052")

    def test_string_site_without_leading_zeros_padded(self):
        _, _, site = _normalise_site_filter("35", "001", "52")
        self.assertEqual(site, "0052")

    def test_county_code_zero_padded_to_three_digits(self):
        _, county, _ = _normalise_site_filter("35", "1", "0052")
        self.assertEqual(county, "001")

    def test_state_code_zero_padded_to_two_digits(self):
        state, _, _ = _normalise_site_filter("5", "001", "0052")
        self.assertEqual(state, "05")

    def test_all_codes_padded_together(self):
        state, county, site = _normalise_site_filter("5", "1", 7)
        self.assertEqual(state, "05")
        self.assertEqual(county, "001")
        self.assertEqual(site, "0007")

    def test_already_padded_values_unchanged(self):
        state, county, site = _normalise_site_filter("35", "001", "0052")
        self.assertEqual(state, "35")
        self.assertEqual(county, "001")
        self.assertEqual(site, "0052")

    def test_compound_station_id_split_and_padded(self):
        # station_id format "35-1-52" → split into parts and zero-pad each
        state, county, site = _normalise_site_filter(None, None, "35-1-52")
        self.assertEqual(state, "35")
        self.assertEqual(county, "001")
        self.assertEqual(site, "0052")

    def test_compound_station_id_already_padded(self):
        state, county, site = _normalise_site_filter(None, None, "35-001-0052")
        self.assertEqual(state, "35")
        self.assertEqual(county, "001")
        self.assertEqual(site, "0052")

    def test_compound_station_id_wrong_part_count_raises(self):
        with self.assertRaises(ValueError):
            _normalise_site_filter(None, None, "35-001")

    def test_placeholder_site_raises(self):
        with self.assertRaises(ValueError):
            _normalise_site_filter("35", "001", "site_number")

    def test_non_numeric_site_raises(self):
        with self.assertRaises(ValueError):
            _normalise_site_filter("35", "001", "abc")


class ResolveFilterTests(unittest.TestCase):
    """Tests for _resolve_filter zero-padding in byState/byCounty branches."""

    def test_bystate_pads_short_state_code(self):
        endpoint, params = _resolve_filter(
            "dailyData", "5", None, None, None, None, None, None, None
        )
        self.assertEqual(endpoint, "dailyData/byState")
        self.assertEqual(params["state"], "05")

    def test_bycounty_pads_state_and_county(self):
        endpoint, params = _resolve_filter(
            "dailyData", "5", "1", None, None, None, None, None, None
        )
        self.assertEqual(endpoint, "dailyData/byCounty")
        self.assertEqual(params["state"], "05")
        self.assertEqual(params["county"], "001")

    def test_bysite_pads_all_three_codes(self):
        endpoint, params = _resolve_filter(
            "dailyData", "5", "1", 52, None, None, None, None, None
        )
        self.assertEqual(endpoint, "dailyData/bySite")
        self.assertEqual(params["state"], "05")
        self.assertEqual(params["county"], "001")
        self.assertEqual(params["site"], "0052")

    def test_bysite_via_compound_station_id(self):
        endpoint, params = _resolve_filter(
            "dailyData", None, None, "35-1-52", None, None, None, None, None
        )
        self.assertEqual(endpoint, "dailyData/bySite")
        self.assertEqual(params["state"], "35")
        self.assertEqual(params["county"], "001")
        self.assertEqual(params["site"], "0052")

    def test_missing_all_filters_raises(self):
        with self.assertRaises(ValueError):
            _resolve_filter(
                "dailyData", None, None, None, None, None, None, None, None
            )


if __name__ == "__main__":
    unittest.main()
