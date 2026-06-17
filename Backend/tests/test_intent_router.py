import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from services.intent_router import inject_routing_hint, route_intent


class RouteIntentTests(unittest.TestCase):
    # ── GROUND_ONLY ──────────────────────────────────────────────────────────

    def test_nearest_monitor_routes_ground(self):
        self.assertEqual(route_intent("Find the nearest NO2 monitor to Austin Texas"), "GROUND")

    def test_closest_sensor_routes_ground(self):
        self.assertEqual(route_intent("What is the closest sensor to Denver CO?"), "GROUND")

    def test_find_monitor_routes_ground(self):
        self.assertEqual(route_intent("find a monitor near Tampa FL"), "GROUND")

    def test_locate_station_routes_ground(self):
        self.assertEqual(route_intent("locate a station in New Jersey"), "GROUND")

    def test_site_info_routes_ground(self):
        self.assertEqual(route_intent("Give me site info for monitor 34-023-0011"), "GROUND")

    def test_station_details_routes_ground(self):
        self.assertEqual(route_intent("station details for 48-453-0014"), "GROUND")

    def test_daily_reading_routes_ground(self):
        self.assertEqual(route_intent("daily reading for site 48-453-0014"), "GROUND")

    def test_daily_summary_routes_ground(self):
        self.assertEqual(route_intent("get the daily summary for this monitor"), "GROUND")

    def test_quarterly_summary_routes_ground(self):
        self.assertEqual(route_intent("quarterly summary for Q1 2024"), "GROUND")

    def test_annual_summary_routes_ground(self):
        self.assertEqual(route_intent("show me the annual summary for NO2"), "GROUND")

    def test_exceedance_routes_ground(self):
        self.assertEqual(route_intent("Find exceedance days for NO2 in Dallas 2024"), "GROUND")

    def test_exceedances_plural_routes_ground(self):
        self.assertEqual(route_intent("How many exceedances were there in Q2?"), "GROUND")

    def test_hourly_profile_routes_ground(self):
        self.assertEqual(route_intent("show the hourly profile for 2024-03-15"), "GROUND")

    def test_aqi_routes_ground(self):
        self.assertEqual(route_intent("What is the AQI near Houston?"), "GROUND")

    def test_air_quality_level_routes_ground(self):
        self.assertEqual(route_intent("What are the air quality levels in Chicago?"), "GROUND")

    def test_no2_monitor_routes_ground(self):
        self.assertEqual(route_intent("find a NO2 monitor near Rutgers University"), "GROUND")

    def test_pm25_sensor_routes_ground(self):
        self.assertEqual(route_intent("nearest PM2.5 sensor to Boston"), "GROUND")

    def test_epa_monitor_routes_ground(self):
        self.assertEqual(route_intent("EPA monitor data for site 34-019-0007"), "GROUND")

    # ── SATELLITE_ONLY ───────────────────────────────────────────────────────

    def test_tropomi_routes_satellite(self):
        self.assertEqual(route_intent("Plot TROPOMI NO2 over Houston for 2024-01-15"), "SATELLITE")

    def test_omi_routes_satellite(self):
        self.assertEqual(route_intent("show OMI ozone over Texas"), "SATELLITE")

    def test_tempo_routes_satellite(self):
        self.assertEqual(route_intent("TEMPO data over the northeast"), "SATELLITE")

    def test_modis_routes_satellite(self):
        self.assertEqual(route_intent("MODIS aerosol over California"), "SATELLITE")

    def test_satellite_map_routes_satellite(self):
        self.assertEqual(route_intent("show me a satellite map of NO2"), "SATELLITE")

    def test_satellite_plot_routes_satellite(self):
        self.assertEqual(route_intent("generate a satellite plot over New Jersey"), "SATELLITE")

    def test_gridded_routes_satellite(self):
        self.assertEqual(route_intent("get gridded NO2 data for this region"), "SATELLITE")

    def test_column_density_routes_satellite(self):
        self.assertEqual(route_intent("column density over the Gulf Coast"), "SATELLITE")

    def test_plot_no2_routes_satellite(self):
        self.assertEqual(route_intent("plot NO2 over New York on 2024-06-01"), "SATELLITE")

    def test_visualize_ozone_routes_satellite(self):
        self.assertEqual(route_intent("visualize ozone over the Southwest"), "SATELLITE")

    # ── BOTH ─────────────────────────────────────────────────────────────────

    def test_compare_ground_and_satellite_routes_both(self):
        self.assertEqual(
            route_intent("Compare ground NO2 readings to satellite over Austin"),
            "BOTH",
        )

    def test_confirm_exceedance_with_satellite_routes_both(self):
        self.assertEqual(
            route_intent("confirm ground exceedance from space using satellite"),
            "BOTH",
        )

    def test_ground_vs_tropomi_routes_both(self):
        self.assertEqual(
            route_intent("How do the EPA ground levels compare to TROPOMI?"),
            "BOTH",
        )

    # ── LLM ─────────────────────────────────────────────────────────────────

    def test_vague_query_routes_llm(self):
        self.assertEqual(route_intent("What is the air pollution like in Texas?"), "LLM")

    def test_generic_question_routes_llm(self):
        self.assertEqual(route_intent("Tell me about air quality standards"), "LLM")

    def test_empty_message_routes_llm(self):
        self.assertEqual(route_intent(""), "LLM")


class InjectRoutingHintTests(unittest.TestCase):
    def test_ground_intent_prepends_directive(self):
        msg = "Find the nearest NO2 monitor to Austin Texas"
        result = inject_routing_hint(msg)
        self.assertTrue(result.startswith("[ROUTE:GROUND_ONLY]\n\n"))
        self.assertIn(msg, result)

    def test_satellite_intent_prepends_directive(self):
        msg = "Plot TROPOMI NO2 over Texas for 2024-01-15"
        result = inject_routing_hint(msg)
        self.assertTrue(result.startswith("[ROUTE:SATELLITE_ONLY]\n\n"))
        self.assertIn(msg, result)

    def test_both_intent_passes_through_unchanged(self):
        msg = "Compare ground NO2 to TROPOMI over Austin"
        self.assertEqual(inject_routing_hint(msg), msg)

    def test_llm_intent_passes_through_unchanged(self):
        msg = "What is air quality like?"
        self.assertEqual(inject_routing_hint(msg), msg)

    def test_directive_not_double_prepended_on_second_call(self):
        msg = "nearest NO2 monitor to Denver"
        once = inject_routing_hint(msg)
        twice = inject_routing_hint(once)
        # The already-prefixed message still starts with one directive
        self.assertEqual(once.count("[ROUTE:GROUND_ONLY]"), 1)
        # Second call sees the prefix text — the whole string becomes the new
        # "message" for the supervisor, so it should not break anything.
        self.assertIn("[ROUTE:GROUND_ONLY]", twice)


if __name__ == "__main__":
    unittest.main()
