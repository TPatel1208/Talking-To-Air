"""The instrument check on satellite workflow calls."""
from __future__ import annotations

import unittest

from tta_backend.harness.supervisor.instruments import (
    NEEDS_INSTRUMENT,
    NOT_REGISTERED,
    USER_NAMED,
    instrument_refusal,
    instruments,
    mentioned,
)

NO2 = "no2_tropospheric_column"


class InstrumentTests(unittest.TestCase):
    def test_each_quantity_lists_its_instruments_in_rank_order(self):
        listed = instruments()
        self.assertEqual(listed[NO2], ("TEMPO", "OMI", "TROPOMI"))
        self.assertIn("MODIS", listed["aod_550nm"])

    def test_only_instrument_names_count_as_mentioned_never_platforms(self):
        self.assertEqual(mentioned("Compare TROPOMI with tempo over Newark."), ("TROPOMI", "TEMPO"))
        self.assertEqual(mentioned("Aqua and Terra pass over Aura Lake."), ())
        self.assertEqual(mentioned("TEMPOrary NO2 over NJ"), ())

    def test_a_missing_name_is_refused_with_the_instruments_for_the_quantity(self):
        for named in (None, "", "  ", 3):
            with self.subTest(named=named):
                self.assertEqual(
                    instrument_refusal(named, NO2, "Map NO2 over NJ."),
                    NEEDS_INSTRUMENT.format(quantity=NO2, instruments="TEMPO, OMI, TROPOMI"),
                )

    def test_a_name_matching_no_registered_product_is_refused(self):
        refusal = instrument_refusal("GOME", NO2, "Map NO2 over NJ.")
        self.assertEqual(refusal, NOT_REGISTERED.format(named="GOME", instruments="TEMPO, OMI, TROPOMI"))

    def test_any_instrument_serves_when_the_user_named_none(self):
        self.assertIsNone(instrument_refusal("TEMPO", NO2, "Map NO2 over NJ."))
        self.assertIsNone(instrument_refusal("TEMPO_NO2", NO2, "Map NO2 over NJ."))

    def test_the_instrument_the_user_named_must_be_the_one_passed(self):
        message = "Compare the ground NO2 monitors with TROPOMI satellite NO2 over Newark NJ for January 2024."
        self.assertEqual(instrument_refusal("TEMPO", NO2, message), USER_NAMED.format(mentioned="TROPOMI", named="TEMPO"))
        for named in ("TROPOMI", "tropomi", "Sentinel-5P", "TROPOMI_NO2", "TROPOMI annual"):
            with self.subTest(named=named):
                self.assertIsNone(instrument_refusal(named, NO2, message))

    def test_with_two_instruments_named_either_is_taken(self):
        message = "Compare TEMPO with TROPOMI NO2 over Newark."
        self.assertIsNone(instrument_refusal("TEMPO", NO2, message))
        self.assertIsNone(instrument_refusal("TROPOMI", NO2, message))
        self.assertIsNotNone(instrument_refusal("OMI", NO2, message))


if __name__ == "__main__":
    unittest.main()
