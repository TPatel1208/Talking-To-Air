"""T64: the satellite agent's own ground tools (validate_against_ground,
exceedance_overlay) run only when its task is a comparison.

Live 2026-10-16: asked for a TEMPO NO2 map of the OTC (intent visualize),
the satellite agent also called validate_against_ground, querying AQS NO2
across the whole OTC -- ground work the supervisor had not dispatched and
the dispatch log could not show."""
import json
import unittest
from unittest.mock import AsyncMock, patch


class GroundValidationScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_outside_a_dispatch_ground_validation_is_allowed(self):
        """The router fast path and every other caller keep today's tools."""
        from tta_backend.utils.streaming import ground_validation_allowed

        self.assertTrue(ground_validation_allowed())

    async def test_both_tools_refuse_out_of_scope_without_opening_data_or_calling_aqs(self):
        from tta_backend.tools.satellite_tools import validation_tools
        from tta_backend.utils.streaming import ground_validation_scope

        open_handle = AsyncMock(side_effect=AssertionError("must not open the handle"))
        monitors = AsyncMock(side_effect=AssertionError("must not call AQS"))
        with patch.object(validation_tools, "open_handle", open_handle), \
             patch.object(validation_tools.aqs, "_fetch_active_monitors", monitors):
            for make in (validation_tools.make_validate_against_ground, validation_tools.make_exceedance_overlay):
                tool = make({})
                with self.subTest(tool=tool.name), ground_validation_scope(False):
                    raw = await tool.ainvoke({"handle": "obs_1", "location": "the OTC"})
                    error = json.loads(raw)["error"]
                    self.assertIn("not part of this task", error)
                    self.assertIn("compare", error)

        open_handle.assert_not_called()
        monitors.assert_not_called()

    async def test_the_scope_ends_with_its_block(self):
        from tta_backend.utils.streaming import ground_validation_allowed, ground_validation_scope

        with ground_validation_scope(False):
            self.assertFalse(ground_validation_allowed())
        self.assertTrue(ground_validation_allowed())


if __name__ == "__main__":
    unittest.main()
