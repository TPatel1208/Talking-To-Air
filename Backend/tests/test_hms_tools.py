
import unittest
from unittest.mock import AsyncMock, patch

from tta_backend.tools.hms_tools import query_hms_smoke


class HMSToolsTests(unittest.IsolatedAsyncioTestCase):

    async def test_unknown_region(self):
        with patch(
            "tta_backend.utils.plotting.RegionResolver"
        ) as mock_resolver:
            mock_resolver.return_value.aresolve_location = AsyncMock(
                return_value=None
            )

            with patch(
                "tta_backend.tools.hms_tools.download_archive",
                new_callable=AsyncMock,
            ) as mock_download:

                result = await query_hms_smoke.ainvoke({
                    "date": "2026-01-01",
                    "region": "xyz123_unknown",
                })

                self.assertIn("invalid_region", result)
                self.assertIn("Unknown region", result)
                mock_download.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
