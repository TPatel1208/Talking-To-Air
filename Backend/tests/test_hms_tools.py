
import unittest
from unittest.mock import AsyncMock, patch

from tta_backend.models import parse_agent_result
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

    async def test_smoke_map_is_registered_as_a_map_artifact(self):
        parsed = {
            "geojson": {"type": "FeatureCollection", "features": []},
            "counts": {"light": 1, "medium": 0, "heavy": 0, "unknown": 0},
            "feature_count": 1,
            "status": "ready",
            "provenance": {"source": "NOAA HMS"},
        }
        emitted = []
        with (
            patch("tta_backend.tools.hms_tools.download_archive", new_callable=AsyncMock, return_value=b"zip"),
            patch("tta_backend.tools.hms_tools.parse_archive", return_value=parsed),
            patch("tta_backend.tools.hms_tools.emit_chart", side_effect=emitted.append),
        ):
            raw = await query_hms_smoke.ainvoke({"date": "2025-08-01", "bounds": [-125.0, 32.0, -114.0, 42.0]})

        result = parse_agent_result(raw)
        self.assertIsNotNone(result)
        [chart] = emitted
        [artifact] = result.artifacts
        self.assertEqual(artifact.id, chart["chart_id"])
        self.assertEqual(artifact.type, "map")
        self.assertEqual(artifact.metadata["bbox"], [-125.0, 32.0, -114.0, 42.0])
        self.assertEqual(artifact.metadata["variable"], "smoke_density")
        self.assertEqual(chart["_artifact_refs"][0]["id"], chart["chart_id"])

    async def test_full_product_artifact_uses_the_whole_globe(self):
        parsed = {
            "geojson": {"type": "FeatureCollection", "features": []},
            "counts": {"light": 0, "medium": 0, "heavy": 0, "unknown": 0},
            "feature_count": 0,
            "status": "no_observed_smoke",
            "provenance": {"source": "NOAA HMS"},
        }
        with (
            patch("tta_backend.tools.hms_tools.download_archive", new_callable=AsyncMock, return_value=b"zip"),
            patch("tta_backend.tools.hms_tools.parse_archive", return_value=parsed),
            patch("tta_backend.tools.hms_tools.emit_chart"),
        ):
            raw = await query_hms_smoke.ainvoke({"date": "2025-08-01"})

        [artifact] = parse_agent_result(raw).artifacts
        self.assertEqual(artifact.metadata["bbox"], [-180.0, -90.0, 180.0, 90.0])


if __name__ == "__main__":
    unittest.main()
