
import unittest
from datetime import date

from tta_backend.services.hms_service import (
    HMSError,
    download_url,
    parse_archive,
    product_date,
)


class HMSServiceTests(unittest.TestCase):

    def test_valid_date(self):
        result = product_date("2026-01-01")
        self.assertEqual(result, date(2026, 1, 1))

    def test_invalid_date(self):
        for value in ("2026-02-30", "not-a-date", "2099-01-01"):
            with self.subTest(value=value):
                with self.assertRaises(HMSError) as ctx:
                    product_date(value)
                self.assertEqual(ctx.exception.status, "invalid_date")

    def test_download_url(self):
        result = download_url("2026-01-01")
        self.assertTrue(
            result.endswith("/2026/01/hms_smoke20260101.zip")
        )

    def test_invalid_zip(self):
        with self.assertRaises(HMSError) as ctx:
            parse_archive(b"this is not a zip", "2026-01-01")
        self.assertEqual(ctx.exception.status, "invalid_data")


if __name__ == "__main__":
    unittest.main()
