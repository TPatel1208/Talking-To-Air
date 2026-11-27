"""Access stamps keep last-access order strict even when the clock does not move."""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

FROZEN_NS = 1_800_000_000_000_000_000


class AccessClockTests(unittest.TestCase):
    def test_stamps_increase_while_the_clock_stands_still(self):
        from tta_backend.infrastructure.access_clock import STAMP_STEP_NS, access_stamp_ns

        with patch("time.time_ns", return_value=FROZEN_NS):
            stamps = [access_stamp_ns() for _ in range(5)]
        self.assertEqual(stamps, sorted(set(stamps)))
        self.assertTrue(all(b - a >= STAMP_STEP_NS for a, b in zip(stamps, stamps[1:], strict=False)))

    def test_a_stamp_is_never_behind_the_clock(self):
        from tta_backend.infrastructure.access_clock import access_stamp_ns

        later = FROZEN_NS + 10**12
        with patch("time.time_ns", return_value=later):
            self.assertGreaterEqual(access_stamp_ns(), later)

    def test_two_files_stamped_under_a_frozen_clock_keep_their_order_on_disk(self):
        from tta_backend.infrastructure.access_clock import stamp_access

        with tempfile.TemporaryDirectory() as root:
            first, second = os.path.join(root, "first"), os.path.join(root, "second")
            os.mkdir(first)
            os.mkdir(second)
            with patch("time.time_ns", return_value=FROZEN_NS):
                stamp_access(first)
                stamp_access(second)
            self.assertLess(os.stat(first).st_mtime_ns, os.stat(second).st_mtime_ns)


if __name__ == "__main__":
    unittest.main()
