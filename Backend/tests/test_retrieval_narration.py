"""What the chat calls an in-flight retrieval while the researcher waits.

A Harmony job spends minutes in "materializing" and the status line had
nothing to say about it beyond "Retrieving data — running...". Every fact that
would make that wait legible — which variable, which place, which dates,
roughly how many bytes — is known synchronously at submit time and was simply
dropped. These tests pin the shape of the description built from it.

The registry itself is deliberately the same two-step, TTL-bounded, in-memory
shape as ``scope_registry``/``variable_choice_registry`` (recorded by
job_handle at submit, read back while awaiting), so — like those two — tests
clear its dict themselves rather than going through ``cache_isolation``: a
narration is keyed by a per-test job handle and has none of the silently-
serve-a-stale-result failure mode that policy exists for.
"""

import unittest


class RetrievalNarrationTests(unittest.TestCase):
    def setUp(self):
        from tta_backend.services import retrieval_narration

        retrieval_narration._narrations.clear()
        self.addCleanup(retrieval_narration._narrations.clear)

    def test_an_unrecorded_job_has_nothing_to_say(self):
        """describe() must never invent a subject. A job nothing was recorded
        for degrades the caller back to the bare "Retrieving data" line."""
        from tta_backend.services import retrieval_narration

        self.assertIsNone(retrieval_narration.describe("job_unknown"))

    def test_it_leads_with_the_science_variable_and_qualifies_it(self):
        """The full point_timeseries shape: variable up front (that's what the
        researcher asked about), then the qualifiers that bound the wait."""
        from tta_backend.services import retrieval_narration

        retrieval_narration.record(
            "job_1",
            variable="nitrogendioxide_tropospheric_column",
            location="Newark, NJ",
            time_range="2024-06-12T00:00:00/2024-06-14T23:59:59",
            estimated_bytes=47_000_000,
        )

        self.assertEqual(
            retrieval_narration.describe("job_1"),
            "nitrogendioxide_tropospheric_column · Newark, NJ · Jun 12–14, 2024 · ~47 MB",
        )

    def test_a_group_qualified_variable_is_shown_by_its_leaf_name(self):
        """Retrievals request group-qualified names ("product/foo"); the group
        is plumbing the researcher never typed and shouldn't have to read."""
        from tta_backend.services import retrieval_narration

        retrieval_narration.record("job_2", variable="product/vertical_column_troposphere")

        self.assertEqual(retrieval_narration.describe("job_2"), "vertical_column_troposphere")

    def test_it_omits_facts_the_caller_did_not_have(self):
        """safe_retrieve's real shape: it only ever sees an opaque aoi_handle,
        never a place name (the T46 note on its scope_registry call). A missing
        fact is omitted, not rendered as an empty slot or a "None"."""
        from tta_backend.services import retrieval_narration

        retrieval_narration.record(
            "job_3",
            variable="AOD_550_Dark_Target_Deep_Blue_Combined",
            time_range="2024-06-12T00:00:00/2024-06-12T23:59:59",
            estimated_bytes=5_200_000,
        )

        self.assertEqual(
            retrieval_narration.describe("job_3"),
            "AOD_550_Dark_Target_Deep_Blue_Combined · Jun 12, 2024 · ~5.2 MB",
        )

    def test_qualifiers_alone_still_describe_the_wait(self):
        """0 or >1 science variables is not a choice worth naming (the same
        rule variable_choice_registry applies), but the scope still is."""
        from tta_backend.services import retrieval_narration

        retrieval_narration.record("job_4", time_range="2024-06-12T00:00:00/2024-07-03T23:59:59")

        self.assertEqual(retrieval_narration.describe("job_4"), "Jun 12 – Jul 3, 2024")

    def test_a_range_spanning_new_year_carries_both_years(self):
        """The year is factored out of a same-year range to keep the line
        short; a range that crosses one must not lose it."""
        from tta_backend.services import retrieval_narration

        retrieval_narration.record("job_5", time_range="2023-12-30T00:00:00/2024-01-02T23:59:59")

        self.assertEqual(retrieval_narration.describe("job_5"), "Dec 30, 2023 – Jan 2, 2024")

    def test_an_unparseable_time_range_is_dropped_rather_than_shown_raw(self):
        """Narration is cosmetic; a malformed range means the MCP's own
        time_range validation is about to reject the job anyway. Showing the
        raw string would put ISO plumbing in front of the researcher for no
        gain, so the fact is simply omitted and the rest still renders."""
        from tta_backend.services import retrieval_narration

        retrieval_narration.record("job_6", variable="NO2", time_range="whenever")

        self.assertEqual(retrieval_narration.describe("job_6"), "NO2")

    def test_recording_nothing_usable_is_a_no_op(self):
        """Same guard as scope_registry.record_pending: an entry with no fact
        in it would render an empty subject line, which reads worse than the
        "Retrieving data" default it would be replacing."""
        from tta_backend.services import retrieval_narration

        retrieval_narration.record("job_7", variable=None, location=None, time_range=None)

        self.assertIsNone(retrieval_narration.describe("job_7"))
        self.assertNotIn("job_7", retrieval_narration._narrations)

    def test_a_falsy_job_handle_is_a_no_op(self):
        """A submit that returned no job_handle is a contract error the caller
        raises on; narration must not be the thing that KeyErrors first."""
        from tta_backend.services import retrieval_narration

        retrieval_narration.record("", variable="NO2")
        retrieval_narration.record(None, variable="NO2")

        self.assertEqual(retrieval_narration._narrations, {})

    def test_a_discarded_job_stops_being_described(self):
        """Unlike the other two registries there is no ``finalize``: a
        narration describes the *wait*, so reaching a terminal state ends it."""
        from tta_backend.services import retrieval_narration

        retrieval_narration.record("job_8", variable="NO2")
        retrieval_narration.discard("job_8")

        self.assertIsNone(retrieval_narration.describe("job_8"))

    def test_discarding_an_unknown_job_is_harmless(self):
        """await_retrieval discards on every terminal poll, including for jobs
        it never recorded (open_handle's rematerialize path awaits a job this
        process never submitted)."""
        from tta_backend.services import retrieval_narration

        retrieval_narration.discard("job_never_seen")  # must not raise

    def test_an_expired_narration_is_dropped_on_read(self):
        """TTL-bounded like its two sibling registries — a job that never
        reaches a terminal state (so never discards) must not pin its entry
        forever."""
        import time

        from tta_backend.services import retrieval_narration

        retrieval_narration.record("job_9", variable="NO2")
        facts, _ = retrieval_narration._narrations["job_9"]
        retrieval_narration._narrations["job_9"] = (facts, time.time() - 1)

        self.assertIsNone(retrieval_narration.describe("job_9"))
        self.assertNotIn("job_9", retrieval_narration._narrations)

    def test_byte_sizes_scale_to_a_readable_unit(self):
        """A retrieval estimate spans kB to GB; the number is an estimate, so
        it carries a "~" and at most one decimal rather than implying the
        provider quoted it to the byte."""
        from tta_backend.services.retrieval_narration import _format_bytes

        self.assertEqual(_format_bytes(2_100_000_000), "~2.1 GB")
        self.assertEqual(_format_bytes(47_000_000), "~47 MB")
        self.assertEqual(_format_bytes(5_200_000), "~5.2 MB")
        self.assertEqual(_format_bytes(4_096), "~4.1 kB")
        self.assertEqual(_format_bytes(512), "~512 B")

    def test_a_missing_or_nonsensical_byte_estimate_is_omitted(self):
        """estimate_retrieval_size can answer with no number at all
        (safe_retrieve's "couldn't price it" branch), and that is not zero."""
        from tta_backend.services.retrieval_narration import _format_bytes

        self.assertIsNone(_format_bytes(None))
        self.assertIsNone(_format_bytes(-1))
        self.assertIsNone(_format_bytes(0))


if __name__ == "__main__":
    unittest.main()
