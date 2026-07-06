import asyncio
import os
import sys
import unittest
from types import SimpleNamespace

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


class JobProgressStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_response_forwards_job_progress_events_in_order(self):
        from utils.streaming import emit_job_progress, stream_response

        class FakeAgent:
            # Real polling/model calls always await genuine I/O between emits;
            # asyncio.sleep(0) here stands in for that so this fake is a
            # faithful shape of production execution, not a synthetic
            # zero-suspension burst.
            async def astream(self, input_, config, stream_mode):
                emit_job_progress("job_1", "queued", 0, "submitting", "Submitting retrieval...")
                await asyncio.sleep(0)
                yield "updates", {}
                emit_job_progress("job_1", "processing", 40, "materializing", "40% complete")
                await asyncio.sleep(0)
                yield "updates", {}
                emit_job_progress("job_1", "materialized", 100, "done", None)
                await asyncio.sleep(0)
                yield "messages", (SimpleNamespace(content="done", type="ai", tool_calls=None), {})

        events = [event async for event in stream_response(FakeAgent(), "do it", "thread-1")]

        job_events = [data for event_type, data in events if event_type == "job_progress"]

        self.assertEqual(
            [e["status"] for e in job_events],
            ["queued", "processing", "materialized"],
        )
        self.assertEqual(job_events[0]["job_handle"], "job_1")
        self.assertEqual(job_events[1]["progress"], 40)
        self.assertEqual(job_events[1]["phase"], "materializing")


class UserIdContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_user_id_context_sets_and_resets_current_user_id(self):
        from utils.streaming import current_user_id, user_id_context

        self.assertIsNone(current_user_id())
        with user_id_context("user-1"):
            self.assertEqual(current_user_id(), "user-1")
        self.assertIsNone(current_user_id())


if __name__ == "__main__":
    unittest.main()
