"""T64 Phase 4: narration lanes. Sub-agents a dispatch call runs in parallel
narrate into one stream; each status/job_progress event they emit carries the
lane it belongs to, so the workflow strip can show one row per request
instead of interleaving them."""
import asyncio
import unittest
from types import SimpleNamespace


def _sat(location):
    from tta_backend.models import SatelliteRequest

    return SatelliteRequest.model_validate(
        {"intent": "visualize", "figure_type": "map", "variable": "NO2", "location": location},
    )


def _ground(location):
    from tta_backend.models import GroundRequest

    return GroundRequest.model_validate({"intent": "readings", "pollutant": "O3", "location": location})


async def _narrate(requests, run_one):
    """Run one dispatch inside a real stream_response turn and return the
    narration it streamed."""
    from tta_backend.services.dispatch_executor import execute_dispatch
    from tta_backend.utils.streaming import stream_response

    class Supervisor:
        async def astream(self, input_, config, stream_mode):
            await asyncio.create_task(execute_dispatch(requests, run_one))
            yield "messages", (SimpleNamespace(content="done", type="ai", tool_calls=None), {})

    return [
        (event, data) async for event, data in stream_response(Supervisor(), "hi", "thread-1")
        if event in ("status", "job_progress")
    ]


class DispatchLaneTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_requests_narration_carries_its_own_lane(self):
        from tta_backend.models import AgentResult
        from tta_backend.utils.streaming import emit_status

        async def run_one(request):
            if request.agent == "satellite":
                emit_status("Searching datasets...", stage="search")
            await asyncio.sleep(0)
            return AgentResult(text="ok")

        events = await _narrate([_sat("the OTC"), _ground("New Jersey")], run_one)

        search = [d for e, d in events if d.get("stage") == "search"]
        self.assertEqual([(d["lane"], d["lane_label"]) for d in search], [("1.0", "Satellite · NO2 · the OTC")])
        lanes = {d["lane"]: d["lane_label"] for _, d in events}
        self.assertEqual(lanes, {"1.0": "Satellite · NO2 · the OTC", "1.1": "Ground · O3 · New Jersey"})

    async def test_every_lane_opens_running_and_closes_finished_saying_how_it_ended(self):
        """The ground agent narrates nothing itself; without the executor's
        own markers its lane would never appear, and a finished lane would
        look stuck on its last stage."""
        from tta_backend.models import AgentResult

        async def run_one(request):
            if request.location == "New York":
                raise RuntimeError("boom")
            return AgentResult(text="ok")

        events = await _narrate([_ground("New Jersey"), _ground("New York")], run_one)

        by_lane = {}
        for _, d in events:
            by_lane.setdefault(d["lane"], []).append((d["stage"], d["message"]))
        self.assertEqual(by_lane["1.0"], [("running", "Running"), ("finished", "Done")])
        self.assertEqual(by_lane["1.1"], [("running", "Running"), ("finished", "Failed")])

    async def test_a_run_held_back_by_a_cap_shows_queued_before_running(self):
        """Two satellite slots: the third satellite request waits, and its
        lane says so instead of sitting silent."""
        from tta_backend.models import AgentResult

        async def run_one(request):
            await asyncio.sleep(0.02)
            return AgentResult(text="ok")

        events = await _narrate([_sat("NJ"), _sat("NY"), _sat("PA")], run_one)

        stages = {}
        for _, d in events:
            stages.setdefault(d["lane"], []).append(d["stage"])
        self.assertEqual(stages["1.0"], ["running", "finished"])
        self.assertEqual(stages["1.2"], ["queued", "running", "finished"])

    async def test_job_progress_inside_a_run_carries_its_lane(self):
        from tta_backend.models import AgentResult
        from tta_backend.utils.streaming import emit_job_progress

        async def run_one(request):
            emit_job_progress("job_1", "processing", 40)
            return AgentResult(text="ok")

        events = await _narrate([_sat("the OTC")], run_one)

        (progress,) = [d for e, d in events if e == "job_progress"]
        self.assertEqual((progress["lane"], progress["job_handle"]), ("1.0", "job_1"))

    async def test_narration_outside_a_dispatch_carries_no_lane(self):
        """The router fast path's events keep their exact shape."""
        from tta_backend.utils.streaming import emit_status, stream_response

        class Agent:
            async def astream(self, input_, config, stream_mode):
                emit_status("Searching datasets...", stage="search")
                await asyncio.sleep(0)
                yield "messages", (SimpleNamespace(content="done", type="ai", tool_calls=None), {})

        statuses = [d for e, d in [x async for x in stream_response(Agent(), "hi", "t")] if e == "status"]

        self.assertEqual(statuses, [{"message": "Searching datasets...", "stage": "search"}])


if __name__ == "__main__":
    unittest.main()
