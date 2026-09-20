import asyncio
import importlib.util
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


class SilentGroundAgent:
    """A ground agent that thinks for a while and narrates nothing.

    The honest shape of ``run_ground``: one awaited ``ainvoke`` with no
    ``on_event``, so the fast path has nothing to forward until it returns.
    """

    def __init__(self, think_seconds):
        self.think_seconds = think_seconds

    async def ainvoke(self, input_, config):
        await asyncio.sleep(self.think_seconds)
        return {"messages": [SimpleNamespace(content=json.dumps({
            "summary": "The closest NO2 monitor is Rutgers University.",
            "artifact_ids": [],
            "handles": [],
        }), type="ai")]}


class ChattySatelliteAgent:
    """Narrates steadily for longer than the heartbeat interval.

    Satellite rather than ground because ``run_satellite`` is the one that
    takes an ``on_event`` -- this is the only way the fast path's own loop
    sees activity while a sub-agent is still working.
    """

    def __init__(self, rounds, gap_seconds):
        self.rounds = rounds
        self.gap_seconds = gap_seconds

    async def astream(self, input_, config, stream_mode):
        from tta_backend.utils.streaming import emit_status

        for _ in range(self.rounds):
            emit_status("Fetching a granule (a real stage)...", stage="retrieve")
            await asyncio.sleep(self.gap_seconds)
        yield "messages", (SimpleNamespace(content=json.dumps({
            "summary": "Plotted NO2 over New Jersey.", "artifact_ids": [], "handles": [],
        }), type="ai", tool_calls=None), {})


def _no_monitor_context():
    from tta_backend.services import subagent_dispatch

    return (
        patch.object(subagent_dispatch, "get_ground_monitor_context", AsyncMock(return_value={})),
        patch.object(subagent_dispatch, "save_ground_monitor_context", AsyncMock()),
    )


GROUND_MESSAGE = "Find the nearest NO2 monitor to Tampa FL"
SATELLITE_MESSAGE = "Plot TROPOMI NO2 over New Jersey for 2024-01-15"


async def _fast_path_frames(*, ground=None, satellite=None):
    """Frames from one fast-path turn, routed to whichever agent was given.

    Which slot the agent goes in matters: passing a satellite agent as the
    ground one leaves the real satellite an AsyncMock, the sub-agent raises,
    and the turn answers "hit an internal error" -- which looks exactly like
    a quiet, well-behaved turn to an assertion counting heartbeats.
    """
    from tta_backend.services.chart_service import ChartService
    from tta_backend.services.chat_stream_service import ChatStreamService

    assert (ground is None) != (satellite is None), "exactly one agent per turn"
    service = ChatStreamService(ChartService(), long_request_seconds=999)
    get_ctx, save_ctx = _no_monitor_context()
    with get_ctx, save_ctx:
        frames = [
            frame
            async for frame in service.stream_chat_events(
                AsyncMock(),
                ground if ground is not None else UntouchedAgent(),
                satellite if satellite is not None else UntouchedAgent(),
                GROUND_MESSAGE if ground is not None else SATELLITE_MESSAGE,
                "thread-1", "user-1", "req-1",
            )
        ]
    # A sub-agent that blew up answers in prose rather than raising, so every
    # assertion below would otherwise pass against a turn that never ran.
    assert "internal error" not in "".join(frames), "".join(frames)
    return frames


class UntouchedAgent:
    """Fails loudly if the route dispatches to the agent it should not."""

    def __getattr__(self, name):
        raise AssertionError(f"unexpected access to untouched agent: {name}")


def _working_statuses(frames):
    out = []
    for frame in frames:
        if not frame.startswith("event: status"):
            continue
        payload = json.loads(frame.split("data: ", 1)[1])
        if payload.get("stage") == "working":
            out.append(payload)
    return out


@unittest.skipIf(importlib.util.find_spec("langchain") is None, "langchain is not installed")
class FastPathHeartbeatTests(unittest.IsolatedAsyncioTestCase):
    """The fast path must write during a silence, not just after it.

    T63 D14 makes a reader infer "the replica that owned this turn died" from
    a stream that has gone quiet past the heartbeat. That inference is only
    as good as the guarantee that a live turn keeps writing -- and the fast
    path never had one: it dispatches sub-agents directly and so never enters
    ``stream_response``, where the supervisor route's watchdog lives.
    """

    async def asyncSetUp(self):
        from tta_backend.services import subagent_dispatch

        subagent_dispatch.get_call_budget().clear()

    async def test_a_silent_ground_turn_still_writes_while_it_thinks(self):
        import tta_backend.utils.streaming as streaming

        with patch.object(streaming, "HEARTBEAT_INTERVAL_SECONDS", 0.05), \
             patch.object(streaming, "HEARTBEAT_CHECK_SECONDS", 0.02):
            frames = await _fast_path_frames(ground=SilentGroundAgent(think_seconds=0.25))

        beats = _working_statuses(frames)
        self.assertGreaterEqual(len(beats), 1, "".join(frames))
        self.assertIn("elapsed", beats[0]["message"])
        self.assertIsInstance(beats[0]["detail"], int)

    async def test_the_turn_still_ends_normally_around_the_heartbeat(self):
        """The beat is additive: it must not displace or reorder the answer."""
        import tta_backend.utils.streaming as streaming

        with patch.object(streaming, "HEARTBEAT_INTERVAL_SECONDS", 0.05), \
             patch.object(streaming, "HEARTBEAT_CHECK_SECONDS", 0.02):
            frames = await _fast_path_frames(ground=SilentGroundAgent(think_seconds=0.25))

        joined = "".join(frames)
        self.assertIn("Agent consulted: GROUND", joined)
        self.assertEqual(joined.count("event: done"), 1, joined)
        self.assertTrue(frames[-1].startswith("event: done"), frames[-1])

    async def test_a_turn_that_never_goes_quiet_gets_no_heartbeat(self):
        """The beat measures silence, not elapsed time.

        Mutation guard for the activity clock: a watchdog that fires on a
        bare timer would talk over a sub-agent that is already narrating,
        which is the thing the supervisor route's watchdog is careful not to
        do. The sub-agent here speaks every 0.03s for 0.30s -- five times the
        heartbeat interval, and never once silent for one.
        """
        import tta_backend.utils.streaming as streaming

        with patch.object(streaming, "HEARTBEAT_INTERVAL_SECONDS", 0.06), \
             patch.object(streaming, "HEARTBEAT_CHECK_SECONDS", 0.01):
            frames = await _fast_path_frames(
                satellite=ChattySatelliteAgent(rounds=10, gap_seconds=0.03),
            )

        self.assertEqual(_working_statuses(frames), [], "".join(frames))

    async def test_the_heartbeat_does_not_outlive_the_turn(self):
        """One leaked task per turn, for the life of the process.

        The same shape as the ``_tasks`` leak T63 Phase 2 closed: the loop
        never ends on its own, so a beat left running after the turn answers
        keeps its frame -- and everything that frame captured -- resident.
        """
        import tta_backend.utils.streaming as streaming

        before = len(asyncio.all_tasks())
        with patch.object(streaming, "HEARTBEAT_INTERVAL_SECONDS", 0.05), \
             patch.object(streaming, "HEARTBEAT_CHECK_SECONDS", 0.02):
            await _fast_path_frames(ground=SilentGroundAgent(think_seconds=0.15))

        # One hand-back so a just-cancelled task reaches its done state.
        await asyncio.sleep(0)
        self.assertEqual(len(asyncio.all_tasks()), before)


if __name__ == "__main__":
    unittest.main()
