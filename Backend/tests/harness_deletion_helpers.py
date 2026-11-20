"""A conversation's harness records, and its neighbours', for the deletion tests on either store."""
from __future__ import annotations

import datetime
import os
from typing import Any

import pandas as pd

from tta_backend.harness.deletion import delete_conversation
from tta_backend.harness.handles import Descriptor, HandleRecord, ServicePayload, new_handle_id
from tta_backend.harness.handles.errors import HandleError
from tta_backend.harness.handles.payloads import PayloadStore
from tta_backend.harness.job_store import SavedJob
from tta_backend.harness.results import Mode, Result
from tta_backend.harness.results_store import ResultNotFound
from tta_backend.harness.storage import Stores
from tta_backend.harness.trace import NewEntry, Run, RunNotFound, new_run_id

NOW = datetime.datetime(2026, 1, 15, 12, tzinfo=datetime.UTC)


def _on_disk(root: str, relative: str) -> bool:
    return os.path.exists(os.path.join(root, relative))


def _job(fingerprint: str, job_handle: str) -> SavedJob:
    return SavedJob(fingerprint=fingerprint, tool="retrieve_subset", job_handle=job_handle,
                    answer={"job_handle": job_handle}, submitted=NOW)


class Scenario:
    """User A's conversation conv-1 with two runs; A's conv-2, which reuses one of conv-1's jobs; user B's conv-1."""

    def __init__(self, stores: Stores, payloads: PayloadStore) -> None:
        self.stores = stores
        self.payloads = payloads
        self.runs: dict[str, str] = {}
        self.handles: dict[str, str] = {}
        self.payload_paths: dict[str, str] = {}

    async def _run(self, name: str, user_id: str, conversation_id: str, jobs: list[str]) -> None:
        run_id = new_run_id()
        self.runs[name] = run_id
        traces = self.stores.traces
        await traces.start_run(Run(id=run_id, user_id=user_id, conversation_id=conversation_id,
                                   mode=Mode.exploratory(), started=NOW))
        if jobs:
            # A retrieval's begin step, saved as waiting with the jobs it submitted.
            await traces.append(user_id, NewEntry(
                run_id=run_id, kind="use", status="waiting", started=NOW, seconds=1.0,
                capability="sources.satellite.get_data", params={"area": "aoi_1"},
                result={"job": {"product": "TEMPO_NO2", "jobs": [{"job": job, "day": None} for job in jobs]}},
            ))
        await traces.finish_run(user_id, run_id, "done", NOW)
        await self.stores.results.put(user_id, Result(status="done", mode=Mode.exploratory(), trace_id=run_id))

    async def _handles(self, name: str, user_id: str, conversation_id: str) -> None:
        retrieval = f"obs_{name}"
        await self.stores.handles.record_retrieval(HandleRecord(
            id=retrieval, kind="grid", origin="retrieval", user_id=user_id, conversation_id=conversation_id,
            descriptor=Descriptor(summary="NO2"), created_at=NOW, payload=ServicePayload(),
        ))
        derived = new_handle_id("table")
        payload = await self.payloads.write(user_id, derived, "table", pd.DataFrame({"value": [1.0, 2.0]}))
        await self.stores.handles.mint(HandleRecord(
            id=derived, kind="table", origin="derived", user_id=user_id, conversation_id=conversation_id,
            descriptor=Descriptor(summary="A table"), parents=(retrieval,), operation="tabulate",
            created_at=NOW, payload=payload,
        ))
        self.handles[name] = retrieval
        self.handles[f"{name}-derived"] = derived
        self.payload_paths[name] = payload.path  # type: ignore[union-attr]

    async def build(self) -> None:
        await self._run("a1", "user-a", "conv-1", ["job_only_here", "job_shared"])
        await self._run("a2", "user-a", "conv-1", [])
        await self._run("a-other", "user-a", "conv-2", ["job_shared"])
        await self._run("b1", "user-b", "conv-1", ["job_only_here"])
        await self._handles("a", "user-a", "conv-1")
        await self._handles("aother", "user-a", "conv-2")
        await self._handles("b", "user-b", "conv-1")
        for user_id in ("user-a", "user-b"):
            for fingerprint, job in (("fp-1", "job_only_here"), ("fp-2", "job_shared"), ("fp-3", "job_unrelated")):
                await self.stores.jobs.save(user_id, _job(fingerprint, job))


class DeletionChecks:
    """The checks both stores pass; ``stores`` and ``root`` are set by the test."""

    stores: Stores
    root: str

    async def scenario(self) -> Scenario:
        built = Scenario(self.stores, PayloadStore(self.root))
        await built.build()
        await delete_conversation(self.stores, PayloadStore(self.root), "user-a", "conv-1")
        return built

    async def check_the_conversations_runs_results_handles_payloads_and_its_own_jobs_go(self) -> None:
        case: Any = self
        built = await self.scenario()
        for name in ("a1", "a2"):
            with case.assertRaises(RunNotFound):
                await self.stores.traces.get_run("user-a", built.runs[name])
            with case.assertRaises(RunNotFound):
                await self.stores.traces.entries("user-a", built.runs[name])
            with case.assertRaises(ResultNotFound):
                await self.stores.results.get("user-a", built.runs[name])
        case.assertEqual(await self.stores.traces.list_runs("user-a", "conv-1"), [])
        for name in ("a", "a-derived"):
            with case.assertRaises(HandleError):
                await self.stores.handles.get("user-a", built.handles[name])
        case.assertFalse(_on_disk(self.root, built.payload_paths["a"]))
        case.assertIsNone(await self.stores.jobs.find("user-a", "fp-1"))

    async def check_a_job_another_conversation_used_and_everything_else_stays(self) -> None:
        case: Any = self
        built = await self.scenario()
        # Reused by conv-2, so kept for it; never used here, so kept.
        case.assertIsNotNone(await self.stores.jobs.find("user-a", "fp-2"))
        case.assertIsNotNone(await self.stores.jobs.find("user-a", "fp-3"))
        other = built.runs["a-other"]
        case.assertEqual((await self.stores.traces.get_run("user-a", other)).conversation_id, "conv-2")
        case.assertEqual(len(await self.stores.traces.entries("user-a", other)), 1)
        await self.stores.results.get("user-a", other)
        for name in ("aother", "aother-derived"):
            await self.stores.handles.get("user-a", built.handles[name])
        case.assertTrue(_on_disk(self.root, built.payload_paths["aother"]))
        # Another user's conversation of the same id is untouched.
        await self.stores.traces.get_run("user-b", built.runs["b1"])
        await self.stores.results.get("user-b", built.runs["b1"])
        for name in ("b", "b-derived"):
            await self.stores.handles.get("user-b", built.handles[name])
        case.assertTrue(_on_disk(self.root, built.payload_paths["b"]))
        for fingerprint in ("fp-1", "fp-2", "fp-3"):
            case.assertIsNotNone(await self.stores.jobs.find("user-b", fingerprint))

    async def check_deleting_again_does_nothing(self) -> None:
        await self.scenario()
        await delete_conversation(self.stores, PayloadStore(self.root), "user-a", "conv-1")
        case: Any = self
        case.assertEqual(await self.stores.traces.list_runs("user-a", "conv-1"), [])


__all__ = ["DeletionChecks", "Scenario"]
