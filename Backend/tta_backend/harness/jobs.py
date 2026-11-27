"""Retrieval jobs saved across runs, and the state of the jobs waiting runs are on.

A run's idempotency key covers that run only, so a second run asking for the
same data would submit a second job. ``ReusingEarthdata`` wraps a session and,
before submitting a retrieval, looks for a job the same user already
submitted for the same request; while the provider still holds it, the saved
submit answer is returned instead of submitting again.

``check_waiting`` asks the provider, once per job and without waiting, about
the jobs that the conversation's waiting runs are on.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from typing import Any, Literal

from tta_backend.domain.earthdata.retrieval import PAUSED_STATUS, TERMINAL_STATUSES, annotate_paused
from tta_backend.domain.errors import CATEGORY_NOT_FOUND, CATEGORY_PROVIDER_UNAVAILABLE
from tta_backend.harness.capabilities.invoke import JOB_PARAM
from tta_backend.harness.job_store import RetrievalJobStore, SavedJob
from tta_backend.harness.run_context import Clock, EarthdataSession, RunContext, request_fingerprint
from tta_backend.harness.trace import InvalidTrace, RunNotFound, TraceEntry, serialize_json

logger = logging.getLogger(__name__)

SUBMIT_TOOLS = frozenset({"retrieve_subset", "retrieve_timeseries"})
STATUS_TOOL = "get_retrieval_status"

# A job the provider will never finish: failed, cancelled, paused or no longer held.
_DEAD = "dead"

# Each (user, fingerprint) being submitted in this process: its lock and how
# many calls hold or wait on it. Shared by every ReusingEarthdata, so runs
# with their own instances still submit an identical request once; an entry
# is removed when its last call leaves.
_LOCKS: dict[tuple[str, str], tuple[asyncio.Lock, int]] = {}


def _category(exc: BaseException) -> str | None:
    category = getattr(exc, "category", None)
    return category if isinstance(category, str) else None


def _classify(status: Mapping[str, Any]) -> str:
    """``ready`` or ``running`` for a job the provider holds, ``dead`` for one it never will finish."""
    state = annotate_paused(dict(status)).get("status", "")
    if state == "ready":
        return "ready"
    if state == PAUSED_STATUS or state in TERMINAL_STATUSES:
        return _DEAD
    return "running"


class ReusingEarthdata:
    """An earthdata session that collects a saved retrieval job instead of submitting it again.

    Tools outside ``SUBMIT_TOOLS`` pass straight through. For a submit, a job
    this user saved for the same tool and arguments within ``max_age`` is asked
    about once with ``get_retrieval_status``:

    - queued, running, ready or any other state the provider can still finish:
      the saved submit answer is returned and nothing is submitted;
    - failed, cancelled, paused, no longer held, or a ``not_found`` error: the
      job is forgotten and the request submitted;
    - a ``provider_unavailable`` error: the saved answer is returned, since the
      job may well be fine; the source's own polling meets the outage next and
      handles it as for any job;
    - any other error is raised, as the submit itself would have failed.

    A real submit passes the idempotency key to the inner session, and the
    answer is saved before it is returned, when it holds a job handle. A
    submit whose answer is the job just forgotten (the inner session repeating
    an answered key) is not saved again. The store is best effort: a failure
    reading or saving it is logged and the call goes on as without it.

    A job forgotten as dead is resubmitted only because the caller asked
    again, so a step's retry after ``JobFailed``, which uses a new attempt key,
    submits a fresh job.

    Identical submits for one user are serialised by a lock per request
    shared by every instance in the process, so two concurrent ones submit
    once even from different runs. Only two processes racing on the same
    request can both submit; the later save wins.
    """

    def __init__(
        self,
        inner: EarthdataSession,
        store: RetrievalJobStore,
        user_id: str,
        clock: Clock,
        *,
        max_age: datetime.timedelta = datetime.timedelta(hours=24),
    ) -> None:
        self._inner = inner
        self._store = store
        self._user_id = user_id
        self._clock = clock
        self._max_age = max_age

    async def call(
        self, tool: str, arguments: dict[str, Any], *, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        if tool not in SUBMIT_TOOLS:
            return await self._inner.call(tool, arguments, idempotency_key=idempotency_key)
        fingerprint = request_fingerprint(tool, arguments)
        async with self._held(fingerprint):
            saved = await self._find(fingerprint)
            forgotten: str | None = None
            if saved is not None:
                if await self._reusable(saved):
                    return saved.answer
                forgotten = saved.job_handle
                await self._forget(fingerprint)
            answer = await self._inner.call(tool, arguments, idempotency_key=idempotency_key)
            job_handle = answer.get("job_handle")
            if isinstance(job_handle, str) and job_handle and job_handle != forgotten:
                await self._save(SavedJob(
                    fingerprint=fingerprint, tool=tool, job_handle=job_handle,
                    answer=answer, submitted=self._clock.now(),
                ))
            return answer

    @asynccontextmanager
    async def _held(self, fingerprint: str) -> AsyncIterator[None]:
        key = (self._user_id, fingerprint)
        lock, users = _LOCKS.get(key) or (asyncio.Lock(), 0)
        _LOCKS[key] = (lock, users + 1)
        try:
            async with lock:
                yield
        finally:
            lock, users = _LOCKS[key]
            if users == 1:
                del _LOCKS[key]
            else:
                _LOCKS[key] = (lock, users - 1)

    async def _reusable(self, saved: SavedJob) -> bool:
        if self._clock.now() - saved.submitted >= self._max_age:
            return False
        try:
            status = await self._inner.call(STATUS_TOOL, {"job_handle": saved.job_handle})
        except Exception as exc:
            category = _category(exc)
            if category == CATEGORY_PROVIDER_UNAVAILABLE:
                return True
            if category == CATEGORY_NOT_FOUND:
                return False
            raise
        return _classify(status) != _DEAD

    async def _find(self, fingerprint: str) -> SavedJob | None:
        try:
            return await self._store.find(self._user_id, fingerprint)
        except Exception as exc:
            logger.warning("could not read a saved retrieval job: %s", type(exc).__name__)
            return None

    async def _forget(self, fingerprint: str) -> None:
        try:
            await self._store.forget(self._user_id, fingerprint)
        except Exception as exc:
            logger.warning("could not forget a saved retrieval job: %s", type(exc).__name__)

    async def _save(self, job: SavedJob) -> None:
        try:
            # The answer is a small reply; one too large or not JSON is not kept.
            serialize_json(job.answer, "answer")
            await self._store.save(self._user_id, job)
        except InvalidTrace:
            logger.warning("a submit answer could not be saved for reuse")
        except Exception as exc:
            logger.warning("could not save a retrieval job: %s", type(exc).__name__)


JobState = Literal["ready", "running", "failed", "unknown"]


def _key(job: Any) -> str:
    return json.dumps(job, sort_keys=True, separators=(",", ":"), default=str)


def _begun_job(entry: TraceEntry) -> dict[str, Any] | None:
    """The job reference a ``begin`` entry saved, or None for any other entry."""
    if entry.kind != "use" or entry.status != "waiting" or JOB_PARAM in entry.params:
        return None
    result = entry.result
    if isinstance(result, dict) and set(result) == {"job"} and isinstance(result["job"], dict):
        return result["job"]
    return None


def _still_open(entry: TraceEntry) -> bool:
    """Whether a ``complete`` entry left its job to be collected later: it
    stopped waiting while the job ran, or failed before learning the job's
    fate. A completed, dead, paused or cancelled job is settled."""
    if entry.status != "failed":
        return False
    result = entry.result
    return result is None or (isinstance(result, dict) and "still_running" in result)


def _job_ids(job: dict[str, Any]) -> list[str]:
    listed = job.get("jobs")
    ids: list[str] = []
    for item in listed if isinstance(listed, list) else ():
        job_id = item.get("job") if isinstance(item, dict) else None
        if isinstance(job_id, str) and job_id:
            ids.append(job_id)
    return ids


def begun_jobs(entries: Iterable[TraceEntry]) -> list[str]:
    """Every provider job id the entries' ``begin`` steps submitted or reused, settled or not, in first-seen order."""
    ids = [job_id for entry in entries if (job := _begun_job(entry)) is not None for job_id in _job_ids(job)]
    return list(dict.fromkeys(ids))


def waiting_jobs(entries: Iterable[TraceEntry]) -> list[str]:
    """The provider job ids a run's entries are still waiting on, in first-seen order.

    A ``begin`` entry opens its job reference; a later ``complete`` entry for
    the same reference (its params holding it under ``_job``) settles it,
    unless it only stopped waiting.
    """
    open_jobs: dict[str, dict[str, Any]] = {}
    for entry in sorted(entries, key=lambda e: e.seq):
        begun = _begun_job(entry)
        if begun is not None:
            open_jobs[_key(begun)] = begun
            continue
        if entry.kind == "use" and isinstance(entry.params.get(JOB_PARAM), dict):
            key = _key(entry.params[JOB_PARAM])
            if key in open_jobs and not _still_open(entry):
                del open_jobs[key]
    ids = [job_id for job in open_jobs.values() for job_id in _job_ids(job)]
    return list(dict.fromkeys(ids))


async def _job_state(earthdata: EarthdataSession, job_id: str, gate: asyncio.Semaphore) -> JobState:
    async with gate:
        try:
            status = await earthdata.call(STATUS_TOOL, {"job_handle": job_id})
        except Exception as exc:
            category = _category(exc)
            if category is None:
                raise
            return "failed" if category == CATEGORY_NOT_FOUND else "unknown"
    state = _classify(status)
    return "failed" if state == _DEAD else "ready" if state == "ready" else "running"


def _run_state(states: list[JobState]) -> JobState:
    if not states:
        return "unknown"
    if "failed" in states:
        return "failed"
    if "unknown" in states:
        return "unknown"
    return "ready" if all(state == "ready" for state in states) else "running"


async def check_waiting(
    context: RunContext, earthdata: EarthdataSession, *, max_concurrent: int = 4
) -> dict[str, JobState]:
    """Each of the conversation's ``waiting`` runs mapped to the state of the jobs it waits on.

    Every job is asked about once with ``get_retrieval_status``, at most
    ``max_concurrent`` at a time, and nothing waits or sleeps. A run is
    ``ready`` when all its jobs are, ``failed`` when any failed, was
    cancelled or paused, or is no longer held, ``unknown`` when a status
    call failed with a classified error or no job was found, and
    ``running`` otherwise. An unclassified error is raised.
    """
    if max_concurrent < 1:
        raise ValueError("max_concurrent is at least 1")
    identity = context.identity
    traces = context.storage.traces
    waiting: dict[str, list[str]] = {}
    for run in await traces.list_runs(identity.user_id, identity.conversation_id):
        if run.status != "waiting":
            continue
        try:
            entries = await traces.entries(identity.user_id, run.id)
        except RunNotFound:
            continue
        waiting[run.id] = waiting_jobs(entries)
    unique = list(dict.fromkeys(job_id for ids in waiting.values() for job_id in ids))
    gate = asyncio.Semaphore(max_concurrent)
    try:
        async with asyncio.TaskGroup() as group:
            tasks = {job_id: group.create_task(_job_state(earthdata, job_id, gate)) for job_id in unique}
    except ExceptionGroup as failed:
        # The first unclassified error, as a single status call would raise it.
        raise failed.exceptions[0] from None
    states = {job_id: task.result() for job_id, task in tasks.items()}
    return {run_id: _run_state([states[job_id] for job_id in ids]) for run_id, ids in waiting.items()}
