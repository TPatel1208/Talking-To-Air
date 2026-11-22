"""The runs a conversation has paused, for the app's runs panel: listing them and cancelling one.

Both work outside a turn and call no model. Listing first does what a turn
does at its start: cancels the runs paused past their expiry and checks, once
and without waiting, the jobs the waiting runs are on, so a conversation
opened after its retrievals finished shows them ready.
"""
from __future__ import annotations

import datetime
from typing import Any, Protocol, cast

from tta_backend.harness.jobs import JobState
from tta_backend.harness.results import Result
from tta_backend.harness.run_context import EarthdataSession, RunContext
from tta_backend.harness.trace import Run, RunClosed, RunNotFound
from tta_backend.harness.workflows.runner import EnvFactory, RunNotResumable, paused_since
from tta_backend.services.harness_chat import HarnessChat, question_payload
from tta_backend.utils.streaming import user_id_context

PAUSED = ("needs_input", "waiting")


class RunNotPaused(Exception):
    """The run is not one of this conversation's paused runs."""


class RunsUnavailable(Exception):
    """The chat host's supervisor does not list or cancel paused runs."""


class _PausedRunSupervisor(Protocol):
    async def expire(self, context: RunContext, build_env: EnvFactory) -> None: ...

    async def job_states(self, context: RunContext, earthdata: EarthdataSession) -> dict[str, JobState] | None: ...

    async def cancel_paused_run(self, context: RunContext, build_env: EnvFactory, run_id: str) -> Result: ...


def _supervisor(chat: HarnessChat) -> _PausedRunSupervisor:
    if not all(hasattr(chat.supervisor, name) for name in ("expire", "job_states", "cancel_paused_run")):
        raise RunsUnavailable()
    return cast(_PausedRunSupervisor, chat.supervisor)


def _iso(moment: datetime.datetime) -> str:
    return moment.astimezone(datetime.UTC).isoformat()


def run_row(run: Run, saved: Result | None, state: JobState | None) -> dict[str, Any]:
    """One paused run as the app lists it. The question is the one the run waits on, in the
    shape of the ``done`` frame's; ``ready`` says whether a waiting run's jobs are finished."""
    # The result is read only when it says what the trace says.
    current = saved if saved is not None and saved.status == run.status else None
    workflow = run.mode.workflow is not None
    row: dict[str, Any] = {
        "run_id": run.id,
        "mode": run.mode.view(),
        "status": run.status,
        "started": _iso(run.started),
        "paused": _iso(paused_since(run)),
        "reason": current.reason if current is not None else None,
    }
    if run.status == "needs_input":
        row["question"] = question_payload(current.question) if current is not None and workflow else None
    else:
        # An exploration cannot be continued.
        row["ready"] = workflow and state == "ready"
    return row


async def list_paused_runs(chat: HarnessChat, thread_id: str, user_id: str) -> list[dict[str, Any]]:
    """The user's ``needs_input`` and ``waiting`` runs in the conversation, oldest first."""
    supervisor = _supervisor(chat)
    context, ports = chat.quiet_context(thread_id, user_id)
    with user_id_context(user_id):
        await supervisor.expire(context, ports.build_env)
        states = await supervisor.job_states(context, ports.earthdata)
    traces, results = chat.stores.traces, chat.stores.results
    paused = [run for run in await traces.list_runs(user_id, thread_id) if run.status in PAUSED]
    saved = await results.for_runs(user_id, [run.id for run in paused])
    return [run_row(run, saved.get(run.id), (states or {}).get(run.id)) for run in paused]


async def cancel_paused_run(chat: HarnessChat, user_id: str, run_id: str) -> dict[str, Any]:
    """Cancel one of the user's paused runs, in the conversation it belongs to, and say what it is now.
    Provider jobs are not cancelled. Raises RunNotPaused for a run that is not the user's or not paused."""
    supervisor = _supervisor(chat)
    try:
        run = await chat.stores.traces.get_run(user_id, run_id)
        context, ports = chat.quiet_context(run.conversation_id, user_id)
        with user_id_context(user_id):
            result = await supervisor.cancel_paused_run(context, ports.build_env, run_id)
    except (RunNotFound, RunNotResumable, RunClosed):
        raise RunNotPaused() from None
    return {"run_id": run_id, "status": result.status, "reason": result.reason}
