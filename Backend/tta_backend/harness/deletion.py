"""Deleting what the harness kept for one of a user's conversations.

``delete_conversation`` removes the conversation's runs with their trace
entries, their saved results, the handles recorded in it with their payload
files, and the saved retrieval jobs only its runs used. A job a run in
another of the user's conversations also used is kept for that conversation
to reuse.

The runs are deleted last: they are how everything else is found, so a
delete that fails part way can be run again and finishes the rest. A payload
file that cannot be deleted is logged and left; its record is already gone,
so nothing reads it again.
"""
from __future__ import annotations

import logging

from tta_backend.harness.handles.model import StoredPayload
from tta_backend.harness.handles.payloads import PayloadStore
from tta_backend.harness.jobs import begun_jobs
from tta_backend.harness.storage import Stores

logger = logging.getLogger(__name__)


async def delete_conversation(
    stores: Stores, payloads: PayloadStore | None, user_id: str, conversation_id: str,
) -> None:
    """Delete the user's harness records for ``conversation_id``; another user's are never touched."""
    runs = await stores.traces.list_runs(user_id, conversation_id)
    used_here: list[str] = []
    for run in runs:
        used_here.extend(begun_jobs(await stores.traces.entries(user_id, run.id)))
    used_elsewhere = set(begun_jobs(await stores.traces.waiting_uses_elsewhere(user_id, conversation_id)))
    await stores.jobs.forget_jobs(user_id, [job for job in dict.fromkeys(used_here) if job not in used_elsewhere])
    await stores.results.delete_runs(user_id, [run.id for run in runs])
    for record in await stores.handles.delete_conversation(user_id, conversation_id):
        if payloads is None or not isinstance(record.payload, StoredPayload):
            continue
        try:
            await payloads.delete(user_id, record.id, record.payload.format)
        except Exception as exc:
            logger.warning("a deleted handle's payload could not be removed: %s", type(exc).__name__)
    await stores.traces.delete_conversation(user_id, conversation_id)


__all__ = ["delete_conversation"]
