from __future__ import annotations

from typing import Any

from tta_backend.repositories.artifact_repository import delete_artifacts_for_session
from tta_backend.repositories.chart_repository import delete_charts_for_session
from tta_backend.repositories.session_metadata_repository import (
    delete_session_metadata,
    list_session_metadata,
    session_belongs_to_user,
)
from tta_backend.utils.db import pg_connection


class SessionRepository:
    async def list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        return await list_session_metadata(user_id)

    async def delete_session(self, thread_id: str, user_id: str) -> bool:
        # Ownership is established by a read, and the row that proves it is
        # destroyed last. These deletes share no transaction and each commits
        # on its own, so deleting the metadata row first meant any later
        # failure left the thread's charts, artifacts and checkpoints alive
        # with nothing left to authorise removing them -- and the retry
        # reporting "not found" as though the delete had succeeded.
        if not await session_belongs_to_user(thread_id, user_id):
            return False
        await delete_charts_for_session(thread_id, user_id)
        await delete_artifacts_for_session(thread_id, user_id)
        # LangGraph does not currently expose a session-delete helper here.
        # These table names are internal to LangGraph's Postgres checkpointer
        # and should be revisited when upgrading LangGraph.
        async with pg_connection() as conn:
            for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                await conn.execute(
                    f"DELETE FROM {table} WHERE thread_id = %s",
                    (thread_id,),
                )
            await conn.commit()
        # Last: once this row is gone the thread is unauthorisable, so
        # nothing that still needs authorising may follow it.
        await delete_session_metadata(thread_id, user_id)
        return True
