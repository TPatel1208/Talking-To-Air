from __future__ import annotations

from typing import Any

from repositories.chart_repository import delete_charts_for_session
from repositories.session_metadata_repository import delete_session_metadata, list_session_metadata
from utils.db import pg_connection


class SessionRepository:
    async def list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        return await list_session_metadata(user_id)

    async def delete_session(self, thread_id: str, user_id: str) -> bool:
        deleted = await delete_session_metadata(thread_id, user_id)
        if not deleted:
            return False
        await delete_charts_for_session(thread_id, user_id)
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
        return True
