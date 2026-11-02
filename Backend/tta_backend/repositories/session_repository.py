from __future__ import annotations

import asyncio
import logging
from typing import Any

from tta_backend.repositories.artifact_repository import delete_artifacts_for_session
from tta_backend.repositories.chart_repository import delete_charts_for_session
from tta_backend.repositories.session_metadata_repository import (
    delete_session_metadata,
    list_session_metadata,
    session_belongs_to_user,
)
from tta_backend.services import overlay_store
from tta_backend.utils.db import pg_connection

logger = logging.getLogger(__name__)


class SessionRepository:
    async def list_sessions(
        self, user_id: str, *, limit: int, cursor: str | None = None
    ) -> dict[str, Any]:
        return await list_session_metadata(user_id, limit=limit, cursor=cursor)

    async def delete_session(self, thread_id: str, user_id: str) -> bool:
        # Ownership is established by a read, and the row that proves it is
        # destroyed last. These deletes share no transaction and each commits
        # on its own, so deleting the metadata row first meant any later
        # failure left the thread's charts, artifacts and checkpoints alive
        # with nothing left to authorise removing them -- and the retry
        # reporting "not found" as though the delete had succeeded.
        if not await session_belongs_to_user(thread_id, user_id):
            return False
        # Checkpoints before the charts and artifacts they name, so a partial
        # failure never leaves a loadable thread pointing at deleted rows.
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
        deleted_charts = await delete_charts_for_session(thread_id, user_id)
        # The overlay PNGs after the rows that name them, so a surviving chart
        # never loses its overlay. Best effort: a PNG left behind here is an
        # orphan the store's LRU cap reclaims, not a reason to stop before the
        # ownership row.
        overlay_paths = [p for chart in deleted_charts for p in overlay_store.chart_overlay_paths(chart)]
        if overlay_paths:
            try:
                await asyncio.to_thread(overlay_store.delete_overlays, overlay_paths)
            except Exception:
                logger.warning("session_overlay_delete_failed", extra={"_thread_id": thread_id}, exc_info=True)
        await delete_artifacts_for_session(thread_id, user_id)
        # Last: once this row is gone the thread is unauthorisable, so
        # nothing that still needs authorising may follow it.
        await delete_session_metadata(thread_id, user_id)
        return True
