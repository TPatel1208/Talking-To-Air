from __future__ import annotations

import base64
import json
import re
from datetime import datetime
from typing import Any

from tta_backend.utils.db import pg_connection

MAX_TITLE_LENGTH = 60
SESSION_PAGE_SIZE = 50

# When a thread was last used: the listing's sort key and page cursor. The
# index that serves the listing is built on this exact expression, so the two
# must not drift apart.
_RECENCY = "COALESCE(last_event_at, first_frame_at, created_at)"


def generate_session_title(message: str) -> str:
    title = re.sub(r"\s+", " ", message or "").strip()
    if not title:
        return "Untitled session"
    if len(title) <= MAX_TITLE_LENGTH:
        return title
    return title[: MAX_TITLE_LENGTH - 3].rstrip() + "..."


def _serialize_created_at(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value is not None else None


async def ensure_session_metadata_table() -> None:
    async with pg_connection() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_metadata (
                thread_id TEXT PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                user_id TEXT NOT NULL DEFAULT '__legacy__'
            )
            """
        )
        await conn.execute(
            """
            ALTER TABLE session_metadata
            ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT '__legacy__'
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_session_metadata_user_id
            ON session_metadata(user_id)
            """
        )
        await conn.execute(
            """
            ALTER TABLE session_metadata
            ADD COLUMN IF NOT EXISTS ground_monitor_context JSONB NOT NULL DEFAULT '{}'::jsonb
            """
        )
        await conn.execute(
            """
            ALTER TABLE session_metadata
            ADD COLUMN IF NOT EXISTS satellite_context JSONB NOT NULL DEFAULT '{}'::jsonb
            """
        )
        # Asked before the column is added, because afterwards the answer is
        # always yes. A NULL in an existing row and a NULL in a new one mean
        # opposite things -- "predates this column" and "this thread's turn
        # has not produced yet" -- and only the first may be backfilled. This
        # runs on every startup, so backfilling unconditionally would relist
        # every empty thread the day after it was hidden.
        cursor = await conn.execute(
            """
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'session_metadata' AND column_name = 'first_frame_at'
            """
        )
        column_existed = await cursor.fetchone() is not None
        await conn.execute(
            """
            ALTER TABLE session_metadata
            ADD COLUMN IF NOT EXISTS first_frame_at TIMESTAMPTZ
            """
        )
        if not column_existed:
            await conn.execute(
                """
                UPDATE session_metadata
                SET first_frame_at = created_at
                WHERE first_frame_at IS NULL
                """
            )
        # Deliberately added without a backfill, unlike the column above. A
        # NULL here is answered at read time by the COALESCE in
        # list_session_metadata, so an existing thread sorts by the newest
        # fact it does have. Writing one on startup would need the same
        # probe-then-fill dance, and would reset the stamp of every thread
        # whose turn is running across a restart.
        await conn.execute(
            """
            ALTER TABLE session_metadata
            ADD COLUMN IF NOT EXISTS last_event_at TIMESTAMPTZ
            """
        )
        # Serves list_session_metadata's ORDER BY row for row, so a page is a
        # LIMIT read off the index instead of a sort of the user's history.
        await conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_session_metadata_user_recency
            ON session_metadata (user_id, {_RECENCY} DESC, thread_id DESC)
            """
        )
        await conn.commit()


async def save_session_metadata_once(thread_id: str, first_message: str, user_id: str) -> None:
    async with pg_connection() as conn:
        await conn.execute(
            """
            INSERT INTO session_metadata (thread_id, title, created_at, user_id)
            VALUES (%s, %s, now(), %s)
            ON CONFLICT (thread_id) DO NOTHING
            """,
            (thread_id, generate_session_title(first_message), user_id),
        )
        await conn.commit()


async def mark_session_activity(thread_id: str) -> None:
    """Record that a turn on this thread has produced its first frame.

    Two stamps, one event, one write. They answer different questions and
    must not be collapsed into a single column:

    - ``first_frame_at`` means "this conversation began" and decides whether
      the thread is listed at all. The COALESCE keeps it write-once: every
      turn's first frame calls this, and a later one must not move it.
    - ``last_event_at`` means "this conversation was last used" and decides
      where in the list it sits. It moves on every turn, which is the whole
      point -- a thread answered an hour ago belongs above one answered last
      week, however long ago either was started.

    Called once per turn, not once per frame: the caller stops asking after
    the first one.
    """
    async with pg_connection() as conn:
        await conn.execute(
            """
            UPDATE session_metadata
            SET first_frame_at = COALESCE(first_frame_at, now()),
                last_event_at = now()
            WHERE thread_id = %s
            """,
            (thread_id,),
        )
        await conn.commit()


async def get_session_metadata(thread_id: str) -> dict[str, Any] | None:
    async with pg_connection() as conn:
        cursor = await conn.execute(
            """
            SELECT thread_id, title, created_at, user_id
            FROM session_metadata
            WHERE thread_id = %s
            """,
            (thread_id,),
        )
        row = await cursor.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "title": row[1],
        "created_at": _serialize_created_at(row[2]),
        "user_id": row[3],
    }


async def session_belongs_to_user(thread_id: str, user_id: str) -> bool:
    metadata = await get_session_metadata(thread_id)
    return metadata is not None and metadata["user_id"] == user_id


class InvalidSessionCursor(ValueError):
    """A page cursor this module did not issue."""


def _encode_session_cursor(sort_at: datetime, thread_id: str) -> str:
    payload = json.dumps({"at": sort_at.isoformat(), "id": thread_id})
    return base64.urlsafe_b64encode(payload.encode()).decode()


def _decode_session_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        sort_at, thread_id = datetime.fromisoformat(payload["at"]), payload["id"]
    except (ValueError, TypeError, KeyError) as exc:
        raise InvalidSessionCursor(cursor) from exc
    if not isinstance(thread_id, str) or sort_at.tzinfo is None:
        raise InvalidSessionCursor(cursor)
    return sort_at, thread_id


async def list_session_metadata(
    user_id: str, *, limit: int = SESSION_PAGE_SIZE, cursor: str | None = None
) -> dict[str, Any]:
    """One page of this user's threads that have something in them.

    A row exists from the moment its message is posted -- it is the only
    record of who owns the thread, and the stream and stop endpoints refuse
    without it -- but that says nothing about whether the conversation has
    any content. Two facts answer that, and neither covers both routes:

    - ``first_frame_at``: the turn has narrated. The fast path checkpoints
      its transcript once, at the end, so this is the only evidence a turn
      still running leaves.
    - a checkpoint row: the thread has a transcript. The supervisor route
      checkpoints the human message at its first superstep, before the first
      frame is yielded -- so a turn stopped in that window holds the user's
      question and has narrated nothing.

    Either one lists the thread; neither leaves it out of the sidebar, still
    reachable by id. The checkpoint tables are LangGraph's own, named here
    for the same reason ``SessionRepository.delete_session`` names them.

    Ordered by when the thread was last *used*, not when it was created: a
    thread you returned to this morning sorts above one you started
    yesterday and abandoned. The COALESCE is what lets the column go in
    without a backfill, and it is not only for old rows -- a thread listed
    by the checkpoint branch above was stopped before it ever narrated, so
    it has neither stamp and sorts by ``created_at``, which is the only
    thing that ever happened to it.

    Paged by key, not offset: ``next_cursor`` is the recency and id of the
    last row returned. ``last_event_at`` moves on every turn, so an offset
    would repeat or skip a row whenever a thread was used between fetches.
    One row past ``limit`` is read to learn whether another page exists.
    """
    after: tuple[Any, ...] = ()
    resume = ""
    if cursor is not None:
        after = _decode_session_cursor(cursor)
        resume = f"AND ({_RECENCY}, thread_id) < (%s, %s)"
    async with pg_connection() as conn:
        result = await conn.execute(
            f"""
            SELECT thread_id, title, created_at, {_RECENCY}
            FROM session_metadata
            WHERE user_id = %s
              AND (
                first_frame_at IS NOT NULL
                OR EXISTS (
                    SELECT 1 FROM checkpoints
                    WHERE checkpoints.thread_id = session_metadata.thread_id
                )
              )
              {resume}
            ORDER BY {_RECENCY} DESC, thread_id DESC
            LIMIT %s
            """,
            (user_id, *after, limit + 1),
        )
        rows = await result.fetchall()

    page, more = rows[:limit], len(rows) > limit
    return {
        "sessions": [
            {
                "id": row[0],
                "title": row[1],
                "created_at": _serialize_created_at(row[2]),
            }
            for row in page
        ],
        "next_cursor": _encode_session_cursor(page[-1][3], page[-1][0]) if more else None,
    }


async def get_ground_monitor_context(thread_id: str) -> dict[str, str]:
    """The ground path's cross-turn monitor context (last monitor discussed
    on this thread) — per-thread, not process-wide, so concurrent
    conversations never bleed into each other (T14)."""
    async with pg_connection() as conn:
        cursor = await conn.execute(
            "SELECT ground_monitor_context FROM session_metadata WHERE thread_id = %s",
            (thread_id,),
        )
        row = await cursor.fetchone()
    if not row or not row[0]:
        return {}
    return dict(row[0])


async def save_ground_monitor_context(thread_id: str, context: dict[str, str]) -> None:
    """Best-effort — a thread with no session_metadata row yet (should not
    happen on the normal chat flow, which always saves metadata first)
    simply does not persist the context."""
    async with pg_connection() as conn:
        await conn.execute(
            "UPDATE session_metadata SET ground_monitor_context = %s WHERE thread_id = %s",
            (json.dumps(context), thread_id),
        )
        await conn.commit()


async def get_satellite_context(thread_id: str) -> dict[str, str]:
    """The satellite path's cross-turn retrieval context for this thread — the
    dataset/AOI last worked with and the handles minted for them. Per-thread,
    so concurrent conversations never bleed into each other (mirrors
    ``get_ground_monitor_context``). Injected into a follow-up earthdata task
    so a continuation ("pick a date in that range") arrives with the concrete
    dataset/location/handles instead of only the prose answer the fast path
    wrote back — it is never treated as an availability verdict (the earthdata
    agent re-checks coverage; see its Availability-must-be-tool-grounded rule)."""
    async with pg_connection() as conn:
        cursor = await conn.execute(
            "SELECT satellite_context FROM session_metadata WHERE thread_id = %s",
            (thread_id,),
        )
        row = await cursor.fetchone()
    if not row or not row[0]:
        return {}
    return dict(row[0])


async def save_satellite_context(thread_id: str, context: dict[str, str]) -> None:
    """Best-effort — a thread with no session_metadata row yet (should not
    happen on the normal chat flow, which always saves metadata first)
    simply does not persist the context."""
    async with pg_connection() as conn:
        await conn.execute(
            "UPDATE session_metadata SET satellite_context = %s WHERE thread_id = %s",
            (json.dumps(context), thread_id),
        )
        await conn.commit()


async def delete_session_metadata(thread_id: str, user_id: str) -> bool:
    async with pg_connection() as conn:
        cursor = await conn.execute(
            "DELETE FROM session_metadata WHERE thread_id = %s AND user_id = %s",
            (thread_id, user_id),
        )
        await conn.commit()
    return cursor.rowcount > 0
