"""Durable handle records, isolated per user.

Every read takes the requesting user. A handle owned by someone else raises
exactly what a handle that does not exist raises.
"""
from __future__ import annotations

import datetime
from typing import Any, Protocol

from psycopg.types.json import Jsonb

from tta_backend.harness.handles.errors import HandleError
from tta_backend.harness.handles.model import STORED, EvictedPayload, HandleRecord, StoredPayload, is_handle_id
from tta_backend.harness.database import Connect


class HandleRegistry(Protocol):
    async def mint(self, record: HandleRecord) -> HandleRecord:
        """Store a new derived handle; its parents must exist for the same user."""
        ...

    async def record_retrieval(self, record: HandleRecord) -> HandleRecord:
        """Record a retrieval handle's descriptor; recording it again returns the first."""
        ...

    async def get(self, user_id: str, handle_id: str) -> HandleRecord:
        """The user's handle, or ``not_found``."""
        ...

    async def mark_evicted(self, user_id: str, handle_id: str, at: datetime.datetime) -> HandleRecord:
        """Mark a file-backed derived payload as evicted; the record stays."""
        ...


def _check_mint(record: HandleRecord) -> None:
    if record.origin != "derived":
        raise ValueError("mint takes derived handles; use record_retrieval")
    if not isinstance(record.payload, StoredPayload) and record.kind != "scalar":
        raise ValueError("a derived handle is minted with its payload stored")


def _check_retrieval(record: HandleRecord) -> None:
    if record.origin != "retrieval":
        raise ValueError("record_retrieval takes retrieval handles; use mint")


def _evict(record: HandleRecord, at: datetime.datetime) -> HandleRecord:
    if not isinstance(record.payload, StoredPayload):
        if record.payload.state == "evicted":
            return record
        raise HandleError("not_evictable", record.id)
    # Only the payload changes, to a validated one of the same format; the rest
    # is the stored record, which is not rechecked against today's rules.
    return record.model_copy(update={"payload": EvictedPayload(format=record.payload.format, evicted_at=at)})


class InMemoryHandleRegistry:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], HandleRecord] = {}

    async def mint(self, record: HandleRecord) -> HandleRecord:
        _check_mint(record)
        key = (record.user_id, record.id)
        if key in self._records:
            raise HandleError("already_exists", record.id)
        for parent in record.parents:
            if (record.user_id, parent) not in self._records:
                raise HandleError("parent_not_found", parent)
        self._records[key] = record.model_copy(deep=True)
        return record

    async def record_retrieval(self, record: HandleRecord) -> HandleRecord:
        _check_retrieval(record)
        existing = self._records.setdefault((record.user_id, record.id), record.model_copy(deep=True))
        return existing.model_copy(deep=True)

    async def get(self, user_id: str, handle_id: str) -> HandleRecord:
        record = self._records.get((user_id, handle_id))
        if record is None:
            raise HandleError("not_found", handle_id)
        # A copy, so a caller mutating a parameters dict cannot change the stored record.
        return record.model_copy(deep=True)

    async def mark_evicted(self, user_id: str, handle_id: str, at: datetime.datetime) -> HandleRecord:
        record = _evict(await self.get(user_id, handle_id), at)
        self._records[(user_id, handle_id)] = record
        return record.model_copy(deep=True)


async def ensure_tables(connect: Connect) -> None:
    async with connect() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS harness_handles (
                user_id TEXT NOT NULL,
                id TEXT NOT NULL,
                kind TEXT NOT NULL,
                origin TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                record JSONB NOT NULL,
                PRIMARY KEY (user_id, id)
            )
            """
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_harness_handles_conversation ON harness_handles (user_id, conversation_id, created_at)"
        )


def _dump(record: HandleRecord) -> Jsonb:
    # The model has already replaced non-finite floats, which JSONB rejects.
    return Jsonb(record.model_dump(mode="json"))


def _load(value: Any) -> HandleRecord:
    # Parsed for shape only, so a row valid when written still loads after a
    # check is tightened.
    return HandleRecord.model_validate(value, context=STORED)


class PostgresHandleRegistry:
    def __init__(self, connect: Connect) -> None:
        self._connect = connect

    async def mint(self, record: HandleRecord) -> HandleRecord:
        _check_mint(record)
        async with self._connect() as conn:
            if record.parents:
                cur = await conn.execute(
                    "SELECT id FROM harness_handles WHERE user_id = %s AND id = ANY(%s)",
                    (record.user_id, list(record.parents)),
                )
                found = {row[0] for row in await cur.fetchall()}
                for parent in record.parents:
                    if parent not in found:
                        raise HandleError("parent_not_found", parent)
            cur = await conn.execute(
                """
                INSERT INTO harness_handles (user_id, id, kind, origin, conversation_id, created_at, record)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, id) DO NOTHING
                RETURNING id
                """,
                (record.user_id, record.id, record.kind, record.origin, record.conversation_id, record.created_at, _dump(record)),
            )
            if await cur.fetchone() is None:
                raise HandleError("already_exists", record.id)
        return record

    async def record_retrieval(self, record: HandleRecord) -> HandleRecord:
        _check_retrieval(record)
        async with self._connect() as conn:
            await conn.execute(
                """
                INSERT INTO harness_handles (user_id, id, kind, origin, conversation_id, created_at, record)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, id) DO NOTHING
                """,
                (record.user_id, record.id, record.kind, record.origin, record.conversation_id, record.created_at, _dump(record)),
            )
            cur = await conn.execute(
                "SELECT record FROM harness_handles WHERE user_id = %s AND id = %s",
                (record.user_id, record.id),
            )
            row = await cur.fetchone()
        if row is None:
            raise HandleError("not_found", record.id)
        return _load(row[0])

    async def get(self, user_id: str, handle_id: str) -> HandleRecord:
        if not is_handle_id(handle_id):
            raise HandleError("not_found", handle_id)
        async with self._connect() as conn:
            cur = await conn.execute(
                "SELECT record FROM harness_handles WHERE user_id = %s AND id = %s",
                (user_id, handle_id),
            )
            row = await cur.fetchone()
        if row is None:
            raise HandleError("not_found", handle_id)
        return _load(row[0])

    async def mark_evicted(self, user_id: str, handle_id: str, at: datetime.datetime) -> HandleRecord:
        if not is_handle_id(handle_id):
            raise HandleError("not_found", handle_id)
        async with self._connect() as conn:
            cur = await conn.execute(
                "SELECT record FROM harness_handles WHERE user_id = %s AND id = %s FOR UPDATE",
                (user_id, handle_id),
            )
            row = await cur.fetchone()
            if row is None:
                raise HandleError("not_found", handle_id)
            record = _evict(_load(row[0]), at)
            await conn.execute(
                "UPDATE harness_handles SET record = %s WHERE user_id = %s AND id = %s",
                (_dump(record), user_id, handle_id),
            )
        return record
