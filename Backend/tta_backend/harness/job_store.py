"""Retrieval jobs saved across runs, keyed by user and request fingerprint."""
from __future__ import annotations

import datetime
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from psycopg.types.json import Jsonb

from tta_backend.harness.database import Connect


@dataclass(frozen=True)
class SavedJob:
    """One submitted retrieval: the request's fingerprint, the submit tool, the
    job it started and the service's submit answer, which holds ``job_handle``."""

    fingerprint: str
    tool: str
    job_handle: str
    answer: dict[str, Any]
    submitted: datetime.datetime

    def __post_init__(self) -> None:
        if self.submitted.tzinfo is None:
            raise ValueError("submitted must be timezone-aware")
        if not self.job_handle:
            raise ValueError("a saved job needs its job handle")


class RetrievalJobStore(Protocol):
    """Saved jobs keyed by user and request fingerprint; another user's job
    behaves exactly as a missing one."""

    async def find(self, user_id: str, fingerprint: str) -> SavedJob | None: ...

    async def save(self, user_id: str, job: SavedJob) -> None:
        """Insert, or replace the job saved for the same fingerprint."""
        ...

    async def forget(self, user_id: str, fingerprint: str) -> None:
        """Remove the job; forgetting one that is not saved does nothing."""
        ...

    async def forget_jobs(self, user_id: str, job_handles: Sequence[str]) -> None:
        """Remove every saved job of the user's whose job handle is in ``job_handles``."""
        ...


class InMemoryRetrievalJobStore:
    def __init__(self) -> None:
        self._jobs: dict[tuple[str, str], SavedJob] = {}

    async def find(self, user_id: str, fingerprint: str) -> SavedJob | None:
        job = self._jobs.get((user_id, fingerprint))
        # A copy of the answer, so a caller changing it cannot change the saved job.
        return None if job is None else replace(job, answer=json.loads(json.dumps(job.answer)))

    async def save(self, user_id: str, job: SavedJob) -> None:
        self._jobs[(user_id, job.fingerprint)] = replace(job, answer=json.loads(json.dumps(job.answer)))

    async def forget(self, user_id: str, fingerprint: str) -> None:
        self._jobs.pop((user_id, fingerprint), None)

    async def forget_jobs(self, user_id: str, job_handles: Sequence[str]) -> None:
        doomed = set(job_handles)
        for key in [k for k, job in self._jobs.items() if k[0] == user_id and job.job_handle in doomed]:
            del self._jobs[key]


async def ensure_tables(connect: Connect) -> None:
    async with connect() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS harness_retrieval_jobs (
                user_id TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                tool TEXT NOT NULL,
                job_handle TEXT NOT NULL,
                answer JSONB NOT NULL,
                submitted TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (user_id, fingerprint)
            )
            """
        )


class PostgresRetrievalJobStore:
    def __init__(self, connect: Connect) -> None:
        self._connect = connect

    async def find(self, user_id: str, fingerprint: str) -> SavedJob | None:
        async with self._connect() as conn:
            cur = await conn.execute(
                "SELECT fingerprint, tool, job_handle, answer, submitted FROM harness_retrieval_jobs "
                "WHERE user_id = %s AND fingerprint = %s",
                (user_id, fingerprint),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return SavedJob(fingerprint=row[0], tool=row[1], job_handle=row[2], answer=row[3], submitted=row[4])

    async def save(self, user_id: str, job: SavedJob) -> None:
        async with self._connect() as conn:
            await conn.execute(
                """
                INSERT INTO harness_retrieval_jobs (user_id, fingerprint, tool, job_handle, answer, submitted)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, fingerprint) DO UPDATE SET
                    tool = EXCLUDED.tool, job_handle = EXCLUDED.job_handle,
                    answer = EXCLUDED.answer, submitted = EXCLUDED.submitted
                """,
                (user_id, job.fingerprint, job.tool, job.job_handle, Jsonb(job.answer), job.submitted),
            )

    async def forget(self, user_id: str, fingerprint: str) -> None:
        async with self._connect() as conn:
            await conn.execute(
                "DELETE FROM harness_retrieval_jobs WHERE user_id = %s AND fingerprint = %s",
                (user_id, fingerprint),
            )

    async def forget_jobs(self, user_id: str, job_handles: Sequence[str]) -> None:
        doomed = list(dict.fromkeys(job_handles))
        if not doomed:
            return
        async with self._connect() as conn:
            await conn.execute(
                "DELETE FROM harness_retrieval_jobs WHERE user_id = %s AND job_handle = ANY(%s)",
                (user_id, doomed),
            )
