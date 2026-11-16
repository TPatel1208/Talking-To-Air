"""Saved results: one ``Result`` per run, isolated per user.

A result is keyed by (user, run id). Saving it again replaces it, so a paused
run's result is overwritten when the run later finishes. Every read takes the
requesting user; another user's result behaves exactly as a missing one.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol

from psycopg.types.json import Jsonb

from tta_backend.harness.database import Connect
from tta_backend.harness.results import RUN_ID_PATTERN, Result


class ResultNotFound(Exception):
    """Raised for a missing result and for another user's alike."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"no result for run: {run_id}")
        self.run_id = run_id


class ResultStore(Protocol):
    async def put(self, user_id: str, result: Result) -> None:
        """Save ``result`` for run ``result.trace_id``, replacing any saved before."""
        ...

    async def get(self, user_id: str, run_id: str) -> Result:
        """The user's result for the run; ResultNotFound when there is none."""
        ...

    async def for_runs(self, user_id: str, run_ids: Sequence[str]) -> dict[str, Result]:
        """The user's results for ``run_ids`` by run id; runs with none are absent."""
        ...


def _dump(user_id: str, result: Result) -> dict[str, Any]:
    if not user_id:
        raise ValueError("a result is saved for a user")
    data = result.model_dump(mode="json")
    # Postgres JSONB cannot hold NUL; refused in both stores so they agree.
    if "\\u0000" in json.dumps(data):
        raise ValueError("no string in a result may contain NUL")
    return data


def _load(data: Any) -> Result:
    return Result.model_validate(data)


class InMemoryResultStore:
    def __init__(self) -> None:
        # JSON text, so callers never alias what is stored.
        self._results: dict[tuple[str, str], str] = {}

    async def put(self, user_id: str, result: Result) -> None:
        self._results[(user_id, result.trace_id)] = json.dumps(_dump(user_id, result))

    async def get(self, user_id: str, run_id: str) -> Result:
        stored = self._results.get((user_id, run_id))
        if stored is None:
            raise ResultNotFound(run_id)
        return _load(json.loads(stored))

    async def for_runs(self, user_id: str, run_ids: Sequence[str]) -> dict[str, Result]:
        found: dict[str, Result] = {}
        for run_id in run_ids:
            stored = self._results.get((user_id, run_id))
            if stored is not None:
                found[run_id] = _load(json.loads(stored))
        return found


async def ensure_tables(connect: Connect) -> None:
    async with connect() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS harness_results (
                user_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                status TEXT NOT NULL,
                result JSONB NOT NULL,
                PRIMARY KEY (user_id, run_id)
            )
            """
        )


class PostgresResultStore:
    def __init__(self, connect: Connect) -> None:
        self._connect = connect

    async def put(self, user_id: str, result: Result) -> None:
        data = _dump(user_id, result)
        async with self._connect() as conn:
            await conn.execute(
                """
                INSERT INTO harness_results (user_id, run_id, status, result)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, run_id) DO UPDATE SET status = EXCLUDED.status, result = EXCLUDED.result
                """,
                (user_id, result.trace_id, result.status, Jsonb(data)),
            )

    async def get(self, user_id: str, run_id: str) -> Result:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ResultNotFound(run_id)
        async with self._connect() as conn:
            cur = await conn.execute(
                "SELECT result FROM harness_results WHERE user_id = %s AND run_id = %s", (user_id, run_id)
            )
            row = await cur.fetchone()
        if row is None:
            raise ResultNotFound(run_id)
        return _load(row[0])

    async def for_runs(self, user_id: str, run_ids: Sequence[str]) -> dict[str, Result]:
        wanted = list(dict.fromkeys(run_ids))
        if not wanted:
            return {}
        async with self._connect() as conn:
            cur = await conn.execute(
                "SELECT run_id, result FROM harness_results WHERE user_id = %s AND run_id = ANY(%s)",
                (user_id, wanted),
            )
            rows = await cur.fetchall()
        return {str(row[0]): _load(row[1]) for row in rows}
