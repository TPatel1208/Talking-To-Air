from __future__ import annotations

import csv
import io
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from tta_backend.models.artifact import ArtifactReference, TableArtifactPayload
from tta_backend.repositories import artifact_repository


@dataclass
class StoredArtifact:
    payload: TableArtifactPayload
    created_at: float
    expires_at: float
    user_id: str | None = None
    thread_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ArtifactStore:
    """Table artifacts, write-through to Postgres (T39).

    ``put_table`` (mint) only ever touches the in-memory dict -- the tool
    that mints a table doesn't yet know who owns it. ``claim`` (the stream
    service attaches ownership moments later) is the durability boundary:
    it upserts the row to ``agent_artifacts`` so it survives a restart and
    is readable from any worker. Reads check memory first and rehydrate
    from Postgres on a miss, so memory is a hot cache, not the system of
    record, for anything already claimed.
    """

    def __init__(self, ttl_seconds: int = 30 * 60):
        self.ttl_seconds = ttl_seconds
        self._artifacts: dict[str, StoredArtifact] = {}

    def put_table(
        self,
        title: str,
        columns: list[str],
        rows: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactReference:
        self.cleanup()
        artifact_id = f"tbl_{uuid.uuid4().hex[:12]}"
        payload = TableArtifactPayload(
            title=title,
            columns=columns,
            rows=rows,
            metadata=metadata or {},
        )
        now = time.time()
        stored = StoredArtifact(
            payload=payload,
            created_at=now,
            expires_at=now + self.ttl_seconds,
        )
        self._artifacts[artifact_id] = stored
        return self._build_reference(artifact_id, stored)

    async def reference(self, artifact_id: str, user_id: str) -> ArtifactReference:
        stored = await self._owned_artifact(artifact_id, user_id)
        return self._build_reference(artifact_id, stored)

    async def claim(self, artifact_id: str, user_id: str, thread_id: str) -> ArtifactReference:
        stored = await self._active_artifact(artifact_id)
        if stored.user_id is None:
            stored.user_id = user_id
            stored.thread_id = thread_id
        elif stored.user_id != user_id:
            raise KeyError(artifact_id)
        await artifact_repository.save_artifact(
            artifact_id,
            stored.user_id,
            stored.thread_id,
            stored.payload.title,
            stored.payload.columns,
            stored.payload.rows,
            stored.payload.metadata,
        )
        # T39 user story #4: rows are only ever written here (mint stays
        # in-memory-only), so no unclaimed row should ever exist -- this is
        # a defensive sweep against a future write path that changes that.
        await artifact_repository.delete_expired_unclaimed(self.ttl_seconds)
        return self._build_reference(artifact_id, stored)

    async def get_page(
        self,
        artifact_id: str,
        user_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        stored = await self._owned_artifact(artifact_id, user_id)
        rows = stored.payload.rows
        offset = max(offset, 0)
        limit = min(max(limit, 1), 1000)
        return {
            "id": artifact_id,
            "type": stored.payload.type,
            "title": stored.payload.title,
            "columns": stored.payload.columns,
            "total_rows": len(rows),
            "offset": offset,
            "limit": limit,
            "rows": rows[offset:offset + limit],
            "metadata": stored.payload.metadata,
        }

    async def iter_csv_chunks(self, artifact_id: str, user_id: str) -> AsyncIterator[bytes]:
        stored = await self._owned_artifact(artifact_id, user_id)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=stored.payload.columns, extrasaction="ignore")
        writer.writeheader()
        yield output.getvalue().encode("utf-8")
        output.seek(0)
        output.truncate(0)

        for row in stored.payload.rows:
            writer.writerow(row)
            if output.tell() >= 64 * 1024:
                yield output.getvalue().encode("utf-8")
                output.seek(0)
                output.truncate(0)
        if output.tell():
            yield output.getvalue().encode("utf-8")

    def cleanup(self) -> None:
        now = time.time()
        expired = [artifact_id for artifact_id, stored in self._artifacts.items() if stored.expires_at <= now]
        for artifact_id in expired:
            self._artifacts.pop(artifact_id, None)

    async def _active_artifact(self, artifact_id: str) -> StoredArtifact:
        self.cleanup()
        stored = self._artifacts.get(artifact_id)
        if stored is not None:
            return stored
        row = await artifact_repository.get_artifact(artifact_id)
        if row is None:
            raise KeyError(artifact_id)
        return self._rehydrate(artifact_id, row)

    async def _owned_artifact(self, artifact_id: str, user_id: str) -> StoredArtifact:
        stored = await self._active_artifact(artifact_id)
        # Fail closed: an artifact with no owner yet (minted, not claimed)
        # belongs to nobody, so it is readable by nobody. Treating "no
        # owner" as "not someone else's owner" handed every unclaimed
        # artifact to any authenticated caller who knew its id. claim()
        # runs before the id is ever emitted to a client, so no legitimate
        # read path observes the unclaimed window.
        if stored.user_id != user_id:
            raise KeyError(artifact_id)
        return stored

    def _build_reference(self, artifact_id: str, stored: StoredArtifact) -> ArtifactReference:
        payload = stored.payload
        return ArtifactReference(
            id=artifact_id,
            type=payload.type,
            title=payload.title,
            row_count=len(payload.rows),
            metadata=payload.metadata,
        )

    def _rehydrate(self, artifact_id: str, row: dict[str, Any]) -> StoredArtifact:
        payload = TableArtifactPayload(
            title=row["title"],
            columns=row["columns"],
            rows=row["rows"],
            metadata=row.get("metadata") or {},
        )
        now = time.time()
        stored = StoredArtifact(
            payload=payload,
            created_at=now,
            expires_at=now + self.ttl_seconds,
            user_id=row["user_id"],
            thread_id=row["thread_id"],
        )
        self._artifacts[artifact_id] = stored
        return stored


artifact_store = ArtifactStore()
