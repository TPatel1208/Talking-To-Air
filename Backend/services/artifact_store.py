from __future__ import annotations

import csv
import io
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from models.artifact import ArtifactReference, TableArtifactPayload


@dataclass
class StoredArtifact:
    payload: TableArtifactPayload
    created_at: float
    expires_at: float
    user_id: str | None = None
    thread_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ArtifactStore:
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
        self._artifacts[artifact_id] = StoredArtifact(
            payload=payload,
            created_at=now,
            expires_at=now + self.ttl_seconds,
        )
        return self.reference(artifact_id)

    def reference(self, artifact_id: str) -> ArtifactReference:
        stored = self._active_artifact(artifact_id)
        payload = stored.payload
        return ArtifactReference(
            id=artifact_id,
            type=payload.type,
            title=payload.title,
            row_count=len(payload.rows),
            metadata=payload.metadata,
        )

    def claim(self, artifact_id: str, user_id: str, thread_id: str) -> ArtifactReference:
        stored = self._active_artifact(artifact_id)
        if stored.user_id is None:
            stored.user_id = user_id
            stored.thread_id = thread_id
        elif stored.user_id != user_id:
            raise KeyError(artifact_id)
        return self.reference(artifact_id)

    def get_page(
        self,
        artifact_id: str,
        user_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        stored = self._owned_artifact(artifact_id, user_id)
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
        stored = self._owned_artifact(artifact_id, user_id)
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

    def _active_artifact(self, artifact_id: str) -> StoredArtifact:
        self.cleanup()
        stored = self._artifacts.get(artifact_id)
        if stored is None:
            raise KeyError(artifact_id)
        return stored

    def _owned_artifact(self, artifact_id: str, user_id: str) -> StoredArtifact:
        stored = self._active_artifact(artifact_id)
        if stored.user_id is not None and stored.user_id != user_id:
            raise KeyError(artifact_id)
        return stored


artifact_store = ArtifactStore()
