"""One way in to any handle: mint derived ones, open either origin, evict payloads."""
from __future__ import annotations

import datetime
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import JsonValue

from tta_backend.harness.handles.errors import HandleError, require_kind
from tta_backend.harness.handles.model import (
    Descriptor,
    EvictedPayload,
    HandleKind,
    HandleRecord,
    StoredPayload,
    new_handle_id,
)
from tta_backend.harness.handles.payloads import PayloadStore
from tta_backend.harness.handles.registry import HandleRegistry

# Opens a retrieval handle through the earthdata service in the given user's
# workspace. It is called only after that user's ownership is checked.
OpenRetrieval = Callable[[str, HandleRecord], Awaitable[Any]]


class HandleResolver:
    def __init__(self, registry: HandleRegistry, payloads: PayloadStore, open_retrieval: OpenRetrieval) -> None:
        self._registry = registry
        self._payloads = payloads
        self._open_retrieval = open_retrieval

    async def get(self, user_id: str, handle_id: str, *kinds: HandleKind) -> HandleRecord:
        """The user's record, checked against ``kinds`` when given."""
        record = await self._registry.get(user_id, handle_id)
        return require_kind(record, *kinds) if kinds else record

    async def open(self, user_id: str, handle_id: str, *kinds: HandleKind) -> Any:
        """The handle's data; ownership and kind are checked before anything is read.

        A grid comes back lazily loaded from its Zarr store. The caller loads
        what it needs before the handle can be evicted: once the bytes are
        deleted, a later read of that grid fails with Zarr's own error.
        """
        record = await self.get(user_id, handle_id, *kinds)
        if record.origin == "retrieval":
            return await self._open_retrieval(user_id, record)
        return await self._payloads.open(record)

    async def mint(
        self,
        *,
        user_id: str,
        conversation_id: str,
        kind: HandleKind,
        data: Any,
        descriptor: Descriptor,
        operation: str,
        created_at: datetime.datetime,
        parents: tuple[str, ...] = (),
        parameters: dict[str, JsonValue] | None = None,
        run_id: str | None = None,
    ) -> HandleRecord:
        """Write the payload, then the record; a refused record leaves no payload behind."""
        # Checked before the payload write, which may be large.
        for parent in parents:
            try:
                await self._registry.get(user_id, parent)
            except HandleError as error:
                raise HandleError("parent_not_found", parent) from error
        handle_id = new_handle_id(kind)
        payload = await self._payloads.write(user_id, handle_id, kind, data)
        try:
            record = HandleRecord(
                id=handle_id,
                kind=kind,
                origin="derived",
                user_id=user_id,
                conversation_id=conversation_id,
                descriptor=descriptor,
                parents=parents,
                operation=operation,
                parameters=parameters or {},
                run_id=run_id,
                created_at=created_at,
                payload=payload,
            )
            return await self._registry.mint(record)
        except BaseException:
            if isinstance(payload, StoredPayload):
                await self._payloads.delete(user_id, handle_id, payload.format)
            raise

    async def evict(self, user_id: str, handle_id: str, at: datetime.datetime) -> HandleRecord:
        """Drop a derived handle's payload bytes; the record stays and opening it says ``evicted``."""
        # The record changes first, so nothing is sent to a file that is going
        # away. The bytes are deleted on every call, even for a record already
        # evicted, so a retry finishes a delete that failed or was interrupted.
        updated = await self._registry.mark_evicted(user_id, handle_id, at)
        if isinstance(updated.payload, EvictedPayload):
            await self._payloads.delete(user_id, handle_id, updated.payload.format)
        return updated
