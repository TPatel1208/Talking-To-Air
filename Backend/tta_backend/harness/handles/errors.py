"""The one error a handle lookup or operation refuses with."""
from __future__ import annotations

from typing import Literal

from tta_backend.harness.handles.model import HandleKind, HandleRecord

Reason = Literal[
    "not_found",
    "wrong_kind",
    "evicted",
    "already_exists",
    "parent_not_found",
    "not_evictable",
]

# Fixed wording: the message never depends on whether a handle exists for
# someone else, so it cannot reveal another user's handles.
MESSAGES: dict[Reason, str] = {
    "not_found": "No such handle.",
    "wrong_kind": "The handle is not of the kind this step needs.",
    "evicted": "The handle's data is no longer stored.",
    "already_exists": "A handle with this id already exists.",
    "parent_not_found": "A handle this one is made from does not exist.",
    "not_evictable": "This handle's data cannot be evicted.",
}


class HandleError(Exception):
    """A refusal with a stable machine-readable ``reason``."""

    def __init__(self, reason: Reason, handle_id: str | None = None) -> None:
        super().__init__(MESSAGES[reason])
        self.reason: Reason = reason
        self.message = MESSAGES[reason]
        self.handle_id = handle_id


def require_kind(record: HandleRecord, *allowed: HandleKind) -> HandleRecord:
    """Refuse with ``wrong_kind`` unless the handle is one of ``allowed``."""
    if record.kind not in allowed:
        raise HandleError("wrong_kind", record.id)
    return record
