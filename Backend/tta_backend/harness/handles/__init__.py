"""Typed handles, the per-user registry that holds them, their payloads and lineage."""
from tta_backend.harness.handles.errors import HandleError, require_kind
from tta_backend.harness.handles.lineage import Provenance, dataset_handles, lineage
from tta_backend.harness.handles.model import (
    KINDS,
    Descriptor,
    HandleKind,
    HandleRecord,
    InlineScalar,
    Period,
    ServicePayload,
    new_handle_id,
)
from tta_backend.harness.handles.payloads import PayloadStore, Scalar, ScalarInput
from tta_backend.harness.handles.registry import (
    HandleRegistry,
    InMemoryHandleRegistry,
    PostgresHandleRegistry,
    ensure_tables,
)
from tta_backend.harness.handles.resolver import HandleResolver, OpenRetrieval

__all__ = [
    "KINDS",
    "Descriptor",
    "HandleError",
    "HandleKind",
    "HandleRecord",
    "HandleRegistry",
    "HandleResolver",
    "InMemoryHandleRegistry",
    "InlineScalar",
    "OpenRetrieval",
    "PayloadStore",
    "Period",
    "PostgresHandleRegistry",
    "Provenance",
    "Scalar",
    "ScalarInput",
    "ServicePayload",
    "dataset_handles",
    "ensure_tables",
    "lineage",
    "new_handle_id",
    "require_kind",
]
