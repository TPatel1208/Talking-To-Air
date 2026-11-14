"""The stores a run reads and writes, and the tables behind them."""
from __future__ import annotations

from dataclasses import dataclass

from tta_backend.harness import trace
from tta_backend.harness.database import Connect
from tta_backend.harness.handles import registry
from tta_backend.harness.handles.registry import HandleRegistry, InMemoryHandleRegistry, PostgresHandleRegistry
from tta_backend.harness.trace import InMemoryTraceStore, PostgresTraceStore, TraceStore


@dataclass
class Stores:
    handles: HandleRegistry
    traces: TraceStore


async def ensure_tables(connect: Connect) -> None:
    """Create every table the harness owns; safe to call on each startup."""
    await registry.ensure_tables(connect)
    await trace.ensure_tables(connect)


def postgres_stores(connect: Connect) -> Stores:
    return Stores(handles=PostgresHandleRegistry(connect), traces=PostgresTraceStore(connect))


def memory_stores() -> Stores:
    """Stores that live in the process, for tests and benchmarks."""
    return Stores(handles=InMemoryHandleRegistry(), traces=InMemoryTraceStore())
