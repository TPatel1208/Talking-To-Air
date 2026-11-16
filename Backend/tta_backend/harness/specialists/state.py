"""What one exploration task holds as it runs: the handles it may use, and what its calls reported."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tta_backend.harness.capabilities.definition import Capability
from tta_backend.harness.capabilities.result import CapabilityResult
from tta_backend.harness.handles.model import HandleRecord, InlineScalar


def scalar_value(record: HandleRecord) -> int | float | None:
    """The value a scalar handle holds inline, or None for any other handle."""
    if record.kind != "scalar" or not isinstance(record.payload, InlineScalar):
        return None
    return record.payload.value


def product_disclosure(result: CapabilityResult) -> str | None:
    """The product line a result owes when it names its product and rule but no disclosure mentions the product."""
    rule = result.values.get("rule")
    if not result.product or not isinstance(rule, str):
        return None
    if any(result.product in line for line in result.disclosures):
        return None
    return f"Product: {result.product}, chosen by rule {rule}."


class TaskState:
    """The task's handle allowlist and what its calls reported, in the order they ran.

    A handle is on the allowlist when the task was given it, resolved it before
    the model ran, or a call during the task returned it.
    """

    def __init__(self) -> None:
        self._records: dict[str, HandleRecord] = {}
        self._inputs: list[str] = []
        self._minted: list[str] = []
        self._disclosures: list[str] = []
        self._charts: list[str] = []
        self.shown_notes: set[str] = set()
        # Calls refused for naming a handle outside the allowlist.
        self.refusals = 0
        self.calls = 0

    def record(self, handle_id: str) -> HandleRecord | None:
        return self._records.get(handle_id)

    def add_input(self, record: HandleRecord) -> None:
        """Add an input handle; one a call already returned, such as a pre-resolved area, becomes an input."""
        if record.id in self._minted:
            self._minted.remove(record.id)
        if record.id not in self._inputs:
            self._records[record.id] = record
            self._inputs.append(record.id)

    def add_minted(self, record: HandleRecord) -> None:
        if record.id not in self._records:
            self._records[record.id] = record
            self._minted.append(record.id)

    def made_here(self, handle_id: str) -> bool:
        """Whether a call during the task returned the handle, as opposed to the task being given it."""
        return handle_id in self._minted

    def inputs(self) -> tuple[HandleRecord, ...]:
        return tuple(self._records[handle] for handle in self._inputs)

    def foreign(self, capability: Capability[Any], params: Mapping[str, Any]) -> list[str]:
        """Handle ids in ``params``' handle parameters that are not on the allowlist."""
        found: list[str] = []
        for name in capability.handle_params:
            value = params.get(name)
            values = value if isinstance(value, list | tuple) else [value]
            found.extend(v for v in values if isinstance(v, str) and v not in self._records)
        return list(dict.fromkeys(found))

    def disclose(self, line: str) -> None:
        self._disclosures.append(line)

    def take(self, result: CapabilityResult) -> None:
        """Keep a call's disclosures and charts."""
        self._disclosures.extend(result.disclosures)
        line = product_disclosure(result)
        if line is not None:
            self._disclosures.append(line)
        self._charts.extend(result.charts)

    def handles(self) -> tuple[str, ...]:
        """Input handles first, then those the calls returned."""
        return tuple(dict.fromkeys([*self._inputs, *self._minted]))

    def disclosures(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self._disclosures))

    def charts(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self._charts))

    def minted_scalars(self) -> tuple[HandleRecord, ...]:
        return tuple(self._records[h] for h in self._minted if self._records[h].kind == "scalar")
