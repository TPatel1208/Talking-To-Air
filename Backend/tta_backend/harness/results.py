"""The envelope workflows and specialists both return."""
from __future__ import annotations

import math
import re
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tta_backend.harness.release import HARNESS_RELEASE

ResultStatus = Literal[
    "done", "needs_input", "waiting", "insufficient_evidence", "failed", "cancelled",
    # An exploration that ran out of its model-call or tool-call budget; it returns what it found.
    "incomplete",
]
ModeKind = Literal["validated", "draft", "exploratory"]

# The one run-id format: trace runs and the result that points at one share it.
RUN_ID_PATTERN = re.compile(r"^run_[0-9a-f]{32}$")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WorkflowRef(_Model):
    name: str = Field(min_length=1)
    version: int = Field(ge=1)

    def label(self) -> str:
        return f"{self.name} v{self.version}"


class Mode(_Model):
    kind: ModeKind
    workflow: WorkflowRef | None = None
    # Exact child versions the run used, so a parent's label can be audited.
    child_workflows: tuple[WorkflowRef, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.kind == "exploratory":
            if self.workflow is not None or self.child_workflows:
                raise ValueError("an exploratory result names no workflow")
        elif self.workflow is None:
            raise ValueError(f"a {self.kind} result must name its workflow")
        return self

    @classmethod
    def exploratory(cls) -> Mode:
        return cls(kind="exploratory")

    @classmethod
    def validated(
        cls, name: str, version: int, children: tuple[WorkflowRef, ...] = ()
    ) -> Mode:
        return cls(
            kind="validated",
            workflow=WorkflowRef(name=name, version=version),
            child_workflows=children,
        )

    @classmethod
    def draft(
        cls, name: str, version: int, children: tuple[WorkflowRef, ...] = ()
    ) -> Mode:
        return cls(
            kind="draft",
            workflow=WorkflowRef(name=name, version=version),
            child_workflows=children,
        )

    def view(self) -> dict[str, Any]:
        """The mode as the frontend reads it: kind, label and the workflow, if any."""
        workflow = None if self.workflow is None else {"name": self.workflow.name, "version": self.workflow.version}
        return {"kind": self.kind, "label": self.label(), "workflow": workflow}

    def label(self) -> str:
        """The wording a user sees for how the result was made."""
        if self.workflow is None:
            return "Exploratory"
        if self.kind == "draft":
            return f"Draft workflow: {self.workflow.label()}"
        return self.workflow.label()


class Finding(_Model):
    name: str = Field(min_length=1)
    value: float | int | str
    unit: str | None = None
    # Kept as opaque JSON until the shape of uncertainty is settled.
    uncertainty: Any = None
    region: str | None = None
    period: str | None = None
    source_handle: str = Field(min_length=1)

    @field_validator("value")
    @classmethod
    def _finite(cls, value: float | int | str) -> float | int | str:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("a finding value must be finite")
        return value


class Gap(_Model):
    step: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class Result(_Model):
    status: ResultStatus
    mode: Mode
    findings: tuple[Finding, ...] = ()
    handles: tuple[str, ...] = ()
    charts: tuple[str, ...] = ()
    disclosures: tuple[str, ...] = ()
    gaps: tuple[Gap, ...] = ()
    trace_id: str = Field(pattern=RUN_ID_PATTERN.pattern)
    harness_release: str = Field(default=HARNESS_RELEASE, min_length=1)
    # Why the result is not done, in fixed wording.
    reason: str | None = None
    # The typed question a needs_input result waits on, as plain JSON.
    question: dict[str, Any] | None = None
    # Model-written text, always shown as interpretation, apart from the methods.
    interpretation: str | None = Field(default=None, min_length=1)
    # Follow-up questions the model suggests.
    followups: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.question is not None and self.status != "needs_input":
            raise ValueError("a question comes only with needs_input")
        if self.reason is not None and self.status == "done":
            raise ValueError("a done result has no reason")
        return self
