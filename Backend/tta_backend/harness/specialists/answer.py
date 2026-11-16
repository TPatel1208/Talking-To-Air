"""The specialist's final answer: its shape, the checks code makes on it, one re-prompt, and salvage.

The answer is given through the ``SpecialistAnswer`` tool. Code checks each
finding against the task: its handle must be on the allowlist, and a scalar
handle's finding must carry the value the scalar holds. An answer that fails
gets exactly one re-prompt naming the errors, with only the answer tool
bound. If that fails too, ``salvage`` keeps what can be used.
"""
from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphBubbleUp
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tta_backend.harness.handles.model import HandleRecord
from tta_backend.harness.results import Finding
from tta_backend.harness.specialists.state import TaskState, scalar_value

ANSWER_TOOL = "SpecialistAnswer"
MAX_SUMMARY_CHARS = 4000
MAX_FINDINGS = 20
MAX_FOLLOWUPS = 3
MAX_FOLLOWUP_CHARS = 300
_QUOTED_CHARS = 80
# Relative tolerance for a scalar finding's value: float noise, not rounding.
_SCALAR_TOLERANCE = 1e-9

# The result of a tool call made in the same turn as an answer.
NOT_RUN = "Not run: a turn that answers runs no other tool."
CORRECTION = (
    "Your answer could not be accepted:\n{errors}\n"
    f"Call {ANSWER_TOOL} once more with these fixed. No other tools are available now."
)
PLAIN_TEXT = f"the answer was plain text; give it by calling {ANSWER_TOOL}"
WRONG_SHAPE = f"the answer did not fit the fields of {ANSWER_TOOL}"
EMPTY_SUMMARY = "the summary is empty"


class AnswerFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200, description="What the value is, in a few words.")
    value: Annotated[float, Field(allow_inf_nan=False)] | int | str = Field(
        description="The value; for a scalar handle, exactly the value the tool returned."
    )
    unit: str | None = Field(default=None, max_length=50, description="The value's unit, when it has one.")
    handle: str = Field(min_length=1, max_length=200, description="The handle id the value comes from.")


class SpecialistAnswer(BaseModel):
    """The final answer: what the task established, the findings behind it, and follow-up questions."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        min_length=1, max_length=MAX_SUMMARY_CHARS,
        description="What you established and what it does not show, in a few sentences.",
    )
    findings: list[AnswerFinding] = Field(default_factory=list, max_length=MAX_FINDINGS)
    followups: list[str] = Field(
        default_factory=list, max_length=MAX_FOLLOWUPS, description="Follow-up questions worth asking next."
    )


def _quoted(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= _QUOTED_CHARS else text[: _QUOTED_CHARS - 1] + "…"


def _equal(given: Any, held: int | float) -> bool:
    if isinstance(given, bool) or not isinstance(given, int | float):
        return False
    return math.isclose(float(given), float(held), rel_tol=_SCALAR_TOLERANCE, abs_tol=0.0)


def _region(record: HandleRecord) -> str | None:
    region = record.descriptor.region
    name = region.get("name") if isinstance(region, dict) else None
    return name if isinstance(name, str) and name.strip() else None


def _period(record: HandleRecord) -> str | None:
    period = record.descriptor.period
    if period is None:
        return None
    return period.start if period.start == period.end else f"{period.start} to {period.end}"


def to_finding(
    record: HandleRecord, name: str, value: float | int | str, unit: str | None
) -> Finding | None:
    """A finding over ``record``, with its region and period from the descriptor."""
    try:
        return Finding(
            name=name, value=value, unit=unit, region=_region(record), period=_period(record), source_handle=record.id,
        )
    except ValidationError:
        return None


def check_finding(finding: AnswerFinding, state: TaskState) -> tuple[Finding | None, str | None]:
    """The checked finding, or the error that refuses it."""
    label = f"finding '{_quoted(finding.name)}'"
    record = state.record(finding.handle)
    if record is None:
        return None, f"{label}: handle {_quoted(finding.handle)} isn't part of this task"
    value, unit = finding.value, finding.unit
    if record.kind == "scalar":
        held = scalar_value(record)
        if held is None:
            return None, f"{label}: the scalar {record.id} holds no value"
        if not _equal(value, held):
            return None, f"{label}: {record.id} is a scalar holding {held!r}; give exactly that value"
        # The value and unit the scalar holds, not the model's copy of them.
        value, unit = held, record.descriptor.unit or unit
    checked = to_finding(record, finding.name.strip(), value, unit)
    if checked is None:
        return None, f"{label}: the value is not a finite number or a text"
    return checked, None


def check_findings(findings: Sequence[AnswerFinding], state: TaskState) -> tuple[list[Finding], list[str]]:
    kept: list[Finding] = []
    errors: list[str] = []
    for finding in findings:
        checked, error = check_finding(finding, state)
        if checked is not None:
            kept.append(checked)
        if error is not None:
            errors.append(error)
    return kept, errors


def _followups(items: Any) -> tuple[str, ...]:
    if not isinstance(items, list):
        return ()
    texts = [" ".join(item.split()) for item in items if isinstance(item, str)]
    kept = [text for text in texts if text and len(text) <= MAX_FOLLOWUP_CHARS]
    return tuple(dict.fromkeys(kept))[:MAX_FOLLOWUPS]


def _summary(text: Any) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    return text.strip()[:MAX_SUMMARY_CHARS]


@dataclass
class Attempt:
    """One answer the model gave: parsed when it fit the shape, else its raw arguments or prose."""

    parsed: SpecialistAnswer | None
    raw: Mapping[str, Any] | None
    prose: str
    findings: list[Finding] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Answer:
    """What the task reports from the model's answer."""

    summary: str | None
    findings: tuple[Finding, ...]
    followups: tuple[str, ...]


def _lenient(attempt: Attempt, state: TaskState) -> Answer:
    """What can be used from an attempt that did not fit: a summary, the findings that check, the follow-ups."""
    if attempt.parsed is not None:
        return Answer(_summary(attempt.parsed.summary), tuple(attempt.findings), _followups(attempt.parsed.followups))
    raw = attempt.raw
    if raw is None:
        return Answer(_summary(attempt.prose), (), ())
    findings: list[Finding] = []
    items = raw.get("findings")
    for item in items if isinstance(items, list) else []:
        try:
            given = AnswerFinding.model_validate(item)
        except ValidationError:
            continue
        checked, _ = check_finding(given, state)
        if checked is not None:
            findings.append(checked)
    return Answer(_summary(raw.get("summary")), tuple(findings), _followups(raw.get("followups")))


def accepted(attempts: Sequence[Attempt]) -> Answer | None:
    """The answer from the last attempt, when it passed every check."""
    if not attempts or attempts[-1].errors or attempts[-1].parsed is None:
        return None
    last = attempts[-1]
    assert last.parsed is not None
    return Answer(_summary(last.parsed.summary), tuple(last.findings), _followups(last.parsed.followups))


def salvage(attempts: Sequence[Attempt], state: TaskState) -> Answer | None:
    """The most recent attempt with a summary or a checked finding, bad findings dropped; None when none has either."""
    for attempt in reversed(attempts):
        answer = _lenient(attempt, state)
        if answer.summary is not None or answer.findings:
            return answer
    return None


def _text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    parts = [part.get("text", "") if isinstance(part, dict) else str(part) for part in content]
    return "".join(p for p in parts if isinstance(p, str))


class AnswerGuard(AgentMiddleware[Any, Any, Any]):
    """Checks each answer the model gives and asks once more when it fails.

    It goes after the meter and before the call limits in the middleware
    list, so its ``after_model`` runs last of the after-model hooks and its
    jump never skips a limit's count. It also keeps the last error any model
    attempt raised, so the caller can tell a model failure from any other.
    """

    def __init__(self, state: TaskState) -> None:
        super().__init__()
        self._state = state
        self.attempts: list[Attempt] = []
        self.reprompted = False
        self.last_model_error: Exception | None = None

    def _attempt(self, last: AIMessage) -> Attempt:
        """This turn's answer, read from its own answer call: the agent's
        ``structured_response`` can still hold an earlier turn's answer."""
        answer_calls = [call for call in last.tool_calls if call["name"] == ANSWER_TOOL]
        raw = answer_calls[0]["args"] if answer_calls else None
        parsed = None
        if len(answer_calls) == 1:
            try:
                parsed = SpecialistAnswer.model_validate(raw)
            except ValidationError:
                parsed = None
        attempt = Attempt(parsed=parsed, raw=raw, prose=_text(last))
        if parsed is not None:
            attempt.findings, attempt.errors = check_findings(parsed.findings, self._state)
            if not parsed.summary.strip():
                attempt.errors.insert(0, EMPTY_SUMMARY)
        else:
            attempt.errors = [WRONG_SHAPE if answer_calls else PLAIN_TEXT]
        return attempt

    @hook_config(can_jump_to=["model", "end"])
    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        last = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if last is None:
            return None
        answered = any(call["name"] == ANSWER_TOOL for call in last.tool_calls)
        if last.tool_calls and not answered:
            return None
        attempt = self._attempt(last)
        self.attempts.append(attempt)
        # Every other tool call in an answering turn gets a result instead of
        # running, so nothing runs after the answer and the history stays valid.
        answered_ids = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
        skipped = [
            ToolMessage(content=NOT_RUN, tool_call_id=call["id"], name=call["name"], status="error")
            for call in last.tool_calls
            if call["name"] != ANSWER_TOOL and call["id"] not in answered_ids
        ]
        if not attempt.errors:
            return {"messages": skipped, "jump_to": "end"} if skipped else None
        if self.reprompted:
            return {"messages": skipped, "jump_to": "end"}
        self.reprompted = True
        errors = "\n".join(f"- {error}" for error in attempt.errors)
        return {"messages": [*skipped, HumanMessage(CORRECTION.format(errors=errors))], "jump_to": "model"}

    def _request(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        # The re-prompt binds the answer tool alone.
        return request.override(tools=[]) if self.reprompted else request

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        try:
            return handler(self._request(request))
        except GraphBubbleUp:
            raise
        except Exception as exc:
            self.last_model_error = exc
            raise

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        try:
            return await handler(self._request(request))
        except GraphBubbleUp:
            raise
        except Exception as exc:
            self.last_model_error = exc
            raise
