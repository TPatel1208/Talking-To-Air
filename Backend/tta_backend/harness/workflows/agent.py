"""What an ``agent`` step sends a model and what it accepts back.

An agent step reads and writes only: it gets its instruction, the values its
``reads`` resolve to and the knowledge notes for the datasets and quantities
behind them, and answers with the shape its ``returns`` declares. It has no
tools besides the one that carries the answer.

- ``answer_model(returns)``: the pydantic model an answer must fit.
- ``build_prompt(...)``: the system prompt and the one human message.
- ``ask_model(choice, ...)``: one agent run through retry and fallback,
  with a wrong-shaped answer retried once. ``Asked.outcome`` says whether an
  answer fit, none did, the models failed, or something else went wrong.

Size bounds: each read value is at most ``MAX_READ_BYTES`` of JSON and all of
them together at most ``MAX_READS_BYTES``.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse, hook_config
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.errors import GraphBubbleUp
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StringConstraints, create_model

from tta_backend.harness.knowledge import Note
from tta_backend.harness.models import (
    DEFAULT_BACKOFF,
    Backoff,
    ModelChoice,
    Usage,
    UsageMeter,
    resilience_middleware,
    system_message,
)
from tta_backend.harness.workflows.spec import ReturnType

logger = logging.getLogger(__name__)

# The tool an answer is given through, and the model it is checked against.
ANSWER_TOOL = "answer"
MAX_TEXT_CHARS = 2000
MAX_LIST_ITEMS = 10
MAX_ITEM_CHARS = 300
MAX_READ_BYTES = 4096
MAX_READS_BYTES = 16384
# Model turns per step: the first answer and one retry of a wrong-shaped one.
MAX_TURNS = 2
_RECURSION_LIMIT = 12

_Text = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=MAX_TEXT_CHARS)]
_Item = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=MAX_ITEM_CHARS)]
_Number = Annotated[StrictInt | StrictFloat, Field(allow_inf_nan=False)]

_FIELDS: dict[str, tuple[Any, str]] = {
    "string": (_Text, f"text, at most {MAX_TEXT_CHARS} characters"),
    "number": (_Number, "a number"),
    "boolean": (StrictBool, "true or false"),
    "list": (
        Annotated[list[_Item], Field(max_length=MAX_LIST_ITEMS)],
        f"a list of at most {MAX_LIST_ITEMS} short texts, each at most {MAX_ITEM_CHARS} characters",
    ),
}


def answer_model(returns: Mapping[str, ReturnType]) -> type[BaseModel]:
    """The model an answer must fit: every declared field, nothing else."""
    fields: dict[str, Any] = {
        name: (_FIELDS[kind][0], Field(description=_FIELDS[kind][1])) for name, kind in returns.items()
    }
    model: type[BaseModel] = create_model(
        ANSWER_TOOL,
        __config__=ConfigDict(extra="forbid"),
        __doc__="The step's answer.",
        **fields,
    )
    return model


def shape_text(returns: Mapping[str, ReturnType]) -> str:
    """The declared shape in words, as the model is told it."""
    return "; ".join(f"{name} ({_FIELDS[kind][1]})" for name, kind in returns.items())


def value_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def json_size(value: Any) -> int:
    return len(value_json(value).encode())


@dataclass(frozen=True)
class Prompt:
    system: str
    human: str


def build_prompt(
    instruction: str, reads: Sequence[tuple[str, Any]], notes: Sequence[Note], returns: Mapping[str, ReturnType]
) -> Prompt:
    """The step's instruction, then the values it reads and the notes that apply."""
    system = (
        f"You carry out one step of a workflow. {instruction}\n\n"
        "Use only the values and notes you are given; you cannot fetch data or call other tools. "
        f"Give your answer by calling {ANSWER_TOOL} once, with these fields: {shape_text(returns)}."
    )
    lines = ["Values:"]
    lines.extend(f"{ref}: {value_json(value)}" for ref, value in reads)
    if not reads:
        lines.append("none")
    if notes:
        lines.append("")
        lines.append("Notes:")
        for note in notes:
            lines.append(f"{note.title}: {note.body}")
    return Prompt(system=system, human="\n".join(lines))


def _correction(returns: Mapping[str, ReturnType]) -> str:
    return f"That answer did not fit the declared shape. Call {ANSWER_TOOL} once with exactly these fields: {shape_text(returns)}."


class _OneRetry(AgentMiddleware[Any, Any, Any]):
    """Ends the run after ``MAX_TURNS`` model turns without an answer, and
    asks again after a turn that answered in plain text."""

    def __init__(self, correction: str) -> None:
        super().__init__()
        self._correction = correction

    @hook_config(can_jump_to=["end", "model"])
    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if state.get("structured_response") is not None:
            return None
        messages = state.get("messages") or []
        turns = sum(1 for message in messages if isinstance(message, AIMessage))
        if turns >= MAX_TURNS:
            return {"jump_to": "end"}
        last = messages[-1] if messages else None
        if isinstance(last, AIMessage) and not last.tool_calls:
            return {"messages": [HumanMessage(self._correction)], "jump_to": "model"}
        return None


class _ModelErrors(AgentMiddleware[Any, Any, Any]):
    """Keeps the last error a model call raised, so a failure elsewhere in the
    agent run is told apart from the models failing."""

    def __init__(self) -> None:
        super().__init__()
        self.last: Exception | None = None

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        try:
            return handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            self.last = exc
            raise

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            self.last = exc
            raise


Outcome = Literal["answered", "invalid", "model_failed", "internal_error"]


@dataclass(frozen=True)
class Asked:
    """One agent run: the answer when one fit, else None, how it ended, and what it used."""

    outcome: Outcome
    answer: dict[str, Any] | None
    usage: Usage


def _agent_context() -> contextvars.Context:
    """The current context without langchain's run config, so the step's agent
    is its own run and not a subgraph of the workflow's graph: nothing of it
    reaches the workflow's checkpoint or streams into the workflow's events.
    Logging and tracing context are kept."""
    context = contextvars.copy_context()
    context.run(var_child_runnable_config.set, None)
    return context


async def ask_model(
    choice: ModelChoice,
    prompt: Prompt,
    returns: Mapping[str, ReturnType],
    *,
    max_retries: int = 2,
    backoff: Backoff = DEFAULT_BACKOFF,
) -> Asked:
    """Ask the step's model for an answer, with retry and fallback around each call.
    With ``choice.prompt_cache`` the system prompt carries a cache breakpoint."""
    schema = answer_model(returns)
    correction = _correction(returns)
    meter = UsageMeter(choice)
    errors = _ModelErrors()
    agent: Any = create_agent(
        choice.model,
        tools=[],
        system_prompt=system_message(prompt.system, choice),
        response_format=ToolStrategy(schema, handle_errors=lambda _: correction),
        # Innermost after retry and fallback, so it sees the model's own errors.
        middleware=[
            *resilience_middleware(choice, max_retries=max_retries, backoff=backoff), meter, errors,
            _OneRetry(correction),
        ],
        checkpointer=False,
    )
    run = agent.ainvoke({"messages": [HumanMessage(prompt.human)]}, config={"recursion_limit": _RECURSION_LIMIT})
    try:
        state = await asyncio.create_task(run, context=_agent_context())
    except GraphBubbleUp:
        raise
    except Exception as exc:
        # The type name only: provider errors can carry request details.
        if exc is errors.last:
            logger.warning("an agent step's models all failed: %s", type(exc).__name__)
            return Asked("model_failed", None, meter.usage())
        logger.error("an agent step's run failed outside its model calls: %s", type(exc).__name__)
        return Asked("internal_error", None, meter.usage())
    answer = state.get("structured_response") if isinstance(state, Mapping) else None
    if not isinstance(answer, schema):
        return Asked("invalid", None, meter.usage())
    return Asked("answered", answer.model_dump(mode="json"), meter.usage())
