"""Assembling a stateless specialist from its tools, prompt and budgets.

Middleware order, outermost first::

    [*resilience_middleware(choice), meter, guard, model-call limit, tool-call budget]

Retry and fallback wrap each model call, and the meter sits inside them so it
sees every attempt. The guard's ``wrap_model_call`` is innermost, so it sees
the error the last attempt raised. With ``choice.prompt_cache`` the system
prompt carries a cache breakpoint; the host's model marks the conversation. The limits use ``before_model`` and
``after_model`` hooks, not wraps, so they count one per model turn whatever
the retries. After-model hooks run last to first: the tool-call limit, then
the model-call count, then the guard, whose jump to re-prompt or to end
therefore never skips a count. Both raise when hit: the model-call limit
with ``exit_behavior="error"`` raises ``ModelCallLimitExceededError``, the
tool-call budget ``ToolCallLimitExceededError``; the caller tells which by
the exception type.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from tta_backend.harness.models import (
    DEFAULT_BACKOFF,
    Backoff,
    ModelChoice,
    UsageMeter,
    resilience_middleware,
    system_message,
)
from tta_backend.harness.specialists.answer import ANSWER_TOOL, AnswerGuard, SpecialistAnswer
from tta_backend.harness.specialists.task import ExploreLimits


class ToolCallBudget(AgentMiddleware[Any, Any, Any]):
    """At most ``limit`` capability tool calls per run; build one per run.

    After each model turn it counts the turn's tool calls and raises
    ``ToolCallLimitExceededError`` before any of them runs when they would go
    over the limit. A turn that answers is not counted: the answer is not a
    tool call, and the guard runs none of the turn's other calls.
    """

    def __init__(self, limit: int) -> None:
        super().__init__()
        if limit < 1:
            raise ValueError("limit is at least 1")
        self._limit = limit
        self._count = 0

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        last = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if last is None or any(call["name"] == ANSWER_TOOL for call in last.tool_calls):
            return None
        wanted = self._count + len(last.tool_calls)
        if wanted > self._limit:
            raise ToolCallLimitExceededError(
                thread_count=wanted, run_count=wanted, thread_limit=None, run_limit=self._limit,
            )
        self._count = wanted
        return None


@dataclass(frozen=True)
class Specialist:
    """A compiled agent with the meter and guard built for it; build one per task."""

    agent: Any
    meter: UsageMeter
    guard: AnswerGuard


def build_specialist(
    choice: ModelChoice,
    tools: Sequence[BaseTool],
    system_prompt: str,
    guard: AnswerGuard,
    limits: ExploreLimits,
    *,
    max_retries: int = 2,
    backoff: Backoff = DEFAULT_BACKOFF,
) -> Specialist:
    meter = UsageMeter(choice)
    middleware = [
        *resilience_middleware(choice, max_retries=max_retries, backoff=backoff),
        meter,
        guard,
        ModelCallLimitMiddleware(run_limit=limits.model_calls, exit_behavior="error"),
        ToolCallBudget(limits.tool_calls),
    ]
    agent = create_agent(
        choice.model,
        tools=list(tools),
        system_prompt=system_message(system_prompt, choice),
        middleware=middleware,
        response_format=ToolStrategy(SpecialistAnswer),
        # Stateless: nothing of the task goes into a caller's checkpoint.
        checkpointer=False,
    )
    return Specialist(agent=agent, meter=meter, guard=guard)
