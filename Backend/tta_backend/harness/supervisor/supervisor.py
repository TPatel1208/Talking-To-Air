"""The supervisor: one agent per conversation that reads the ledger, delegates and writes the answer.

A host builds one ``Supervisor`` at startup and calls ``turn`` per message.
Each turn:

1. checks the jobs waiting runs are on and builds the session ledger, rendered once;
2. runs the agent on the message. Its checkpoint thread is
   ``supervisor/{user}/{conversation}``, so no two users share one. The
   ledger is put just before the newest user message on every model call
   and never written to the checkpoint;
3. streams a preview through ``events.text`` when the host has it: the
   model's text line by line as it is produced, the sources line before the
   first line that comes once a result exists, then code's parts in one
   final chunk. Narration before a tool call may stream too;
4. builds ``Turn.text`` from the sources line of all the turn's results,
   the prose of the final model message (the one that calls no tool) and
   code's parts. Narration is never part of it.

Middleware, outermost first::

    [summarisation, *resilience_middleware(choice), meter, guard, context editing, unfinished calls, ledger]

The guard counts one model call per model turn. The last call the limit
allows, and any call once the tool-call limit is spent, is made with
tool_choice "none"; further tool calls are refused. It sits inside retry and
fallback, so the error it records last is the one that leaves the agent
when every model failed. Unfinished calls answers, in the request only, a
tool call an interrupted turn left without a result.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import quote

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    SummarizationMiddleware,
    hook_config,
)
from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphBubbleUp
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from tta_backend.harness.capabilities.registry import CapabilityRegistry
from tta_backend.harness.context.ledger import Ledger, build_ledger, render_ledger
from tta_backend.harness.specialists.prompt import date_preamble
from tta_backend.harness.jobs import check_waiting
from tta_backend.harness.knowledge import NoteCatalogue
from tta_backend.harness.models import (
    DEFAULT_BACKOFF,
    Backoff,
    ModelChoice,
    Models,
    UsageMeter,
    resilience_middleware,
    system_message,
)
from tta_backend.harness.results import Result
from tta_backend.harness.run_context import NO_MODEL_REASON, EarthdataSession, Identity, RunContext
from tta_backend.harness.specialists.task import ExploreLimits
from tta_backend.harness.supervisor.answer import (
    ProseCleaner,
    appended,
    clean_prose,
    data_sources,
    sources_line,
    unmatched_numbers,
)
from tta_backend.harness.supervisor.index import supervisor_index
from tta_backend.harness.supervisor.prompt import supervisor_prompt
from tta_backend.harness.supervisor.tools import TurnTools, latest_runnable, tool_schemas
from tta_backend.harness.workflows.loader import WorkflowCatalogue
from tta_backend.harness.workflows.runner import EnvFactory, WorkflowRunner

logger = logging.getLogger(__name__)

AGENT_NAME = "supervisor"
# Marks the injected ledger message, which is never part of the saved state.
LEDGER_SOURCE = "ledger"
SUMMARY_SOURCE = "summarization"
# The trace step of the numbers check, an ``agent`` entry on the turn's lookup run.
NUMBERS_STEP = "supervisor.numbers_check"

LIMIT_WORDING = "This answer stopped at the limit of {limit} model calls for one turn; it reports only what had finished."
FAILED_WORDING = "The language model could not be reached, so this answer was not finished; what had already run is kept."
INTERNAL_WORDING = "The answer stopped on an internal error; what had already run is kept."
UNFINISHED_CALL = "This call did not finish."
TOOL_LIMIT = "The limit of {limit} tool calls for this turn is reached, so this call was not run. Answer from what you have."

# The most follow-ups a turn suggests, after merging its results' in order.
MAX_FOLLOWUPS = 3

TurnStatus = Literal["answered", "limit", "failed"]


@dataclass(frozen=True, kw_only=True)
class SupervisorLimits:
    """Per-turn budgets and the history sizes past which context is cut down."""

    model_calls: int = 6
    tool_calls: int = 4
    summarise_above_tokens: int = 12_000
    keep_messages: int = 20
    clear_tool_results_above_tokens: int = 8_000
    keep_tool_results: int = 3

    def __post_init__(self) -> None:
        if self.model_calls < 1 or self.tool_calls < 1:
            raise ValueError("model_calls and tool_calls are at least 1")
        if min(self.summarise_above_tokens, self.keep_messages, self.clear_tool_results_above_tokens) < 1:
            raise ValueError("token thresholds and keep_messages are at least 1")
        if self.keep_tool_results < 0:
            raise ValueError("keep_tool_results is at least 0")


@dataclass(frozen=True, kw_only=True)
class Turn:
    """One turn as the user sees it, with what it produced and what it used."""

    # Prose and the code-appended parts, as the user sees them.
    text: str
    # Every result the turn's tools produced, in the order they came back.
    results: tuple[Result, ...]
    # Those results' runs, then the turn's lookup run, if any.
    run_ids: tuple[str, ...]
    followups: tuple[str, ...]
    # Connectors whose service was down when work needing them was refused.
    unavailable: tuple[str, ...] = ()
    # The first needs_input question, if any.
    question: dict[str, Any] | None
    # For a question with options: run_id, prompt and options, for the app's picker.
    variable_choice: dict[str, Any] | None
    status: TurnStatus
    model_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int


def thread_id(identity: Identity) -> str:
    """The checkpoint thread of a user's conversation; the parts are escaped so two users never meet."""
    return f"supervisor/{quote(identity.user_id, safe='')}/{quote(identity.conversation_id, safe='')}"


# Middleware


def turn_context(context: RunContext, ledger: Ledger) -> str:
    """The per-turn message: today's date, which relative dates in workflow
    inputs are worked out from, then the ledger. It sits after the cached prefix."""
    return f"{date_preamble(context.clock.today())}\n\n{render_ledger(ledger)}"


def _with_ledger(messages: Sequence[AnyMessage], ledger: HumanMessage) -> list[AnyMessage]:
    """``messages`` with ``ledger`` just before the newest user message."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, HumanMessage) and message.additional_kwargs.get("lc_source") != SUMMARY_SOURCE:
            return [*messages[:index], ledger, *messages[index:]]
    return [*messages, ledger]


class LedgerMiddleware(AgentMiddleware[Any, Any, Any]):
    """Puts the rendered ledger before the newest user message of each model
    call. It changes only the request, so the saved state never holds it."""

    def __init__(self, rendered: str) -> None:
        super().__init__()
        self._message = HumanMessage(content=rendered, additional_kwargs={"lc_source": LEDGER_SOURCE})

    def wrap_model_call(
        self, request: ModelRequest[Any], handler: Callable[[ModelRequest[Any]], ModelResponse[Any]]
    ) -> ModelResponse[Any]:
        return handler(request.override(messages=_with_ledger(request.messages, self._message)))

    async def awrap_model_call(
        self, request: ModelRequest[Any], handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]]
    ) -> ModelResponse[Any]:
        return await handler(request.override(messages=_with_ledger(request.messages, self._message)))


def _answer_only(message: Any) -> Any:
    if not isinstance(message, AIMessage) or not message.tool_calls:
        return message
    content = message.content
    if isinstance(content, list):
        content = [block for block in content if not (isinstance(block, dict) and block.get("type") == "tool_use")]
    return message.model_copy(update={"tool_calls": [], "invalid_tool_calls": [], "content": content})


def _without_tool_calls(response: Any) -> Any:
    if isinstance(response, ExtendedModelResponse):
        return ExtendedModelResponse(
            model_response=_without_tool_calls(response.model_response), command=response.command
        )
    if isinstance(response, ModelResponse):
        return ModelResponse(
            result=[_answer_only(m) for m in response.result], structured_response=response.structured_response
        )
    if isinstance(response, AIMessage):
        return _answer_only(response)
    return response


class TurnGuard(AgentMiddleware[Any, Any, Any]):
    """The turn's model-call and tool-call limits; build one per turn.

    The last model call the limit allows, and any call once the tool calls
    are spent, is made with tool_choice "none" and answered from its text
    alone. ``limited`` is set when the last allowed call still wanted tools.
    """

    def __init__(self, limits: SupervisorLimits) -> None:
        super().__init__()
        self._limits = limits
        self.model_turns = 0
        self.tool_calls = 0
        self.limited = False
        self.last_model_error: BaseException | None = None

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if self.model_turns >= self._limits.model_calls:
            self.limited = True
            return {"jump_to": "end"}
        self.model_turns += 1
        return None

    async def awrap_model_call(
        self, request: ModelRequest[Any], handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]]
    ) -> ModelResponse[Any]:
        last_call = self.model_turns >= self._limits.model_calls
        answer_now = last_call or self.tool_calls >= self._limits.tool_calls
        if answer_now:
            request = request.override(tool_choice="none")
        try:
            response = await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            self.last_model_error = exc
            raise
        if not answer_now:
            return response
        if last_call and _wants_tools(response):
            self.limited = True
        # A model that calls tools anyway is answered from its text alone.
        return _without_tool_calls(response)  # type: ignore[no-any-return]

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        if self.tool_calls >= self._limits.tool_calls:
            call = request.tool_call
            return ToolMessage(
                content=TOOL_LIMIT.format(limit=self._limits.tool_calls), tool_call_id=call["id"], name=call["name"],
            )
        self.tool_calls += 1
        return await handler(request)


def _wants_tools(response: Any) -> bool:
    if isinstance(response, ExtendedModelResponse):
        response = response.model_response
    messages = response.result if isinstance(response, ModelResponse) else [response]
    return any(isinstance(m, AIMessage) and m.tool_calls for m in messages)


def _answered(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """``messages`` with a fixed-wording result after each tool call that has none."""
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    out: list[AnyMessage] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        out.append(message)
        index += 1
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        missing = [call for call in message.tool_calls if call.get("id") not in answered]
        if not missing:
            continue
        # After the results the call did get, so each call's results stay together.
        while index < len(messages) and isinstance(messages[index], ToolMessage):
            out.append(messages[index])
            index += 1
        out.extend(
            ToolMessage(content=UNFINISHED_CALL, tool_call_id=str(call.get("id")), name=call.get("name"))
            for call in missing
        )
    return out


class UnfinishedCalls(AgentMiddleware[Any, Any, Any]):
    """Answers, in the request only, any tool call a cancelled or failed turn
    left without a result, since providers refuse such a history. The saved
    state keeps the call as it was."""

    def wrap_model_call(
        self, request: ModelRequest[Any], handler: Callable[[ModelRequest[Any]], ModelResponse[Any]]
    ) -> ModelResponse[Any]:
        return handler(request.override(messages=_answered(request.messages)))

    async def awrap_model_call(
        self, request: ModelRequest[Any], handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]]
    ) -> ModelResponse[Any]:
        return await handler(request.override(messages=_answered(request.messages)))


# Streaming the answer


def _from_supervisor_model(chunk: Any, metadata: dict[str, Any]) -> bool:
    """Whether a streamed message is the supervisor's own model output, not a nested agent's."""
    return (
        isinstance(chunk, AIMessage)
        and metadata.get("langgraph_node") == "model"
        and metadata.get("lc_agent_name") == AGENT_NAME
        and "|" not in str(metadata.get("langgraph_checkpoint_ns") or "")
    )


class _Message:
    def __init__(self) -> None:
        self.buffer = ""
        self.tools = False
        self.emitted = False


class AnswerStream:
    """Writes the answer through ``events.text`` as it is produced.

    The model's text goes out a line at a time, cleaned by ``ProseCleaner``;
    a message stops going out once a tool call shows up in it, so narration
    before a tool call may already have gone. The sources line goes just
    before the first prose line that comes once a result exists. ``finish``
    sends the sources line if it has not gone, then code's parts, as one chunk.

    What streams is a preview: the turn's text is built from its final
    message alone, by ``turn_text``.
    """

    def __init__(
        self, context: RunContext, sources: Callable[[], Awaitable[list[str]]], has_results: Callable[[], bool]
    ) -> None:
        self.chunks: list[str] = []
        self._context = context
        self._sources = sources
        self._has_results = has_results
        self._cleaner = ProseCleaner()
        self._opened = False
        self._messages: dict[tuple[Any, str], _Message] = {}
        self._current: tuple[Any, str] | None = None
        self._count = 0

    def _emit(self, content: str) -> None:
        self.chunks.append(content)
        write = getattr(self._context.events, "text", None)
        if callable(write):
            try:
                write(content)
            except Exception as exc:
                logger.warning("the host could not take streamed text: %s", type(exc).__name__)

    async def _open(self) -> bool:
        """Sends the sources line once; whether it was sent now."""
        if self._opened:
            return False
        self._opened = True
        line = sources_line(await self._sources())
        if line is None:
            return False
        self._emit(("\n\n" if self.chunks else "") + line)
        return True

    async def _line(self, message: _Message, line: str) -> None:
        out = self._cleaner.line(line)
        if not out:
            return
        if self._has_results() and await self._open():
            # The prose after the sources line starts a new paragraph.
            out = "\n\n" + out.lstrip("\n")
        message.emitted = True
        self._emit(out)

    async def _flush(self, key: tuple[Any, str] | None) -> None:
        message = self._messages.get(key) if key is not None else None
        if message is None or message.tools or not message.buffer:
            return
        line, message.buffer = message.buffer, ""
        await self._line(message, line)

    async def feed(self, chunk: AIMessage, metadata: dict[str, Any]) -> None:
        # Keyed by graph step as well as id, so no two steps' messages merge;
        # a chunk without an id continues the message in hand of its step.
        step = metadata.get("langgraph_step")
        if chunk.id is None and self._current is not None and self._current[0] == step:
            key = self._current
        else:
            key = (step, chunk.id or f"unnamed-{self._count}")
        if key != self._current:
            if self._current is not None and self._current[0] == step and step is not None:
                # A new message in the same model step: the one in hand was an attempt that failed.
                self._messages[self._current].buffer = ""
            else:
                await self._flush(self._current)
            self._current = key
            self._count += 1
            self._cleaner.paragraph()
        message = self._messages.setdefault(key, _Message())
        if chunk.tool_calls or getattr(chunk, "tool_call_chunks", None):
            message.tools, message.buffer = True, ""
        if message.tools:
            return
        message.buffer += chunk.text
        while "\n" in message.buffer:
            line, message.buffer = message.buffer.split("\n", 1)
            await self._line(message, line)

    async def end(self) -> None:
        """The stream has ended: the last message's final line goes out."""
        await self._flush(self._current)
        self._current = None

    async def ensure(self, final: AIMessage) -> None:
        """Sends the final answer when the stream did not, as when its tool calls were dropped."""
        if any(key[1] == final.id and message.emitted for key, message in self._messages.items()):
            return
        message = _Message()
        self._cleaner.paragraph()
        for line in final.text.split("\n"):
            await self._line(message, line)

    async def finish(self, tail: str) -> None:
        await self._open()
        if tail:
            self._emit(("\n\n" if self.chunks else "") + tail)


def final_message(messages: Sequence[AnyMessage]) -> AIMessage | None:
    """The turn's answer: the last message, when it is the model's, calls no tool and follows the user's message."""
    if not messages:
        return None
    last = messages[-1]
    if not isinstance(last, AIMessage) or last.tool_calls:
        return None
    newest = max(
        (i for i, m in enumerate(messages)
         if isinstance(m, HumanMessage) and m.additional_kwargs.get("lc_source") != SUMMARY_SOURCE),
        default=-1,
    )
    return last if newest < len(messages) - 1 else None


def turn_text(
    sources: Sequence[str], prose: str, wording: str | None, results: Sequence[Result],
    unavailable: Sequence[str] = (),
) -> str:
    """The turn as the user sees it: the sources line, the final message's prose, then code's parts."""
    parts = (sources_line(sources), clean_prose(prose), wording, appended(results, unavailable))
    return "\n\n".join(part for part in parts if part)


# The supervisor


class Supervisor:
    """Builds and runs the supervisor agent for each turn; one is shared by every request."""

    def __init__(
        self,
        *,
        capabilities: CapabilityRegistry,
        catalogue: WorkflowCatalogue,
        runner: WorkflowRunner,
        notes: NoteCatalogue,
        checkpointer: BaseCheckpointSaver[Any],
        models: Models,
        limits: SupervisorLimits | None = None,
        explore_limits: ExploreLimits | None = None,
        backoff: Backoff = DEFAULT_BACKOFF,
    ) -> None:
        self._capabilities = capabilities
        self._runner = runner
        self._notes = notes
        self._checkpointer = checkpointer
        self._models = models
        self._limits = limits or SupervisorLimits()
        self._explore_limits = explore_limits
        self._backoff = backoff
        self._workflows = latest_runnable(catalogue)
        # Built once, so the cached prefix is the same on every call.
        self._prompt = supervisor_prompt(supervisor_index(catalogue, capabilities))
        self._schemas = tool_schemas(sorted(self._workflows), capabilities)

    @property
    def prompt(self) -> str:
        return self._prompt

    def config(self, identity: Identity) -> RunnableConfig:
        # Each model turn passes a few graph steps; this leaves room for every one the limits allow.
        return {
            "configurable": {"thread_id": thread_id(identity)},
            "recursion_limit": 10 * (self._limits.model_calls + 2),
        }

    def _agent(self, choice: ModelChoice, tools: TurnTools, guard: TurnGuard, meter: UsageMeter, ledger: str) -> Any:
        summary = self._models.for_role("agent_step") or choice
        limits = self._limits
        middleware: list[AgentMiddleware[Any, Any, Any]] = [
            SummarizationMiddleware(
                summary.model,
                trigger=("tokens", limits.summarise_above_tokens),
                keep=("messages", limits.keep_messages),
            ),
            *resilience_middleware(choice, backoff=self._backoff),
            meter,
            guard,
            ContextEditingMiddleware(edits=[ClearToolUsesEdit(
                trigger=limits.clear_tool_results_above_tokens, keep=limits.keep_tool_results,
            )]),
            UnfinishedCalls(),
            LedgerMiddleware(ledger),
        ]
        return create_agent(
            choice.model,
            tools=tools.tools(),
            system_prompt=system_message(self._prompt, choice),
            middleware=middleware,
            checkpointer=self._checkpointer,
            name=AGENT_NAME,
        )

    async def turn(self, context: RunContext, build_env: EnvFactory, earthdata: EarthdataSession, message: str) -> Turn:
        """Answer ``message`` in the context's conversation."""
        try:
            job_states = await check_waiting(context, earthdata)
        except Exception as exc:
            # Unchecked jobs show as unknown in the ledger; the turn goes on.
            logger.warning("the jobs of waiting runs could not be checked: %s", type(exc).__name__)
            job_states = {}
        ledger = await build_ledger(context, job_states=job_states)
        tools = TurnTools(
            context=context, build_env=build_env, ledger=ledger, runner=self._runner, workflows=self._workflows,
            capabilities=self._capabilities, notes=self._notes, schemas=self._schemas,
            explore_limits=self._explore_limits, backoff=self._backoff, message=message,
        )
        user_id = context.identity.user_id

        async def sources() -> list[str]:
            return await data_sources(context.storage.handles, user_id, tools.results)

        stream = AnswerStream(context, sources, lambda: bool(tools.results))
        guard = TurnGuard(self._limits)
        choice = self._models.for_role("supervisor")
        meter = UsageMeter(choice) if choice is not None else None
        status: TurnStatus = "answered"
        wording: str | None = None
        prose = ""
        try:
            if choice is None or meter is None:
                status, wording = "failed", NO_MODEL_REASON.format(role="supervisor")
            else:
                status, wording, prose = await self._run(
                    self._agent(choice, tools, guard, meter, turn_context(context, ledger)), context, stream, guard,
                    message,
                )
        except asyncio.CancelledError:
            await asyncio.shield(tools.close())
            raise
        unmatched = unmatched_numbers(prose, tools.results, message)
        if unmatched:
            await self._record_numbers(tools, unmatched)
        lookup = await tools.close()
        shown = [*tools.results, *([lookup] if lookup is not None else [])]
        tail = "\n\n".join(part for part in (wording, appended(shown, tools.unavailable)) if part)
        await stream.finish(tail)
        text = turn_text(await sources(), prose, wording, shown, tools.unavailable)
        return self._turn(text, tools, guard, meter, status, lookup)

    async def _run(
        self, agent: Any, context: RunContext, stream: AnswerStream, guard: TurnGuard, message: str
    ) -> tuple[TurnStatus, str | None, str]:
        """How the turn ended, any fixed wording for that, and the final message's prose."""
        config = self.config(context.identity)
        status: TurnStatus = "answered"
        wording: str | None = None
        try:
            async for chunk, metadata in agent.astream(
                {"messages": [HumanMessage(content=message)]}, config, stream_mode="messages",
            ):
                if _from_supervisor_model(chunk, metadata):
                    await stream.feed(chunk, metadata)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            # The type name only: provider errors can carry request details.
            if exc is guard.last_model_error:
                logger.warning("every supervisor model failed: %s", type(exc).__name__)
                status, wording = "failed", FAILED_WORDING
            else:
                logger.error("the supervisor turn hit an internal error: %s", type(exc).__name__)
                status, wording = "failed", INTERNAL_WORDING
        await stream.end()
        final = None
        try:
            snapshot = await agent.aget_state(config)
            final = final_message((snapshot.values or {}).get("messages") or [])
        except Exception as exc:
            logger.error("the supervisor's saved state could not be read: %s", type(exc).__name__)
        if final is not None and status == "answered":
            await stream.ensure(final)
        if status == "answered" and guard.limited:
            status, wording = "limit", LIMIT_WORDING.format(limit=self._limits.model_calls)
        return status, wording, final.text if final is not None and status != "failed" else ""

    async def _record_numbers(self, tools: TurnTools, unmatched: list[str]) -> None:
        """One trace entry on the lookup run naming the prose's unmatched numbers; never blocks the answer."""
        try:
            env = await tools.lookup_env()
            async with env.recorder.step(kind="agent", step=NUMBERS_STEP, params={"checked": "answer prose"}) as scope:
                scope.result = {"unmatched": unmatched}
        except Exception as exc:
            logger.warning("the numbers check could not be recorded: %s", type(exc).__name__)

    def _turn(
        self,
        text: str,
        tools: TurnTools,
        guard: TurnGuard,
        meter: UsageMeter | None,
        status: TurnStatus,
        lookup: Result | None,
    ) -> Turn:
        results = tuple(tools.results)
        question = next((r.question for r in results if r.status == "needs_input" and r.question), None)
        asking = next((r for r in results if r.question is question), None) if question else None
        options = question.get("options") if question else None
        variable_choice = None
        if asking is not None and question is not None and isinstance(options, list) and options:
            variable_choice = {"run_id": asking.trace_id, "prompt": question.get("prompt"), "options": list(options)}
        run_ids = list(tools.run_ids)
        if lookup is not None:
            run_ids.append(lookup.trace_id)
        usage = meter.usage() if meter is not None else None
        return Turn(
            text=text,
            results=results,
            run_ids=tuple(dict.fromkeys(run_ids)),
            followups=tuple(dict.fromkeys(f for r in results for f in r.followups))[:MAX_FOLLOWUPS],
            unavailable=tuple(tools.unavailable),
            question=question,
            variable_choice=variable_choice,
            status=status,
            model_calls=usage.model_calls if usage else 0,
            tool_calls=guard.tool_calls,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            cache_read_tokens=usage.cache_read_tokens if usage else 0,
            cache_creation_tokens=usage.cache_creation_tokens if usage else 0,
        )
