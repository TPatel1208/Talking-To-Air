"""In-memory run-context parts for tests and benchmarks."""
from __future__ import annotations

import datetime
import itertools
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, get_args

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.exceptions import ModelRateLimitError
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from tta_backend.harness.handles.errors import HandleError
from tta_backend.harness.handles.model import HandleRecord
from tta_backend.harness.handles.payloads import PayloadStore
from tta_backend.harness.handles.resolver import OpenRetrieval
from tta_backend.harness.models import ModelChoice, ModelRole, Models
from tta_backend.harness.results import Mode
from tta_backend.harness.run_context import (
    AnsweredRequests,
    ArtifactRef,
    Artifacts,
    Clock,
    Compute,
    Credentials,
    EarthdataSession,
    Events,
    Identity,
    Limits,
    PlaceRefused,
    Places,
    Reachable,
    Region,
    RunContext,
    RunEnv,
    all_reachable,
    run_env,
)
from tta_backend.harness.storage import Stores, memory_stores
from tta_backend.harness.trace import Run, new_run_id


class RecordingEvents:
    def __init__(self) -> None:
        self.progress_events: list[dict[str, Any]] = []
        self.charts: list[dict[str, Any]] = []
        self.questions: list[dict[str, Any]] = []
        self.texts: list[str] = []

    def progress(
        self, message: str, *, stage: str | None = None, detail: Any = None, run_id: str | None = None
    ) -> None:
        self.progress_events.append({"message": message, "stage": stage, "detail": detail, "run_id": run_id})

    def chart(self, payload: dict[str, Any]) -> None:
        self.charts.append(payload)

    def question(self, payload: dict[str, Any]) -> None:
        self.questions.append(payload)

    def text(self, content: str) -> None:
        self.texts.append(content)


class MemoryArtifacts:
    def __init__(self) -> None:
        self.stored: dict[str, dict[str, Any]] = {}

    def _mint(self, kind: str) -> str:
        return f"{kind}-{len(self.stored) + 1}"

    async def store_chart(self, payload: dict[str, Any]) -> ArtifactRef:
        ref = ArtifactRef(id=self._mint("chart"), type="chart", title=payload.get("title"))
        self.stored[ref.id] = payload
        return ref

    async def store_table(
        self,
        title: str,
        columns: list[str],
        rows: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        ref = ArtifactRef(id=self._mint("table"), type="table", title=title)
        self.stored[ref.id] = {
            "title": title,
            "columns": columns,
            "rows": rows,
            "metadata": metadata,
        }
        return ref


class StaticCredentials(dict[str, str]):
    async def secret(self, connector: str) -> str | None:
        return self.get(connector)


class FixedClock:
    def __init__(self, now: datetime.datetime) -> None:
        if now.tzinfo is None:
            raise ValueError("FixedClock needs a timezone-aware datetime")
        self._now = now

    def today(self) -> datetime.date:
        return self._now.date()

    def now(self) -> datetime.datetime:
        return self._now


class FakeToolError(Exception):
    """A classified source failure, shaped like the earthdata errors: a string ``category``."""

    def __init__(self, category: str, message: str = "") -> None:
        super().__init__(message)
        self.category = category


class FakeEarthdata:
    """Scripted earthdata tools: each maps arguments to a result, or raises.

    Keyed calls behave as the host's session does: a repeat of an answered
    key and request returns the first answer without running the tool again,
    and the same key with a different request is refused as a contract failure.
    """

    def __init__(self, tools: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] | None = None) -> None:
        self.tools = dict(tools or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.keys: list[str | None] = []
        self.answered = AnsweredRequests()

    async def call(
        self, tool: str, arguments: dict[str, Any], *, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        if idempotency_key is not None:
            answer = self.answered.get(idempotency_key, tool, arguments)
            if answer is not None:
                return answer
        self.calls.append((tool, dict(arguments)))
        self.keys.append(idempotency_key)
        if tool not in self.tools:
            raise FakeToolError("contract", f"unscripted tool {tool}")
        result = self.tools[tool](arguments)
        if idempotency_key is not None:
            self.answered.put(idempotency_key, tool, arguments, result)
        return result


class FakePlaces:
    """Resolves the places it was given, matched case-insensitively; any other wording resolves to nothing.

    Wording listed in ``refusals`` raises that PlaceRefused instead.
    """

    def __init__(
        self, regions: dict[str, Region] | None = None, refusals: dict[str, PlaceRefused] | None = None
    ) -> None:
        self.regions = {name.lower(): region for name, region in (regions or {}).items()}
        self.refusals = {name.lower(): refused for name, refused in (refusals or {}).items()}
        self.asked: list[str] = []

    async def resolve(self, wording: str) -> Region | None:
        self.asked.append(wording)
        key = " ".join(wording.lower().split())
        if key in self.refusals:
            raise self.refusals[key]
        return self.regions.get(key)


def box_region(name: str, west: float, south: float, east: float, north: float, **kw: Any) -> Region:
    """A rectangular region, for tests."""
    ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    return Region(
        name=name, bounds=(west, south, east, north),
        geometry={"type": "Polygon", "coordinates": [ring]}, **kw,
    )


class InlineCompute:
    """Runs the work on the calling thread, so tests stay deterministic."""

    def __init__(self) -> None:
        self.runs = 0

    async def run[**A, T](self, fn: Callable[A, T], /, *args: A.args, **kwargs: A.kwargs) -> T:
        self.runs += 1
        return fn(*args, **kwargs)


TEST_LIMITS = Limits(
    retrieval_soft_bytes=50_000_000,
    retrieval_hard_bytes=200_000_000,
    max_point_timeseries_days=366,
    retrieval_timeout_seconds=60.0,
    poll_min_seconds=0.01,
    poll_max_seconds=0.1,
)


_CALL_IDS = itertools.count(1)


def _usage(usage: tuple[int, int] | None) -> UsageMetadata | None:
    if usage is None:
        return None
    tokens_in, tokens_out = usage
    return UsageMetadata(input_tokens=tokens_in, output_tokens=tokens_out, total_tokens=tokens_in + tokens_out)


def answer(text: str, *, usage: tuple[int, int] | None = None) -> AIMessage:
    """A scripted final answer; ``usage`` is (input tokens, output tokens)."""
    return AIMessage(content=text, usage_metadata=_usage(usage))


def tool_call(
    name: str,
    args: dict[str, Any],
    *,
    call_id: str | None = None,
    content: str = "",
    usage: tuple[int, int] | None = None,
) -> AIMessage:
    """A scripted turn that calls tool ``name``. A structured answer under
    ``ToolStrategy`` is a call to the tool named after its schema."""
    return AIMessage(
        content=content,
        tool_calls=[{"name": name, "args": args, "id": call_id or f"call_{next(_CALL_IDS)}", "type": "tool_call"}],
        usage_metadata=_usage(usage),
    )


def rate_limited() -> ModelRateLimitError:
    """A scripted rate-limit failure, the kind a retry repeats."""
    return ModelRateLimitError("rate limited (scripted)")


class ScriptExhausted(AssertionError):
    """The scripted model was called more times than it has turns."""


class ScriptedChatModel(BaseChatModel):
    """A chat model that answers from a script, for agents under test.

    Each call takes the next turn: an AIMessage is returned (a copy), an
    exception is raised. ``received`` holds the messages each call was sent,
    ``tools_offered`` the tool names bound for it and ``tool_choices`` the
    tool choice it was bound with.
    """

    script: list[Any] = Field(default_factory=list)
    received: list[list[BaseMessage]] = Field(default_factory=list)
    tools_offered: list[list[str]] = Field(default_factory=list)
    tool_choices: list[Any] = Field(default_factory=list)
    position: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    @property
    def calls(self) -> int:
        return len(self.received)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        names = [convert_to_openai_tool(tool)["function"]["name"] for tool in tools]
        return self.bind(tool_names=names, tool_choice=tool_choice, **kwargs)

    def _turn(self, messages: list[BaseMessage], kwargs: dict[str, Any]) -> ChatResult:
        self.received.append(list(messages))
        self.tools_offered.append(list(kwargs.get("tool_names", [])))
        self.tool_choices.append(kwargs.get("tool_choice"))
        if self.position >= len(self.script):
            raise ScriptExhausted(f"the script has {len(self.script)} turns")
        turn = self.script[self.position]
        self.position += 1
        if isinstance(turn, BaseException):
            raise turn
        if not isinstance(turn, AIMessage):
            raise TypeError("a script turn is an AIMessage or an exception")
        return ChatResult(generations=[ChatGeneration(message=turn.model_copy(deep=True))])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._turn(messages, kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._turn(messages, kwargs)


class ScriptedModels:
    """Models keyed by role; a role not given gets ``default``. ``asked`` records each role asked for."""

    def __init__(
        self, choices: Mapping[ModelRole, ModelChoice] | None = None, *, default: ModelChoice | None = None
    ) -> None:
        self.choices = dict(choices or {})
        self.default = default
        self.asked: list[ModelRole] = []

    @classmethod
    def every_role(cls, choice: ModelChoice) -> ScriptedModels:
        return cls({role: choice for role in get_args(ModelRole)})

    def for_role(self, role: ModelRole) -> ModelChoice | None:
        self.asked.append(role)
        return self.choices.get(role, self.default)


async def _no_retrievals(user_id: str, record: HandleRecord) -> Any:
    raise HandleError("not_found", record.id)


async def make_run_env(
    payload_root: str | os.PathLike[str],
    *,
    context: RunContext | None = None,
    earthdata: EarthdataSession | None = None,
    compute: Compute | None = None,
    limits: Limits | None = None,
    open_retrieval: OpenRetrieval | None = None,
    places: Places | None = None,
    models: Models | None = None,
    mode: Mode | None = None,
) -> RunEnv:
    """A run started in the context's trace store, with an environment on fakes."""
    context = context or make_run_context()
    run = Run(
        id=new_run_id(),
        user_id=context.identity.user_id,
        conversation_id=context.identity.conversation_id,
        mode=mode or Mode.exploratory(),
        started=context.clock.now(),
    )
    await context.storage.traces.start_run(run)
    return run_env(
        context,
        run_id=run.id,
        payloads=PayloadStore(payload_root),
        open_retrieval=open_retrieval or _no_retrievals,
        earthdata=earthdata or FakeEarthdata(),
        compute=compute or InlineCompute(),
        limits=limits or TEST_LIMITS,
        places=places or FakePlaces(),
        models=models,
    )


def env_factory(
    payload_root: str | os.PathLike[str],
    *,
    earthdata: EarthdataSession | None = None,
    compute: Compute | None = None,
    limits: Limits | None = None,
    open_retrieval: OpenRetrieval | None = None,
    places: Places | None = None,
    models: Models | None = None,
) -> Callable[[RunContext, str], Awaitable[RunEnv]]:
    """Builds the environment for a run already started, as a workflow runner asks for it."""
    shared_earthdata = earthdata or FakeEarthdata()
    shared_compute = compute or InlineCompute()
    shared_places = places or FakePlaces()

    async def build(context: RunContext, run_id: str) -> RunEnv:
        return run_env(
            context,
            run_id=run_id,
            payloads=PayloadStore(payload_root),
            open_retrieval=open_retrieval or _no_retrievals,
            earthdata=shared_earthdata,
            compute=shared_compute,
            limits=limits or TEST_LIMITS,
            places=shared_places,
            models=models,
        )

    return build


def make_run_context(
    *,
    user_id: str = "user-a",
    conversation_id: str = "conv-1",
    events: Events | None = None,
    artifacts: Artifacts | None = None,
    credentials: Credentials | None = None,
    clock: Clock | None = None,
    stores: Stores | None = None,
    reachable: Reachable | None = None,
) -> RunContext:
    """A run context on fakes. Pass one ``stores`` to several contexts to test users against each other.
    Every connector is reachable unless ``reachable`` says otherwise."""
    return RunContext(
        identity=Identity(user_id=user_id, conversation_id=conversation_id),
        events=events or RecordingEvents(),
        artifacts=artifacts or MemoryArtifacts(),
        credentials=credentials or StaticCredentials(),
        storage=stores or memory_stores(),
        clock=clock or FixedClock(datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)),
        reachable=reachable or all_reachable,
    )
