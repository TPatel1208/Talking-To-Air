"""
subagent_dispatch.py
---------------------
Shared sub-agent invocation logic for the ground sensor and earthdata agents.

Both the supervisor's ``ask_ground_sensor_agent``/``ask_earthdata_agent`` tool
wrappers (agents/supervisor_agent.py) and the router fast path
(services/chat_stream_service.py, T14) call into ``run_ground``/
``run_satellite`` here — context enrichment, envelope finalization, the
per-request call budget, and metrics recording are identical regardless of
who initiated the call.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from config.error_templates import (
    CATEGORY_RATE_LIMITED,
    CATEGORY_RECURSION_EXHAUSTED,
    render_error_answer,
    render_scope_note,
)
from config.model_factory import structured_output
from earthdata_mcp.connection import STATE_READY
from earthdata_mcp.results import (
    CATEGORY_CONTRACT,
    CATEGORY_PROVIDER_UNAVAILABLE,
    CATEGORY_USER_INPUT,
    MCPToolError,
    parse_tool_result,
)
from models import AgentResult, SubAgentEnvelope, parse_agent_result, parse_chart_payload, parse_sub_agent_envelope
from models.artifact import ArtifactReference
from repositories.session_metadata_repository import (
    get_ground_monitor_context,
    get_satellite_context,
    save_ground_monitor_context,
    save_satellite_context,
)
from tools import GROUND_TOOLS
from tools.satellite_tools.factory import sanctioned_tool_names
from utils.message_utils import extract_last_text, truncate_text
from utils.metrics import record_agent_request, record_envelope_salvaged
from utils.streaming import get_call_budget, stream_response

logger = logging.getLogger(__name__)

OnEvent = Callable[[str, Any], Awaitable[None]]


async def run_ground(
    ground_agent: Any,
    task: str,
    conversation_thread_id: str | None = None,
) -> AgentResult:
    """
    Invoke the ground sensor agent once for ``task``, budget-checked,
    context-enriched from the thread's persisted monitor context, and
    envelope-finalized.

    ``conversation_thread_id`` keys the cross-turn monitor context in
    per-thread persisted metadata (session_metadata_repository) — shared by
    every caller on that thread, whether it took the supervisor path or the
    fast path.
    """
    budget = get_call_budget()
    count = budget.get("ground", 0)
    if count >= 1:
        logger.warning(
            "agent_budget_exceeded",
            extra={
                "_event": "agent_budget_exceeded",
                "_agent_type": "ground_sensor",
                "_call_count": count,
            },
        )
        return AgentResult(text=(
            "The worker has already returned a result. "
            "Do not call the supervisor_agent again. "
            "Synthesize your answer from the result already received."
        ))
    budget["ground"] = count + 1

    monitor_context = await get_ground_monitor_context(conversation_thread_id) if conversation_thread_id else {}

    async def _invoke(task_text: str) -> AgentResult:
        sub_thread_id = str(uuid.uuid4())
        outcome = "success"
        try:
            from langchain_core.messages import HumanMessage

            result = await ground_agent.ainvoke(
                {"messages": [HumanMessage(content=task_text)]},
                config={"configurable": {"thread_id": sub_thread_id}},
            )
            # Untruncated — _finalize_sub_agent_result parses the
            # {summary, artifact_ids, handles} envelope out of this
            # before any length limit is applied (T11).
            text = extract_last_text(
                result,
                "Ground sensor agent returned no response.",
                agent_name="ground_sensor",
                truncate=False,
            )
            artifact_refs = _extract_artifact_refs(result.get("messages", []))
        except TimeoutError:
            outcome = "timeout"
            raise
        except Exception as exc:
            outcome = "failure"
            # T37: taxonomy answer, never a bare exception string (see
            # _render_unexpected_exception).
            text = _render_unexpected_exception(exc, "ground sensor agent")
            artifact_refs = []
        finally:
            record_agent_request("ground_sensor", outcome)
        return AgentResult(text=text, artifacts=artifact_refs)

    enriched_task = _inject_ground_context(task, monitor_context)
    result = await _invoke(enriched_task)
    if _is_ground_tool_failure(result.text):
        logger.warning(
            "llm_tool_call_refusal",
            extra={
                "_event": "llm_tool_call_refusal",
                "_agent_type": "ground_sensor",
            },
        )
        retry_task = _ground_retry_task(enriched_task)
        result = await _reprompt_final_envelope(ground_agent, retry_task, "ground_sensor")

    result = _finalize_sub_agent_result(result, "ground sensor")

    new_context = _extract_ground_monitor_context(result.text)
    if new_context and conversation_thread_id:
        await save_ground_monitor_context(conversation_thread_id, {**monitor_context, **new_context})

    return result


async def run_satellite(
    satellite_agent: Any,
    task: str,
    conversation_thread_id: str | None = None,
    on_event: OnEvent | None = None,
    mcp_manager: Any = None,
) -> AgentResult:
    """
    Invoke the earthdata agent once for ``task``, budget-checked and
    envelope-finalized.

    ``on_event``, if given, is awaited with every ``(event_type, data)`` tuple
    the sub-agent's own stream produces — the router fast path (T14) uses
    this to forward tool_call/status/job_progress/chart_payload events to the
    SSE stream live instead of buffering them until the turn finishes.

    ``mcp_manager``, if given, gates dispatch on the earthdata-retrieval MCP
    connection manager's state (T17 story #13): when it isn't ``ready``,
    ``satellite_agent`` is never touched — the deterministic unavailable
    answer is returned immediately, at zero model-call cost. ``None`` (the
    default) preserves prior behavior for every existing caller that doesn't
    pass one.
    """
    if mcp_manager is not None and mcp_manager.state != STATE_READY:
        logger.info(
            "satellite_dispatch_skipped_mcp_not_ready",
            extra={
                "_event": "satellite_dispatch_skipped_mcp_not_ready",
                "_mcp_state": mcp_manager.state,
            },
        )
        return AgentResult(
            text=render_error_answer(
                CATEGORY_PROVIDER_UNAVAILABLE,
                "satellite data layer",
                "Ground/EPA air-quality questions still work.",
            ),
            metadata={"data_layer": mcp_manager.state},
        )

    budget = get_call_budget()
    count = budget.get("satellite", 0)
    if count >= 1:
        logger.warning(
            "agent_budget_exceeded",
            extra={
                "_event": "agent_budget_exceeded",
                "_agent_type": "satellite",
                "_call_count": count,
            },
        )
        return AgentResult(text=(
            "[STOP] Satellite agent has already been called for this request — "
            "this call is blocked. Do NOT call ask_earthdata_agent again. "
            "Synthesize your answer from the result already received."
        ))
    budget["satellite"] = count + 1

    # Cross-turn retrieval context (dataset/AOI/handles the last satellite turn
    # on this thread worked with). The fast path writes only the prose answer
    # back into the supervisor's thread, so a follow-up that drops to the
    # supervisor ("pick a date in that range") otherwise arrives with no
    # concrete dataset/AOI/handles to continue from — the earthdata agent then
    # re-derives from the paraphrased answer and can confabulate. Injected here
    # as UNVERIFIED context (never an availability verdict) so the agent reuses
    # the handles but still re-checks coverage.
    satellite_context = await _load_satellite_context(conversation_thread_id)
    enriched_task = _inject_satellite_context(_current_date_preamble() + task, satellite_context)
    captured_context: dict[str, str] = {}
    # T46: the first define_area_of_interest user_input rejection seen this
    # turn. Its presence alongside a delivered chart means the agent improvised
    # a substitute region instead of relaying the rejection — the no-
    # substitution guard below refuses that answer deterministically.
    aoi_rejection: dict[str, MCPToolError] = {}
    # T49: the deterministic variable-choice picker a tool emitted out-of-band
    # (emit_variable_choice) when the T48 resolver couldn't confidently choose.
    # Kept last-wins; attached to the finalized answer below with each
    # candidate's auto-send prompt filled from the ORIGINAL request (task).
    variable_choice_box: dict[str, dict] = {}

    async def _invoke(task_text: str) -> AgentResult:
        charts = []
        artifacts = []
        text_parts = []
        sub_thread_id = str(uuid.uuid4())
        outcome = "success"

        try:
            async for event_type, data in stream_response(
                satellite_agent, task_text, thread_id=sub_thread_id
            ):
                if on_event is not None:
                    await on_event(event_type, data)
                if event_type == "tool_call":
                    _capture_satellite_context(data.get("name"), data.get("args"), captured_context)
                    continue
                if event_type == "variable_choice":
                    # T49: the deterministic picker rides out-of-band, exactly
                    # like a chart_payload — captured here, never parsed from the
                    # model's prose, so the candidate list is a guarantee.
                    if isinstance(data, dict):
                        variable_choice_box["value"] = data
                    continue
                if event_type == "chart_payload":
                    # T13: plot/statistics tools emit the full render
                    # payload out-of-band via emit_chart and return the
                    # model only a compact summary — harvest charts from
                    # this event, not from tool_result content.
                    chart = parse_chart_payload(data)
                    if chart is not None:
                        charts.append(chart)
                    # Carry every chart id across turns so a later reliability
                    # question ("why trust this?") can name one to
                    # explain_measurement — the satellite agent is stateless
                    # (fresh thread each dispatch), so without this the chart ids
                    # from a prior turn are unrecoverable and P3 can't ground.
                    _capture_chart_id(data, captured_context)
                    continue
                if event_type == "tool_result":
                    content = data.get("content", "")
                    _capture_aoi_rejection(data.get("name"), content, aoi_rejection)
                    artifacts.extend(_artifact_refs_from_content(content))
                    nested = parse_agent_result(content)
                    if nested is not None:
                        text_parts.append(nested.text)
                        charts.extend(nested.charts)
                        artifacts.extend(nested.artifacts)
                elif event_type in ("text", "done"):
                    t = data if isinstance(data, str) else data.get("response", "")
                    if t:
                        text_parts.append(t)
        except TimeoutError as exc:
            outcome = "timeout"
            text_parts = [str(exc)]
        except Exception as exc:
            outcome = "failure"
            if exc.__class__.__name__ == "HarmonyTimeoutError":
                outcome = "timeout"
            # T37: an arbitrary exception string must never become the
            # sub-agent's "answer" the supervisor may dress up — render the
            # taxonomy's honest error answer instead (classified category
            # when it's an MCPToolError, contract otherwise); the real
            # exception goes to the logs.
            text_parts = [_render_unexpected_exception(exc, "earthdata agent")]
        finally:
            record_agent_request("satellite", outcome)

        # Joined with "" (not " ") — text_parts holds successive streamed
        # deltas of one final message, and this string must round-trip
        # through parse_sub_agent_envelope as valid JSON; a space
        # injected between two delta chunks would corrupt it. Left
        # untruncated here — _finalize_sub_agent_result parses the
        # envelope before any length limit is applied (T11).
        text = "".join(text_parts) or "Earthdata agent returned no response."
        return AgentResult(text=text, charts=charts, artifacts=artifacts)

    result = await _invoke(enriched_task)
    refusal_markers = (
        "necessary tools are not present",
        "don't have access to fetch_environmental_data",
        "do not have access to fetch_environmental_data",
        "failed to call a function",
        "failed_generation",
    )
    if any(marker in result.text.lower() for marker in refusal_markers):
        logger.warning(
            "llm_tool_call_refusal",
            extra={
                "_event": "llm_tool_call_refusal",
                "_agent_type": "satellite",
            },
        )
        result = await _reprompt_final_envelope(satellite_agent, _satellite_retry_task(enriched_task), "satellite")

    finalized = _finalize_sub_agent_result(result, "earthdata")
    finalized = _guard_aoi_substitution(finalized, aoi_rejection.get("error"))
    finalized = _attach_variable_choice(finalized, variable_choice_box.get("value"), task)
    if captured_context and conversation_thread_id:
        await _persist_satellite_context(
            conversation_thread_id, {**satellite_context, **captured_context}
        )
    return finalized


# ── Helpers ───────────────────────────────────────────────────────────────────

_GROUND_TOOL_FAILURE_MARKERS = (
    "failed to call a function",
    "failed_generation",
    "tool call validation failed",
    "parameters for tool",
    "did not match schema",
    "please adjust your prompt",
)

# Derived from GROUND_TOOLS/sanctioned_tool_names (the actual registered
# toolsets) rather than hand-maintained, so a retry can never name a tool
# that has left the surface.
_GROUND_RETRY_TOOL_GUIDANCE = (
    "The ground sensor tools are registered and available in this runtime: "
    f"{', '.join(t.name for t in GROUND_TOOLS)}. Retry using valid tool arguments. "
    "For by-site summaries, pass either station_id as site_number like "
    "'34-023-0011' or split it into state_code='34', county_code='023', "
    "site_number='0011'. Always pass pollutant_standard exactly as one of: "
    "'NO2 1-hour 2010', 'PM25 24-hour 2024', 'Ozone 8-hour 2015', "
    "'SO2 1-hour 2010', 'CO 8-hour 1971'. Use integer k values."
)


# Message-shape signals for a language-model provider rate-limit/quota failure
# (Gemini free-tier: "429 Too Many Requests" / "RESOURCE_EXHAUSTED" / quota
# prose). Matched on text, not a provider-specific exception class, so this
# stays correct across providers (google/groq/openai) and across the wrapping
# layers (langchain re-raises the provider error under its own class) without
# importing any of them. Live-observed in the 2026-07-20 AOD session logs.
_RATE_LIMIT_SIGNALS: tuple[str, ...] = (
    "429",
    "too many requests",
    "resource_exhausted",
    "rate limit",
    "ratelimit",
    "quota",
)


def _classify_escaped_exception(exc: Exception) -> str:
    """Category for a non-MCPToolError exception that escaped a sub-agent turn.
    Two LLM/graph-layer failures get their own honest answer instead of the
    generic contract "internal error": a LangGraph recursion-limit stop
    (matched by class name, mirroring the HarmonyTimeoutError pattern above, so
    langgraph need not be imported here) and a provider rate-limit/quota
    (matched by message shape). Everything else stays ``contract``."""
    if exc.__class__.__name__ == "GraphRecursionError":
        return CATEGORY_RECURSION_EXHAUSTED
    text = f"{exc.__class__.__name__}: {exc}".lower()
    if any(signal in text for signal in _RATE_LIMIT_SIGNALS):
        return CATEGORY_RATE_LIMITED
    return CATEGORY_CONTRACT


def _render_unexpected_exception(exc: Exception, stage: str) -> str:
    """T37: the honest taxonomy answer for an exception that escaped a
    sub-agent turn. A classified MCPToolError keeps its own category and
    message; a recursion-limit stop or a provider rate-limit gets its own
    actionable category; anything else is a contract failure whose text
    (possibly empty, possibly an internal path) never reaches the researcher —
    the real exception goes to the logs here."""
    if isinstance(exc, MCPToolError):
        return render_error_answer(exc.category, stage, exc.message)
    category = _classify_escaped_exception(exc)
    logger.exception(
        "subagent_unexpected_exception",
        extra={"_event": "subagent_unexpected_exception", "_stage": stage, "_category": category},
        exc_info=exc,
    )
    return render_error_answer(category, stage)


def _finalize_sub_agent_result(result: AgentResult, agent_label: str) -> AgentResult:
    """
    Validate a sub-agent's final message against the {summary, artifact_ids,
    handles} envelope contract. A missing/invalid envelope on its own is no
    longer fatal (T15) — it is salvaged from the raw prose, since the work
    collected from the tool stream during the turn (charts, artifacts) was
    never contingent on the final message parsing cleanly.

    ``result.text`` must be the untruncated final message — envelope parsing
    happens first; truncation applies afterwards, to the extracted summary
    only, so a compliant answer is never destroyed by a length limit fired
    before it was ever parsed (T11).
    """
    envelope = parse_sub_agent_envelope(result.text)
    if envelope is None:
        return _append_disclosures(_salvage_sub_agent_result(result, agent_label))
    discovered = {ref.id: ref for ref in result.artifacts}
    artifacts = [discovered[artifact_id] for artifact_id in envelope.artifact_ids if artifact_id in discovered]
    return _append_disclosures(AgentResult(
        text=truncate_text(envelope.summary, 2000, agent_name=agent_label),
        charts=result.charts,
        artifacts=artifacts,
        handles=envelope.handles,
        suggested_followups=envelope.suggested_followups,
    ))


def _append_disclosures(result: AgentResult) -> AgentResult:
    """Apply every deterministic, provenance-driven disclosure the answer owes
    the researcher: the T48 variable-resolution note (which product was picked)
    and the T46 scope-substitution note (what scope was actually served). Both
    are templated from chart provenance, never model prose."""
    return _append_scope_note(_append_variable_note(result))


def _capture_aoi_rejection(name: Any, content: Any, into: dict[str, MCPToolError]) -> None:
    """Record the first define_area_of_interest user_input rejection seen in a
    turn's tool stream (T46). The rejection travels as bind_workspace's own
    error envelope in the tool_result content, which parse_tool_result
    re-raises as a typed MCPToolError. Non-AOI tools, non-error results, and
    non-user_input errors (a transient provider outage isn't the researcher's
    fault) are ignored; only the first rejection is kept."""
    if name != "define_area_of_interest" or into.get("error") is not None:
        return
    try:
        parse_tool_result(content)
    except MCPToolError as exc:
        if exc.category == CATEGORY_USER_INPUT:
            into["error"] = exc
    except Exception:
        pass


def _guard_aoi_substitution(result: AgentResult, rejection: MCPToolError | None) -> AgentResult:
    """T46 story #1: when define_area_of_interest rejected the requested area
    with a user_input error but the turn still delivered a chart, the agent
    improvised a substitute region instead of relaying the rejection — the
    live 2026-07-17 incident answered an impossible bbox with a confident
    'North America' map. Replace the answer with the deterministic T18 error
    relay (the exact mechanism as salvage — no model in the loop) and drop the
    wrong-region chart, so a researcher never receives a confident map of a
    region they did not ask about. A rejection with no delivered chart (the
    agent relayed it and asked a clarifying question) is left untouched."""
    if rejection is None or not result.charts:
        return result
    detail = rejection.message
    if rejection.suggestion:
        detail = f"{detail.rstrip('.')}. {rejection.suggestion}"
    return AgentResult(
        text=render_error_answer(CATEGORY_USER_INPUT, "area-of-interest request", detail),
        metadata={"aoi_substitution_refused": True},
    )


def _chart_provenance(chart: Any) -> dict:
    """The provenance dict off a finalized chart, whether it survived as a
    ChartPayload (extra='allow', provenance is an attribute) or a plain
    dict."""
    if isinstance(chart, dict):
        prov = chart.get("provenance")
    else:
        prov = getattr(chart, "provenance", None)
    return prov if isinstance(prov, dict) else {}


def _attach_variable_choice(
    result: AgentResult, payload: dict | None, original_request: str,
) -> AgentResult:
    """T49: attach the deterministic variable-choice picker to the finalized
    answer, reconstructing each candidate's auto-send prompt from the ORIGINAL
    request text (never the date/context-enriched task the sub-agent actually
    ran, so no injected preamble leaks into the message a click would resend).

    Two sources feed one attach point:
    - LOW confidence: ``payload`` is the picker a tool emitted out-of-band
      (emit_variable_choice) in place of a chart.
    - MEDIUM confidence: the answer arrived WITH a chart; its provenance
      carries the picker (stashed by AggregationService._resolution_facts) so
      the guess can be overridden in one click, without blocking the answer.

    The out-of-band payload wins when both exist. A no-op when neither source
    has a picker. Failure to parse is non-fatal — the answer still stands."""
    if result.variable_choice is not None:
        return result
    payload = payload or _medium_variable_choice_from_charts(result)
    if not payload:
        return result
    from models import VariableChoice
    from preprocessing.variable_choice_builder import fill_prompts

    try:
        choice = VariableChoice.model_validate(payload)
    except Exception:
        logger.warning("variable_choice_parse_failed", exc_info=True)
        return result
    result.variable_choice = fill_prompts(choice, original_request)
    return result


def _medium_variable_choice_from_charts(result: AgentResult) -> dict | None:
    """The medium-confidence picker stashed in a delivered chart's provenance
    (variable_resolution.variable_choice), or None. This is how a medium
    auto-pick attaches its override picker without a separate out-of-band
    emit — the resolver already rode its facts out on the chart."""
    for chart in result.charts:
        resolution = _chart_provenance(chart).get("variable_resolution")
        if isinstance(resolution, dict):
            picker = resolution.get("variable_choice")
            if isinstance(picker, dict) and picker.get("candidates"):
                return picker
    return None


def _append_variable_note(result: AgentResult) -> AgentResult:
    """T48: append the VariableResolver's deterministic disclosure to a
    finalized answer when a chart's provenance records that the resolver
    auto-picked a variable from a wide, ambiguous product. The disclosure is
    built by the resolver from its own facts (chosen field + ranked
    alternatives) -- never model prose -- so a scientific choice made on the
    researcher's behalf can't be paraphrased away. A no-op when no resolver
    disclosure was recorded (an explicit/registry/single-var pick)."""
    for chart in result.charts:
        provenance = _chart_provenance(chart)
        resolution = provenance.get("variable_resolution")
        note = (resolution or {}).get("disclosure") if isinstance(resolution, dict) else None
        if note:
            result.text = f"{result.text}\n\n{note}" if result.text else note
            return result
    return result


def _append_scope_note(result: AgentResult) -> AgentResult:
    """T46: append the deterministic scope-substitution disclosure to a
    finalized answer when any chart's provenance records a requested scope
    that differs materially from what was delivered. Rendered by the T18/T37
    template machinery (render_scope_note) from provenance facts — never
    composed by the model, so an inconvenient correction can't be paraphrased
    away. A no-op when nothing was substituted (don't nag on exact matches)."""
    for chart in result.charts:
        provenance = _chart_provenance(chart)
        note = render_scope_note(provenance.get("requested_scope"), provenance.get("delivered_scope"))
        if note:
            result.text = f"{result.text}\n\n{note}" if result.text else note
            return result
    return result


def _salvage_sub_agent_result(result: AgentResult, agent_label: str) -> AgentResult:
    """
    Policy for a final message that failed envelope parsing (T15). When
    there is prose, the prose speaks for itself: it becomes the summary,
    artifacts/charts already collected from the tool stream are attached,
    and handles named in those artifacts' metadata are carried through.
    Salvage never invents a cause for the failure — when there is nothing
    to salvage (empty/whitespace text), this states the observed fact and
    stops rather than handing the supervisor a free-text error it could
    dress up as an explanation of the researcher's question.
    """
    prose = (result.text or "").strip()
    if not prose:
        return AgentResult(
            text=render_error_answer(
                CATEGORY_CONTRACT,
                f"{agent_label} agent",
                "Its final message did not parse and contained no text.",
            ),
            charts=result.charts,
            metadata={"error": "invalid_envelope"},
        )

    preview = truncate_text(prose, 300, agent_name=agent_label)
    logger.warning(
        "envelope_salvaged",
        extra={
            "_event": "envelope_salvaged",
            "_agent_type": agent_label,
            "_raw_preview": preview,
            "_artifact_count": len(result.artifacts),
        },
    )
    record_envelope_salvaged(agent_label)
    return AgentResult(
        text=truncate_text(prose, 2000, agent_name=agent_label),
        charts=result.charts,
        artifacts=result.artifacts,
        handles=_handles_from_artifacts(result.artifacts),
        metadata={"salvaged": True, "raw_preview": preview},
    )


def _handles_from_artifacts(artifacts: list[ArtifactReference]) -> list[str]:
    """Handles named in collected artifacts' metadata (e.g. a map's
    source_handles) — attached unambiguously since they come straight from
    the tool results, not from the unparsed prose."""
    handles: list[str] = []
    for artifact in artifacts:
        for handle in artifact.metadata.get("source_handles") or []:
            if handle not in handles:
                handles.append(handle)
    return handles


async def _reprompt_final_envelope(agent: Any, task_text: str, metric_agent_type: str) -> AgentResult:
    """
    T15 retry demotion. When a sub-agent wrongly claims its tools are
    missing, recovery used to be a full second tool-workflow run — doubling
    the slowest path exactly when the provider is already throttling.
    Instead, make exactly one structured-output call (routed through the
    model factory's provider-aware ``structured_output`` hook, T12) asking
    for the final envelope directly — never a second tool-workflow run.

    ``agent`` is the compiled sub-agent graph; ``subagent_model`` is the raw
    chat model attached to it at build time (agents/earthdata_agent.py,
    agents/ground_sensor_agent.py) since these agents are stateless (no
    checkpointer) and hold no other handle to it.
    """
    model = getattr(agent, "subagent_model", None)
    if model is None:
        record_agent_request(metric_agent_type, "failure")
        return AgentResult(text="")

    outcome = "success"
    try:
        bound = structured_output(model, SubAgentEnvelope)
        envelope = await bound.ainvoke(task_text)
    except Exception:
        outcome = "failure"
        return AgentResult(text="")
    else:
        if not isinstance(envelope, SubAgentEnvelope):
            outcome = "failure"
            return AgentResult(text="")
        return AgentResult(text=envelope.model_dump_json())
    finally:
        record_agent_request(metric_agent_type, outcome)


def _is_ground_tool_failure(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in _GROUND_TOOL_FAILURE_MARKERS)


def _ground_retry_task(task: str) -> str:
    return f"{_GROUND_RETRY_TOOL_GUIDANCE}\n\nTask: {task}"


def _satellite_retry_task(task: str) -> str:
    # Derived from sanctioned_tool_names() (the actual built toolset) rather
    # than hand-maintained, so a retry can never name a tool that has left
    # the model-facing surface.
    guidance = (
        "The satellite tools are registered and available in this runtime: "
        f"{', '.join(sanctioned_tool_names())}. Retry the task using those tools exactly as needed."
    )
    return f"{guidance}\n\nTask: {task}"


def _inject_ground_context(task: str, context: dict[str, str]) -> str:
    if not context:
        return task

    bits = []
    if context.get("name"):
        bits.append(f"monitor_name={context['name']}")
    if context.get("site_id"):
        bits.append(f"station_id={context['site_id']}")
    if context.get("latitude") and context.get("longitude"):
        bits.append(f"coordinates=({context['latitude']}, {context['longitude']})")

    if not bits:
        return task
    return "Prior ground monitor context: " + "; ".join(bits) + ".\n\n" + task


def _extract_ground_monitor_context(text: str) -> dict[str, str]:
    text = str(text or "")
    station_match = re.search(r"\b(?:station_id|site_id)\s*[:=]?\s*([0-9]{2}-[0-9]{3}-[0-9]{4})\b", text, re.I)
    coord_match = re.search(
        r"(?:coordinates|located at coordinates)?\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?",
        text,
        re.I,
    )
    name_match = re.search(
        r"(?:monitor(?:\s+name)?|station(?:\s+name)?)\s*(?:is|:)\s*"
        r"(.+?)(?=\s+with\s+(?:station_id|site_id)|\s+located\s+at\s+coordinates|[.\n]|$)",
        text,
        re.I,
    )

    context: dict[str, str] = {}
    if name_match:
        name = name_match.group(1).strip(" '\"")
        if name and not re.search(r"\b(not available|unknown|n/a)\b", name, re.I):
            context["name"] = name
    if station_match:
        context["site_id"] = station_match.group(1)
    if coord_match:
        context["latitude"] = coord_match.group(1)
        context["longitude"] = coord_match.group(2)
    return context


async def _load_satellite_context(conversation_thread_id: str | None) -> dict[str, str]:
    """Best-effort read of this thread's satellite retrieval context. Memory
    degradation (no thread id, or a transient DB failure) is acceptable — the
    turn still answers, just without the continuation shortcut — but it is
    logged, never silent (same policy as the fast-path write-back)."""
    if not conversation_thread_id:
        return {}
    try:
        return await get_satellite_context(conversation_thread_id)
    except Exception:
        logger.warning(
            "satellite_context_load_failed",
            exc_info=True,
        )
        return {}


async def _persist_satellite_context(conversation_thread_id: str, context: dict[str, str]) -> None:
    try:
        await save_satellite_context(conversation_thread_id, context)
    except Exception:
        logger.warning(
            "satellite_context_save_failed",
            exc_info=True,
        )


def _capture_satellite_context(name: Any, args: Any, captured: dict[str, str]) -> None:
    """Record the dataset/AOI/handles a satellite turn worked with, read from
    the tool-call arguments the agent already emits (never from parsed tool
    results — the call args are structured and stable). ``search_datasets``'
    query and ``define_area_of_interest``'s location name the dataset/place in
    the researcher's own terms; the ``dataset_handle``/``aoi_handle``/
    ``time_range`` that later coverage/preview/retrieve calls carry are the
    workspace handles a follow-up can reuse to skip re-searching."""
    if not isinstance(args, dict):
        return
    if name == "search_datasets" and args.get("query"):
        captured["dataset_query"] = str(args["query"])
    elif name == "define_area_of_interest" and args.get("location"):
        captured["location"] = str(args["location"])
    if args.get("dataset_handle"):
        captured["dataset_handle"] = str(args["dataset_handle"])
    if args.get("aoi_handle"):
        captured["aoi_handle"] = str(args["aoi_handle"])
    if args.get("time_range"):
        captured["last_time_range"] = str(args["time_range"])


def _capture_chart_id(data: Any, captured: dict[str, str]) -> None:
    """Accumulate every chart id one dispatch streams, ordered, most recent
    last, comma-joined (the persisted satellite context is a flat str->str
    map). A single last-write-wins slot would misattribute a follow-up
    reliability question whenever one turn mints several distinct charts
    ("plot AOD and also NO2") — the streaming order of chart events is an
    implementation artifact, not the chart the researcher means. A re-emitted
    id moves to the most-recent slot rather than duplicating."""
    if not isinstance(data, dict) or not data.get("chart_id"):
        return
    chart_id = str(data["chart_id"])
    ids = [i for i in captured.get("last_chart_ids", "").split(",") if i and i != chart_id]
    ids.append(chart_id)
    captured["last_chart_ids"] = ",".join(ids)


_EXPLICIT_AREA_PATTERN = re.compile(
    r"\b(?:bounding box|bbox|longitude|latitude|lon|lat)\b", re.I
)


def _task_specifies_new_area(task: str) -> bool:
    """True when this turn's own task text spells out a location/bounding
    box (lon/lat wording or numeric bounds). Live-testing bug: a follow-up
    that named a brand-new bounding box was still handed the prior turn's
    aoi_handle framed as 'reuse to skip re-searching', and the sub-agent
    trusted the stale handle over the new numbers, silently re-rendering the
    old AOI instead of calling define_area_of_interest again. Same principle
    as the ground-monitor pollutant fix: the researcher's current wording is
    always authoritative over carried-over context."""
    return bool(_EXPLICIT_AREA_PATTERN.search(str(task or "")))


def _current_date_preamble(now: datetime | None = None) -> str:
    """Authoritative current-date banner prepended to every satellite task.

    Root cause A (live 2026-07-19): the sub-agent runs on a fast model whose
    training prior can trail the real clock by years. Shown only a bare
    ``[Current UTC time: ...]`` stamp, it was observed refusing valid
    present-day dates outright — "those dates are in the future, no
    observations exist" — for L3 AOD requested in the current week. So the
    date is stated as authoritative ground truth, with the future-refusal
    wording explicitly disarmed and doubt routed to a tool check, not a
    refusal from memory. The system prompt (earthdata_agent_prompt) reinforces
    the same rule; this banner is what makes the concrete date available."""
    now = now or datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        f"[Current date/time: {stamp} (AUTHORITATIVE — this is the real current "
        "date; trust it over any assumption you hold about what year it is). Any "
        "date on or before it is a valid past/present date: NEVER tell the "
        "researcher a requested date is \"in the future\" or that \"no "
        "observations exist yet\" from your own prior — if you doubt a date has "
        "data, check with check_availability/check_coverage and report what the "
        "tool returns.]\n\n"
    )


def _inject_satellite_context(task: str, context: dict[str, str]) -> str:
    """Prepend prior retrieval context to a satellite task, framed as
    UNVERIFIED — the handles likely still exist in the researcher's workspace
    (they persist per-user across threads), so reusing them skips re-searching,
    but availability is never carried as a verdict: the agent must re-run
    check_coverage/check_availability for the current request (its prompt's
    Availability-must-be-tool-grounded rule). Without this framing a follow-up
    that dropped to the supervisor path would re-derive everything from the
    paraphrased prior answer and drift.

    The AOI (location/aoi_handle) bits are withheld when the task itself
    names a new area — unlike the dataset, which stays valid across an entire
    conversation, the AOI is exactly what a "wider box" / "different region"
    follow-up is asking to change, so reuse must not be offered as an option
    in that case."""
    new_area = _task_specifies_new_area(task)
    bits = []
    if context.get("dataset_query"):
        bits.append(f"dataset={context['dataset_query']}")
    if context.get("dataset_handle"):
        bits.append(f"dataset_handle={context['dataset_handle']}")
    if context.get("location") and not new_area:
        bits.append(f"location={context['location']}")
    if context.get("aoi_handle") and not new_area:
        bits.append(f"aoi_handle={context['aoi_handle']}")
    if context.get("last_time_range"):
        bits.append(f"previously_checked_range={context['last_time_range']}")
    chart_ids = [i for i in context.get("last_chart_ids", "").split(",") if i]
    if not chart_ids and context.get("last_chart_id"):
        chart_ids = [context["last_chart_id"]]  # context persisted before the multi-id capture
    if len(chart_ids) == 1:
        bits.append(f"most_recent_chart_id={chart_ids[0]}")
    elif chart_ids:
        # Several charts in play: the stateless agent has no other signal that
        # ambiguity exists, so the disambiguation instruction must travel with
        # the ids — a bare single id would confidently explain the wrong chart.
        bits.append(
            f"chart_ids_from_previous_turns={','.join(chart_ids)} (most recent last; "
            "if the researcher's question could refer to more than one of these "
            "charts, ask which chart is meant before calling explain_measurement)"
        )
    if not bits:
        return task
    preamble = (
        "Prior retrieval context from earlier in this conversation (UNVERIFIED — "
        "these handles likely already exist in your workspace, so reuse them to "
        "skip re-searching, but you MUST re-run check_coverage/check_availability "
        "for the current request before stating any availability): "
        + "; ".join(bits) + ".\n\n"
    )
    return preamble + task


def _task_summary(task: str, max_chars: int = 200) -> str:
    return " ".join(str(task).split())[:max_chars]


def _artifact_refs_from_content(content: Any) -> list[ArtifactReference]:
    """Collect _artifact_refs embedded in one ToolMessage's content.

    Shared by the ground path (_extract_artifact_refs, which walks a full
    ainvoke() message list) and the satellite path (_invoke, which sees
    tool_result events one at a time via stream_response).
    """
    parsed = _parse_tool_content(content)
    if parsed is None:
        return []
    refs = []
    for ref in parsed.get("_artifact_refs") or []:
        if isinstance(ref, dict) and ref.get("id") and ref.get("type"):
            try:
                refs.append(ArtifactReference(**ref))
            except Exception:
                pass
    return refs


def _extract_artifact_refs(messages: list) -> list[ArtifactReference]:
    """Collect _artifact_refs from ground agent ToolMessages after ainvoke."""
    refs = []
    for msg in messages:
        if not (hasattr(msg, "name") and msg.name):
            continue
        refs.extend(_artifact_refs_from_content(getattr(msg, "content", "")))
    return refs


def _parse_tool_content(content: Any) -> dict | None:
    """Normalize a ToolMessage content value to a dict, or return None."""
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    if isinstance(content, list):
        # LangChain 0.3+ may wrap content in a list of blocks; check each block.
        for block in content:
            result = _parse_tool_content(block)
            if result is not None:
                return result
    return None
