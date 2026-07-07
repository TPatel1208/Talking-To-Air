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
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from config.model_factory import structured_output
from models import AgentResult, SubAgentEnvelope, parse_agent_result, parse_chart_payload, parse_sub_agent_envelope
from models.artifact import ArtifactReference
from repositories.session_metadata_repository import get_ground_monitor_context, save_ground_monitor_context
from tools import GROUND_TOOLS
from tools.satellite_tools.factory import sanctioned_tool_names
from utils.message_utils import extract_last_text, truncate_text
from utils.metrics import record_agent_request, record_envelope_salvaged
from utils.streaming import stream_response

logger = logging.getLogger(__name__)

# Per-request call counters — one integer per asyncio Task (one per HTTP
# request). Each Task inherits a copy of the current context, so the default
# of 0 is always seen at the start of a new request without manual resets.
_ground_call_count: ContextVar[int] = ContextVar("_ground_call_count", default=0)
_satellite_call_count: ContextVar[int] = ContextVar("_satellite_call_count", default=0)

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
    count = _ground_call_count.get()
    if count >= 1:
        logger.warning(
            "agent_budget_exceeded",
            extra={
                "_event": "agent_budget_exceeded",
                "_agent_type": "ground_sensor",
                "_call_count": count,
                "_thread_id": conversation_thread_id,
            },
        )
        return AgentResult(text=(
            "The worker has already returned a result. "
            "Do not call the supervisor_agent again. "
            "Synthesize your answer from the result already received."
        ))
    _ground_call_count.set(count + 1)

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
            text = str(exc)
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
                "_task_summary": _task_summary(enriched_task),
                "_thread_id": conversation_thread_id,
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
) -> AgentResult:
    """
    Invoke the earthdata agent once for ``task``, budget-checked and
    envelope-finalized.

    ``on_event``, if given, is awaited with every ``(event_type, data)`` tuple
    the sub-agent's own stream produces — the router fast path (T14) uses
    this to forward tool_call/status/job_progress/chart_payload events to the
    SSE stream live instead of buffering them until the turn finishes.
    """
    count = _satellite_call_count.get()
    if count >= 1:
        logger.warning(
            "agent_budget_exceeded",
            extra={
                "_event": "agent_budget_exceeded",
                "_agent_type": "satellite",
                "_call_count": count,
                "_thread_id": conversation_thread_id,
            },
        )
        return AgentResult(text=(
            "[STOP] Satellite agent has already been called for this request — "
            "this call is blocked. Do NOT call ask_earthdata_agent again. "
            "Synthesize your answer from the result already received."
        ))
    _satellite_call_count.set(count + 1)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    enriched_task = f"[Current UTC time: {now}]\n\n{task}"

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
                if event_type == "chart_payload":
                    # T13: plot/statistics tools emit the full render
                    # payload out-of-band via emit_chart and return the
                    # model only a compact summary — harvest charts from
                    # this event, not from tool_result content.
                    chart = parse_chart_payload(data)
                    if chart is not None:
                        charts.append(chart)
                    continue
                if event_type == "tool_result":
                    content = data.get("content", "")
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
            text_parts = [str(exc)]
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
                "_task_summary": _task_summary(enriched_task),
                "_thread_id": conversation_thread_id,
            },
        )
        result = await _reprompt_final_envelope(satellite_agent, _satellite_retry_task(enriched_task), "satellite")

    return _finalize_sub_agent_result(result, "earthdata")


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
        return _salvage_sub_agent_result(result, agent_label)
    discovered = {ref.id: ref for ref in result.artifacts}
    artifacts = [discovered[artifact_id] for artifact_id in envelope.artifact_ids if artifact_id in discovered]
    return AgentResult(
        text=truncate_text(envelope.summary, 2000, agent_name=agent_label),
        charts=result.charts,
        artifacts=artifacts,
        handles=envelope.handles,
    )


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
            text=f"The {agent_label} agent's final message did not parse and contained no text.",
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
