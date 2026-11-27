"""The supervisor: its generated index, its system prompt, its tools and the agent that answers each turn."""
from tta_backend.harness.supervisor.answer import assemble, result_view, sources_line, unmatched_numbers
from tta_backend.harness.supervisor.index import supervisor_index
from tta_backend.harness.supervisor.prompt import supervisor_prompt
from tta_backend.harness.supervisor.supervisor import Supervisor, SupervisorLimits, Turn, thread_id
from tta_backend.harness.supervisor.tools import MAX_TASKS, TOOL_NAMES, provider_schema

__all__ = [
    "MAX_TASKS",
    "TOOL_NAMES",
    "Supervisor",
    "SupervisorLimits",
    "Turn",
    "assemble",
    "provider_schema",
    "result_view",
    "sources_line",
    "supervisor_index",
    "supervisor_prompt",
    "thread_id",
    "unmatched_numbers",
]
