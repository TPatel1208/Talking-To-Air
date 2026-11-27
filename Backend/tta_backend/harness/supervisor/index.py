"""The supervisor's index of workflows, exploration bundles and capability groups."""
from __future__ import annotations

from tta_backend.harness.capabilities.instruments import instruments
from tta_backend.harness.capabilities.registry import CapabilityRegistry
from tta_backend.harness.capabilities.sources.satellite import QUANTITIES
from tta_backend.harness.specialists.bundles import BUNDLES
from tta_backend.harness.workflows.loader import WorkflowCatalogue
from tta_backend.harness.workflows.spec import Input, WorkflowSpec


def _input_line(name: str, declared: Input) -> str:
    kind: str = declared.type
    if declared.options:
        kind = f"{kind}: {'|'.join(declared.options)}"
    return f"{name} ({kind}, {'optional' if declared.optional else 'required'})"


# How run_workflow takes each input type. Places stay in the user's words; code resolves them.
_FORMATS = {
    "quantity": (
        "quantity: one of " + ", ".join(QUANTITIES) + ". Pick the one for what the user asked about. "
        "These are the quantities workflows take, not all the data there is."
    ),
    "date": "date: an ISO date, YYYY-MM-DD.",
    "date_range": (
        'date_range: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}; one day has the same start and end. '
        "Work out relative dates such as yesterday from today's date in the session ledger message."
    ),
    "date_list": "date_list: a list of ISO dates.",
    "area": "area: the place exactly as the user wrote it, or an area handle from the session ledger.",
    "choice": "choice: one of the options listed with the input.",
    "text": (
        "text: free text. named is always given: the instrument or product the user named, as they wrote "
        "it; if they named none, one that serves the quantity: "
        + "; ".join(f"{quantity}: {', '.join(names)}" for quantity, names in instruments().items())
        + "."
    ),
    "number": "number: a plain number.",
    "monitor": "monitor: a monitor id from the session ledger.",
}


def _formats(specs: list[WorkflowSpec]) -> list[str]:
    used = {declared.type for spec in specs for declared in spec.inputs.values()}
    return [f"- {_FORMATS[kind]}" for kind in _FORMATS if kind in used]


def _workflow_line(spec: WorkflowSpec) -> str:
    inputs = ", ".join(_input_line(name, declared) for name, declared in spec.inputs.items())
    return f"- {spec.name} [{spec.status}]: {spec.description} Inputs: {inputs or 'none'}."


def supervisor_index(catalogue: WorkflowCatalogue, registry: CapabilityRegistry) -> str:
    """The index text: the same catalogue and registry always give the same text."""
    runnable = [
        spec
        for spec in (catalogue.latest(name) for name in catalogue.names())
        if spec.not_runnable_reason is None
    ]
    workflows = [_workflow_line(spec) for spec in runnable]
    bundles = [f"- {name}: {BUNDLES[name].description}" for name in sorted(BUNDLES)]
    used = sorted({group.name for bundle in BUNDLES.values() for group in bundle.groups(registry)})
    groups = [f"- {name}: {registry.group(name).description}" for name in used]
    return "\n".join(
        [
            "Workflows (run_workflow):",
            *workflows,
            "",
            "Input formats (run_workflow):",
            *_formats(runnable),
            "",
            "Exploration bundles (explore):",
            *bundles,
            "",
            "Capability areas the bundles draw on:",
            *groups,
        ]
    )
