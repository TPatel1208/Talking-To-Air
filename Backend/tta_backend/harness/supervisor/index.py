"""The supervisor's index of workflows, exploration bundles and capability groups."""
from __future__ import annotations

from tta_backend.harness.capabilities.registry import CapabilityRegistry
from tta_backend.harness.specialists.bundles import BUNDLES
from tta_backend.harness.workflows.loader import WorkflowCatalogue
from tta_backend.harness.workflows.spec import Input, WorkflowSpec


def _input_line(name: str, declared: Input) -> str:
    kind: str = declared.type
    if declared.options:
        kind = f"{kind}: {'|'.join(declared.options)}"
    return f"{name} ({kind}, {'optional' if declared.optional else 'required'})"


def _workflow_line(spec: WorkflowSpec) -> str:
    inputs = ", ".join(_input_line(name, declared) for name, declared in spec.inputs.items())
    return f"- {spec.name} [{spec.status}]: {spec.description} Inputs: {inputs or 'none'}."


def supervisor_index(catalogue: WorkflowCatalogue, registry: CapabilityRegistry) -> str:
    """The index text: the same catalogue and registry always give the same text."""
    workflows = [
        _workflow_line(spec)
        for spec in (catalogue.latest(name) for name in catalogue.names())
        if spec.not_runnable_reason is None
    ]
    bundles = [f"- {name}: {BUNDLES[name].description}" for name in sorted(BUNDLES)]
    used = sorted({group.name for bundle in BUNDLES.values() for group in bundle.groups(registry)})
    groups = [f"- {name}: {registry.group(name).description}" for name in used]
    return "\n".join(
        [
            "Workflows (run_workflow):",
            *workflows,
            "",
            "Exploration bundles (explore):",
            *bundles,
            "",
            "Capability areas the bundles draw on:",
            *groups,
        ]
    )
