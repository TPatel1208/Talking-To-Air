"""The framework every capability plugs into: declaration, result, invoke path and registry."""
from tta_backend.harness.capabilities.call import Call
from tta_backend.harness.capabilities.failures import InputEvicted, JobStillRunning
from tta_backend.harness.capabilities.definition import Capability, HandleParam, Params, handle_param
from tta_backend.harness.capabilities.invoke import JOB_PARAM, begin, complete, invoke
from tta_backend.harness.capabilities.registry import CapabilityRegistry, Group, UnknownCapability, UnknownGroup
from tta_backend.harness.capabilities.result import CapabilityResult, FieldNotFound, JobRef, Question, needs_choice

__all__ = [
    "JOB_PARAM",
    "Call",
    "Capability",
    "CapabilityRegistry",
    "CapabilityResult",
    "FieldNotFound",
    "Group",
    "HandleParam",
    "InputEvicted",
    "JobRef",
    "JobStillRunning",
    "Params",
    "Question",
    "UnknownCapability",
    "UnknownGroup",
    "begin",
    "complete",
    "handle_param",
    "invoke",
    "needs_choice",
]
