"""The named capability bundles a specialist can be given, computed from the registry."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from tta_backend.harness.capabilities.definition import Capability
from tta_backend.harness.capabilities.registry import CapabilityRegistry, Group

BundleName = Literal["analyse", "investigate", "find-data"]


class UnknownBundle(LookupError):
    def __init__(self, name: str) -> None:
        super().__init__(f"no bundle named {name!r}")
        self.name = name


def _in_prefix(group: str, prefix: str) -> bool:
    """Whether ``group`` is ``prefix`` or sits under it on a dotted boundary."""
    return group == prefix or group.startswith(prefix + ".")


def _any(_: Capability[Any]) -> bool:
    return True


# A find-data task's places are resolved in code before its model runs, so the model resolves none itself.
_RESOLVE = "sources.areas.resolve"


def _discovery(capability: Capability[Any]) -> bool:
    return not capability.retrieves and capability.name != _RESOLVE


@dataclass(frozen=True)
class Bundle:
    """A fixed capability set: group prefixes, narrowed by ``where``, plus named ``extra`` ones.

    ``workflow_only`` capabilities are never included.
    """

    name: BundleName
    description: str  # one line
    prefixes: tuple[str, ...] = ()
    where: Callable[[Capability[Any]], bool] = _any
    extra: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.description.strip() or "\n" in self.description:
            raise ValueError("a bundle description is one non-empty line")

    def capabilities(self, registry: CapabilityRegistry) -> tuple[Capability[Any], ...]:
        chosen: dict[str, Capability[Any]] = {}
        for group in registry.groups():
            if any(_in_prefix(group.name, p) for p in self.prefixes):
                for capability in registry.in_group(group.name):
                    if self.where(capability):
                        chosen[capability.name] = capability
        for name in self.extra:
            chosen[name] = registry.get(name)  # raises UnknownCapability if absent
        return tuple(c for _, c in sorted(chosen.items()) if not c.workflow_only)

    def groups(self, registry: CapabilityRegistry) -> tuple[Group, ...]:
        names = {c.group for c in self.capabilities(registry)}
        return tuple(registry.group(n) for n in sorted(names))


BUNDLES: Mapping[BundleName, Bundle] = MappingProxyType(
    {
        "analyse": Bundle(
            name="analyse",
            description="Retrieve from any source, then compute and draw: sources, operations and render.",
            prefixes=("sources", "operations", "render"),
        ),
        "investigate": Bundle(
            name="investigate",
            description="Work on existing handles only: operations, explain and render, no sources.",
            prefixes=("operations", "explain", "render"),
        ),
        "find-data": Bundle(
            name="find-data",
            description=(
                "Discover what data exists: source capabilities that do not retrieve, plus monitor listing; "
                "no place resolving."
            ),
            prefixes=("sources",),
            where=_discovery,
            extra=("sources.monitors.find",),
        ),
    }
)


def get_bundle(name: str) -> Bundle:
    try:
        return BUNDLES[name]  # type: ignore[index]
    except KeyError:
        raise UnknownBundle(name) from None


__all__ = ["BUNDLES", "Bundle", "BundleName", "UnknownBundle", "get_bundle"]
