"""The capabilities a host offers, grouped as the supervisor and specialists see them."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel

from tta_backend.harness.capabilities.definition import Capability

P = TypeVar("P", bound=BaseModel)

_GROUP = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*")


class UnknownCapability(LookupError):
    def __init__(self, name: str) -> None:
        super().__init__(f"no capability named {name!r}")
        self.name = name


class UnknownGroup(LookupError):
    def __init__(self, name: str) -> None:
        super().__init__(f"no capability group named {name!r}")
        self.name = name


@dataclass(frozen=True)
class Group:
    """A capability group: one index line for the supervisor, usage notes for whoever is given it."""

    name: str
    description: str
    notes: str = ""

    def __post_init__(self) -> None:
        if not _GROUP.fullmatch(self.name):
            raise ValueError("a group name is dotted lower case, like sources.satellite")
        if not self.description.strip() or "\n" in self.description:
            raise ValueError("a group description is one non-empty line")


class CapabilityRegistry:
    def __init__(self) -> None:
        self._groups: dict[str, Group] = {}
        self._capabilities: dict[str, Capability[Any]] = {}

    def add_group(self, group: Group) -> Group:
        if group.name in self._groups:
            raise ValueError(f"group {group.name} is already declared")
        self._groups[group.name] = group
        return group

    def register(self, capability: Capability[P]) -> Capability[P]:
        """Add a capability to its declared group; a name is registered once."""
        if capability.name in self._capabilities:
            raise ValueError(f"capability {capability.name} is already registered")
        if capability.group not in self._groups:
            raise UnknownGroup(capability.group)
        self._capabilities[capability.name] = capability
        return capability

    def get(self, name: str) -> Capability[Any]:
        try:
            return self._capabilities[name]
        except KeyError:
            raise UnknownCapability(name) from None

    def group(self, name: str) -> Group:
        try:
            return self._groups[name]
        except KeyError:
            raise UnknownGroup(name) from None

    def groups(self) -> tuple[Group, ...]:
        return tuple(self._groups[name] for name in sorted(self._groups))

    def in_group(self, name: str) -> tuple[Capability[Any], ...]:
        self.group(name)
        return tuple(c for _, c in sorted(self._capabilities.items()) if c.group == name)

    def index(self) -> str:
        """One line per group, for the supervisor's prompt."""
        return "\n".join(f"{g.name}: {g.description}" for g in self.groups())

    def schema(self, name: str) -> dict[str, Any]:
        """The capability's parameter JSON schema."""
        return self.get(name).schema()
