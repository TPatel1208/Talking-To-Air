"""What an exploration task is given, and the budgets it runs under."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from tta_backend.harness.handles.model import is_handle_id
from tta_backend.harness.specialists.bundles import BundleName

MAX_OBJECTIVE_CHARS = 1000
MAX_HANDLES = 12
MAX_PLACES = 4
MAX_PLACE_CHARS = 200
MAX_WORDING_CHARS = 200

_Place = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_PLACE_CHARS)]
_Wording = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_WORDING_CHARS)]


class ExploreTask(BaseModel):
    """One task: what to establish, with which bundle, from which inputs.

    ``handles`` are ids already resolved. ``places``, ``variable`` and
    ``dates`` are the user's own wording, verbatim; places are resolved in
    code before the model is called.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    objective: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_OBJECTIVE_CHARS)]
    bundle: BundleName
    handles: tuple[str, ...] = Field(default=(), max_length=MAX_HANDLES)
    places: tuple[_Place, ...] = Field(default=(), max_length=MAX_PLACES)
    variable: _Wording | None = None
    dates: _Wording | None = None

    @field_validator("handles")
    @classmethod
    def _handle_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not all(is_handle_id(handle) for handle in value):
            raise ValueError("every handle is a handle id")
        return tuple(dict.fromkeys(value))

    def inputs(self) -> dict[str, str | list[str]]:
        """The inputs as the explore trace entry records them."""
        recorded: dict[str, str | list[str]] = {"handles": list(self.handles), "places": list(self.places)}
        if self.variable is not None:
            recorded["variable"] = self.variable
        if self.dates is not None:
            recorded["dates"] = self.dates
        return recorded


@dataclass(frozen=True, kw_only=True)
class ExploreLimits:
    """Budgets per task. ``retrieval_wait_seconds``, when set, shortens how
    long a retrieval is waited for inline below the host's own timeout."""

    model_calls: int = 12
    tool_calls: int = 20
    retrieval_wait_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.model_calls < 1 or self.tool_calls < 1:
            raise ValueError("model_calls and tool_calls are at least 1")
        if self.retrieval_wait_seconds is not None and not self.retrieval_wait_seconds > 0:
            raise ValueError("retrieval_wait_seconds is positive")
