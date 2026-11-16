"""Declarative checks on an Outcome.

Each check says what must be true of the outcome, never how a system got
there. A check never names a system's internal tools; it speaks only of the
boundary (the data service and EPA AQS) and of what the user sees.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol

from benchmark.outcome import BoundaryCall, Outcome


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    reason: str


class Check(Protocol):
    def evaluate(self, outcome: Outcome) -> CheckResult: ...


def contains_subsequence(trace: list[str], expected: list[str]) -> bool:
    """True if ``expected`` appears in ``trace`` in order, not necessarily contiguously."""
    it = iter(trace)
    return all(name in it for name in expected)


def _calls_to(outcome: Outcome, tool: str) -> list[BoundaryCall]:
    return [call for call in outcome.boundary_calls if call.tool == tool]


def _normalize(text: str) -> str:
    return text.replace("’", "'").lower()


def _range_bounds(time_range: object) -> tuple[date, date] | None:
    """The inclusive (start, end) days of an ISO "start/end" range, or None if unparseable."""
    if not isinstance(time_range, str):
        return None
    days = re.findall(r"\d{4}-\d{2}-\d{2}", time_range)
    if not days:
        return None
    try:
        start, end = date.fromisoformat(days[0]), date.fromisoformat(days[-1])
    except ValueError:
        return None
    return (start, end) if start <= end else (end, start)


@dataclass(frozen=True)
class Completed:
    """The turn finished and reported no error."""

    def evaluate(self, outcome: Outcome) -> CheckResult:
        if outcome.error:
            return CheckResult(False, f"the run errored: {outcome.error}")
        if not outcome.done:
            if outcome.unfinished:
                return CheckResult(False, f"the run never finished: {outcome.unfinished}")
            return CheckResult(False, "the run never finished")
        return CheckResult(True, "completed")


@dataclass(frozen=True)
class BoundaryCalled:
    """The data service received a call to ``tool``.

    ``contains`` pairs an argument name with text that argument must contain
    (case-insensitive), all within the same call.
    """

    tool: str
    contains: tuple[tuple[str, str], ...] = ()

    def evaluate(self, outcome: Outcome) -> CheckResult:
        calls = _calls_to(outcome, self.tool)
        if not calls:
            return CheckResult(False, f"{self.tool} was never called")
        for call in calls:
            if all(needle.lower() in str(call.args.get(arg, "")).lower() for arg, needle in self.contains):
                return CheckResult(True, f"{self.tool} called")
        wanted = ", ".join(f"{arg} containing {needle!r}" for arg, needle in self.contains)
        seen = [[call.args.get(arg) for arg, _ in self.contains] for call in calls]
        return CheckResult(False, f"{self.tool} was called but never with {wanted}; saw {seen}")


def _bbox(location: object) -> tuple[float, float, float, float] | None:
    """A "min_lon,min_lat,max_lon,max_lat" location as numbers, or None."""
    if not isinstance(location, str):
        return None
    parts = location.split(",")
    if len(parts) != 4:
        return None
    try:
        min_lon, min_lat, max_lon, max_lat = (float(part) for part in parts)
    except ValueError:
        return None
    return min_lon, min_lat, max_lon, max_lat


@dataclass(frozen=True)
class AreaRequested:
    """The data service was asked for an area covering ``place``.

    A system may send the place by name or resolve it first and send its
    bounding box, so either passes: a location naming ``place``, or a
    "min_lon,min_lat,max_lon,max_lat" box containing ``point`` (lat, lon).
    """

    place: str
    point: tuple[float, float]

    def evaluate(self, outcome: Outcome) -> CheckResult:
        calls = _calls_to(outcome, "define_area_of_interest")
        if not calls:
            return CheckResult(False, "define_area_of_interest was never called")
        lat, lon = self.point
        for call in calls:
            location = call.args.get("location")
            if self.place.lower() in str(location).lower():
                return CheckResult(True, f"an area named {self.place!r} was requested")
            box = _bbox(location)
            if box is not None and box[0] <= lon <= box[2] and box[1] <= lat <= box[3]:
                return CheckResult(True, f"a box covering {self.place} was requested")
        seen = [call.args.get("location") for call in calls]
        return CheckResult(False, f"no requested area names or covers {self.place}; saw {seen}")


@dataclass(frozen=True)
class BoundaryNotCalled:
    tool: str

    def evaluate(self, outcome: Outcome) -> CheckResult:
        calls = _calls_to(outcome, self.tool)
        if calls:
            return CheckResult(False, f"{self.tool} was called {len(calls)} time(s); first args {calls[0].args}")
        return CheckResult(True, f"{self.tool} not called")


@dataclass(frozen=True)
class TimeRangeCovers:
    """Some ``tool`` call asked for a time range that includes ``day``."""

    tool: str
    day: date

    def evaluate(self, outcome: Outcome) -> CheckResult:
        calls = _calls_to(outcome, self.tool)
        if not calls:
            return CheckResult(False, f"{self.tool} was never called, so no range covers {self.day}")
        ranges = [call.args.get("time_range") for call in calls]
        for time_range in ranges:
            bounds = _range_bounds(time_range)
            if bounds is not None and bounds[0] <= self.day <= bounds[1]:
                return CheckResult(True, f"a {self.tool} range covers {self.day}")
        return CheckResult(False, f"no {self.tool} range covers {self.day}; ranges were {ranges}")


@dataclass(frozen=True)
class SatelliteSourceUsed:
    """The satellite data service was called at all."""

    def evaluate(self, outcome: Outcome) -> CheckResult:
        if outcome.boundary_calls:
            return CheckResult(True, "the satellite source was used")
        return CheckResult(False, "the satellite source was never used")


@dataclass(frozen=True)
class SatelliteSourceNotUsed:
    def evaluate(self, outcome: Outcome) -> CheckResult:
        if outcome.boundary_calls:
            return CheckResult(False, f"the satellite source was used ({outcome.boundary_calls[0].tool} called)")
        return CheckResult(True, "the satellite source was not used")


@dataclass(frozen=True)
class GroundSourceUsed:
    """EPA AQS was queried at all."""

    def evaluate(self, outcome: Outcome) -> CheckResult:
        if outcome.aqs_calls:
            return CheckResult(True, "the ground source was used")
        return CheckResult(False, "the ground source was never used")


@dataclass(frozen=True)
class GroundSourceNotUsed:
    def evaluate(self, outcome: Outcome) -> CheckResult:
        if outcome.aqs_calls:
            return CheckResult(False, f"the ground source was used ({outcome.aqs_calls[0][0]} queried)")
        return CheckResult(True, "the ground source was not used")


@dataclass(frozen=True)
class HasResult:
    """The user got something back: a handle, a chart or an artifact."""

    def evaluate(self, outcome: Outcome) -> CheckResult:
        if outcome.handles or outcome.charts or outcome.artifact_ids:
            return CheckResult(True, "a result was returned")
        return CheckResult(False, "no handle, chart or artifact was returned")


@dataclass(frozen=True)
class ChartProduced:
    """A chart or an artifact was shown to the user."""

    def evaluate(self, outcome: Outcome) -> CheckResult:
        if outcome.charts or outcome.artifact_ids:
            return CheckResult(True, "a chart was produced")
        return CheckResult(False, "no chart or artifact was produced")


@dataclass(frozen=True)
class AnswerMentionsAny:
    """The answer text contains at least one of ``markers`` (case-insensitive)."""

    markers: tuple[str, ...]

    def evaluate(self, outcome: Outcome) -> CheckResult:
        text = _normalize(outcome.answer_text)
        for marker in self.markers:
            if _normalize(marker) in text:
                return CheckResult(True, f"the answer mentions {marker!r}")
        return CheckResult(False, f"the answer mentions none of {list(self.markers)}")


@dataclass(frozen=True)
class AnswerMatches:
    """The answer text matches the regular expression ``pattern`` (case-insensitive)."""

    pattern: str
    description: str

    def evaluate(self, outcome: Outcome) -> CheckResult:
        match = re.search(self.pattern, _normalize(outcome.answer_text), re.IGNORECASE)
        if match:
            return CheckResult(True, f"the answer {self.description}: {match.group(0)!r}")
        return CheckResult(False, f"the answer never {self.description}")


@dataclass(frozen=True)
class StageSubsequence:
    """The narrated stages include ``stages`` in order, with others allowed between."""

    stages: tuple[str, ...]

    def evaluate(self, outcome: Outcome) -> CheckResult:
        if contains_subsequence(outcome.stages, list(self.stages)):
            return CheckResult(True, "the stages appeared in order")
        return CheckResult(False, f"stages {list(self.stages)} not found in order; saw {outcome.stages}")


@dataclass(frozen=True)
class StageSequenceAny:
    """The narrated stages include one of the ``alternatives`` in order, with others allowed between.

    Each alternative is one system's narration of the same course of events,
    so the check asks what happened, not how a system words it.

    An alternative in ``needing_result`` narrates capabilities rather than
    steps, so a stage alone proves little: it counts only when the data
    service also handed back a ready handle and a chart or artifact was shown.
    The outcome does not order those two against the stages, so this proves
    data was retrieved and something was rendered, not that one fed the other.
    """

    alternatives: tuple[tuple[str, ...], ...]
    needing_result: tuple[tuple[str, ...], ...] = ()

    def evaluate(self, outcome: Outcome) -> CheckResult:
        produced = bool(outcome.handles) and bool(outcome.charts or outcome.artifact_ids)
        for stages in (*self.alternatives, *(s for s in self.needing_result if produced)):
            if contains_subsequence(outcome.stages, list(stages)):
                return CheckResult(True, f"the stages {list(stages)} appeared in order")
        everything = [list(stages) for stages in (*self.alternatives, *self.needing_result)]
        note = "" if produced or not self.needing_result else "; no data handle and chart were produced"
        return CheckResult(False, f"none of {everything} found in order; saw {outcome.stages}{note}")


@dataclass(frozen=True)
class AqsDateRelativeToClock:
    """An AQS call used the run's date shifted by ``days``.

    The run's start and end dates are both accepted, so a run that crosses
    midnight passes, but no other day does: "yesterday" sent as today fails.
    """

    days: int = -1

    def evaluate(self, outcome: Outcome) -> CheckResult:
        clocks = {outcome.clock, outcome.clock_end or outcome.clock}
        accepted = sorted((clock + timedelta(days=self.days)).strftime("%Y%m%d") for clock in clocks)
        used = [params.get(key) for _, params in outcome.aqs_calls for key in ("bdate", "edate") if params.get(key)]
        if set(accepted).intersection(used):
            return CheckResult(True, f"an AQS call used {' or '.join(accepted)}")
        return CheckResult(False, f"no AQS call used {' or '.join(accepted)}; saw {used}")
