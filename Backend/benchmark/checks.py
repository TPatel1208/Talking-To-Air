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


def _days_of(time_range: object) -> set[date]:
    bounds = _range_bounds(time_range)
    if bounds is None:
        return set()
    start, end = bounds
    return {start + timedelta(days=n) for n in range((end - start).days + 1)}


def _covered_once(calls: list[BoundaryCall], days: tuple[date, ...], what: str) -> CheckResult:
    """Whether ``calls`` ask for exactly ``days``, each day in one call only."""
    counts: dict[date, int] = {}
    for call in calls:
        for day in _days_of(call.args.get("time_range")):
            counts[day] = counts.get(day, 0) + 1
    wanted = set(days)
    missing = sorted(day.isoformat() for day in wanted - counts.keys())
    extra = sorted(day.isoformat() for day in counts.keys() - wanted)
    repeated = sorted(day.isoformat() for day, count in counts.items() if count > 1)
    if not (missing or extra or repeated):
        return CheckResult(True, f"{what} covered exactly the {len(wanted)} days expected, each once")
    return CheckResult(
        False, f"{what} days differ from those expected: missing {missing}, extra {extra}, repeated {repeated}",
    )


@dataclass(frozen=True)
class RetrievedExactlyDays:
    """The data service was asked to retrieve exactly ``days``, each in one retrieval call only."""

    days: tuple[date, ...]
    tool: str = "retrieve_subset"

    def evaluate(self, outcome: Outcome) -> CheckResult:
        return _covered_once(_calls_to(outcome, self.tool), self.days, self.tool)


@dataclass(frozen=True)
class RetrievedAroundAnswer:
    """Before the user answered on ``step``, the data service was asked to retrieve exactly ``before``;
    after it, exactly ``after``; each day once. So what the answer chose is told apart from what was
    retrieved without it."""

    step: str
    before: tuple[date, ...]
    after: tuple[date, ...]
    tool: str = "retrieve_subset"

    def evaluate(self, outcome: Outcome) -> CheckResult:
        sent = next((a for a in outcome.answers if a.get("step") == self.step), None)
        if sent is None or not isinstance(sent.get("calls_before"), int):
            return CheckResult(False, f"no answer on {self.step} was sent, so nothing was retrieved after one")
        split = int(sent["calls_before"])
        earlier = [c for c in outcome.boundary_calls[:split] if c.tool == self.tool]
        later = [c for c in outcome.boundary_calls[split:] if c.tool == self.tool]
        first = _covered_once(earlier, self.before, f"{self.tool} before the answer")
        if not first.passed:
            return first
        second = _covered_once(later, self.after, f"{self.tool} after the answer")
        if not second.passed:
            return second
        return CheckResult(True, f"{first.reason}; {second.reason}")


@dataclass(frozen=True)
class AnsweredThenDone:
    """A run asked the user on ``step`` (a question of ``kind``), was answered there, and then finished done."""

    step: str
    kind: str | None = None

    def evaluate(self, outcome: Outcome) -> CheckResult:
        asked = [q for q in outcome.questions if q.get("step") == self.step]
        if not asked:
            seen = [q.get("step") for q in outcome.questions]
            return CheckResult(False, f"no run asked the user on {self.step}; questions answered: {seen}")
        for question in asked:
            if self.kind is not None and question.get("kind") != self.kind:
                continue
            status = outcome.run_statuses.get(str(question.get("run_id")))
            if status == "done":
                return CheckResult(True, f"the run asked on {self.step}, was answered and finished done")
            return CheckResult(False, f"the run asked on {self.step} and was answered, but ended {status}")
        return CheckResult(False, f"the question on {self.step} was not a {self.kind} question")


@dataclass(frozen=True)
class FindingReported:
    """The answer reports a finding named ``name`` equal to ``value``, within ``tolerance`` for a number."""

    name: str
    value: float | int | str
    tolerance: float = 0.0

    def _matches(self, found: object) -> bool:
        if isinstance(found, bool):
            return False
        if isinstance(self.value, str) or not isinstance(found, int | float):
            return found == self.value
        return abs(found - self.value) <= self.tolerance

    def evaluate(self, outcome: Outcome) -> CheckResult:
        found = [f.get("value") for f in outcome.findings if isinstance(f, dict) and f.get("name") == self.name]
        if not found:
            return CheckResult(False, f"no finding named {self.name} was reported")
        if any(self._matches(value) for value in found):
            return CheckResult(True, f"the finding {self.name} was {self.value!r}")
        within = f" within {self.tolerance:g}" if self.tolerance else ""
        return CheckResult(False, f"the finding {self.name} was {found}, not {self.value!r}{within}")
