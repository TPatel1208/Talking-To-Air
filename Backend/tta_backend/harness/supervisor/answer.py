"""What the supervisor's model is shown of a result, and the parts of the answer code writes.

The model writes prose only. Code writes the rest, in fixed wording:

- ``result_view``: the compact JSON a tool returns for a result, never its
  charts or data;
- ``sources_line``: the line that opens an answer, naming the products
  behind the turn's result handles, EPA AQS last;
- ``assemble``: the prose with any sources line the model wrote removed,
  then each result's disclosures, gaps, reason and question, de-duplicated
  across results;
- ``unmatched_numbers``: the numbers in the prose that match no finding.
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Iterable, Sequence
from typing import Any

from tta_backend.harness.handles.errors import HandleError
from tta_backend.harness.handles.registry import HandleRegistry
from tta_backend.harness.results import Result

logger = logging.getLogger(__name__)

SOURCES_PREFIX = "Data sources:"
GROUND_SOURCE = "EPA AQS"
# A sources line the model wrote: "Data sources: ..." or "Sources: ...", in any emphasis.
MODEL_SOURCES_LINE = re.compile(r"^[\s>*_#-]*(?:data\s+)?sources?[*_\s]*:", re.IGNORECASE)

NOTES_HEADING = "Notes:"
GAPS_HEADING = "Gaps:"
NOT_DONE_HEADING = "Not finished:"
QUESTION_LINE = "Question: {prompt}"
# The sources a connector names, as the user knows them.
SOURCE_NAMES = {"earthdata": "NASA Earthdata", "epa_aqs": "EPA AQS"}
UNAVAILABLE = "{source} is unavailable right now, so nothing was started; try again in a few minutes."
OPTIONS_LINE = "Options: {options}"

# Results whose handles name no source: nothing they hold was delivered as an answer.
_NO_SOURCES = frozenset({"failed", "cancelled"})
# How far up a handle's parents the products behind it are looked for.
_MAX_ANCESTORS = 200
_OPTION_CHARS = 80
_MAX_UNMATCHED = 20


def _line(text: str) -> str:
    return " ".join(text.split())


# The compact view


def _finding(finding: Any) -> dict[str, Any]:
    view: dict[str, Any] = {"name": finding.name, "value": finding.value}
    for name in ("unit", "region", "period"):
        value = getattr(finding, name)
        if value is not None:
            view[name] = value
    view["source_handle"] = finding.source_handle
    return view


def view_of(result: Result) -> dict[str, Any]:
    """The compact view as a dict; ``result_view`` is its JSON."""
    view: dict[str, Any] = {"run_id": result.trace_id, "status": result.status, "mode": result.mode.label()}
    if result.findings:
        view["findings"] = [_finding(finding) for finding in result.findings]
    if result.handles:
        view["handles"] = list(result.handles)
    if result.charts:
        # References only: the charts themselves go to the user directly.
        view["charts"] = list(result.charts)
    if result.reason is not None:
        view["reason"] = result.reason
    if result.question is not None:
        view["question"] = result.question
    if result.interpretation is not None:
        view["interpretation"] = result.interpretation
    if result.followups:
        view["followups"] = list(result.followups)
    return view


def result_view(result: Result) -> str:
    """What the supervisor's model sees of a result: status, mode label,
    findings, handles, chart references, the fixed reason or question,
    interpretation and follow-ups. Disclosures and gaps are left out, since
    code appends them to the answer."""
    return dumps(view_of(result))


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# The sources line


async def data_sources(registry: HandleRegistry, user_id: str, results: Iterable[Result]) -> list[str]:
    """The products behind the results' handles and findings, each once in
    first-seen order, EPA AQS last. A failed or cancelled result names none;
    a handle the user does not own is skipped. A registry that cannot be
    read gives none, logged, so the answer still goes out without the line."""
    seen: list[str] = []
    visited: set[str] = set()
    for result in results:
        if result.status in _NO_SOURCES:
            continue
        pending = [*result.handles, *(finding.source_handle for finding in result.findings)]
        while pending and len(visited) < _MAX_ANCESTORS:
            handle = pending.pop(0)
            if handle in visited:
                continue
            visited.add(handle)
            try:
                record = await registry.get(user_id, handle)
            except HandleError:
                continue
            except Exception as exc:
                # The type name only: an exception's text can carry values or secrets.
                logger.error("the sources behind a result could not be read: %s", type(exc).__name__)
                return []
            product = record.descriptor.product
            if product and product not in seen:
                seen.append(product)
            pending.extend(record.parents)
    return [s for s in seen if s != GROUND_SOURCE] + ([GROUND_SOURCE] if GROUND_SOURCE in seen else [])


def sources_line(sources: Sequence[str]) -> str | None:
    """The line that opens an answer drawing on ``sources``; None for none."""
    return f"{SOURCES_PREFIX} {' · '.join(sources)}" if sources else None


# The prose


class ProseCleaner:
    """Cleans the prose one line at a time, so it can be streamed.

    Sources lines the model wrote are dropped, blank lines are collapsed to
    one, and leading and trailing blank lines are never written. ``line``
    returns the text to emit for a line, with the break before it.
    """

    def __init__(self) -> None:
        self._started = False
        self._blank = False
        self._parts: list[str] = []

    @property
    def text(self) -> str:
        return "".join(self._parts)

    def line(self, line: str) -> str:
        line = line.rstrip()
        if not line.strip():
            self._blank = self._started
            return ""
        if MODEL_SOURCES_LINE.match(line):
            return ""
        prefix = "" if not self._started else "\n\n" if self._blank else "\n"
        self._started, self._blank = True, False
        self._parts.append(prefix + line)
        return prefix + line

    def paragraph(self) -> None:
        """Starts the next text, such as another model message's, as a new paragraph."""
        self._blank = self._started


def clean_prose(text: str) -> str:
    """``text`` cleaned as ``ProseCleaner`` cleans a stream of it."""
    cleaner = ProseCleaner()
    for line in text.split("\n"):
        cleaner.line(line)
    return cleaner.text


# The parts code appends


def _option(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    text = _line(text)
    return text if len(text) <= _OPTION_CHARS else text[: _OPTION_CHARS - 1] + "…"


def _question_lines(question: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    prompt = question.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        lines.append(QUESTION_LINE.format(prompt=_line(prompt)))
    options = question.get("options")
    if isinstance(options, list | tuple) and options:
        lines.append(OPTIONS_LINE.format(options=" | ".join(_option(option) for option in options)))
    return lines


def unavailable_wording(connector: str) -> str:
    """The fixed wording of work refused because ``connector``'s service is down."""
    return UNAVAILABLE.format(source=SOURCE_NAMES.get(connector, connector))


def appended(results: Sequence[Result], unavailable: Sequence[str] = ()) -> str:
    """The parts code writes after the prose, in a fixed layout: every
    disclosure, every gap, the reason of each result not done, the wording for
    each unavailable connector, and each question with its options. A text
    already written for an earlier result is not repeated, a question
    included; empty when there is nothing to add."""
    seen: set[str] = set()

    def fresh(text: str) -> bool:
        key = _line(text)
        if not key or key in seen:
            return False
        seen.add(key)
        return True

    notes = [f"- {_line(d)}" for r in results for d in r.disclosures if fresh(d)]
    gaps = [f"- {_line(g.reason)}" for r in results for g in r.gaps if fresh(g.reason)]
    not_done: list[str] = []
    questions: list[str] = []
    for result in results:
        prompt = result.question.get("prompt") if result.question else None
        if isinstance(prompt, str):
            # A reason that only repeats the question is said once, as the question.
            seen.add(_line(prompt))
        if result.status != "done" and result.reason and fresh(result.reason):
            not_done.append(f"- {_line(result.reason)}")
        if result.question is not None:
            asked = _question_lines(result.question)
            if asked and fresh("\n".join(asked)):
                questions.extend(asked)
    not_done.extend(f"- {line}" for line in map(unavailable_wording, unavailable) if fresh(line))
    sections = []
    for heading, lines in ((NOTES_HEADING, notes), (GAPS_HEADING, gaps), (NOT_DONE_HEADING, not_done)):
        if lines:
            sections.append("\n".join([heading, *lines]))
    if questions:
        sections.append("\n".join(questions))
    return "\n\n".join(sections)


def assemble(prose: str, results: Sequence[Result], unavailable: Sequence[str] = ()) -> str:
    """The answer without its sources line: the prose with any sources line
    the model wrote removed, then ``appended(results, unavailable)``."""
    return "\n\n".join(part for part in (clean_prose(prose), appended(results, unavailable)) if part)


# The numbers check

_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?Z?)?\b")
_TIME = re.compile(r"\b\d{1,2}:\d{2}\b")
# A numbered list's marker, such as "1." or "2)" opening a line.
_LIST_MARKER = re.compile(r"^(\s*)\d+[.)](?=\s)", re.MULTILINE)
# A number directly followed by one of these counts something or says how long; it is not a claim.
# Units such as ppb, µg/m³ or % are not among them, so those numbers are checked.
_COUNTED = re.compile(
    r"\s*(?:hours?|days?|weeks?|months?|years?|minutes?|seconds?|sites?|monitors?|stations?|times?|steps?|"
    r"tasks?|runs?|samples?|granules?|points?|locations?|counties|county|states?|datasets?|products?|files?)\b",
    re.IGNORECASE,
)
# A number standing on its own: not part of a name such as NO2, PM2.5, a
# handle id or a hyphenated id such as a monitor's 34-003-0010.
_NUMBER = re.compile(
    r"(?<![\w.-])(?P<n>-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?)(?:\s*[×x]\s*10\^?(?P<e>[-−]?\d+))?"
    r"(?P<pct>\s*%)?(?![\w]|-\d)"
)


def _numbers(text: str, *, claims: bool = False) -> list[tuple[str, float, int, bool]]:
    """Each number in ``text``: its wording, value, decimal places and whether it is a percentage.

    With ``claims``, list markers and numbers that count or measure time are left out.
    """
    text = _TIME.sub(" ", _ISO_DATE.sub(" ", text))
    if claims:
        text = _LIST_MARKER.sub(lambda m: m[1], text)
    found = []
    for match in _NUMBER.finditer(text):
        if claims and _COUNTED.match(text, match.end()):
            continue
        digits = match["n"].replace(",", "")
        try:
            value = float(digits)
        except ValueError:
            continue
        mantissa = digits.split("e")[0].split("E")[0]
        decimals = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
        if match["e"]:
            exponent = int(match["e"].replace("−", "-"))
            value *= 10.0**exponent
            decimals -= exponent
        if not math.isfinite(value):
            continue
        found.append((match[0].strip(), value, decimals, bool(match["pct"])))
    return found


def _is_year(wording: str, value: float) -> bool:
    return wording.isdigit() and len(wording) == 4 and 1900 <= value <= 2100


def _matches(number: float, decimals: int, value: float) -> bool:
    if value == number:
        return True
    if value != 0 and abs(number - value) <= 0.01 * abs(value):
        return True
    if decimals >= 0:
        return round(value, decimals) == round(number, decimals)
    scale = 10.0 ** (-decimals)
    return round(value / scale) == round(number / scale)


def finding_values(results: Iterable[Result]) -> list[float]:
    values: list[float] = []
    for result in results:
        for finding in result.findings:
            value = finding.value
            if isinstance(value, bool):
                continue
            if isinstance(value, int | float):
                values.append(float(value))
            else:
                values.extend(number for _, number, _, _ in _numbers(value))
    return values


def unmatched_numbers(prose: str, results: Iterable[Result], message: str = "") -> list[str]:
    """The numbers in ``prose`` that match no finding value of ``results``,
    as written, at most 20. A match is within 1% of a value, or the value
    rounded to the prose's precision; a percentage also matches a fraction.
    Dates, times, years, list markers, counts and durations (a number followed
    by a word such as days or sites) and numbers the user wrote are not checked."""
    values = finding_values(results)
    asked = {number for _, number, _, _ in _numbers(message)}
    unmatched: list[str] = []
    for wording, number, decimals, percent in _numbers(prose, claims=True):
        if _is_year(wording, number) or number in asked:
            continue
        candidates = [number, number / 100] if percent else [number]
        if any(_matches(n, decimals + (2 if n != number else 0), v) for n in candidates for v in values):
            continue
        if wording not in unmatched:
            unmatched.append(wording)
        if len(unmatched) >= _MAX_UNMATCHED:
            break
    return unmatched
