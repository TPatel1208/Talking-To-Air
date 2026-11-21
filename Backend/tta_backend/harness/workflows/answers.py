"""Questions a paused workflow run asks, and checking the answers given to them.

A run pauses on a question when an ``ask`` step runs, or when a capability
comes back with a question whose answer can be passed back to it. The
question reaches the user, the result and the ledger as one payload::

    {"run_id": "run_...", "step": "days", "kind": "confirm_list",
     "question": "These days will be used...", "prompt": "These days will be used...",
     "items": ["2024-06-01", ...]}

``options`` replaces ``items`` for ``pick_one`` and ``pick_many``; a ``text``
question has neither. ``prompt`` repeats ``question`` for the readers that
show a result's question by its prompt.

``check_answer`` checks an answer against the question in code and returns it
in canonical form, or raises ``AnswerRejected`` with fixed wording. No model
reads or repairs an answer.

A capability's question is answered into the run only when code knows how
the answer reaches the capability: the question names the parameter it sets
(``Question.param``), or it is the size confirmation of a retrieval, whose
``confirm`` sets ``confirmed`` and whose ``narrow`` stops the run.
``capability_rule`` gives that rule; ``apply_rule`` the parameters an answer sets.
"""
from __future__ import annotations

import datetime
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from pydantic import JsonValue

from tta_backend.harness.capabilities.definition import Capability
from tta_backend.harness.capabilities.result import Question

AskKind = Literal["pick_one", "pick_many", "confirm_list", "text"]
ASK_KINDS: tuple[AskKind, ...] = ("pick_one", "pick_many", "confirm_list", "text")

MAX_TEXT_CHARS = 1000
MAX_ITEM_CHARS = 500
MAX_ITEMS = 366
MAX_OPTIONS = 100
# How many options a rejection's wording lists before saying how many more there are.
_LISTED_OPTIONS = 10
_OPTION_CHARS = 80

# The size confirmation a retrieval asks before a large request.
CONFIRMED = "confirmed"
CONFIRM = "confirm"
NARROW = "narrow"
CONFIRM_OPTIONS = (CONFIRM, NARROW)

PICK_ONE = "The answer must be exactly one of the options: {options}."
PICK_MANY = "The answer must be one or more of the options, each at most once: {options}."
CONFIRM_LIST = "The answer must be a list of 1 to {limit} {what}."
TEXT = "The answer must be text of 1 to {limit} characters."
_ITEM_WORDING = {
    "date": "ISO dates, like 2024-06-01",
    "number": "numbers",
    "boolean": "true or false values",
    "text": f"texts of at most {MAX_ITEM_CHARS} characters",
    "object": "items shaped like the ones offered",
    "any": "items",
}


class AnswerRejected(ValueError):
    """An answer that does not fit its question; ``reason`` is fixed wording."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def option_text(value: Any) -> str:
    """An option as the user reads it: a string as it is, a mapping by its label, name or value."""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("label", "name", "value"):
            if isinstance(value.get(key), str):
                return str(value[key])
    return json.dumps(value, sort_keys=True, default=str)


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _OPTION_CHARS else text[: _OPTION_CHARS - 1] + "…"


def _listed(options: Sequence[Any]) -> str:
    shown = [_short(option_text(option)) for option in options[:_LISTED_OPTIONS]]
    more = len(options) - len(shown)
    return " | ".join(shown) + (f" and {more} more" if more > 0 else "")


# The payload


def question_payload(
    run_id: str,
    step: str,
    kind: AskKind,
    prompt: str,
    *,
    options: Sequence[JsonValue] = (),
    items: Sequence[JsonValue] = (),
) -> dict[str, Any]:
    """The question a run waits on, as the result, the ledger and the app carry it."""
    payload: dict[str, Any] = {"run_id": run_id, "step": step, "kind": kind, "question": prompt, "prompt": prompt}
    if kind in ("pick_one", "pick_many"):
        payload["options"] = list(options)
    elif kind == "confirm_list":
        payload["items"] = list(items)
    return payload


def answerable(question: Mapping[str, Any] | None, run_id: str | None = None) -> bool:
    """Whether ``question`` is one a paused run waits on, so an answer goes into the run
    (of ``run_id``, when given), rather than one that ended its run."""
    if not question:
        return False
    kind, step, asked = question.get("kind"), question.get("step"), question.get("run_id")
    if kind not in ASK_KINDS or not isinstance(step, str) or not isinstance(asked, str):
        return False
    return run_id is None or asked == run_id


# Checking


def _matches(option: Any, value: Any) -> bool:
    if isinstance(option, bool) or isinstance(value, bool):
        return isinstance(option, bool) and isinstance(value, bool) and option == value
    if option == value:
        return True
    if isinstance(value, str):
        wanted = " ".join(value.split()).casefold()
        return bool(wanted) and " ".join(option_text(option).split()).casefold() == wanted
    return False


def _option(options: Sequence[Any], value: Any) -> Any:
    """The option ``value`` names; a missing one raises KeyError."""
    for option in options:
        if _matches(option, value):
            return option
    raise KeyError("no option")


def _iso_date(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value.strip()).isoformat()
    except ValueError:
        return None


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _plain_text(value: Any, limit: int) -> str | None:
    if not isinstance(value, str) or "\x00" in value:
        return None
    text = value.strip()
    return text if 1 <= len(text) <= limit else None


def item_type(items: Sequence[Any]) -> str:
    """The type every offered item has, which the answer's items must have too."""
    if not items:
        return "any"
    if all(_iso_date(item) is not None for item in items):
        return "date"
    if all(isinstance(item, bool) for item in items):
        return "boolean"
    if all(_is_number(item) for item in items):
        return "number"
    if all(isinstance(item, str) for item in items):
        return "text"
    if all(isinstance(item, Mapping) for item in items):
        keys = {frozenset(item) for item in items}
        return "object" if len(keys) == 1 else "any"
    return "any"


def _item(kind: str, value: Any, offered: Sequence[Any]) -> JsonValue:
    """One item of a confirm_list answer in canonical form; a wrong one raises ValueError."""
    if kind == "date":
        day = _iso_date(value)
        if day is None:
            raise ValueError("not a date")
        return day
    if kind == "boolean":
        if not isinstance(value, bool):
            raise ValueError("not a boolean")
        return value
    if kind == "number":
        if not _is_number(value):
            raise ValueError("not a number")
        return cast(JsonValue, value)
    if kind == "text":
        text = _plain_text(value, MAX_ITEM_CHARS)
        if text is None:
            raise ValueError("not text")
        return text
    if kind == "object":
        if not isinstance(value, Mapping) or set(value) != set(offered[0]):
            raise ValueError("not shaped like the offered items")
        return cast(JsonValue, dict(value))
    if value is None or isinstance(value, float) and not math.isfinite(value):
        raise ValueError("empty")
    return cast(JsonValue, value)


def _once(values: Sequence[Any]) -> list[Any]:
    kept: list[Any] = []
    for value in values:
        if not any(_same(value, seen) for seen in kept):
            kept.append(value)
    return kept


def _same(a: Any, b: Any) -> bool:
    return type(a) is type(b) and a == b


def check_answer(question: Mapping[str, Any], value: Any) -> JsonValue:
    """``value`` as an answer to ``question``, in canonical form; raises AnswerRejected.

    pick_one: exactly one of the options, matched as given or by its text,
    ignoring case and spacing. pick_many: one or more of the options, each
    once; one option given alone counts as a list of one. confirm_list: 1 to
    ``MAX_ITEMS`` items of the type the offered items have, dates as ISO
    dates; the user may drop items and add new ones, and a repeat is kept
    once. text: 1 to ``MAX_TEXT_CHARS`` characters once trimmed.
    """
    kind = question.get("kind")
    if kind == "pick_one":
        options = list(question.get("options") or ())
        try:
            return cast(JsonValue, _option(options, value))
        except KeyError:
            raise AnswerRejected(PICK_ONE.format(options=_listed(options))) from None
    if kind == "pick_many":
        options = list(question.get("options") or ())
        given = list(value) if isinstance(value, list | tuple) else [value]
        try:
            picked = [_option(options, item) for item in given]
        except KeyError:
            raise AnswerRejected(PICK_MANY.format(options=_listed(options))) from None
        if not picked or len(_once(picked)) != len(picked):
            raise AnswerRejected(PICK_MANY.format(options=_listed(options)))
        return cast(JsonValue, picked)
    if kind == "confirm_list":
        offered = list(question.get("items") or ())
        what = item_type(offered)
        wording = CONFIRM_LIST.format(limit=MAX_ITEMS, what=_ITEM_WORDING[what])
        if not isinstance(value, list | tuple) or not value or len(value) > MAX_ITEMS:
            raise AnswerRejected(wording)
        try:
            items = [_item(what, item, offered) for item in value]
        except ValueError:
            raise AnswerRejected(wording) from None
        return cast(JsonValue, _once(items))
    if kind == "text":
        text = _plain_text(value, MAX_TEXT_CHARS)
        if text is None:
            raise AnswerRejected(TEXT.format(limit=MAX_TEXT_CHARS))
        return text
    raise AnswerRejected("The question cannot be answered.")


# A capability's question


def capability_rule(capability: Capability[Any], question: Question) -> dict[str, JsonValue] | None:
    """How the answer to ``capability``'s question reaches it when its step runs again, as plain
    JSON for the checkpoint; None when code knows no way, and the question then ends the run."""
    fields = capability.params.model_fields
    if question.param is not None:
        return {"param": question.param} if question.param in fields else None
    if CONFIRMED in fields and question.kind == "pick_one" and tuple(question.options) == CONFIRM_OPTIONS:
        return {"param": CONFIRMED, "values": {CONFIRM: True}, "stops": [NARROW]}
    return None


def apply_rule(rule: Mapping[str, Any], answer: JsonValue) -> dict[str, JsonValue] | None:
    """The parameters ``answer`` sets under ``rule``; None when the answer stops the step."""
    if isinstance(answer, str) and answer in (rule.get("stops") or ()):
        return None
    values = rule.get("values")
    if isinstance(values, Mapping) and isinstance(answer, str) and answer in values:
        return {str(rule["param"]): values[answer]}
    return {str(rule["param"]): answer}
