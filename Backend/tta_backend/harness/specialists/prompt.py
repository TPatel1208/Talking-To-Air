"""What a specialist is told: a short role prompt with its notes, and the task message.

The system prompt holds only what is stable for the task, so its cached
prefix survives: the role, the bundle's group notes and the knowledge notes
for the input handles. The date goes in the task message. A note that comes
into play later is attached to the tool result that brought it.
"""
from __future__ import annotations

import datetime
from collections.abc import Iterable, Sequence

from tta_backend.harness.capabilities.registry import Group
from tta_backend.harness.handles.model import HandleRecord
from tta_backend.harness.knowledge import Note, NoteCatalogue
from tta_backend.harness.specialists.answer import ANSWER_TOOL
from tta_backend.harness.specialists.task import ExploreTask

ROLE = (
    "You are a specialist working one exploration task for an air quality analyst. Your answer is "
    "exploratory: it is labelled as such and is not a validated method.\n\n"
    "Work through handles. Each tool takes handle ids and small values and returns a new handle with a "
    "short summary, never the data itself. Use only the handles listed in the task and the ones your "
    "calls return. Tool names are capability names with each dot written as two underscores.\n\n"
    "A tool result with status failed or insufficient_evidence states its reason: report it, and do not "
    "substitute another area, dates or product unless the task asks for that. A failed call marked "
    "retry: false will not succeed if repeated.\n\n"
    f"Finish by calling {ANSWER_TOOL} once. Its summary says what you established and what it does not "
    "show. Each finding names the handle it comes from; for a scalar handle, give its value exactly as "
    "the tool returned it. Add up to three follow-up questions worth asking next."
)

DATE_PREAMBLE = (
    "Today's date is {today} (UTC). It is authoritative: trust it over any assumption about the current "
    "date, and work out relative dates such as \"yesterday\" or \"last week\" from it. Never refuse a date "
    "on or before today as being in the future. If you doubt that data exists for a date, check with a "
    "tool and report what it returns."
)


def note_text(note: Note) -> str:
    return f"{note.title}: {note.body}"


def system_prompt(groups: Sequence[Group], notes: Sequence[Note]) -> str:
    """The role, then each group's usage notes, then the knowledge notes in play."""
    parts = [ROLE]
    usage = [f"{group.name}: {group.notes}" for group in groups if group.notes.strip()]
    if usage:
        parts.append("Capability notes:\n" + "\n\n".join(usage))
    if notes:
        parts.append("Knowledge notes:\n" + "\n\n".join(note_text(note) for note in notes))
    return "\n\n".join(parts)


UNRESOLVED_NOTE = (
    "These places could not be resolved to an area. Search without an area for them, and do not use "
    "another area in their place."
)


def date_preamble(today: datetime.date) -> str:
    return DATE_PREAMBLE.format(today=today.isoformat())


def task_message(
    task: ExploreTask, today: datetime.date, inputs: Sequence[HandleRecord], unresolved: Sequence[str] = ()
) -> str:
    """The date preamble, the objective, the user's own wording, each input handle and each place that
    could not be resolved, as the user wrote it."""
    lines = [date_preamble(today), "", f"Objective: {task.objective}"]
    if task.variable is not None:
        lines.append(f"Variable, as the user wrote it: {task.variable}")
    if task.dates is not None:
        lines.append(f"Dates, as the user wrote them: {task.dates}")
    lines.append("")
    if inputs:
        lines.append("Handles you can use:")
        lines.extend(f"- {record.id} ({record.kind}): {record.descriptor.summary}" for record in inputs)
    elif unresolved:
        lines.append("No handles were given.")
    else:
        lines.append("No handles were given; resolve or retrieve what the objective needs.")
    if unresolved:
        lines.append("")
        lines.append("Places as the user wrote them:")
        lines.extend(f"- {place}" for place in unresolved)
        lines.append(UNRESOLVED_NOTE)
    return "\n".join(lines)


def keys_in_play(record: HandleRecord) -> tuple[set[str], set[str], set[str]]:
    """The datasets, quantities and pollutants a handle brings into play.

    A monitor handle names no pollutant key in its descriptor, so the
    pollutant comes from the parameters that made it.
    """
    descriptor = record.descriptor
    datasets = {descriptor.product} if descriptor.product else set()
    quantities = {descriptor.quantity} if descriptor.quantity else set()
    pollutant = record.parameters.get("pollutant")
    pollutants = {pollutant} if isinstance(pollutant, str) else set()
    return datasets, quantities, pollutants


def notes_in_play(
    catalogue: NoteCatalogue,
    records: Iterable[HandleRecord] = (),
    *,
    datasets: Iterable[str] = (),
    quantities: Iterable[str] = (),
    pollutants: Iterable[str] = (),
) -> tuple[Note, ...]:
    """The notes for what ``records`` and the given keys bring into play."""
    in_datasets, in_quantities, in_pollutants = set(datasets), set(quantities), set(pollutants)
    for record in records:
        d, q, p = keys_in_play(record)
        in_datasets |= d
        in_quantities |= q
        in_pollutants |= p
    return catalogue.notes_for(datasets=in_datasets, quantities=in_quantities, pollutants=in_pollutants)
