"""Loading knowledge notes and looking them up by what is in play.

On disk a pack holds ``<root>/<pack>/notes/<id>.md``: YAML frontmatter, then a
short markdown body. The frontmatter has ``id`` (the file stem), ``title`` and
at least one of ``datasets`` (dataset registry keys), ``quantities`` (the
registry's canonical quantities) and ``pollutants`` (keys of ``standards.yaml``).
A note applies when any of its keys is in play.

Every problem found is a ``NoteError``; all of them under a root are raised
together as ``NoteErrors``.

Entry points:

- ``load_notes(root, datasets)``: every note of every pack under a root.
- ``load_note(source, datasets, ...)``: one note from its text.
- ``NoteCatalogue.notes_for(datasets=..., quantities=..., pollutants=...)``.
"""
from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from tta_backend.datasets.registry import CollectionConfig
from tta_backend.datasets.standards import load_standards

INLINE = "<inline>"
MAX_BODY_CHARS = 1000
_ID = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)(.*)\Z", re.DOTALL)
_ERROR_PREFIX = "Value error, "


class NoteError(Exception):
    """One problem with a note, in fixed wording filled from the note's own names."""

    def __init__(self, message: str, *, source: str, note: str | None = None) -> None:
        self.message = message
        self.source = source
        self.note = note
        super().__init__(str(self))

    def __str__(self) -> str:
        where = [self.source] if self.note is None else [self.source, self.note]
        return ": ".join([*where, self.message])


class NoteErrors(Exception):
    """Every problem found, raised together so an author sees them all at once."""

    def __init__(self, errors: Sequence[NoteError]) -> None:
        self.errors = tuple(errors)
        super().__init__("\n".join(str(error) for error in self.errors))


class _Frontmatter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    title: str
    datasets: tuple[str, ...] = ()
    quantities: tuple[str, ...] = ()
    pollutants: tuple[str, ...] = ()

    @field_validator("id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        if not _ID.match(value):
            raise ValueError("an id is lower-case words joined by hyphens")
        return value

    @field_validator("title")
    @classmethod
    def _title_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a title is not empty")
        return value.strip()

    @field_validator("datasets", "quantities", "pollutants")
    @classmethod
    def _keys(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("a key is listed more than once")
        if any(not key.strip() for key in value):
            raise ValueError("a key is not empty")
        return value


@dataclass(frozen=True)
class Note:
    id: str
    title: str
    body: str
    datasets: tuple[str, ...] = ()
    quantities: tuple[str, ...] = ()
    pollutants: tuple[str, ...] = ()
    pack: str = ""

    def applies(self, datasets: Collection[str], quantities: Collection[str], pollutants: Collection[str]) -> bool:
        return (
            any(key in datasets for key in self.datasets)
            or any(key in quantities for key in self.quantities)
            or any(key in pollutants for key in self.pollutants)
        )


class NoteCatalogue:
    """Checked notes by id."""

    def __init__(self, notes: Iterable[Note] = ()) -> None:
        self._notes: dict[str, Note] = {}
        for note in notes:
            if note.id in self._notes:
                raise ValueError(f"note {note.id} is given more than once")
            self._notes[note.id] = note

    def get(self, note_id: str) -> Note:
        try:
            return self._notes[note_id]
        except KeyError:
            raise LookupError(f"no note {note_id}") from None

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._notes))

    def notes_for(
        self,
        *,
        datasets: Iterable[str] = (),
        quantities: Iterable[str] = (),
        pollutants: Iterable[str] = (),
    ) -> tuple[Note, ...]:
        """The notes whose keys meet what is in play, once each, ordered by id."""
        in_datasets, in_quantities, in_pollutants = frozenset(datasets), frozenset(quantities), frozenset(pollutants)
        return tuple(
            self._notes[note_id]
            for note_id in sorted(self._notes)
            if self._notes[note_id].applies(in_datasets, in_quantities, in_pollutants)
        )

    def __contains__(self, note_id: object) -> bool:
        return note_id in self._notes

    def __iter__(self) -> Iterator[Note]:
        for note_id in sorted(self._notes):
            yield self._notes[note_id]

    def __len__(self) -> int:
        return len(self._notes)


def _field_path(loc: Sequence[int | str]) -> str:
    return ".".join(str(part) for part in loc)


def _validation_errors(error: ValidationError, label: str, note: str | None) -> list[NoteError]:
    found: list[NoteError] = []
    for item in error.errors():
        path = _field_path(item["loc"])
        kind, msg = item["type"], item["msg"]
        if kind == "missing":
            text = f"{path} is required"
        elif kind == "extra_forbidden":
            text = f"{path} is not a known field"
        else:
            if msg.startswith(_ERROR_PREFIX):
                msg = msg[len(_ERROR_PREFIX) :]
            text = f"{path}: {msg}" if path else msg
        found.append(NoteError(text, source=label, note=note))
    return found


def load_note(
    source: str,
    datasets: Mapping[str, CollectionConfig],
    *,
    label: str = INLINE,
    pack: str = "",
    pollutants: Collection[str] | None = None,
) -> Note:
    """One checked note from its text. ``pollutants`` defaults to the keys of ``standards.yaml``."""
    match = _FRONTMATTER.match(source)
    if match is None:
        raise NoteErrors([NoteError("a note starts with a YAML frontmatter block between --- lines", source=label)])
    try:
        raw = yaml.safe_load(match.group(1))
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        where = f" at line {mark.line + 2}" if mark is not None else ""
        raise NoteErrors([NoteError(f"the frontmatter is not valid YAML{where}", source=label)]) from None
    if not isinstance(raw, Mapping):
        raise NoteErrors([NoteError("the frontmatter is a mapping of fields", source=label)])
    named = raw.get("id") if isinstance(raw.get("id"), str) else None
    try:
        front = _Frontmatter.model_validate(raw)
    except ValidationError as error:
        raise NoteErrors(_validation_errors(error, label, named)) from None

    errors: list[NoteError] = []

    def add(message: str) -> None:
        errors.append(NoteError(message, source=label, note=front.id))

    body = match.group(2).strip()
    if not (front.datasets or front.quantities or front.pollutants):
        add("a note names at least one dataset, quantity or pollutant")
    if not body:
        add("the body is empty")
    elif len(body) > MAX_BODY_CHARS:
        add(f"the body is {len(body)} characters; the limit is {MAX_BODY_CHARS}")
    known_quantities = {cfg.quantity for cfg in datasets.values() if cfg.quantity is not None}
    known_pollutants = set(load_standards()) if pollutants is None else set(pollutants)
    for key in front.datasets:
        if key not in datasets:
            add(f"dataset {key} is not in the dataset registry")
    for key in front.quantities:
        if key not in known_quantities:
            add(f"quantity {key} is not measured by any registered dataset")
    for key in front.pollutants:
        if key not in known_pollutants:
            add(f"pollutant {key} is not in the standards")
    if errors:
        raise NoteErrors(errors)
    return Note(
        id=front.id,
        title=front.title,
        body=body,
        datasets=front.datasets,
        quantities=front.quantities,
        pollutants=front.pollutants,
        pack=pack,
    )


def load_notes(
    root: Path,
    datasets: Mapping[str, CollectionConfig],
    *,
    pollutants: Collection[str] | None = None,
) -> NoteCatalogue:
    """Every note at ``<root>/<pack>/notes/<id>.md``. Raises NoteErrors."""
    if not root.is_dir():
        raise NoteErrors([NoteError("the pack root is not a directory", source=str(root))])
    errors: list[NoteError] = []
    loaded: list[tuple[str, Note]] = []
    for pack in sorted(p for p in root.iterdir() if p.is_dir()):
        notes = pack / "notes"
        if not notes.is_dir():
            continue
        for path in sorted(notes.iterdir()):
            label = path.relative_to(root).as_posix()
            if not path.is_file() or path.suffix != ".md":
                errors.append(NoteError("a note file is named by its id, like column-vs-surface.md", source=label))
                continue
            try:
                note = load_note(path.read_text(encoding="utf-8"), datasets, label=label, pack=pack.name, pollutants=pollutants)
            except NoteErrors as failure:
                errors.extend(failure.errors)
                continue
            if note.id != path.stem:
                errors.append(NoteError(f"the note's id is {note.id} but its file is {path.name}", source=label, note=note.id))
                continue
            loaded.append((label, note))
    seen: dict[str, str] = {}
    kept: list[Note] = []
    for label, note in loaded:
        if note.id in seen:
            errors.append(NoteError(f"the id {note.id} is also used in {seen[note.id]}", source=label, note=note.id))
            continue
        seen[note.id] = label
        kept.append(note)
    if errors:
        raise NoteErrors(errors)
    return NoteCatalogue(kept)
