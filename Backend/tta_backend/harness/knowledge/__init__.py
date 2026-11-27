"""Knowledge notes: short, model-facing text injected when its dataset, quantity or pollutant is in play."""
from tta_backend.harness.knowledge.notes import (
    MAX_BODY_CHARS,
    Note,
    NoteCatalogue,
    NoteError,
    NoteErrors,
    load_note,
    load_notes,
)

__all__ = [
    "MAX_BODY_CHARS",
    "Note",
    "NoteCatalogue",
    "NoteError",
    "NoteErrors",
    "load_note",
    "load_notes",
]
