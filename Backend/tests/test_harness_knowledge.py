"""Knowledge notes: the shipped notes load, every validation error is named, and lookup is stable."""
import tempfile
import unittest
from pathlib import Path

from tta_backend.datasets.registry import load_registry
from tta_backend.harness.knowledge import MAX_BODY_CHARS, Note, NoteCatalogue, NoteErrors, load_note, load_notes

PACKS = Path(__file__).resolve().parent.parent / "packs"
DATASETS = load_registry()


def note_text(
    *,
    id: str = "a-note",
    extra: str = "",
    keys: str = "pollutants: [no2]\n",
    body: str = "What a number means.\n",
) -> str:
    return f"---\nid: {id}\ntitle: A note\n{keys}{extra}---\n{body}"


def problems(text: str) -> str:
    with unittest.TestCase().assertRaises(NoteErrors) as caught:
        load_note(text, DATASETS)
    return str(caught.exception)


def make(**fields: object) -> Note:
    base: dict[str, object] = {"id": "x", "title": "X", "body": "b"}
    base.update(fields)
    return Note(**base)  # type: ignore[arg-type]


class ShippedNotesTest(unittest.TestCase):
    def test_every_core_note_loads(self) -> None:
        catalogue = load_notes(PACKS, DATASETS)
        files = sorted(p.stem for p in (PACKS / "core" / "notes").glob("*.md"))
        self.assertTrue(files)
        self.assertEqual(list(catalogue.ids()), files)
        for note in catalogue:
            self.assertEqual(note.pack, "core")
            self.assertLessEqual(len(note.body), MAX_BODY_CHARS)

    def test_a_pollutant_note_appears_for_that_pollutant(self) -> None:
        catalogue = load_notes(PACKS, DATASETS)
        self.assertIn("aqs-monitor-network", [n.id for n in catalogue.notes_for(pollutants=["pm25"])])
        self.assertEqual(catalogue.notes_for(pollutants=["pm10"]), ())

    def test_a_quantity_and_a_dataset_pull_their_notes(self) -> None:
        catalogue = load_notes(PACKS, DATASETS)
        by_quantity = [n.id for n in catalogue.notes_for(quantities=["no2_tropospheric_column"])]
        self.assertEqual(by_quantity, ["column-vs-surface"])
        by_dataset = [n.id for n in catalogue.notes_for(datasets=["TEMPO_O3PROF"])]
        self.assertEqual(by_dataset, ["ozone-profile-layers", "tempo-coverage-and-cadence"])


class ValidationTest(unittest.TestCase):
    def test_a_valid_note_loads(self) -> None:
        note = load_note(note_text(keys="datasets: [TEMPO_NO2]\npollutants: [no2]\n"), DATASETS, pack="core")
        self.assertEqual((note.id, note.title, note.pack), ("a-note", "A note", "core"))
        self.assertEqual((note.datasets, note.pollutants, note.quantities), (("TEMPO_NO2",), ("no2",), ()))
        self.assertEqual(note.body, "What a number means.")

    def test_unknown_field(self) -> None:
        self.assertIn("colour is not a known field", problems(note_text(extra="colour: red\n")))

    def test_missing_required_field(self) -> None:
        self.assertIn("title is required", problems("---\nid: a-note\npollutants: [no2]\n---\nbody\n"))

    def test_no_key(self) -> None:
        self.assertIn("names at least one dataset, quantity or pollutant", problems(note_text(keys="")))

    def test_empty_key_lists_are_no_key(self) -> None:
        self.assertIn("at least one", problems(note_text(keys="datasets: []\npollutants: []\n")))

    def test_repeated_key(self) -> None:
        self.assertIn("a key is listed more than once", problems(note_text(keys="pollutants: [no2, no2]\n")))

    def test_over_long_body(self) -> None:
        text = problems(note_text(body="x" * (MAX_BODY_CHARS + 1)))
        self.assertIn(f"the body is {MAX_BODY_CHARS + 1} characters; the limit is {MAX_BODY_CHARS}", text)

    def test_a_body_at_the_limit_loads(self) -> None:
        self.assertEqual(len(load_note(note_text(body="x" * MAX_BODY_CHARS), DATASETS).body), MAX_BODY_CHARS)

    def test_empty_body(self) -> None:
        self.assertIn("the body is empty", problems(note_text(body="\n")))

    def test_unknown_dataset(self) -> None:
        self.assertIn("dataset NOT_A_DATASET is not in the dataset registry", problems(note_text(keys="datasets: [NOT_A_DATASET]\n")))

    def test_unknown_quantity(self) -> None:
        self.assertIn("quantity soil_moisture is not measured", problems(note_text(keys="quantities: [soil_moisture]\n")))

    def test_unknown_pollutant(self) -> None:
        self.assertIn("pollutant radon is not in the standards", problems(note_text(keys="pollutants: [radon]\n")))

    def test_bad_id(self) -> None:
        self.assertIn("lower-case words joined by hyphens", problems(note_text(id="Bad_Id")))

    def test_no_frontmatter(self) -> None:
        self.assertIn("starts with a YAML frontmatter block", problems("just text\n"))

    def test_invalid_yaml(self) -> None:
        self.assertIn("not valid YAML", problems("---\nid: [unclosed\n---\nbody\n"))

    def test_all_problems_are_reported_together(self) -> None:
        text = problems(note_text(keys="datasets: [NOPE]\nquantities: [nope]\n", body="x" * (MAX_BODY_CHARS + 1)))
        self.assertEqual(len(text.splitlines()), 3)


class PackLoadingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def write(self, pack: str, name: str, text: str) -> None:
        notes = self.root / pack / "notes"
        notes.mkdir(parents=True, exist_ok=True)
        (notes / name).write_text(text, encoding="utf-8")

    def failure(self) -> str:
        with self.assertRaises(NoteErrors) as caught:
            load_notes(self.root, DATASETS)
        return str(caught.exception)

    def test_duplicate_id_across_packs(self) -> None:
        self.write("one", "a-note.md", note_text())
        self.write("two", "a-note.md", note_text())
        self.assertIn("the id a-note is also used in one/notes/a-note.md", self.failure())

    def test_id_must_match_file(self) -> None:
        self.write("one", "other.md", note_text())
        self.assertIn("the note's id is a-note but its file is other.md", self.failure())

    def test_stray_file(self) -> None:
        self.write("one", "readme.txt", "x")
        self.assertIn("a note file is named by its id", self.failure())

    def test_a_pack_without_notes_is_skipped(self) -> None:
        (self.root / "empty").mkdir()
        self.write("one", "a-note.md", note_text())
        self.assertEqual(load_notes(self.root, DATASETS).ids(), ("a-note",))

    def test_root_must_be_a_directory(self) -> None:
        with self.assertRaises(NoteErrors):
            load_notes(self.root / "missing", DATASETS)

    def test_errors_from_many_files_come_together(self) -> None:
        self.write("one", "a-note.md", note_text(extra="colour: red\n"))
        self.write("one", "b-note.md", note_text(id="b-note", keys=""))
        self.assertEqual(len(self.failure().splitlines()), 2)


class LookupTest(unittest.TestCase):
    def test_order_is_by_id_and_notes_appear_once(self) -> None:
        catalogue = NoteCatalogue([
            make(id="zeta", pollutants=("no2",), datasets=("TEMPO_NO2",)),
            make(id="alpha", pollutants=("no2", "o3")),
            make(id="mid", quantities=("no2_tropospheric_column",)),
            make(id="other", pollutants=("co",)),
        ])
        found = catalogue.notes_for(
            datasets=["TEMPO_NO2", "TEMPO_NO2"], quantities=["no2_tropospheric_column"], pollutants=["o3", "no2"]
        )
        self.assertEqual([n.id for n in found], ["alpha", "mid", "zeta"])

    def test_nothing_in_play_gives_nothing(self) -> None:
        catalogue = NoteCatalogue([make(pollutants=("no2",))])
        self.assertEqual(catalogue.notes_for(), ())

    def test_a_generator_is_accepted(self) -> None:
        catalogue = NoteCatalogue([make(pollutants=("no2",))])
        self.assertEqual(len(catalogue.notes_for(pollutants=(p for p in ["no2"]))), 1)

    def test_get_and_membership(self) -> None:
        catalogue = NoteCatalogue([make(id="x", pollutants=("no2",))])
        self.assertIn("x", catalogue)
        self.assertEqual(catalogue.get("x").id, "x")
        with self.assertRaises(LookupError):
            catalogue.get("y")

    def test_duplicate_ids_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            NoteCatalogue([make(id="x"), make(id="x")])


if __name__ == "__main__":
    unittest.main()
