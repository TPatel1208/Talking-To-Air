"""Source comments explain the code as it is now; ticket history lives in git.

Fails if a ticket-style reference (T52, D14, "Phase 3", "PRD", "story #4",
"Finding #12") appears in production source. Write the reason instead, e.g.
"never start a cube write during a turn", not "see T52".
"""
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

ROOTS = [
    (REPO / "Backend" / "tta_backend", ("*.py",)),
    (REPO / "Frontend" / "src", ("*.js", "*.jsx", "*.css")),
]

# Not yet cleaned: edited on an unmerged branch. Remove once it lands.
SKIP = {REPO / "Backend" / "tta_backend" / "services" / "subagent_dispatch.py"}

PATTERN = re.compile(
    r"\bT\d{1,3}\b(?!:\d)"  # T52, but not ISO times like T00:00
    r"|\bD\d{1,2}[a-z]?\b"  # D14, D6a
    r"|\bPRD\b|\bPhase \d+|\bstory #\d+|\bFinding #\d+"
)


def _offending_lines():
    for root, globs in ROOTS:
        for glob in globs:
            for path in sorted(root.rglob(glob)):
                if path in SKIP or "node_modules" in path.parts:
                    continue
                for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if PATTERN.search(line):
                        yield f"{path.relative_to(REPO)}:{number}: {line.strip()}"


class NoTicketReferencesTests(unittest.TestCase):
    def test_production_source_has_no_ticket_references(self):
        offending = list(_offending_lines())
        self.assertEqual(offending, [], "ticket references in source:\n" + "\n".join(offending))


if __name__ == "__main__":
    unittest.main()
