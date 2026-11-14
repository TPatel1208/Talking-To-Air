"""The harness never imports the application that hosts it.

Only ``tta_backend.harness`` and the data layer (``tta_backend.datasets``) may
be imported from inside ``tta_backend/harness``.
"""
import ast
import unittest
from pathlib import Path

PACKAGE = "tta_backend"
HARNESS_DIR = Path(__file__).resolve().parents[1] / PACKAGE / "harness"
ALLOWED = ("tta_backend.harness", "tta_backend.datasets")


def _allowed(module: str) -> bool:
    return any(module == ok or module.startswith(ok + ".") for ok in ALLOWED)


def find_violations(source: str, module: str) -> list[tuple[int, str]]:
    """(line, imported module) for each import that leaves the allowed set.

    ``module`` is the dotted name of the file being parsed, used to resolve
    relative imports.
    """
    package_parts = module.split(".")[:-1]
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        targets: list[str] = []
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package_parts[: len(package_parts) - (node.level - 1)]
                prefix = ".".join(base + ([node.module] if node.module else []))
            else:
                prefix = node.module or ""
            if not node.module and (node.level or prefix == PACKAGE):
                targets = [f"{prefix}.{alias.name}" for alias in node.names]
            elif prefix == PACKAGE:
                targets = [f"{prefix}.{alias.name}" for alias in node.names]
            else:
                targets = [prefix]
        for target in targets:
            if (target == PACKAGE or target.startswith(PACKAGE + ".")) and not _allowed(target):
                found.append((node.lineno, target))
    return found


class HarnessImportBoundaryTests(unittest.TestCase):
    def test_harness_imports_only_harness_and_datasets(self):
        offending = []
        for path in sorted(HARNESS_DIR.rglob("*.py")):
            relative = path.relative_to(HARNESS_DIR.parent.parent)
            module = ".".join(relative.with_suffix("").parts)
            for line, target in find_violations(path.read_text(encoding="utf-8"), module):
                offending.append(f"{relative}:{line}: imports {target}")
        self.assertEqual(offending, [], "harness reaches into the app:\n" + "\n".join(offending))

    def test_detector_catches_violations(self):
        module = "tta_backend.harness.thing"
        cases = {
            "import tta_backend.services.x": "tta_backend.services.x",
            "from tta_backend.utils.db import pg_connection": "tta_backend.utils.db",
            "from tta_backend import api": "tta_backend.api",
            "from .. import services": "tta_backend.services",
            "from ..services import x": "tta_backend.services",
            "def f():\n    from tta_backend.config import settings": "tta_backend.config",
        }
        for source, target in cases.items():
            with self.subTest(source=source):
                self.assertEqual([t for _, t in find_violations(source, module)], [target])

    def test_detector_allows_harness_datasets_and_third_party(self):
        module = "tta_backend.harness.thing"
        source = (
            "import psycopg\n"
            "from tta_backend.harness.storage import Stores\n"
            "from tta_backend.datasets import registry\n"
            "from . import storage\n"
            "from .storage import Stores\n"
        )
        self.assertEqual(find_violations(source, module), [])


if __name__ == "__main__":
    unittest.main()
