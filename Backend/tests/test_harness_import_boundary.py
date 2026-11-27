"""Each layer imports only the layers below it.

| Layer          | May import                                        |
|----------------|---------------------------------------------------|
| harness        | harness, domain, datasets, infrastructure         |
| domain         | domain, datasets, infrastructure                  |
| datasets       | datasets                                          |
| infrastructure | infrastructure                                    |

Third-party imports are always allowed. The app (everything else in the
package) may import anything, so it has no check here.
"""
import ast
import unittest
from pathlib import Path

PACKAGE = "tta_backend"
PACKAGE_DIR = Path(__file__).resolve().parents[1] / PACKAGE
HARNESS_DIR = PACKAGE_DIR / "harness"
DOMAIN_DIR = PACKAGE_DIR / "domain"
DATASETS_DIR = PACKAGE_DIR / "datasets"
INFRASTRUCTURE_DIR = PACKAGE_DIR / "infrastructure"

HARNESS_ALLOWED = ("tta_backend.harness", "tta_backend.domain", "tta_backend.datasets", "tta_backend.infrastructure")
DOMAIN_ALLOWED = ("tta_backend.domain", "tta_backend.datasets", "tta_backend.infrastructure")
DATASETS_ALLOWED = ("tta_backend.datasets",)
INFRASTRUCTURE_ALLOWED = ("tta_backend.infrastructure",)
ALLOWED = HARNESS_ALLOWED


def _allowed(module: str, allowed: tuple[str, ...] = ALLOWED) -> bool:
    return any(module == ok or module.startswith(ok + ".") for ok in allowed)


def find_violations(source: str, module: str, allowed: tuple[str, ...] = ALLOWED) -> list[tuple[int, str]]:
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
            if (target == PACKAGE or target.startswith(PACKAGE + ".")) and not _allowed(target, allowed):
                found.append((node.lineno, target))
    return found


def layer_imports_outside(layer_dir: Path, allowed: tuple[str, ...]) -> list[tuple[str, int, str]]:
    """(file, line, imported module) for each import in ``layer_dir`` that leaves ``allowed``."""
    found = []
    for path in sorted(layer_dir.rglob("*.py")):
        relative = path.relative_to(PACKAGE_DIR.parent)
        module = ".".join(relative.with_suffix("").parts)
        for line, target in find_violations(path.read_text(encoding="utf-8"), module, allowed):
            found.append((relative.as_posix(), line, target))
    return found


def layer_violations(layer_dir: Path, allowed: tuple[str, ...]) -> list[str]:
    return [f"{file}:{line}: imports {target}" for file, line, target in layer_imports_outside(layer_dir, allowed)]


class HarnessImportBoundaryTests(unittest.TestCase):
    def test_harness_imports_only_harness_domain_datasets_and_infrastructure(self):
        offending = layer_violations(HARNESS_DIR, HARNESS_ALLOWED)
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
            "from tta_backend.domain.geo import find_lat_coord\n"
            "from tta_backend.infrastructure.settings import get_settings\n"
            "from . import storage\n"
            "from .storage import Stores\n"
        )
        self.assertEqual(find_violations(source, module), [])


class DomainImportBoundaryTests(unittest.TestCase):
    def test_domain_imports_only_domain_datasets_and_infrastructure(self):
        offending = layer_violations(DOMAIN_DIR, DOMAIN_ALLOWED)
        self.assertEqual(offending, [], "domain reaches into the app or the harness:\n" + "\n".join(offending))

    def test_detector_rejects_harness_and_app_from_domain(self):
        module = "tta_backend.domain.masking.geometry"
        cases = {
            "from tta_backend.harness.handles import Handle": "tta_backend.harness.handles",
            "from tta_backend.utils.plotting import plot_map": "tta_backend.utils.plotting",
            "from ...services import x": "tta_backend.services",
        }
        for source, target in cases.items():
            with self.subTest(source=source):
                self.assertEqual([t for _, t in find_violations(source, module, DOMAIN_ALLOWED)], [target])


class DatasetsImportBoundaryTests(unittest.TestCase):
    def test_datasets_imports_only_datasets(self):
        offending = layer_violations(DATASETS_DIR, DATASETS_ALLOWED)
        self.assertEqual(offending, [], "datasets imports another layer:\n" + "\n".join(offending))

    def test_detector_rejects_every_other_layer_from_datasets(self):
        module = "tta_backend.datasets.registry"
        cases = {
            "from tta_backend.domain.geo import find_lat_coord": "tta_backend.domain.geo",
            "from tta_backend.infrastructure.settings import get_settings": "tta_backend.infrastructure.settings",
            "from tta_backend.harness.handles import Handle": "tta_backend.harness.handles",
            "from ..services import x": "tta_backend.services",
        }
        for source, target in cases.items():
            with self.subTest(source=source):
                self.assertEqual([t for _, t in find_violations(source, module, DATASETS_ALLOWED)], [target])


class InfrastructureImportBoundaryTests(unittest.TestCase):
    def test_infrastructure_imports_only_infrastructure(self):
        offending = layer_violations(INFRASTRUCTURE_DIR, INFRASTRUCTURE_ALLOWED)
        self.assertEqual(offending, [], "infrastructure imports another layer:\n" + "\n".join(offending))

    def test_detector_rejects_every_other_layer_from_infrastructure(self):
        module = "tta_backend.infrastructure.metrics"
        cases = {
            "from tta_backend.domain.geo import find_lat_coord": "tta_backend.domain.geo",
            "from tta_backend.datasets import registry": "tta_backend.datasets",
            "def f():\n    from tta_backend.services.cube_cache import store_size_bytes": "tta_backend.services.cube_cache",
        }
        for source, target in cases.items():
            with self.subTest(source=source):
                self.assertEqual([t for _, t in find_violations(source, module, INFRASTRUCTURE_ALLOWED)], [target])


if __name__ == "__main__":
    unittest.main()
