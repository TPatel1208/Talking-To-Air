"""Exploration over the real capability catalogue, fake earthdata and the core pack's notes."""
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from tests.test_harness_grids import NJ
from tests.test_harness_sources_satellite import NO2, SECRET, TODAY, FakeService
from tests.test_harness_specialists import (
    Model,
    Turn,
    call_on_area,
    choice_of,
    final,
    on_last_scalar,
    results,
)
from tta_backend.datasets.registry import load_registry
from tta_backend.harness.capabilities.catalogue import build_registry
from tta_backend.harness.handles import HandleRecord
from tta_backend.harness.knowledge import load_notes
from tta_backend.harness.models import NO_BACKOFF
from tta_backend.harness.results import Result
from tta_backend.harness.run_context import RunContext
from tta_backend.harness.specialists import ExploreLimits, ExploreTask, explore, get_bundle, tool_name
from tta_backend.harness.specialists.tools import JOB_WAITING
from tta_backend.harness.storage import Stores, memory_stores
from tta_backend.harness.testing import (
    FakePlaces,
    FixedClock,
    MemoryArtifacts,
    ScriptedModels,
    StaticCredentials,
    env_factory,
    make_run_context,
    tool_call,
)

PACKS = Path(__file__).resolve().parent.parent / "packs"
NOTES = load_notes(PACKS, load_registry())
OBJECTIVE = "How high was NO2 over New Jersey on 2024-06-01?"


def satellite_script() -> list[Turn]:
    """Retrieve, take the mean, draw the map, answer with the mean."""
    def grid(messages: list[Any]) -> str:
        return next(r["handle"] for r in results(messages) if r.get("kind") == "grid")

    return [
        call_on_area("sources__satellite__get_data", quantity=NO2, dates="2024-06-01", date_role="subject"),
        lambda m: tool_call("operations__statistic", {"data": grid(m), "stat": "mean"}),
        lambda m: tool_call("render__map", {"grid": grid(m)}),
        final("The mean NO2 column over New Jersey was positive.", findings=on_last_scalar("mean NO2 column")),
    ]


class CatalogueWorld:
    """One user's context on ``stores``, with a fake earthdata service behind the real capabilities."""

    def __init__(self, root: str, stores: Stores) -> None:
        self.service = FakeService()
        self.earthdata = self.service.earthdata()
        self.artifacts = MemoryArtifacts()
        self.stores = stores
        self.registry = build_registry()
        self.root = root
        self.context: RunContext = make_run_context(
            stores=stores, artifacts=self.artifacts, clock=FixedClock(TODAY),
            credentials=StaticCredentials({"earthdata": SECRET}),
        )

    async def open_retrieval(self, user_id: str, record: HandleRecord) -> Any:
        return self.service.files[record.id]

    async def explore(self, *tasks: ExploreTask, script: list[Turn], limits: ExploreLimits | None = None) -> list[Result]:
        self.model = Model(script=script)
        models = ScriptedModels({"explore": choice_of(self.model)})
        build = env_factory(
            self.root, earthdata=self.earthdata, open_retrieval=self.open_retrieval,
            places=FakePlaces({"new jersey": NJ}), models=models,
        )
        return await explore(
            self.context, build, list(tasks), registry=self.registry, notes=NOTES, limits=limits, backoff=NO_BACKOFF,
        )


class CatalogueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.world = CatalogueWorld(tmp.name, memory_stores())

    async def test_an_analyse_task_retrieves_reduces_draws_and_answers(self):
        [result] = await self.world.explore(
            ExploreTask(objective=OBJECTIVE, bundle="analyse", places=("New Jersey",)), script=satellite_script(),
        )
        self.assertEqual(result.status, "done", result.reason)
        area, grid, scalar = result.handles
        self.assertTrue(area.startswith("polygons_") and grid.startswith("obs_") and scalar.startswith("scalar_"))
        [finding] = result.findings
        stat = await self.world.stores.handles.get("user-a", scalar)
        self.assertEqual(finding.value, stat.payload.value)
        self.assertEqual((finding.source_handle, finding.region), (scalar, "New Jersey"))
        self.assertEqual(result.charts, tuple(self.world.artifacts.stored))
        self.assertTrue(any("TEMPO_NO2" in line for line in result.disclosures))

        received = self.world.model.received
        system = str(received[0][0].content)
        self.assertIn("sources.satellite: get_data takes a quantity", system)
        self.assertNotIn("TEMPO covers North America only", system)
        retrieved = results(received[1])[0]
        self.assertEqual(retrieved["product"], "TEMPO_NO2")
        notes = "\n".join(retrieved["notes"])
        self.assertIn("TEMPO covers North America only", notes)
        self.assertIn("Satellite columns are not surface concentrations", notes)
        self.assertNotIn("notes", results(received[2])[1])

    async def test_a_retrieval_that_outlasts_the_wait_ends_the_task_waiting(self):
        self.world.service.statuses = [{"status": "running"}] * 1000
        [result] = await self.world.explore(
            ExploreTask(objective=OBJECTIVE, bundle="analyse", places=("New Jersey",)),
            script=satellite_script(), limits=ExploreLimits(retrieval_wait_seconds=0.05),
        )
        self.assertEqual(result.status, "waiting")
        self.assertEqual(result.reason, JOB_WAITING.format(name="sources.satellite.get_data"))
        entries = await self.world.stores.traces.entries("user-a", result.trace_id)
        waits = [e for e in entries if e.capability == "sources.satellite.get_data"]
        self.assertEqual([e.status for e in waits], ["waiting", "failed"])
        self.assertIn("jobs", waits[0].result["job"])
        self.assertEqual(self.world.model.calls, 1)

    async def test_an_investigate_task_works_on_handles_it_was_given_without_sources(self):
        [first] = await self.world.explore(
            ExploreTask(objective=OBJECTIVE, bundle="analyse", places=("New Jersey",)), script=satellite_script(),
        )
        grid = first.handles[1]
        [result] = await self.world.explore(
            ExploreTask(objective="Where is the NO2 highest?", bundle="investigate", handles=(grid,)), script=[
                tool_call("operations__statistic", {"data": grid, "stat": "max"}),
                final("The highest cell is in the north.", findings=on_last_scalar("max NO2 column")),
            ],
        )
        self.assertEqual(result.status, "done")
        offered = self.world.model.tools_offered[0]
        self.assertFalse(any(name.startswith("sources__") for name in offered))
        self.assertEqual(
            sorted(offered),
            sorted([*(tool_name(c) for c in get_bundle("investigate").capabilities(self.world.registry)), "SpecialistAnswer"]),
        )
        # The given grid brings its product's notes into the prompt from the start.
        self.assertIn("TEMPO covers North America only", str(self.world.model.received[0][0].content))
        self.assertEqual(result.handles[0], grid)
        self.assertEqual(json.loads(json.dumps(result.findings[0].model_dump()))["source_handle"], result.handles[1])
