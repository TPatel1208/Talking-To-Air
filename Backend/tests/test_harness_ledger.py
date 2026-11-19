"""The session ledger, built from runs made with the real capabilities on the fakes, and its rendering."""
import dataclasses
import datetime
import tempfile
from typing import Any

from tests.test_harness_sources_satellite import NO2
from tests.test_harness_specialists import Model, call_on_area, choice_of, final
from tests.test_harness_workflow_packs_run import JULY, JUNE, SMOKE_DATE, PackRun
from tta_backend.harness.capabilities.catalogue import build_registry
from tta_backend.harness.context import Ledger, LedgerArea, LedgerChart, LedgerHandle, build_ledger, render_ledger
from tta_backend.harness.context.ledger import EMPTY, HEADING
from tta_backend.harness.handles import Descriptor, HandleRecord
from tta_backend.harness.knowledge import NoteCatalogue
from tta_backend.harness.models import NO_BACKOFF
from tta_backend.harness.results import Result
from tta_backend.harness.run_context import RunContext
from tta_backend.harness.specialists import ExploreLimits, ExploreTask, explore
from tta_backend.harness.testing import TEST_LIMITS, ScriptedModels, box_region, env_factory, tool_call
from tta_backend.harness.trace import TraceEntry
from tta_backend.harness.workflows.runner import JOB_WAITING

GET_DATA = "sources.satellite.get_data"
RESOLVE = "sources.areas.resolve"
FIND = "sources.monitors.find"
MAP = {"quantity": NO2, "area": "NJ", "dates": JUNE}
GROUND = {"pollutant": "o3", "area": "NJ", "dates": {"start": "2024-06-01", "end": "2024-06-02"}, "view": "map"}
# A retrieval that never finishes, for runs that end waiting.
STILL_RUNNING = [{"status": "running", "phase": "processing"}] * 10_000
SHORT_WAIT = dataclasses.replace(TEST_LIMITS, retrieval_timeout_seconds=0.05, poll_min_seconds=0.01, poll_max_seconds=0.02)


class TickingClock:
    """A clock a millisecond later on each reading, so runs started in turn sort in turn."""

    def __init__(self, start: datetime.datetime) -> None:
        self._now = start

    def today(self) -> datetime.date:
        return self._now.date()

    def now(self) -> datetime.datetime:
        self._now += datetime.timedelta(milliseconds=1)
        return self._now


class LedgerCase(PackRun):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.f.context = dataclasses.replace(self.f.context, clock=TickingClock(self.f.context.clock.now()))
        # A short retrieval wait, so a job that does not finish ends the run waiting at once.
        self.f.build = env_factory(
            self.root, earthdata=self.service.earthdata(), open_retrieval=self.open_retrieval, places=self.f.places,
            limits=SHORT_WAIT,
        )

    async def open_retrieval(self, user_id: str, record: HandleRecord) -> Any:
        return self.service.files[record.id]

    def context_for(self, user_id: str) -> RunContext:
        return dataclasses.replace(self.f.context, identity=dataclasses.replace(self.f.context.identity, user_id=user_id))

    async def ledger(self, context: RunContext | None = None, **kw: Any) -> Ledger:
        return await build_ledger(context or self.f.context, **kw)

    async def text(self, context: RunContext | None = None, *, max_handles: int = 30, **kw: Any) -> str:
        return render_ledger(await self.ledger(context, **kw), max_handles=max_handles)

    async def done_entries(self, run_id: str, capability: str) -> list[TraceEntry]:
        found = await self.f.stores.traces.entries(self.f.context.identity.user_id, run_id)
        return [e for e in found if e.capability == capability and e.status == "done"]

    async def area_of(self, run_id: str) -> str:
        [entry] = await self.done_entries(run_id, RESOLVE)
        return str(entry.result["handle"])

    async def start(self, name: str, inputs: dict[str, Any], context: RunContext | None = None) -> Result:
        return await self.f.runner.start(context or self.f.context, self.f.build, name, 1, inputs)

    async def waiting_map(self) -> Result:
        self.service.statuses = list(STILL_RUNNING)
        result = await self.start("map", MAP)
        self.assertEqual((result.status, result.reason), ("waiting", JOB_WAITING.format(step="data")))
        return result

    def section(self, text: str, heading: str) -> list[str]:
        """The lines under ``heading``, up to the next heading."""
        lines = text.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith(heading))
        body = []
        for line in lines[start + 1:]:
            if not line.startswith(("-", " ")):
                break
            body.append(line)
        return body


class EmptyTests(LedgerCase):
    async def test_a_conversation_with_nothing_yet_renders_a_short_fixed_line(self):
        self.assertEqual(await self.text(), EMPTY)
        self.assertTrue((await self.ledger()).is_empty())


class SectionTests(LedgerCase):
    async def test_a_finished_map_run_lists_its_area_product_handle_and_chart(self):
        result = await self.start("map", MAP)
        self.assertEqual(result.status, "done", result.reason)
        run, area = result.trace_id, await self.area_of(result.trace_id)
        [chart] = result.charts
        [data] = await self.done_entries(run, GET_DATA)
        rule = data.result["values"]["rule"]

        ledger = await self.ledger()
        self.assertEqual(ledger.areas, (LedgerArea(handle=area, name="New Jersey", wording="NJ", run_id=run),))
        [product] = ledger.products
        self.assertEqual((product.product, product.rule, product.run_ids), ("TEMPO_NO2", rule, (run,)))
        [handle] = ledger.handles
        self.assertEqual((handle.id, handle.kind, handle.product, handle.area, handle.period, handle.evicted),
                         ("obs_1", "grid", "TEMPO_NO2", "New Jersey", "2024-06-01 to 2024-06-03", False))
        self.assertEqual(ledger.charts, (LedgerChart(ref=chart, run_id=run, kind="map", title=ledger.charts[0].title,
                                                     handles=("obs_1",)),))
        self.assertEqual((ledger.findings, ledger.open_runs), ((), ()))

        text = render_ledger(ledger)
        self.assertEqual(text.splitlines()[0], HEADING)
        self.assertEqual(self.section(text, "Areas:"), [f'- New Jersey (asked as "NJ"): {area}, run {run}'])
        self.assertEqual(self.section(text, "Products:"), [f"- TEMPO_NO2, chosen by rule {rule}: run {run}"])
        self.assertEqual(self.section(text, "Handles:"), [
            "- obs_1 grid: TEMPO_NO2 no2_tropospheric_column over New Jersey, 2024-06-01 to 2024-06-03; "
            f"unit molecules/cm^2 (run {run})",
        ])
        [chart_line] = self.section(text, "Charts:")
        self.assertTrue(chart_line.startswith(f'- {chart}: map "'), chart_line)
        self.assertTrue(chart_line.endswith(f'" of obs_1 (run {run})'), chart_line)
        self.assertNotIn("Findings:", text)
        self.assertNotIn("Runs not done:", text)

    async def test_findings_are_listed_under_their_run_and_mode_label(self):
        result = await self.start("timeseries", {"quantity": NO2, "area": "New Jersey", "dates": JUNE})
        self.assertEqual(result.status, "done", result.reason)
        text = await self.text()
        expected = [f"- {result.trace_id}, Draft workflow: timeseries v1:"]
        for finding in result.findings:
            unit = f" {finding.unit}" if finding.unit else ""
            where = [part for part in (finding.region, finding.period) if part]
            context = f", {', '.join(where)}" if where else ""
            value = f"{finding.value:.6g}" if isinstance(finding.value, float) else str(finding.value)
            expected.append(f"  - {finding.name} = {value}{unit}{context} (from {finding.source_handle})")
        self.assertEqual(self.section(text, "Findings:"), expected)
        self.assertIn("  - days_with_data = 3 days, New Jersey, 2024-06-01 to 2024-06-03 (from obs_1)", expected)

    async def test_monitors_are_listed_with_their_ids_and_the_area_they_were_searched_in(self):
        result = await self.start("monitor-readings", GROUND)
        self.assertEqual(result.status, "done", result.reason)
        run, area = result.trace_id, await self.area_of(result.trace_id)
        [find] = await self.done_entries(run, FIND)
        points, ids = find.result["handle"], find.result["values"]["monitor_ids"]
        self.assertEqual(len(ids), 2)

        ledger = await self.ledger()
        [monitors] = ledger.monitors
        self.assertEqual((monitors.handle, monitors.monitor_ids, monitors.area, monitors.pollutant),
                         (points, tuple(ids), area, "o3"))
        self.assertNotIn(points, [h.id for h in ledger.handles])
        self.assertEqual(self.section(render_ledger(ledger), "Monitors:"), [
            f"- {points}: 2 o3 monitors in New Jersey ({area}), 2024-06-01 to 2024-06-02; "
            f"monitor ids {', '.join(ids)} (run {run})",
        ])

    async def test_an_area_given_as_a_handle_is_listed_from_the_handle_it_was_used_for(self):
        first = await self.start("map", MAP)
        area = await self.area_of(first.trace_id)
        second = await self.start("map", {**MAP, "area": area, "dates": JULY})
        self.assertEqual(second.status, "done", second.reason)
        ledger = await self.ledger()
        self.assertEqual([a.handle for a in ledger.areas], [area])
        self.assertEqual([h.id for h in ledger.handles], ["obs_1", "obs_2"])
        self.assertEqual(ledger.products[0].run_ids, (first.trace_id, second.trace_id))


class CollapseTests(LedgerCase):
    async def test_past_max_handles_older_handles_are_one_short_line_and_none_is_dropped(self):
        await self.start("map", MAP)
        await self.start("timeseries", {"quantity": NO2, "area": "New Jersey", "dates": JULY})
        handles = (await self.ledger()).handles
        self.assertGreaterEqual(len(handles), 4)
        text = await self.text(max_handles=2)
        lines = self.section(text, "Handles")
        self.assertIn(f"Handles ({len(handles) - 2} oldest shown as id and kind only):", text)
        self.assertEqual(len(lines), len(handles))
        for line, handle in zip(lines[:-2], handles[:-2], strict=True):
            self.assertEqual(line, f"- {handle.id} {handle.kind}")
        for line, handle in zip(lines[-2:], handles[-2:], strict=True):
            self.assertTrue(line.startswith(f"- {handle.id} {handle.kind}: "), line)
        full = self.section(await self.text(), "Handles:")
        self.assertEqual(lines[-2:], full[-2:])

    async def test_past_max_finding_runs_older_runs_findings_are_one_count_line_each(self):
        await self.start("timeseries", {"quantity": NO2, "area": "New Jersey", "dates": JUNE})
        await self.start("monitor-readings", GROUND)
        smoke = await self.start("smoke-map", {"date": SMOKE_DATE, "area": "NJ"})
        ledger = await self.ledger()
        groups = ledger.findings
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[-1].run_id, smoke.trace_id)

        text = render_ledger(ledger, max_finding_runs=1)
        self.assertIn("Findings (2 oldest runs shown as a count only):", text)
        lines = self.section(text, "Findings")
        self.assertEqual(lines[:2], [
            f"- {group.run_id}, {group.mode_label}: {len(group.findings)} findings" for group in groups[:2]
        ])
        full = self.section(render_ledger(ledger), "Findings:")
        self.assertEqual(lines[2:], full[-(1 + len(groups[-1].findings)):])
        self.assertEqual(len(full), 3 + sum(len(g.findings) for g in groups))
        self.assertEqual(len(self.section(render_ledger(ledger, max_finding_runs=0), "Findings")), 3)
        with self.assertRaises(ValueError):
            render_ledger(ledger, max_finding_runs=-1)

    async def test_charts_are_never_collapsed(self):
        await self.start("map", MAP)
        await self.start("timeseries", {"quantity": NO2, "area": "New Jersey", "dates": JULY})
        await self.start("monitor-readings", GROUND)
        ledger = await self.ledger()
        self.assertEqual([c.kind for c in ledger.charts], ["map", "timeseries", "monitor_map"])
        self.assertTrue(all(c.handles for c in ledger.charts))
        charts = self.section(render_ledger(ledger, max_handles=0), "Charts:")
        self.assertEqual(charts, self.section(render_ledger(ledger), "Charts:"))
        for line, chart in zip(charts, ledger.charts, strict=True):
            self.assertTrue(line.startswith(f"- {chart.ref}: {chart.kind} "), line)
            self.assertIn(" of " + ", ".join(chart.handles), line)
        with self.assertRaises(ValueError):
            render_ledger(ledger, max_handles=-1)

    async def test_an_evicted_handle_is_marked_in_full_and_short_lines(self):
        result = await self.start("timeseries", {"quantity": NO2, "area": "New Jersey", "dates": JUNE})
        [series] = [h for h in (await self.ledger()).handles if h.kind == "series"]
        await self.f.stores.handles.mark_evicted("user-a", series.id, self.f.context.clock.now())
        ledger = await self.ledger()
        [evicted] = [h for h in ledger.handles if h.evicted]
        self.assertEqual(evicted.id, series.id)
        full = render_ledger(ledger)
        self.assertIn(f"- {series.id} series, evicted: ", full)
        self.assertIn(f"- {series.id} series, evicted\n", render_ledger(ledger, max_handles=0))
        self.assertIn(result.trace_id, full)

    def test_a_handle_without_a_record_says_so(self):
        ledger = Ledger(conversation_id="conv-1", handles=(LedgerHandle(id="obs_9", run_id="run_" + "0" * 32, known=False),))
        self.assertIn("- obs_9: no record", render_ledger(ledger))


class WaitingTests(LedgerCase):
    async def test_a_waiting_workflow_run_says_what_to_do_for_each_job_state(self):
        result = await self.waiting_map()
        run = result.trace_id
        head = f"- {run} waiting, Draft workflow: map v1: {result.reason} "
        expected = {
            "ready": f"Its retrieval jobs are ready: continue it with resume_run({run}).",
            "running": f"Its retrieval jobs are still running; resume_run({run}) once they are ready.",
            "failed": (
                f"A retrieval job it waits on failed; resume_run({run}) submits it again, "
                "or ends the run once the step has been submitted too often."
            ),
        }
        for state, advice in expected.items():
            with self.subTest(state):
                [line] = self.section(await self.text(job_states={run: state}), "Runs not done:")
                self.assertEqual(line, head + advice)
        unknown = (
            f"Its retrieval jobs have not been checked; resume_run({run}) collects them if they are ready, "
            "or the run waits again."
        )
        for job_states in (None, {}, {run: "unknown"}, {"run_" + "f" * 32: "ready"}):
            with self.subTest(job_states=job_states):
                [line] = self.section(await self.text(job_states=job_states), "Runs not done:")
                self.assertEqual(line, head + unknown)
        [open_run] = (await self.ledger(job_states={run: "ready"})).open_runs
        self.assertEqual((open_run.job_state, open_run.workflow), ("ready", True))

    async def test_a_waiting_exploration_is_not_resumable_and_asking_again_collects_the_job(self):
        self.service.statuses = list(STILL_RUNNING)
        model = Model(script=[call_on_area("sources__satellite__get_data", quantity=NO2, dates=JUNE, date_role="range"),
                              final()])
        build = env_factory(
            self.root, earthdata=self.service.earthdata(), open_retrieval=self.open_retrieval, places=self.f.places,
            models=ScriptedModels({"explore": choice_of(model)}),
        )
        [result] = await explore(
            self.f.context, build, [ExploreTask(objective="Map NO2 over NJ.", bundle="analyse", places=("NJ",))],
            registry=build_registry(), notes=NoteCatalogue([]), limits=ExploreLimits(retrieval_wait_seconds=0.05),
            backoff=NO_BACKOFF,
        )
        self.assertEqual(result.status, "waiting", result.reason)
        run = result.trace_id
        head = f"- {run} waiting, Exploratory: {result.reason} "
        collects = "An exploration cannot be resumed; asking the same question again collects the saved job."
        expected = {
            "ready": f"Its retrieval jobs are ready. {collects}",
            "running": f"Its retrieval jobs are still running. {collects}",
            "unknown": f"Its retrieval jobs have not been checked. {collects}",
            "failed": "A retrieval job it waits on failed. An exploration cannot be resumed; asking again starts a new run.",
        }
        for state, advice in expected.items():
            with self.subTest(state):
                text = await self.text(job_states={run: state})
                [line] = self.section(text, "Runs not done:")
                self.assertEqual(line, head + advice)
                self.assertNotIn("resume_run", text)
        # The area it resolved is still listed.
        self.assertEqual([a.wording for a in (await self.ledger()).areas], ["NJ"])

    async def test_a_run_waiting_on_an_answer_shows_its_reason(self):
        result = await self.start("map", {**MAP, "area": "Atlantis"})
        self.assertEqual(result.status, "needs_input")
        [line] = self.section(await self.text(), "Runs not done:")
        self.assertEqual(line, f"- {result.trace_id} needs_input, Draft workflow: map v1: {result.reason}")

    async def test_a_question_with_options_lists_them(self):
        ledger = Ledger(conversation_id="conv-1", open_runs=(
            dataclasses.replace((await self._open_run()), question={
                "kind": "pick_one", "prompt": "Which product?", "options": ["TEMPO_NO2", {"key": "TROPOMI_NO2"}],
            }),
        ))
        [line] = self.section(render_ledger(ledger), "Runs not done:")
        self.assertTrue(line.endswith('Question: Which product? Options: TEMPO_NO2 | {"key":"TROPOMI_NO2"}'), line)

    async def _open_run(self) -> Any:
        result = await self.start("map", {**MAP, "area": "Atlantis"})
        [open_run] = (await self.ledger()).open_runs
        self.assertEqual(open_run.run_id, result.trace_id)
        return open_run

    async def test_a_cancelled_run_is_listed_as_cancelled(self):
        result = await self.waiting_map()
        await self.f.runner.cancel(await self.f.build(self.f.context, result.trace_id))
        [line] = self.section(await self.text(job_states={result.trace_id: "ready"}), "Runs not done:")
        self.assertEqual(line, f"- {result.trace_id} cancelled, Draft workflow: map v1: The run was cancelled.")


class StabilityTests(LedgerCase):
    async def test_the_text_is_the_same_until_something_new_happens(self):
        await self.start("map", MAP)
        await self.start("monitor-readings", GROUND)
        first = await self.text()
        self.assertEqual(await self.text(), first)

        smoke = await self.start("smoke-map", {"date": SMOKE_DATE, "area": "NJ"})
        self.assertEqual(smoke.status, "done", smoke.reason)
        second = await self.text()
        self.assertNotEqual(second, first)
        # Every earlier line is still there, in the same order.
        later = second.splitlines()
        positions = [later.index(line) for line in first.splitlines()]
        self.assertEqual(positions, sorted(positions))


class IsolationTests(LedgerCase):
    async def test_another_users_runs_in_the_same_conversation_are_absent(self):
        mine = await self.start("map", MAP)
        other = self.context_for("user-b")
        theirs = await self.start("monitor-readings", GROUND, context=other)
        self.assertEqual(theirs.status, "done", theirs.reason)

        my_text, their_text = await self.text(), await self.text(other)
        self.assertIn(mine.trace_id, my_text)
        self.assertNotIn(theirs.trace_id, my_text)
        self.assertNotIn("Monitors:", my_text)
        self.assertIn(theirs.trace_id, their_text)
        self.assertNotIn(mine.trace_id, their_text)
        self.assertNotIn("obs_1", their_text)
        self.assertEqual(await self.text(self.context_for("user-c")), EMPTY)


class ExplorationLedgerCase(LedgerCase):
    async def explore_one(self, task: ExploreTask, script: list[Any]) -> Result:
        model = Model(script=script)
        build = env_factory(
            self.root, earthdata=self.service.earthdata(), open_retrieval=self.open_retrieval, places=self.f.places,
            models=ScriptedModels({"explore": choice_of(model)}),
        )
        [result] = await explore(
            self.f.context, build, [task], registry=build_registry(), notes=NoteCatalogue([]),
            limits=ExploreLimits(retrieval_wait_seconds=0.05), backoff=NO_BACKOFF,
        )
        return result


class MonitorSearchTests(ExplorationLedgerCase):
    async def test_monitors_found_near_a_point_are_listed_with_the_point(self):
        search = {"pollutant": "o3", "lat": 40.8, "lon": -74.1, "start": "2024-06-01", "end": "2024-06-02"}
        result = await self.explore_one(
            ExploreTask(objective="Which ozone monitors are near Bergen?", bundle="analyse"),
            [tool_call("sources__monitors__find", search), final("Two monitors were found.")],
        )
        self.assertEqual(result.status, "done", result.reason)
        [find] = await self.done_entries(result.trace_id, FIND)
        points, ids = find.result["handle"], find.result["values"]["monitor_ids"]

        [monitors] = (await self.ledger()).monitors
        self.assertEqual((monitors.handle, monitors.area, monitors.point, monitors.pollutant),
                         (points, None, (40.8, -74.1), "o3"))
        self.assertEqual(monitors.monitor_ids, tuple(ids))
        self.assertEqual(self.section(await self.text(), "Monitors:"), [
            f"- {points}: {len(ids)} o3 monitors near 40.8, -74.1, 2024-06-01 to 2024-06-02; "
            f"monitor ids {', '.join(ids)} (run {result.trace_id})",
        ])


class CarryOverTests(ExplorationLedgerCase):
    async def test_a_call_that_did_not_happen_lists_nothing(self):
        result = await self.start("map", {**MAP, "area": "Atlantis"})
        self.assertEqual(result.status, "needs_input")

        ledger = await self.ledger()
        self.assertEqual((ledger.areas, ledger.products, ledger.handles, ledger.charts, ledger.monitors), ((),) * 5)
        self.assertEqual([run.run_id for run in ledger.open_runs], [result.trace_id])
        text = render_ledger(ledger)
        for heading in ("Areas:", "Products:", "Handles:", "Charts:", "Monitors:"):
            self.assertNotIn(heading, text)

    async def test_each_place_asked_for_is_its_own_area_with_the_words_it_was_asked_in(self):
        self.f.places.regions["delaware valley"] = box_region("Delaware Valley", -75.6, 39.6, -74.6, 40.3)
        first = await self.start("map", MAP)
        self.assertEqual(first.status, "done", first.reason)
        second = await self.explore_one(
            ExploreTask(objective="Resolve the Delaware Valley.", bundle="find-data", places=("Delaware Valley",)),
            [final("Resolved.")],
        )
        self.assertEqual(second.status, "done", second.reason)

        ledger = await self.ledger()
        self.assertEqual([(a.name, a.wording, a.run_id) for a in ledger.areas], [
            ("New Jersey", "NJ", first.trace_id), ("Delaware Valley", "Delaware Valley", second.trace_id),
        ])
        self.assertEqual(len({a.handle for a in ledger.areas}), 2)
        lines = self.section(render_ledger(ledger), "Areas:")
        self.assertEqual(lines, [
            f'- New Jersey (asked as "NJ"): {ledger.areas[0].handle}, run {first.trace_id}',
            f"- Delaware Valley: {ledger.areas[1].handle}, run {second.trace_id}",
        ])

    async def test_a_polygons_handle_with_no_recorded_place_is_a_handle_not_an_area(self):
        env = await self.f.build(self.f.context, "run_" + "0" * 32)
        unnamed = await env.resolver.mint(
            user_id="user-a", conversation_id="conv-1", kind="polygons",
            data={"type": "FeatureCollection", "features": []},
            descriptor=Descriptor(summary="Some polygons", region={"name": "Somewhere", "bounds": [-75.0, 39.0, -74.0, 40.0]}),
            operation="operations.make", created_at=self.f.context.clock.now(),
        )
        result = await self.explore_one(
            ExploreTask(objective="Look at these polygons.", bundle="investigate", handles=(unnamed.id,)),
            [final("Looked.")],
        )
        self.assertEqual(result.status, "done", result.reason)

        ledger = await self.ledger()
        self.assertEqual(ledger.areas, ())
        self.assertEqual([(h.id, h.kind) for h in ledger.handles], [(unnamed.id, "polygons")])
        self.assertNotIn("Areas:", render_ledger(ledger))
