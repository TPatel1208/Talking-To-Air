"""explain.lineage, explain.citations and explain.reliability on the framework fakes; no network."""
import datetime
import json
import tempfile
import unittest
from typing import Any

from tta_backend.harness.capabilities import CapabilityRegistry, explain, invoke
from tta_backend.harness.handles import Descriptor, HandleRecord, ScalarInput
from tta_backend.harness.handles.model import ServicePayload
from tta_backend.harness.run_context import RunEnv
from tta_backend.harness.storage import memory_stores
from tta_backend.harness.testing import FakeEarthdata, StaticCredentials, make_run_context, make_run_env
from tta_backend.harness.trace import FailureCategory, StepFailure

NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
RESULT_LIMIT = 4096


def provenance_for(answers: dict[str, dict[str, Any]]):
    def tool(arguments: dict[str, Any]) -> dict[str, Any]:
        return answers[arguments["handle"]]

    return tool


def citation_for(records: dict[str, dict[str, Any]]):
    def tool(arguments: dict[str, Any]) -> dict[str, Any]:
        return records[arguments["dataset_handle"]]

    return tool


class _ExplainCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.stores = memory_stores()
        self.fake = FakeEarthdata()
        self.env: RunEnv = await self.make_env("user-a")

    async def make_env(self, user_id: str, *, connected: bool = True) -> RunEnv:
        credentials = StaticCredentials({"earthdata": "edl-token"} if connected else {})
        context = make_run_context(user_id=user_id, stores=self.stores, credentials=credentials)
        return await make_run_env(self._tmp.name, context=context, earthdata=self.fake)

    async def retrieval(self, handle_id: str, kind: Any = "grid", user_id: str = "user-a", **descriptor: Any) -> HandleRecord:
        return await self.stores.handles.record_retrieval(HandleRecord(
            id=handle_id, kind=kind, origin="retrieval", user_id=user_id, conversation_id="conv-1",
            descriptor=Descriptor(summary="A retrieved grid", **descriptor), created_at=NOW, payload=ServicePayload(),
        ))

    async def derived(self, parents: tuple[str, ...], **descriptor: Any) -> HandleRecord:
        return await self.env.resolver.mint(
            user_id="user-a", conversation_id="conv-1", kind="scalar", data=ScalarInput(value=1.0),
            descriptor=Descriptor(summary="A derived value", **descriptor), operation="operations.mean",
            created_at=NOW, parents=parents, parameters={"statistic": "mean"},
        )

    async def entries(self) -> list[Any]:
        return await self.stores.traces.entries("user-a", self.env.run_id)

    def assert_small(self, result: Any) -> None:
        self.assertLess(len(json.dumps(result.recorded())), RESULT_LIMIT)


class LineageTests(_ExplainCase):
    async def test_a_derived_handle_lists_its_retrieval_parent_first(self):
        await self.retrieval("obs_grid1", product="TEMPO_NO2")
        child = await self.derived(("obs_grid1",))
        self.fake.tools["get_provenance"] = provenance_for({"obs_grid1": {
            "handle": "obs_grid1",
            "ancestors": [{"handle": "dataset_tempo_no2", "depth": 1}],
            "events": [
                {"event_type": "materialized", "detail": {"granule_count": 24}, "created_at": "2026-07-01T00:12:00Z"},
                {"event_type": "routed", "detail": {"provider": "GES_DISC"}, "created_at": "2026-07-01T00:00:00Z"},
            ],
        }})

        result = await invoke(self.env, explain.lineage_capability, {"handles": [child.id]})

        steps = result.values["steps"]
        self.assertEqual([s["handle"] for s in steps], ["dataset_tempo_no2", "obs_grid1", child.id])
        self.assertEqual(steps[0]["kind"], "dataset")
        self.assertEqual(steps[1]["operation"], "materialized")
        self.assertEqual(steps[2]["operation"], "operations.mean")
        self.assertEqual(steps[2]["parameters"], {"statistic": "mean"})
        self.assertEqual(steps[2]["parents"], ["obs_grid1"])
        self.assertEqual((result.values["total_steps"], result.values["omitted_steps"]), (3, 0))
        self.assertEqual(self.fake.calls, [("get_provenance", {"handle": "obs_grid1"})])
        self.assertIsNone(result.handle)
        self.assert_small(result)

    async def test_several_handles_merge_into_one_lineage(self):
        first = await self.derived(())
        second = await self.derived((first.id,))

        result = await invoke(self.env, explain.lineage_capability, {"handles": [second.id, first.id]})

        self.assertEqual([s["handle"] for s in result.values["steps"]], [first.id, second.id])
        self.assertEqual(self.fake.calls, [])

    async def test_another_users_handle_is_refused_exactly_as_a_missing_one(self):
        other = await self.make_env("user-b")
        theirs = await other.resolver.mint(
            user_id="user-b", conversation_id="conv-1", kind="scalar", data=ScalarInput(value=2.0),
            descriptor=Descriptor(summary="Theirs"), operation="test.seed", created_at=NOW,
        )
        failures = []
        for handle in (theirs.id, "scalar_" + "0" * 32):
            with self.assertRaises(StepFailure) as caught:
                await invoke(self.env, explain.lineage_capability, {"handles": [handle]})
            failures.append(caught.exception)

        self.assertEqual(failures[0].category, FailureCategory.BAD_INPUT)
        self.assertEqual(failures[0].reason.replace(theirs.id, ""), failures[1].reason.replace("scalar_" + "0" * 32, ""))

    async def test_a_long_lineage_keeps_the_nearest_steps_and_counts_the_rest(self):
        parent: tuple[str, ...] = ()
        for _ in range(60):
            record = await self.derived(parent)
            parent = (record.id,)

        result = await invoke(self.env, explain.lineage_capability, {"handles": [record.id]})

        steps = result.values["steps"]
        self.assertEqual(steps[-1]["handle"], record.id)
        self.assertGreater(result.values["omitted_steps"], 0)
        self.assertEqual(len(steps) + result.values["omitted_steps"], 60)
        self.assertTrue(any("left out" in text for text in result.disclosures))
        self.assert_small(result)

    async def test_the_call_leaves_one_use_entry(self):
        child = await self.derived(())
        await invoke(self.env, explain.lineage_capability, {"handles": [child.id]})

        [entry] = await self.entries()
        self.assertEqual((entry.kind, entry.capability, entry.status), ("use", "explain.lineage", "done"))
        self.assertEqual(entry.params, {"handles": [child.id]})


class ConnectionTests(_ExplainCase):
    async def test_a_retrieval_without_the_earthdata_connection_is_the_not_connected_failure(self):
        await self.retrieval("obs_1")
        env = await self.make_env("user-a", connected=False)
        for capability in (explain.lineage_capability, explain.citations):
            with self.subTest(capability.name):
                with self.assertRaises(StepFailure) as caught:
                    await invoke(env, capability, {"handles": ["obs_1"]})
                self.assertEqual(caught.exception.category, FailureCategory.CREDENTIAL_INVALID)
                self.assertIn("needs the earthdata connector, which is not connected", caught.exception.reason)
        self.assertEqual(self.fake.calls, [])

    async def test_derived_handles_need_no_connection(self):
        record = await self.derived(())
        env = await self.make_env("user-a", connected=False)
        result = await invoke(env, explain.lineage_capability, {"handles": [record.id]})
        self.assertEqual(result.summary, {"steps": 1})


class CitationsTests(_ExplainCase):
    async def test_datasets_behind_handles_are_cited_once_each(self):
        await self.retrieval("obs_a", product="TEMPO_NO2")
        await self.retrieval("obs_b", product="TEMPO_NO2")
        child = await self.derived(("obs_a", "obs_b"))
        shared = {"handle": "obs_x", "ancestors": [{"handle": "dataset_tempo_no2", "depth": 1}], "events": []}
        self.fake.tools["get_provenance"] = provenance_for({"obs_a": {**shared, "handle": "obs_a"}, "obs_b": {**shared, "handle": "obs_b"}})
        self.fake.tools["cite_dataset"] = citation_for({"dataset_tempo_no2": {
            "concept_id": "C1", "doi": "10.5067/X", "doi_authority": "https://doi.org",
            "collection_citations": ["Author (2024). TEMPO NO2. NASA."], "reference_citation_count": 3,
        }})

        result = await invoke(self.env, explain.citations, {"handles": [child.id]})

        [citation] = result.values["citations"]
        self.assertEqual(citation["handle"], "dataset_tempo_no2")
        self.assertEqual(citation["doi"], "10.5067/X")
        self.assertEqual(citation["collection_citations"], ["Author (2024). TEMPO NO2. NASA."])
        self.assertEqual(result.summary, {"datasets": 1})
        cited = [call for call in self.fake.calls if call[0] == "cite_dataset"]
        self.assertEqual(cited, [("cite_dataset", {"dataset_handle": "dataset_tempo_no2"})])
        self.assert_small(result)

    async def test_two_dataset_handles_with_one_doi_are_cited_once(self):
        await self.retrieval("obs_a")
        self.fake.tools["get_provenance"] = provenance_for({"obs_a": {
            "handle": "obs_a", "events": [],
            "ancestors": [{"handle": "dataset_one", "depth": 1}, {"handle": "dataset_two", "depth": 1}],
        }})
        record = {"concept_id": "C1", "doi": "10.5067/X"}
        self.fake.tools["cite_dataset"] = citation_for({"dataset_one": record, "dataset_two": record})

        result = await invoke(self.env, explain.citations, {"handles": ["obs_a"]})

        self.assertEqual(len(result.values["citations"]), 1)

    async def test_no_dataset_in_the_lineage_gives_an_empty_list_and_says_so(self):
        child = await self.derived(())

        result = await invoke(self.env, explain.citations, {"handles": [child.id]})

        self.assertEqual(result.values["citations"], [])
        self.assertIn(explain.NO_DATASETS, result.disclosures)

    async def test_many_long_citations_are_cut_to_fit_and_counted(self):
        await self.retrieval("obs_a")
        names = [f"dataset_{i}" for i in range(30)]
        self.fake.tools["get_provenance"] = provenance_for({"obs_a": {
            "handle": "obs_a", "events": [], "ancestors": [{"handle": n, "depth": 1} for n in names],
        }})
        self.fake.tools["cite_dataset"] = citation_for({
            n: {"concept_id": f"C{i}", "doi": f"10.1/{i}", "collection_citations": ["t" * 500, "u" * 500, "v" * 500]}
            for i, n in enumerate(names)
        })

        result = await invoke(self.env, explain.citations, {"handles": ["obs_a"]})

        self.assertGreater(result.values["omitted_citations"], 0)
        self.assertEqual(len(result.values["citations"]) + result.values["omitted_citations"], 30)
        self.assert_small(result)


class ReliabilityTests(_ExplainCase):
    async def test_a_quality_flagged_beta_product_reports_masking_rates_and_maturity(self):
        masking = {
            "variable": "vertical_column_troposphere", "quality_flag": "main_data_quality_flag",
            "qa_status": "applied", "qa_source": "registry", "qa_pass_rate": 0.82, "valid_fraction": 0.64,
            "region_cells": 100,
        }
        record = await self.derived((), product="TEMPO_O3PROF", masking=masking)

        result = await invoke(self.env, explain.reliability, {"handle": record.id})

        self.assertEqual(result.status, "done")
        self.assertEqual(result.quality, {"valid_fraction": 0.64, "region_cells": 100, "qa_pass_rate": 0.82})
        self.assertEqual(result.summary["masking"]["qa_status"], "applied")
        self.assertEqual(result.summary["maturity"], "beta")
        self.assertIn("not recommended", result.summary["maturity_note"])
        self.assertTrue(result.values["has_evidence"])
        self.assertNotIn("statement", result.values)
        self.assertEqual(self.fake.calls, [])
        self.assert_small(result)

    async def test_a_flagged_product_names_its_registered_flag(self):
        record = await self.derived((), product="TEMPO_NO2", masking={"variable": "v", "qa_status": "applied"})

        result = await invoke(self.env, explain.reliability, {"handle": record.id})

        self.assertEqual(result.summary["registry_quality_flag"], "main_data_quality_flag")

    async def test_a_product_with_no_flag_says_why_none_was_applied(self):
        record = await self.derived(
            (), product="TEMPO_O3TOT",
            masking={"variable": "v", "quality_flag": None, "qa_status": "not applied — this product publishes no quality flag variable"},
        )

        result = await invoke(self.env, explain.reliability, {"handle": record.id})

        self.assertIn("no quality flag", result.summary["masking"]["qa_status"])
        self.assertIsNone(result.summary["registry_quality_flag"])
        self.assertIn("granule-verified", result.summary["no_quality_flag_reason"])
        self.assertIsNone(result.quality)

    async def test_a_handle_with_nothing_recorded_says_so_plainly(self):
        record = await self.derived(())

        result = await invoke(self.env, explain.reliability, {"handle": record.id})

        self.assertEqual(result.status, "done")
        self.assertFalse(result.values["has_evidence"])
        self.assertEqual(result.values["statement"], explain.NO_EVIDENCE)
        self.assertIn(explain.NO_EVIDENCE, result.disclosures)
        self.assertIsNone(result.quality)

    async def test_a_retrieval_reports_its_masking_plan_but_not_as_evidence(self):
        plan = {"variable": "v", "quality_flag": "main_data_quality_flag", "geometry": "area"}
        record = await self.retrieval("obs_p", product="TEMPO_NO2_NRT", masking=plan)

        result = await invoke(self.env, explain.reliability, {"handle": record.id})

        self.assertEqual(result.summary["masking_plan"], plan)
        self.assertNotIn("masking", result.summary)
        self.assertIsNone(result.quality)
        self.assertIn(explain.PLAN_NOTE, result.disclosures)

    async def test_a_plan_alone_is_never_evidence(self):
        record = await self.retrieval("obs_q", masking={"variable": "v", "quality_flag": "flag", "geometry": "area"})

        result = await invoke(self.env, explain.reliability, {"handle": record.id})

        self.assertFalse(result.values["has_evidence"])
        self.assertEqual(result.values["statement"], explain.PLAN_ONLY)
        self.assertEqual(result.disclosures, (explain.PLAN_NOTE, explain.PLAN_ONLY))

    async def test_the_call_leaves_one_use_entry(self):
        record = await self.derived(())
        await invoke(self.env, explain.reliability, {"handle": record.id})

        [entry] = await self.entries()
        self.assertEqual((entry.kind, entry.capability, entry.status), ("use", "explain.reliability", "done"))
        self.assertEqual(entry.result["values"]["subject"], record.id)


class GroupTests(unittest.TestCase):
    def test_the_group_registers_its_capabilities(self):
        registry = CapabilityRegistry()
        registry.add_group(explain.GROUP)
        for capability in explain.CAPABILITIES:
            registry.register(capability)

        self.assertEqual(explain.GROUP.name, "explain")
        self.assertEqual(
            [c.name for c in explain.CAPABILITIES], ["explain.lineage", "explain.citations", "explain.reliability"]
        )
        self.assertTrue(all(not c.retrieves and c.output_kind is None for c in explain.CAPABILITIES))
