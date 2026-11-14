"""Lineage across derived and retrieval handles, in the provenance pane's shape."""
import unittest

from tests.harness_handle_helpers import LATER, FakeProvenance, retrieval_record, scalar_record
from tta_backend.harness.handles import HandleError, HandleRecord, InMemoryHandleRegistry, dataset_handles, lineage

NODE_KEYS = {"handle", "kind", "depth", "events", "note"}


class LineageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.registry = InMemoryHandleRegistry()
        self.provenance = FakeProvenance()

    async def chain(self, user_id="user-a"):
        obs = await self.registry.record_retrieval(retrieval_record(user_id))
        mean = await self.registry.mint(scalar_record(user_id, parents=(obs.id,), parameters={"dims": ["lat", "lon"]}))
        ratio = await self.registry.mint(scalar_record(user_id, parents=(mean.id,), parameters={"divisor": 2}, created_at=LATER))
        return obs, mean, ratio

    async def test_walks_derived_then_hands_off_at_retrieval(self):
        obs, mean, ratio = await self.chain()
        result = await lineage(self.registry, "user-a", [ratio.id], self.provenance)
        handles = [node["handle"] for node in result["nodes"]]
        self.assertEqual(handles, ["dataset_tempo_no2", f"aoi_{obs.id}", obs.id, mean.id, ratio.id])
        self.assertEqual(self.provenance.calls, [("user-a", [obs.id])])
        by_handle = {node["handle"]: node for node in result["nodes"]}
        self.assertEqual(by_handle[obs.id]["kind"], "observation")
        self.assertEqual(by_handle[mean.id]["depth"], 1)
        self.assertNotIn("depth", by_handle[ratio.id])
        event = by_handle[ratio.id]["events"][0]
        self.assertEqual(event["event_type"], "operation")
        self.assertEqual(event["detail"]["operation"], "mean")
        self.assertEqual(event["detail"]["parameters"], {"divisor": 2})
        self.assertEqual(event["detail"]["parents"], [mean.id])
        self.assertEqual(event["created_at"], LATER.isoformat())
        for node in result["nodes"]:
            self.assertLessEqual(set(node), NODE_KEYS)
        self.assertEqual(dataset_handles(result), ["dataset_tempo_no2"])

    async def test_shared_ancestors_appear_once(self):
        obs, mean, ratio = await self.chain()
        sibling = await self.registry.mint(scalar_record("user-a", parents=(mean.id, obs.id)))
        result = await lineage(self.registry, "user-a", [ratio.id, sibling.id, ratio.id], self.provenance)
        handles = [node["handle"] for node in result["nodes"]]
        self.assertEqual(len(handles), len(set(handles)))
        self.assertEqual(self.provenance.calls, [("user-a", [obs.id])])
        self.assertLess(handles.index(mean.id), handles.index(sibling.id))
        self.assertLess(handles.index(mean.id), handles.index(ratio.id))

    async def test_derived_only_lineage_needs_no_service(self):
        root = await self.registry.mint(scalar_record("user-a"))
        result = await lineage(self.registry, "user-a", [root.id], self.provenance)
        self.assertEqual([node["handle"] for node in result["nodes"]], [root.id])
        self.assertEqual(self.provenance.calls, [])
        self.assertEqual(dataset_handles(result), [])

    async def test_another_users_handle_is_not_found_and_never_walked(self):
        _, _, ratio = await self.chain("user-a")
        for handles in ([ratio.id], [ratio.id, "obs_tempo_no2_1"]):
            with self.subTest(handles=handles), self.assertRaises(HandleError) as caught:
                await lineage(self.registry, "user-b", handles, self.provenance)
            self.assertEqual(caught.exception.reason, "not_found")
        self.assertEqual(self.provenance.calls, [])

    async def test_each_user_walks_only_their_own_copy_of_a_retrieval_id(self):
        await self.chain("user-a")
        obs_b = await self.registry.record_retrieval(retrieval_record("user-b"))
        own = await self.registry.mint(scalar_record("user-b", parents=(obs_b.id,)))
        result = await lineage(self.registry, "user-b", [own.id], self.provenance)
        derived = [node["handle"] for node in result["nodes"] if "events" in node and node["events"][0]["event_type"] == "operation"]
        self.assertEqual(derived, [own.id])
        # The service is asked in the requesting user's workspace.
        self.assertEqual(self.provenance.calls, [("user-b", [obs_b.id])])

    async def test_a_cycle_terminates(self):
        a = scalar_record("user-a")
        b = scalar_record("user-a", parents=(a.id,))
        cyclic_a = HandleRecord(**{**a.model_dump(), "parents": (b.id,)})
        # Written past mint, which would refuse the missing parent.
        self.registry._records[("user-a", a.id)] = cyclic_a
        self.registry._records[("user-a", b.id)] = b
        result = await lineage(self.registry, "user-a", [b.id], self.provenance)
        self.assertEqual(sorted(node["handle"] for node in result["nodes"]), sorted([a.id, b.id]))

    async def test_a_missing_parent_is_noted_not_fatal(self):
        orphan = scalar_record("user-a", parents=("obs_gone",))
        self.registry._records[("user-a", orphan.id)] = orphan
        result = await lineage(self.registry, "user-a", [orphan.id], self.provenance)
        self.assertEqual(result["nodes"][0], {"handle": "obs_gone", "kind": "step", "note": "Not available."})
        self.assertEqual(self.provenance.calls, [])


if __name__ == "__main__":
    unittest.main()
