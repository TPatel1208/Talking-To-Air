"""A map drawn through the harness, stored by the app's host, then fetched at the urls its chart row carries."""
from __future__ import annotations

import datetime
import gzip
import os
import sys
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import numpy as np

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import auth_helpers  # noqa: E402
from tests.postgres_helpers import PostgresTestCase, isolated_schema  # noqa: E402
from tests.test_harness_render import NJ, grid_dataset  # noqa: E402
from tta_backend.harness.capabilities import invoke  # noqa: E402
from tta_backend.harness.capabilities.render import maps  # noqa: E402
from tta_backend.harness.handles import Descriptor  # noqa: E402
from tta_backend.harness.handles.model import Period  # noqa: E402
from tta_backend.harness.storage import memory_stores  # noqa: E402
from tta_backend.harness.testing import make_run_context, make_run_env  # noqa: E402
from tta_backend.infrastructure.settings import get_settings  # noqa: E402
from tta_backend.repositories import chart_repository  # noqa: E402
from tta_backend.services import frame_store  # noqa: E402
from tta_backend.services.chart_service import ChartService  # noqa: E402
from tta_backend.services.harness_chat import StoredArtifacts  # noqa: E402
from tta_backend.services.open_handle import OPEN_PIPELINE_VERSION  # noqa: E402

NOW = datetime.datetime(2024, 7, 1, tzinfo=datetime.UTC)
OWNER = "user-a"
OTHER = "user-b"


class HarnessMapMediaTests(PostgresTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        stores = patch.dict(os.environ, {
            "FRAME_STORE_DIR": os.path.join(self._tmp.name, "frames"),
            "OVERLAY_STORE_DIR": os.path.join(self._tmp.name, "overlays"),
        })
        stores.start()
        self.addCleanup(stores.stop)
        get_settings.cache_clear()
        self.addCleanup(get_settings.cache_clear)

        import tta_backend.api as api

        self.api = api
        api.app.state.agent = object()
        api.app.state.earthdata_mcp_tools = {}

    @asynccontextmanager
    async def charts(self) -> AsyncIterator[None]:
        async with isolated_schema() as connect:
            with patch.object(chart_repository, "pg_connection", connect):
                await chart_repository.ensure_chart_table()
                yield

    async def draw(self, times: int) -> tuple[str, StoredArtifacts, list[Any]]:
        """render.map over a grid of ``times`` timesteps, stored by the app's host."""
        stores = memory_stores()
        sent: list[Any] = []
        artifacts = StoredArtifacts(ChartService(), "thread-1", OWNER, sent.append, stores.traces)
        context = make_run_context(user_id=OWNER, stores=stores, artifacts=artifacts)
        env = await make_run_env(os.path.join(self._tmp.name, "payloads"), context=context)
        grid = await env.resolver.mint(
            user_id=OWNER, conversation_id="conv-1", kind="grid", data=grid_dataset(times=times),
            descriptor=Descriptor(
                summary="NO2", product="TEMPO_NO2", unit="molecules/cm^2", region=NJ,
                masking={"variable": "vertical_column_troposphere", "qa_status": "verified", "region_cells": 20},
                period=Period(start="2024-06-01", end="2024-06-03"),
            ),
            operation="test.seed", created_at=NOW,
        )
        result = await invoke(env, maps.render_map, {"grid": grid.id})
        [chart_id] = result.charts
        return chart_id, artifacts, sent

    async def get(self, url: str, user: str) -> Any:
        import httpx

        transport = httpx.ASGITransport(app=self.api.app)
        with auth_helpers.patch_verifier():
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get(url, headers=auth_helpers.auth_header(user))

    async def test_a_harness_maps_frames_and_overlay_are_served_to_its_owner_only(self):
        async with self.charts():
            chart_id, artifacts, _ = await self.draw(times=3)
            row = await chart_repository.get_chart(chart_id)
            assert row is not None
            frames, overlay = row["frames"], row["overlay"]
            # The row the turn sends carries the same urls.
            sent_row = artifacts.stored_charts[chart_id]
            self.assertEqual((sent_row["frames"]["url"], sent_row["overlay"]["url"]), (frames["url"], overlay["url"]))
            self.assertEqual(frames["url"], f"/chart/{chart_id}/frames.f32.gz")
            self.assertEqual(overlay["url"], f"/chart/{chart_id}/overlay.png")
            self.assertTrue(overlay["rescalable"])
            self.assertEqual(row["export"]["frames"]["spec"]["pipeline_version"], OPEN_PIPELINE_VERSION)

            served = await self.get(frames["url"], OWNER)
            self.assertEqual(served.status_code, 200)
            stored = frame_store.read_frames(frames["_key"], pipeline_version=OPEN_PIPELINE_VERSION)
            assert stored is not None
            np.testing.assert_array_equal(
                np.frombuffer(served.content, dtype="float32").reshape(frames["shape"]),
                np.frombuffer(gzip.decompress(stored.gzipped), dtype="float32").reshape(frames["shape"]),
            )
            for statistic, plane in frames["planes"].items():
                self.assertEqual(plane["url"], f"/chart/{chart_id}/frames.{statistic}.f32.gz")
                self.assertEqual((await self.get(plane["url"], OWNER)).status_code, 200, statistic)
            png = await self.get(overlay["url"], OWNER)
            self.assertEqual(png.status_code, 200)
            self.assertTrue(png.content.startswith(b"\x89PNG"))

            for url in (frames["url"], frames["planes"]["max"]["url"], overlay["url"]):
                self.assertEqual((await self.get(url, OTHER)).status_code, 404, url)

    async def test_a_single_timestep_map_has_an_overlay_and_no_frames(self):
        async with self.charts():
            chart_id, artifacts, _ = await self.draw(times=0)
            row = await chart_repository.get_chart(chart_id)
            assert row is not None
            self.assertNotIn("frames", row)
            self.assertEqual(row["overlay"]["url"], f"/chart/{chart_id}/overlay.png")
            self.assertEqual((await self.get(row["overlay"]["url"], OWNER)).status_code, 200)
            self.assertEqual((await self.get(f"/chart/{chart_id}/frames.f32.gz", OWNER)).status_code, 404)
