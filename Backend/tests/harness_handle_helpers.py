"""Records, data and fakes shared by the handle, registry and lineage tests."""
import datetime

import numpy as np
import pandas as pd
import xarray as xr

from tta_backend.harness.handles import Descriptor, HandleRecord, InlineScalar, ServicePayload, new_handle_id
from tta_backend.harness.handles.model import StoredPayload
from tta_backend.harness.handles.payloads import relative_path

NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
LATER = datetime.datetime(2026, 1, 2, tzinfo=datetime.timezone.utc)

TRAVERSAL_IDS = [
    "../../etc/passwd",
    "..",
    "grid_../../x",
    "grid_" + "a" * 32 + "/../" + "b" * 32,
    "grid_" + "a" * 32 + "\n",
    "C:\\Windows\\win.ini",
    "/etc/passwd",
    "obs_../../secret",
    "obs_a/b",
    "",
]


def describe(summary: str = "A handle", **fields) -> Descriptor:
    return Descriptor(summary=summary, **fields)


def retrieval_record(user_id: str, handle_id: str = "obs_tempo_no2_1", kind="grid", summary="TEMPO NO2 over LA") -> HandleRecord:
    return HandleRecord(
        id=handle_id,
        kind=kind,
        origin="retrieval",
        user_id=user_id,
        conversation_id="conv-1",
        descriptor=describe(summary, quantity="no2_tropospheric_column", unit="molec/cm2", product="TEMPO_NO2_L3"),
        created_at=NOW,
        payload=ServicePayload(),
    )


def scalar_record(user_id: str, parents=(), value=1.5, parameters=None, created_at=NOW, run_id=None) -> HandleRecord:
    """A derived record that needs no payload file, for registry-only tests."""
    return HandleRecord(
        id=new_handle_id("scalar"),
        kind="scalar",
        origin="derived",
        user_id=user_id,
        conversation_id="conv-1",
        descriptor=describe("Mean NO2", unit="ppb"),
        parents=tuple(parents),
        operation="mean",
        parameters=parameters or {"dim": "time"},
        run_id=run_id,
        created_at=created_at,
        payload=InlineScalar(value=value),
    )


def table_record(user_id: str, parents=()) -> HandleRecord:
    handle_id = new_handle_id("table")
    return HandleRecord(
        id=handle_id,
        kind="table",
        origin="derived",
        user_id=user_id,
        conversation_id="conv-1",
        descriptor=describe("Exceedance days"),
        parents=tuple(parents),
        operation="exceedance_days",
        created_at=NOW,
        payload=StoredPayload(format="parquet", path=relative_path(user_id, handle_id, "parquet")),
    )


def grid_dataset() -> xr.DataArray:
    return xr.DataArray(
        np.arange(24, dtype="float64").reshape(2, 3, 4),
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2026-01-01", periods=2, freq="D"),
            "lat": [34.0, 34.1, 34.2],
            "lon": [-118.3, -118.2, -118.1, -118.0],
        },
        name="no2",
    )


def monitor_readings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            "site": ["06-037-1103"] * 4,
            "value": [31.0, 35.5, float("nan"), 40.0],
        }
    )


class FakeProvenance:
    """Answers like the app's provenance lookup: ancestors first, then the walked handle."""

    def __init__(self):
        self.calls = []

    async def __call__(self, user_id, source_handles):
        self.calls.append((user_id, list(source_handles)))
        nodes = {}
        for handle in source_handles:
            for node in (
                {"handle": "dataset_tempo_no2", "kind": "dataset", "depth": 2},
                {"handle": f"aoi_{handle}", "kind": "aoi", "depth": 1},
                {"handle": handle, "kind": "observation", "events": [{"event_type": "retrieved", "detail": {}, "created_at": "2026-01-01T00:00:00Z"}]},
            ):
                nodes.setdefault(node["handle"], node)
        return {"nodes": list(nodes.values())}
