"""
In-process fake earthdata-retrieval MCP server, for T02 tests.

Runs a real FastMCP server over streamable HTTP on a background uvicorn
thread, on a free local port, so the real MultiServerMCPClient (and the
langchain-mcp-adapters conversion code) connects to it over the wire exactly
as it would to the production harmony-retrieval-mcp stack. Nothing here
mocks the adapter library's internals — only the MCP server's tool handlers
are fake. Tool signatures mirror the real earthdata-retrieval MCP's schemas.
"""
from __future__ import annotations

import asyncio
import pathlib
import socket
import threading
import time
from typing import Any, Awaitable, Callable

import uvicorn
from fastmcp import FastMCP

# The curated raw tools the model-facing toolset should include.
CURATED_RAW_TOOL_NAMES = (
    "search_datasets",
    "describe_dataset",
    "preview_dataset",
    "summarize_dataset",
    "define_area_of_interest",
    "check_availability",
    "check_coverage",
    "get_retrieval_status",
    "retrieve_timeseries",
    "cite_dataset",
    "get_provenance",
)
# Used internally by the composites but never exposed to the model directly.
INTERNAL_RAW_TOOL_NAMES = (
    "retrieve_subset",
    "estimate_retrieval_size",
    "export_result",
    "rematerialize",
    "list_workspace",
    "cancel_retrieval",
    "align",
)
# Representative sample of raw tools that must never reach the model,
# per the PRD's "Hidden" list (transforms / format / inspection
# plumbing) — enough to prove filtering, not the full real-server surface.
HIDDEN_RAW_TOOL_NAMES = (
    "retrieve_data",
)
ALL_RAW_TOOL_NAMES = CURATED_RAW_TOOL_NAMES + INTERNAL_RAW_TOOL_NAMES + HIDDEN_RAW_TOOL_NAMES

Handler = Callable[..., Awaitable[dict]]


def _default_handler(name: str) -> Handler:
    async def _handler(**kwargs: Any) -> dict:
        return {"tool": name, "echo": kwargs}

    return _handler


def build_fake_mcp(handlers: dict[str, Handler] | None = None, exclude: tuple[str, ...] = ()) -> FastMCP:
    """Build a FastMCP instance exposing every raw tool name with a canned handler.

    ``handlers`` maps tool name -> an async callable receiving that tool's
    keyword arguments (workspace_id included) and returning a JSON-
    serializable dict. Tools not named in ``handlers`` fall back to a
    trivial echo handler. ``exclude`` drops named tools entirely, to
    simulate an MCP server that's missing a tool this backend requires.
    """
    handlers = handlers or {}
    mcp = FastMCP("fake-earthdata")

    def h(name: str) -> Handler:
        return handlers.get(name, _default_handler(name))

    @mcp.tool(name="search_datasets")
    async def search_datasets(query: str, filters: dict | None = None, workspace_id: str = "default") -> dict:
        return await h("search_datasets")(query=query, filters=filters, workspace_id=workspace_id)

    @mcp.tool(name="describe_dataset")
    async def describe_dataset(dataset_handle: str, detail: bool = False, workspace_id: str = "default") -> dict:
        return await h("describe_dataset")(dataset_handle=dataset_handle, detail=detail, workspace_id=workspace_id)

    @mcp.tool(name="preview_dataset")
    async def preview_dataset(
        dataset_handle: str,
        aoi_handle: str | None = None,
        time_range: str | None = None,
        layer: str | None = None,
        workspace_id: str = "default",
    ) -> dict:
        return await h("preview_dataset")(
            dataset_handle=dataset_handle,
            aoi_handle=aoi_handle,
            time_range=time_range,
            layer=layer,
            workspace_id=workspace_id,
        )

    @mcp.tool(name="summarize_dataset")
    async def summarize_dataset(handle: str, workspace_id: str = "default") -> dict:
        return await h("summarize_dataset")(handle=handle, workspace_id=workspace_id)

    @mcp.tool(name="define_area_of_interest")
    async def define_area_of_interest(location: str, workspace_id: str = "default") -> dict:
        return await h("define_area_of_interest")(location=location, workspace_id=workspace_id)

    @mcp.tool(name="check_availability")
    async def check_availability(
        dataset_handle: str, aoi_handle: str, time_range: str, workspace_id: str = "default"
    ) -> dict:
        return await h("check_availability")(
            dataset_handle=dataset_handle, aoi_handle=aoi_handle, time_range=time_range, workspace_id=workspace_id
        )

    @mcp.tool(name="check_coverage")
    async def check_coverage(
        dataset_handle: str, aoi_handle: str, time_range: str, workspace_id: str = "default"
    ) -> dict:
        return await h("check_coverage")(
            dataset_handle=dataset_handle, aoi_handle=aoi_handle, time_range=time_range, workspace_id=workspace_id
        )

    @mcp.tool(name="get_retrieval_status")
    async def get_retrieval_status(job_handle: str, workspace_id: str = "default") -> dict:
        return await h("get_retrieval_status")(job_handle=job_handle, workspace_id=workspace_id)

    @mcp.tool(name="retrieve_timeseries")
    async def retrieve_timeseries(
        dataset_handle: str,
        time_range: str,
        variables: list[str],
        aoi_handle: str | None = None,
        output_format: str | None = None,
        point_sample: bool = False,
        workspace_id: str = "default",
    ) -> dict:
        return await h("retrieve_timeseries")(
            dataset_handle=dataset_handle,
            time_range=time_range,
            variables=variables,
            aoi_handle=aoi_handle,
            output_format=output_format,
            point_sample=point_sample,
            workspace_id=workspace_id,
        )

    @mcp.tool(name="cite_dataset")
    async def cite_dataset(dataset_handle: str, workspace_id: str = "default") -> dict:
        return await h("cite_dataset")(dataset_handle=dataset_handle, workspace_id=workspace_id)

    @mcp.tool(name="get_provenance")
    async def get_provenance(handle: str, workspace_id: str = "default") -> dict:
        return await h("get_provenance")(handle=handle, workspace_id=workspace_id)

    @mcp.tool(name="retrieve_subset")
    async def retrieve_subset(
        dataset_handle: str,
        aoi_handle: str,
        time_range: str,
        variables: list[str],
        output_format: str | None = None,
        workspace_id: str = "default",
    ) -> dict:
        return await h("retrieve_subset")(
            dataset_handle=dataset_handle,
            aoi_handle=aoi_handle,
            time_range=time_range,
            variables=variables,
            output_format=output_format,
            workspace_id=workspace_id,
        )

    @mcp.tool(name="estimate_retrieval_size")
    async def estimate_retrieval_size(
        dataset_handle: str, aoi_handle: str, time_range: str, workspace_id: str = "default"
    ) -> dict:
        return await h("estimate_retrieval_size")(
            dataset_handle=dataset_handle, aoi_handle=aoi_handle, time_range=time_range, workspace_id=workspace_id
        )

    @mcp.tool(name="export_result")
    async def export_result(handle: str, workspace_id: str = "default") -> dict:
        return await h("export_result")(handle=handle, workspace_id=workspace_id)

    @mcp.tool(name="rematerialize")
    async def rematerialize(handle: str, workspace_id: str = "default") -> dict:
        return await h("rematerialize")(handle=handle, workspace_id=workspace_id)

    @mcp.tool(name="retrieve_data")
    async def retrieve_data(dataset_handle: str, workspace_id: str = "default") -> dict:
        return await h("retrieve_data")(dataset_handle=dataset_handle, workspace_id=workspace_id)

    @mcp.tool(name="align")
    async def align(source_handles: list[str], method: str = "outer", workspace_id: str = "default") -> dict:
        return await h("align")(source_handles=source_handles, method=method, workspace_id=workspace_id)

    @mcp.tool(name="cancel_retrieval")
    async def cancel_retrieval(job_handle: str, workspace_id: str = "default") -> dict:
        return await h("cancel_retrieval")(job_handle=job_handle, workspace_id=workspace_id)

    @mcp.tool(name="list_workspace")
    async def list_workspace(workspace_id: str = "default") -> dict:
        return await h("list_workspace")(workspace_id=workspace_id)

    for name in exclude:
        mcp.local_provider.remove_tool(name)

    return mcp


class HandleVolume:
    """A tmp-dir volume of real small Zarr/Parquet fixtures for open_handle tests.

    Models the export/eviction/rematerialize lifecycle against real files (no
    mocked I/O): ``export_result`` reports "ready" with a file:// storage_uri
    while the fixture exists on disk; ``evict`` deletes it to simulate LRU
    eviction, after which export_result reports "expired" until
    ``rematerialize`` regenerates it from the original factory and
    get_retrieval_status reports the job materialized.
    """

    def __init__(self, root):
        self.root = pathlib.Path(root)
        self._factories: dict[str, tuple[str, Callable[[], Any]]] = {}
        self.rematerialize_calls: dict[str, int] = {}

    def _path(self, handle: str) -> pathlib.Path:
        media_type, _ = self._factories[handle]
        ext = {"zarr": ".zarr", "parquet": ".parquet"}[media_type]
        return self.root / f"{handle}{ext}"

    def _write(self, handle: str) -> None:
        media_type, factory = self._factories[handle]
        path = self._path(handle)
        if media_type == "zarr":
            factory().to_zarr(path, mode="w")
        else:
            import pyarrow.parquet as pq

            pq.write_table(factory(), path)

    def add_zarr(self, handle: str, make_dataset: Callable[[], Any]) -> None:
        self._factories[handle] = ("zarr", make_dataset)
        self._write(handle)

    def add_parquet(self, handle: str, make_table: Callable[[], Any]) -> None:
        self._factories[handle] = ("parquet", make_table)
        self._write(handle)

    def evict(self, handle: str) -> None:
        import shutil

        path = self._path(handle)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    async def export_result(self, handle: str, workspace_id: str = "default") -> dict:
        if handle not in self._factories:
            return {"handle": handle, "status": "not_found", "message": f"Unknown handle '{handle}'."}
        path = self._path(handle)
        if not path.exists():
            return {"handle": handle, "status": "expired", "message": f"Handle '{handle}' has been evicted."}
        media_type, _ = self._factories[handle]
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else path.stat().st_size
        return {
            "handle": handle,
            "status": "ready",
            "storage_uri": path.resolve().as_uri(),
            "media_type": media_type,
            "size_bytes": size,
            "rematerialize_hint": None,
        }

    async def rematerialize(self, handle: str, workspace_id: str = "default") -> dict:
        if handle not in self._factories:
            return {"handle": handle, "status": "not_found", "message": f"Unknown handle '{handle}'."}
        self.rematerialize_calls[handle] = self.rematerialize_calls.get(handle, 0) + 1
        self._write(handle)
        return {"job_handle": f"job_{handle}", "obs_handle": handle, "status": "materialized"}

    async def get_retrieval_status(self, job_handle: str, workspace_id: str = "default") -> dict:
        handle = job_handle[len("job_"):] if job_handle.startswith("job_") else job_handle
        return {"job_handle": job_handle, "status": "materialized", "obs_handle": handle}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeEarthdataMCPServer:
    """Starts/stops a real FastMCP streamable-HTTP server in a background thread."""

    def __init__(self, mcp: FastMCP):
        self.mcp = mcp
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}/mcp"
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def start(self, timeout: float = 5.0) -> None:
        app = self.mcp.http_app(path="/mcp")
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning", lifespan="on")
        self._server = uvicorn.Server(config)

        def _run() -> None:
            asyncio.run(self._server.serve())

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._server.started:
                return
            time.sleep(0.02)
        raise RuntimeError("fake earthdata MCP server did not start in time")

    def stop(self, timeout: float = 5.0) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=timeout)
