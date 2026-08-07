"""The render path must not import anything on its first real request.

Diagnosed 2026-08-06: the first `compare` after a restart spent ~0.44 s
importing dask submodules and ~0.10 s importing PIL, both triggered lazily by
the aggregate-and-render chain downstream of the opens. That cost is paid by a
user, once per process, on a request that looks inexplicably slower than every
one after it.

Every assertion here runs in a **subprocess**. ``sys.modules`` is process-wide,
so an in-process check would pass or fail on whether some earlier test in the
same run happened to import dask first -- which is precisely the order
dependence that made the original timing test worthless.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import unittest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED = ("xarray", "numpy", "dask", "PIL")


def _run(*blocks: str) -> list[str]:
    """Run ``blocks`` in a clean interpreter; return the JSON list it prints.

    Each block is dedented independently, so callers can indent them to suit
    their own nesting without the pieces having to agree on a common level.
    """
    prelude = (
        "import json, sys\n"
        f"sys.path.insert(0, {BACKEND_DIR!r})\n"
        f"sys.path.insert(0, {os.path.join(BACKEND_DIR, 'tests')!r})\n"
    )
    script = prelude + "\n".join(textwrap.dedent(b).strip("\n") for b in blocks)
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise AssertionError(f"probe failed:\n{proc.stdout}\n{proc.stderr}")
    out = proc.stdout.strip().splitlines()
    if not out:
        raise AssertionError(f"probe printed nothing:\n{proc.stderr}")
    return json.loads(out[-1])


COMPARE_HARNESS = """
import asyncio
from unittest.mock import AsyncMock, patch
import xarray as xr
from tta_backend.tools.satellite_tools import comparison_tools

def make_ds(v):
    return xr.Dataset(
        {"no2": (("lat", "lon"), [[v, v], [v, v]], {"units": "mol/m^2"})},
        coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
    )

async def fake_open(handle, tools):
    return make_ds(1.0 if handle == "obs_a" else 2.0)

async def run_one_comparison():
    compare = comparison_tools.make_compare({})
    with patch("tta_backend.tools.satellite_tools.plot_tools.emit_chart", lambda p: None), \\
         patch.object(comparison_tools, "open_handle", AsyncMock(side_effect=fake_open)):
        return await compare.ainvoke(
            {"handle_a": "obs_a", "handle_b": "obs_b", "mode": "region"}
        )
"""


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in REQUIRED),
    "render-path warmup test dependencies are not installed",
)
class RenderPathWarmupTests(unittest.TestCase):
    # The stack whose first-use imports the warm-up exists to pay for, and the
    # only one this subprocess can speak to honestly. A real boot imports the
    # supervisor agent, which drags in 11 langchain_core.tracers submodules; a
    # comparison here pulls the two or three stragglers, but that costs 9.8 ms
    # in a booted process (measured) against 15 s to import the agent stack, so
    # it is neither worth warming nor evidence of anything.
    RENDER_STACK = ("dask", "PIL")

    def test_a_comparison_after_warmup_imports_nothing_from_the_render_stack(self):
        """The whole point: warm the path at startup and the first user request
        pays for none of its imports.

        Asserted by which modules arrive rather than by elapsed time,
        deliberately. Timing it would reintroduce a threshold that drifts with
        the machine and with the render path's weight -- the mistake the
        original compare-concurrency test made, where a 0.5 s budget silently
        became a measurement of these very imports. "Imported nothing from the
        render stack" is the property that makes the first request cheap, and
        it cannot go green for an unrelated reason."""
        new = _run(COMPARE_HARNESS, """
            from tta_backend.services.warmup import warm_render_path

            warm_render_path()
            before = set(sys.modules)
            asyncio.run(run_one_comparison())
            print(json.dumps(sorted(set(sys.modules) - before)))
        """)

        cold = [m for m in new if m.split(".")[0] in self.RENDER_STACK]
        self.assertEqual(cold, [])

    def test_without_the_warmup_a_comparison_does_import_the_render_stack(self):
        """The control, and the reason to trust the assertion above.

        Without it, that test would keep passing if these imports ever moved to
        module scope, or if the harness stopped exercising the render path at
        all -- reporting "the warm-up works" while it did nothing. This pins
        that there is real, measurable work for the warm-up to do: 34 dask
        submodules and 8 from PIL, ~0.44 s and ~0.10 s respectively when a user
        pays for them."""
        new = _run(COMPARE_HARNESS, """
            before = set(sys.modules)
            asyncio.run(run_one_comparison())
            print(json.dumps(sorted(set(sys.modules) - before)))
        """)

        cold = {m.split(".")[0] for m in new} & set(self.RENDER_STACK)
        self.assertEqual(cold, set(self.RENDER_STACK))


if __name__ == "__main__":
    unittest.main()
