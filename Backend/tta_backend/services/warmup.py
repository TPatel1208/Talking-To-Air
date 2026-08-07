"""
services/warmup.py
==================
Pay the render path's one-time import cost at boot instead of charging it to
whoever asks the first question.

Measured 2026-08-06 on a cold process: the first ``compare`` spent ~0.44 s
importing dask submodules and ~0.10 s importing PIL, both pulled in lazily by
the aggregate-and-render chain downstream of the opens. Nothing about that work
belongs to the request that happens to arrive first -- it is fixed setup, and a
process has a whole idle startup in which to do it.

Re-measured 2026-08-07 from a booted process, which is the baseline that
matters: the dask half is ~0.4 s and still entirely lazy, but the PIL half has
fallen to ~0.01 s. Both numbers moved for the same reason -- imports migrating
to module scope -- and the split is worth stating because it is easy to read
the dask half as already paid for and drop it:

* ``open_handle`` imports top-level ``dask`` at module scope to bound the
  threaded scheduler. That covers 25 modules, none of them ``dask.array``. The
  34 ``dask.array.*`` modules a reduction actually goes through are untouched
  by it, so ``_warm_dask_reduction`` below is doing the same job it always was.
* ``plot_tools`` imports ``PIL.Image`` at module scope, which is the expensive
  part of PIL; only 8 format plugins are still lazy.

Warms by *performing* a miniature version of the real work rather than by
importing a hand-listed set of module names. A list would be written against
today's call graph and would rot silently the first time the render path
reached for something new; running the operations means whatever they touch
gets touched. ``tests/test_render_path_warmup.py`` pins that in a subprocess by
asserting a real comparison afterwards imports *nothing*.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


def warm_render_path() -> bool:
    """Force the lazy machinery the plot/compare path relies on.

    Returns whether it completed. Never raises: this is an optimization, and a
    boot that dies because a warm-up failed is strictly worse than a first
    request that is half a second slow.
    """
    started = time.perf_counter()
    try:
        _warm_dask_reduction()
        _warm_png_encoder()
    except Exception:  # noqa: BLE001 -- see docstring
        logger.warning("render_path_warmup_failed", exc_info=True)
        return False
    elapsed = time.perf_counter() - started
    logger.info(
        "render_path_warmed",
        extra={"_event": "render_path_warmed", "_duration_seconds": round(elapsed, 3)},
    )
    return True


def _warm_dask_reduction() -> None:
    """A masked, chunked reduction -- the shape every plot, statistic and
    comparison bottoms out in, and what drags in dask's array/scheduler/
    optimization submodules (34 of them under ``dask.array.*``, ~0.4 s; the
    top-level ``dask`` package open_handle already imports is not among them)."""
    import numpy as np
    import xarray as xr

    from tta_backend.preprocessing.aggregation_service import area_weighted_mean

    da = xr.DataArray(
        np.ones((4, 4), dtype="float64"),
        dims=("lat", "lon"),
        coords={"lat": np.arange(4.0), "lon": np.arange(4.0)},
        name="warmup",
    )
    try:
        chunked = da.chunk({"lat": 2, "lon": 2})
    except (ImportError, ValueError):
        chunked = da  # no dask installed; nothing to warm, and the path degrades too
    # ``.where`` then a reduction: the masking step, and the compute that forces
    # dask's scheduler and array machinery to resolve.
    float(chunked.where(chunked > 0).mean(skipna=True))
    # The real cos(latitude) weighted mean rather than a local imitation of it.
    # Calling the function the plot/stat/compare paths actually call is the
    # only version of this that stays true as that function changes -- a
    # hand-rolled copy would warm code nothing else runs.
    area_weighted_mean(chunked)


def _warm_png_encoder() -> None:
    """Encode one tiny RGBA frame, which is what the map-overlay renderer does
    at the end of every plot.

    Only ~0.01 s and 8 format plugins now that ``plot_tools`` imports
    ``PIL.Image`` at module scope. Kept anyway: it is cheap enough that the
    saving does not have to be large to be worth taking, and dropping it would
    leave the encoder as the one step of the render path nothing warms.

    Deliberately to an in-memory buffer: warming must not write into the
    overlay store, which is real, served content keyed by chart id.
    """
    import io

    import numpy as np
    from PIL import Image

    Image.fromarray(np.zeros((2, 2, 4), dtype="uint8"), mode="RGBA").save(
        io.BytesIO(), format="PNG"
    )
