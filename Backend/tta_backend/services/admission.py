"""services/admission.py
=======================
How many heavy reductions may be in memory at once.

Every other memory gate in this backend bounds *one* request. ``frame_stack``
refuses above ``MAX_FRAME_NATIVE_CELLS`` because 17M cells was OOM-killed while
3.75M completed at 1,342 MB; ``open_max_chunk_bytes`` bounds a single dask
chunk. Both are measured, and neither says anything about the second request
that arrives while the first is still running.

Nothing did. Every heavy reducer hops to a worker thread via
``asyncio.to_thread`` (plot_tools, stat_tools, validation_tools,
comparison_tools) so they genuinely overlap, and the only ceiling was the
default executor's ``min(32, cpu_count + 4)`` -- a number derived from CPU
count, which has nothing to do with memory. ``settings.py`` already predicted
what that costs, next to ``open_max_chunk_bytes``:

    The number to watch when tuning is concurrency, not the table: this peak is
    per in-flight request while dask's num_workers cap is process-wide, so N
    researchers plotting continents at once pay it N times.

This module is the missing half of that sentence.

Why a semaphore *and* a private pool
------------------------------------
They bound different things and neither alone is sufficient.

A bare semaphore around ``asyncio.to_thread`` still draws its threads from the
process-wide default executor -- which is also where every request's JWT
verification runs (``api.py``). At N=5 on a two-core container that pool holds
six threads, so heavy work would occupy five of six and arriving requests would
queue *at authentication*. ``export_service`` already documented that exact
failure when it gave exports their own pool; repeating it here would undo that
fix for every other heavy path.

A bare pool has the opposite gap. ``run_in_executor`` hands back a future
immediately and offers no "started" callback, so a caller cannot tell whether it
is running or queued -- and a wait nobody can observe is indistinguishable from
the freeze this module exists to remove. The SSE contract carries ``status``
events and the 10s heartbeat (``utils/streaming.py``) holds the connection open,
but only if something knows to emit one.

So: the semaphore is the admission decision and the thing a caller can observe
waiting on; the executor is what keeps heavy threads out of the default pool.
Both are sized to the same N, which is why :func:`reset_admission` rebuilds them
together rather than letting the two numbers drift.

What is gated, and what deliberately is not
-------------------------------------------
**Gated:** the reductions -- every ``.compute()``/``.values`` path -- and CSV/
PNG/NetCDF export, which holds full-resolution grids of the same kind. Export
keeps its own ``_EXPORT_MAX_WORKERS`` pool for thread isolation, but that number
is now subordinate to this one: without sharing this counter the real ceiling
would be N reductions *plus* four exports, and N would be a number that bounded
nothing.

**Not gated:** ``open_handle``. Opens are lazy -- ``chunks=_lazy_chunks()``
throughout, with no ``.load()`` anywhere in that module -- so an open costs
threads, file descriptors and up to 8 GiB of *disk*, but its RAM is flat. Taking
a memory permit for a multi-minute zip extraction or an MCP rematerialization
would hold the scarcest resource in the process to do work that does not use it,
collapsing effective N under exactly the slow conditions where throughput
matters most. Bounding opens is a real problem; it is a different one, and
solving it here would mean this counter no longer measures memory.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import contextvars
import functools
import logging
import threading
from typing import Any, AsyncIterator, Callable

logger = logging.getLogger(__name__)

# The process floor: memory that is present whether or not anyone is plotting,
# and therefore not available to divide among concurrent reductions.
#
# Measured 2026-09-01 on the live container after three days of uptime:
# uvicorn's VmRSS was 204 MiB with a VmHWM of 346 MiB. Set above the high-water
# mark rather than at the resting value because CPython does not reliably return
# freed memory to the OS -- the floor creeps upward through allocator retention
# after each large reduction, so the number that matters is where the process
# sits *after* heavy use, not where it starts.
#
# Note this is far below the 800 MB an earlier estimate assumed. That estimate
# included ``MEMORY_CACHE_MAX_BYTES`` (500 MB), which was declared in settings
# and read by nothing. That setting has since been deleted rather than left to
# imply a reserve the process never holds -- see the note on
# ``open_max_chunk_bytes`` in config/settings.py, which now records the
# measured warm RSS so this estimate does not have to be made twice.
_RESERVE_MB = 450

# What one admitted request may cost at its worst.
#
# Not an average -- the largest reduction the per-request gates will pass.
# ``frame_stack`` measured 1,342 MB at 3.75M cells (and 1,308 MB for the
# three-statistic build at 1.05M cells) immediately below the extent it refuses.
# Sizing N on a typical request instead would admit five worst cases.
#
# The spread is wide: the same file measured 385 MB on a 535x658 regional grid.
# That waste is deliberate for now and is what a weighted-credit scheme would
# recover, using the native cell count the gates already compute.
_PER_REQUEST_MB = 1400

# How many callers may wait, as a multiple of how many may run.
#
# An unbounded queue does not prevent the failure, it changes its shape: under
# sustained overload every caller waits, spends its LLM tokens, and still loses
# the turn when CHAT_TURN_TIMEOUT_SECONDS (1800s) expires. Shedding past a bound
# means some callers succeed and the rest learn quickly. Three deep absorbs an
# ordinary burst -- nginx already rate-limits arrivals at 10r/s -- without
# letting the wait grow past what a person will sit through.
_QUEUE_DEPTH_MULTIPLE = 3


class AdmissionOverloaded(RuntimeError):
    """Raised instead of queueing when the wait queue is already full.

    Callers translate this into whatever their transport can say: a 503 with
    ``Retry-After`` on the export endpoints, and a terminal stream error on the
    chat path, where the SSE response is already committed and an HTTP status is
    no longer available.

    Deliberately *not* surfaced to the agent as a tool error. A supervisor that
    sees "busy" may simply call the tool again, which turns an overloaded
    backend into a load amplifier bounded only by ``agent_recursion_limit``.
    """


class _Limiter:
    """One generation of admission state.

    A single object rather than a handful of module globals so that
    :func:`reset_admission` can replace the whole generation atomically. Partial
    resets are the failure mode worth designing out: a counter cleared while its
    semaphore is not (or the reverse) reports a lie about capacity, and the
    symptom -- requests that hang rather than fail -- is the hardest kind to
    attribute.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.queue_capacity = limit * _QUEUE_DEPTH_MULTIPLE
        self.semaphore = asyncio.Semaphore(limit)
        self.in_flight = 0
        self.queued = 0


_limiter: _Limiter | None = None
_limiter_lock = threading.Lock()
_heavy_executor: concurrent.futures.ThreadPoolExecutor | None = None


def _derive_limit() -> int:
    """N, from the ceiling the deployment declared.

    Derived rather than hardcoded so the two cannot drift: raising ``mem_limit``
    raises throughput, and lowering it lowers admission. A hardcoded N with the
    ceiling in prose means someone doubling the container's memory gets nothing
    for it, and someone halving it gets an OOM this module was supposed to have
    prevented.

    ``HEAVY_ADMISSION_LIMIT`` overrides the derivation entirely. That is the
    operator's escape hatch for a host whose real headroom differs from what
    compose declares -- and the seam the tests use, since pinning concurrency
    behaviour to the arithmetic would make every one of them fail the day a
    measurement moves.
    """
    from tta_backend.config.settings import get_settings

    settings = get_settings()
    explicit = getattr(settings, "heavy_admission_limit", 0)
    if explicit > 0:
        return explicit

    ceiling_mb = getattr(settings, "backend_mem_limit_mb", 0)
    # Never zero. A ceiling smaller than one request's cost means this
    # deployment cannot serve that request at all -- but a limit of zero does
    # not express that, it deadlocks every heavy call forever while reporting
    # nothing. Admitting one and letting the per-request gates refuse it is the
    # honest failure: the caller gets a measured "too large", not a hang.
    return max(1, (ceiling_mb - _RESERVE_MB) // _PER_REQUEST_MB)


def _get() -> _Limiter:
    global _limiter
    if _limiter is None:
        with _limiter_lock:
            if _limiter is None:
                _limiter = _Limiter(_derive_limit())
                logger.info(
                    "admission_control_configured limit=%d queue_capacity=%d",
                    _limiter.limit, _limiter.queue_capacity,
                )
    return _limiter


def heavy_limit() -> int:
    """How many heavy requests may hold memory at once."""
    return _get().limit


def queue_capacity() -> int:
    """How many may wait before arrivals are shed."""
    return _get().queue_capacity


def in_flight() -> int:
    """How many are inside the heavy section right now.

    Zero when nothing has been admitted yet, rather than building a limiter to
    answer -- a read must not be the thing that fixes the configuration in
    place, or a metrics scrape during startup would pin N before the
    environment is fully assembled.
    """
    return 0 if _limiter is None else _limiter.in_flight


def queued() -> int:
    """How many are waiting for a permit right now."""
    return 0 if _limiter is None else _limiter.queued


@contextlib.asynccontextmanager
async def admit() -> AsyncIterator[None]:
    """Hold a memory permit for the duration of the block.

    Raises :class:`AdmissionOverloaded` immediately when the queue is already
    full, rather than joining it.

    Every exit path returns the permit, which is the property this whole module
    stands on. A leaked permit does not corrupt anything and does not raise --
    it removes one unit of capacity permanently, and after N leaks the backend
    accepts requests and answers none. The cases that matter are ordinary, not
    exotic: reductions raise routinely (a refused extent, an unreadable
    granule), and a client disconnect cancels in-flight work.

    Cancellation *while waiting* is the subtle one, and is handled by
    ``asyncio.Semaphore`` itself on 3.12: a waiter cancelled after being woken
    hands its permit to the next waiter rather than dropping it. The counter
    below must not double-count that -- a waiter holds nothing, so its
    cancellation decrements ``queued`` and touches nothing else.
    """
    limiter = _get()

    if limiter.queued >= limiter.queue_capacity:
        raise AdmissionOverloaded(
            f"{limiter.in_flight} heavy requests are running and "
            f"{limiter.queued} are already waiting."
        )

    limiter.queued += 1
    try:
        await limiter.semaphore.acquire()
    finally:
        # Runs whether the wait succeeded, failed, or was cancelled. Leaving it
        # to the success path would strand the count on cancellation, and a
        # queue that only ever grows sheds every later arrival forever.
        limiter.queued -= 1

    limiter.in_flight += 1
    try:
        yield
    finally:
        limiter.in_flight -= 1
        limiter.semaphore.release()


def _get_heavy_executor() -> concurrent.futures.ThreadPoolExecutor:
    """The pool heavy work runs on, sized to the same N as the semaphore.

    Separate from the default executor so that saturating it cannot queue an
    arriving request at JWT verification -- see the module docstring. Sized to N
    rather than larger because a thread past the Nth could never be admitted
    anyway; it would only be a thread waiting on a semaphore.
    """
    global _heavy_executor

    if _heavy_executor is None:
        with _limiter_lock:
            if _heavy_executor is None:
                _heavy_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=heavy_limit(),
                    thread_name_prefix="heavy",
                )
    return _heavy_executor


async def run_heavy(func: Callable[..., Any], *args: Any) -> Any:
    """Admit, then run ``func`` on the heavy pool.

    The context copy is not incidental, for the same reason ``export_service``
    documents at its own pool: ``run_in_executor`` starts the call with an empty
    context, and ``current_user_id()`` is a ContextVar the workspace-bound MCP
    tools read to decide whose data to open. Losing it does not raise -- the
    call reads the wrong workspace, or none -- so this reproduces exactly what
    ``asyncio.to_thread`` does for its callers.
    """
    async with admit():
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(
            _get_heavy_executor(), functools.partial(ctx.run, func, *args),
        )


def reset_admission() -> None:
    """Drop admission state, for tests.

    Registered in ``tests/cache_isolation.py`` rather than in a pytest fixture,
    for the reason that module documents: ``unittest`` never loads a conftest
    and CI runs ``unittest discover``. The stakes are higher here than for the
    caches it sits beside -- a leaked cache entry serves a stale answer, while a
    leaked permit hangs the runner with no attribution at all, somewhere inside
    a suite that takes around 29 minutes to get there.

    Replaces the limiter rather than mutating it, so a task still holding a
    permit from the previous generation releases it against the object it
    acquired from and cannot over-credit the new one.

    The executor is shut down without waiting: a heavy call still running has no
    reason to be interrupted, and blocking here would make an isolation hook the
    slowest thing in the suite.
    """
    global _limiter, _heavy_executor

    with _limiter_lock:
        _limiter = None
        executor, _heavy_executor = _heavy_executor, None

    if executor is not None:
        executor.shutdown(wait=False)

    # Settings are lru_cached, so a test that just patched BACKEND_MEM_LIMIT_MB
    # or HEAVY_ADMISSION_LIMIT would otherwise rebuild the limiter from the
    # values read before the patch -- and pass or fail on the wrong number.
    try:
        from tta_backend.config.settings import get_settings

        get_settings.cache_clear()
    except Exception:  # noqa: BLE001 - isolation must not gate collection
        pass
