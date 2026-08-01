"""Measure the mask/reduce pipeline three ways against a real bundle (T51).

T52 (the L4 Zarr cube) is blocked on a number nobody has: does the AOI crop's
slice push down to an h5netcdf hyperslab read, or does dask materialize the
whole single chunk (``chunks={}`` -> one chunk per variable per file, see
``services/open_handle._lazy_chunks``) and slice it in memory? If the read
pushes down, the cube needs no spatial chunking; if it doesn't, it does. That
is a measurement, not an argument.

    | case | description                | lands with |
    |------|----------------------------|------------|
    |  1   | current behavior (no crop) | baseline   |
    |  2   | crop-before-mask           | T50        |
    |  3   | crop + Zarr cube           | T52        |

Case 3 is a stub that skips with an explanation until T52 exists; the harness
is written once here and extended once there.

Every case runs the *production* code path -- ``mask_data_by_geometry`` on a
bundle opened by ``services.open_handle`` -- rather than a reimplementation,
so the numbers describe the system rather than the benchmark.

Run it inside the ``backend`` container, where PROJ/GDAL are correct (see
CLAUDE.md); a host run may not have a usable geospatial stack::

    docker compose exec backend python scripts/bench_pipeline.py \
        --variable vertical_column_troposphere --aoi=-74.9,39.1,-73.1,40.9

Note the ``=`` on ``--aoi``: a box starting with a negative longitude is
otherwise parsed as a flag ("expected one argument").

With no ``--bundle``, the newest entry in the bundle-extract cache
(``$TMPDIR/tta_bundle_extract``) is used -- i.e. whatever the last real turn
retrieved.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

CASES = {
    1: "current behavior (crop=False)   [baseline]",
    2: "crop-before-mask (crop=True)    [T50]",
    3: "crop-before-mask + Zarr cube    [T52]",
}

# Case 3 lands with T52 and is no longer a stub. It still *can* skip -- a
# dataset the cube writer refuses (or a cube that fails its own integrity
# check) has no case-3 number to report, and saying so is more useful than a
# fabricated row.


@dataclass
class CaseResult:
    number: int
    description: str
    skipped_reason: str | None = None
    median_seconds: float | None = None
    cells_in: int | None = None
    cells_reduced: int | None = None
    bytes_read: int | None = None
    value: float | None = None


@dataclass
class BenchmarkReport:
    bundle: str
    variable: str
    aoi: str
    members: int
    grid_shape: tuple
    runs: int
    products: list[str] = field(default_factory=list)
    cases: list[CaseResult] = field(default_factory=list)

    def case(self, number: int) -> CaseResult:
        return next(result for result in self.cases if result.number == number)


def _read_bytes_counter() -> int | None:
    """Bytes this process has read through syscalls, from ``/proc/self/io``'s
    ``rchar``.

    ``rchar`` (not ``read_bytes``) is the right counter here: the question is
    how much the reader *asked* for, and a page-cache hit -- which a repeat
    benchmark run always is -- satisfies the request without any disk I/O at
    all, so ``read_bytes`` would report zero for a full-chunk materialization.

    None off Linux, where the file doesn't exist. Reported as unavailable
    rather than as a fabricated zero: a zero here would read as "the crop
    pushed all the way down", which is exactly the conclusion this harness
    exists to establish honestly.
    """
    try:
        with open("/proc/self/io") as fh:
            for line in fh:
                if line.startswith("rchar:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


def _resolve_aoi(aoi: str):
    """A shapely geometry from either a named preset ("northeast us") or an
    explicit ``minx,miny,maxx,maxy`` box. Deliberately no geocoder: a
    benchmark must not depend on a network round-trip, and the T50 motivating
    case (a New-Jersey-sized AOI on a continental grid) has no preset."""
    from shapely.geometry import box

    from tta_backend.utils.plotting import RegionResolver

    parts = aoi.split(",")
    if len(parts) == 4:
        try:
            minx, miny, maxx, maxy = (float(part) for part in parts)
        except ValueError:
            raise SystemExit(f"Could not parse --aoi '{aoi}' as minx,miny,maxx,maxy.")
        return box(minx, miny, maxx, maxy)

    resolver = RegionResolver()
    preset = resolver.global_regions.get(aoi.strip().lower())
    if preset is None:
        raise SystemExit(
            f"Unknown --aoi '{aoi}'. Pass a preset name (e.g. 'northeast us') or an "
            "explicit box as minx,miny,maxx,maxy."
        )
    return resolver._finalize_preset(preset, aoi.strip().lower())["geometry"]


def _newest_cached_bundle() -> str:
    """The newest completed entry in the bundle-extract cache -- i.e. the
    members of whatever bundle the last real turn opened."""
    from tta_backend.services.open_handle import _EXTRACT_CACHE_DIR_NAME, _EXTRACT_COMPLETE_MARKER

    root = os.path.join(tempfile.gettempdir(), _EXTRACT_CACHE_DIR_NAME)
    try:
        entries = [
            entry
            for entry in os.scandir(root)
            if entry.is_dir() and os.path.exists(os.path.join(entry.path, _EXTRACT_COMPLETE_MARKER))
        ]
    except OSError:
        entries = []
    if not entries:
        raise SystemExit(
            f"No cached bundle found under {root}. Run a retrieval turn first, or pass "
            "--bundle with an explicit path to a bundle zip or a directory of NetCDF members."
        )
    return max(entries, key=lambda entry: entry.stat().st_mtime).path


def _open_bundle(path: str):
    """Open ``path`` through the production bundle-open path, whether it is a
    bundle zip or an already-extracted cache directory of members."""
    import xarray as xr

    from tta_backend.services.open_handle import (
        _EXTRACT_COMPLETE_MARKER,
        _lazy_chunks,
        _open_bundle_members_concurrently,
        _open_netcdf_bundle,
        _order_bundle_time,
        _strip_concat_unsafe_coord_attrs,
    )

    if not os.path.isdir(path):
        return _open_netcdf_bundle(path)

    names = sorted(
        name
        for name in os.listdir(path)
        if name != _EXTRACT_COMPLETE_MARKER and os.path.isfile(os.path.join(path, name))
    )
    if not names:
        raise SystemExit(f"Bundle directory '{path}' holds no members.")
    members = _open_bundle_members_concurrently(path, names, _lazy_chunks())
    if len(members) == 1:
        return members[0]
    combined = xr.concat([_strip_concat_unsafe_coord_attrs(ds) for ds in members], dim="time")
    return _order_bundle_time(combined, path)


def _member_count(path: str) -> int:
    import zipfile

    if os.path.isdir(path):
        return sum(1 for name in os.listdir(path) if not name.startswith("."))
    with zipfile.ZipFile(path) as zf:
        return sum(1 for name in zf.namelist() if not name.endswith("/"))


def _select_variable(ds, variable: str | None):
    if variable is None:
        names = list(ds.data_vars)
        if len(names) != 1:
            raise SystemExit(
                f"Pass --variable: this bundle holds {len(names)} data variables ({', '.join(names)})."
            )
        return ds[names[0]]
    if variable not in ds.data_vars:
        raise SystemExit(
            f"Variable '{variable}' is not in this bundle. Available: {', '.join(ds.data_vars)}."
        )
    return ds[variable]


def _products(ds) -> list[str]:
    """Whatever the granules call themselves, so a result names the product
    it describes instead of being an unattributable number."""
    keys = ("ShortName", "short_name", "title", "id", "LongName")
    found = [str(ds.attrs[key]) for key in keys if ds.attrs.get(key)]
    return found or ["(unnamed product)"]


def _time_one_run(da, geometry, crop: bool) -> tuple[float, int, int, int | None, float]:
    """One masked reduction. Returns (seconds, cells_in, cells_reduced,
    bytes_read, value).

    ``.compute()`` is what forces the lazy dask graph, so the span below is
    the read *and* the reduce -- which is the point: a crop that only trims
    an in-memory array after the whole chunk was materialized shows up here
    as bytes-read that didn't shrink.
    """
    import numpy as np

    from tta_backend.utils.plotting import mask_data_by_geometry

    cells_in = int(da.size)
    read_before = _read_bytes_counter()
    started = time.monotonic()
    masked = mask_data_by_geometry(da, geometry, crop=crop)
    value = float(np.asarray(masked.mean(skipna=True).compute()))
    elapsed = time.monotonic() - started
    read_after = _read_bytes_counter()

    bytes_read = None if read_before is None or read_after is None else read_after - read_before
    return elapsed, cells_in, int(masked.size), bytes_read, value


def _cube_and_reopen(ds, variable: str | None):
    """Write ``ds`` through the real cube writer and hand back the DataArray a
    cache *hit* would serve — same code path a third question takes, so case 3
    measures the shipped cache rather than a stand-in."""
    import tempfile
    import unittest.mock

    from tta_backend.config.settings import get_settings
    from tta_backend.services import cube_cache

    store = tempfile.mkdtemp(prefix="bench-cube-store-")
    with unittest.mock.patch.dict(os.environ, {"CUBE_STORE_DIR": store, "CUBE_WRITE_MAX_BYTES": str(64 * 1024 ** 3)}):
        get_settings.cache_clear()
        try:
            if not cube_cache.write_cube(ds, "bench"):
                return None, "the cube writer refused this dataset (see the cube_write_refused log event)"
            cached = cube_cache.lookup("bench")
            if cached is None:
                return None, "the cube did not survive its own integrity check"
            return _select_variable(cached, variable), None
        finally:
            get_settings.cache_clear()


def _run_case(number: int, da, geometry, runs: int) -> CaseResult:
    crop = number in (2, 3)
    samples = [_time_one_run(da, geometry, crop) for _ in range(runs)]
    return CaseResult(
        number=number,
        description=CASES[number],
        # Median, not a single sample: the first run pays cold page cache and
        # any one run can be caught by an unrelated scheduling stall.
        median_seconds=statistics.median(sample[0] for sample in samples),
        cells_in=samples[0][1],
        cells_reduced=samples[0][2],
        bytes_read=(
            None
            if any(sample[3] is None for sample in samples)
            else int(statistics.median(sample[3] for sample in samples))
        ),
        value=samples[0][4],
    )


def run_benchmark(bundle: str | None = None, *, variable: str | None = None, aoi: str, runs: int = 3):
    """Run every case against ``bundle`` and return a :class:`BenchmarkReport`."""
    geometry = _resolve_aoi(aoi)
    path = bundle or _newest_cached_bundle()
    ds = _open_bundle(path)
    da = _select_variable(ds, variable)

    report = BenchmarkReport(
        bundle=path,
        variable=str(da.name),
        aoi=aoi,
        members=_member_count(path),
        grid_shape=tuple(int(size) for size in da.shape),
        runs=runs,
        products=_products(ds),
    )
    cases = []
    for number in sorted(CASES):
        case_da = da
        if number == 3:
            case_da, reason = _cube_and_reopen(ds, variable)
            if case_da is None:
                cases.append(CaseResult(number=3, description=CASES[3], skipped_reason=reason))
                continue
        cases.append(_run_case(number, case_da, geometry, runs))
    report.cases = cases
    return report


def format_report(report: BenchmarkReport) -> str:
    lines = [
        f"bundle:    {report.bundle}",
        f"products:  {', '.join(report.products)}",
        f"variable:  {report.variable}",
        f"grid:      {'x'.join(str(size) for size in report.grid_shape)} "
        f"({report.members} member(s))",
        f"aoi:       {report.aoi}",
        f"runs:      {report.runs} (median reported)",
        "",
        f"{'case':<5}{'description':<44}{'median s':>10}{'cells':>12}{'MiB read':>10}{'value':>14}",
        "-" * 95,
    ]
    for result in report.cases:
        if result.skipped_reason:
            lines.append(f"{result.number:<5}{result.description:<44}  SKIPPED: {result.skipped_reason}")
            continue
        read = "n/a" if result.bytes_read is None else f"{result.bytes_read / 1048576:.1f}"
        lines.append(
            f"{result.number:<5}{result.description:<44}"
            f"{result.median_seconds:>10.3f}{result.cells_reduced:>12,}{read:>10}{result.value:>14.6g}"
        )

    baseline, cropped = report.case(1), report.case(2)
    if baseline.median_seconds and cropped.median_seconds:
        lines += ["", f"crop speedup: {baseline.median_seconds / cropped.median_seconds:.2f}x"]
        if baseline.bytes_read is not None and cropped.bytes_read is not None:
            # The question T52 is blocked on, answered in one line.
            pushed_down = cropped.bytes_read < baseline.bytes_read * 0.9
            lines.append(
                "read push-down: YES -- the crop's slice reached the reader "
                "(the cube needs no spatial chunking for this access pattern)"
                if pushed_down
                else "read push-down: NO -- the same bytes were read either way, so the whole "
                "chunk materializes and the crop only trims in memory (the cube needs "
                "spatial chunking)"
            )

    cubed = report.case(3)
    if cropped.median_seconds and cubed.median_seconds:
        lines.append(f"cube speedup over crop: {cropped.median_seconds / cubed.median_seconds:.2f}x")
        if cropped.bytes_read is not None and cubed.bytes_read is not None:
            saved = 1 - (cubed.bytes_read / cropped.bytes_read) if cropped.bytes_read else 0.0
            lines.append(
                f"cube read reduction: {saved * 100:.0f}% "
                f"({cropped.bytes_read / 1048576:.1f} -> {cubed.bytes_read / 1048576:.1f} MiB) "
                "-- this, not the wall clock, is the number T52 exists to move"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bundle",
        default=None,
        help="Bundle zip or directory of NetCDF members. Defaults to the newest "
        "entry in the bundle-extract cache.",
    )
    parser.add_argument("--variable", default=None, help="Science variable. Required unless the bundle has one.")
    parser.add_argument(
        "--aoi",
        default="-74.9,39.1,-73.1,40.9",
        help="Preset name ('northeast us') or minx,miny,maxx,maxy. Use --aoi=... "
        "for a box starting with a negative longitude, or argparse reads it as a "
        "flag. Defaults to a New-Jersey-sized box -- the T50 motivating case.",
    )
    parser.add_argument("--runs", type=int, default=3, help="Runs per case; the median is reported.")
    args = parser.parse_args()

    print(format_report(run_benchmark(args.bundle, variable=args.variable, aoi=args.aoi, runs=args.runs)))


if __name__ == "__main__":
    main()
