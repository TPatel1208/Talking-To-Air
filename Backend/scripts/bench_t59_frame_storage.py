"""T59 Phase 2 — storage/compression benchmark for the frame blob.

Answers, with real TEMPO frames rather than synthetic arrays:

  * what a frame stack costs raw and gzipped, at 10 / 24 / 60 / 100 frames
  * what compressing and decompressing it costs in time
  * what that implies for ``FRAME_STORE_MAX_BYTES`` and the D5 cell budget

Why real frames matter here: gzip on a float32 geophysical field is dominated
by the NaN fraction (a NaN run is a constant byte pattern and compresses to
nearly nothing) and by how smooth the surviving values are. A synthetic
``np.random`` stack is incompressible and would produce a storage budget several
times too large; a synthetic smooth ramp would produce one several times too
small. TEMPO frames sit between, and the two harvested regimes -- a regional
subset at ~16-22% NaN and a full-domain stack at ~65% NaN -- bracket what
production will actually store.

Frames are never repeated to reach a target N. A duplicated frame is free to
gzip and would silently flatter every number in the table, so a stack short of
its target is reported short rather than padded.

The Float32Array heap measurement is deliberately NOT here -- it belongs in
node (the frontend has no jsdom, so a simulated DOM would measure nothing real).
This script writes the ``.bin``/``.gz`` pairs that ``bench_t59_frame_heap.mjs``
then loads.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time
import zlib

import numpy as np

FRAME_COUNTS = (10, 24, 60, 100)
GZIP_LEVELS = (1, 6, 9)
# Bandwidths the download column is MODELED at. Nothing here measures a real
# network -- a loopback transfer would report the speed of memcpy and read as
# though the blob were free. Modeled numbers with the assumption stated beat a
# measured number that measures the wrong thing.
BANDWIDTHS_MBPS = (5, 25, 100)


def best_of(fn, n=3):
    """Fastest of n runs. Compression is CPU-bound and deterministic, so the
    minimum is the signal and anything above it is scheduler noise from the
    other containers on this host."""
    best, out = float("inf"), None
    for _ in range(n):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best, out


def load_pool(pool_dir: str, target: int, kind: str,
              only: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Every harvested stack at this cell budget, concatenated on frame.

    Only stacks sharing the modal frame shape are pooled: a regional bundle and
    a full-domain bundle coarsen to different grids, and concatenating them
    would silently produce a stack whose "cells per frame" is a fiction.
    """
    paths = sorted(glob.glob(os.path.join(pool_dir, f"*_{kind}{target}.npy")))
    if only:
        paths = [p for p in paths if any(s in os.path.basename(p) for s in only)]
    if not paths:
        return np.empty((0, 0, 0), dtype=np.float32), []
    arrs = [np.load(p) for p in paths]
    shapes = [a.shape[1:] for a in arrs]
    modal = max(set(shapes), key=shapes.count)
    keep = [(p, a) for p, a in zip(paths, arrs) if a.shape[1:] == modal]
    dropped = [os.path.basename(p) for p, a in zip(paths, arrs) if a.shape[1:] != modal]
    if dropped:
        print(f"    (excluded from the {modal} pool, different grid: {', '.join(dropped)})")
    stack = np.concatenate([a for _p, a in keep], axis=0).astype(np.float32)
    return stack, [os.path.basename(p) for p, _a in keep]


def measure(stack: np.ndarray, n_frames: int, out_dir: str, label: str) -> dict:
    sub = np.ascontiguousarray(stack[:n_frames])
    raw = sub.tobytes()
    cells = int(sub.shape[1] * sub.shape[2])
    nan_frac = float(1.0 - np.isfinite(sub).sum() / sub.size)

    row = {
        "label": label,
        "frames_requested": n_frames,
        "frames_actual": int(sub.shape[0]),
        "short": int(sub.shape[0]) < n_frames,
        "cells_per_frame": cells,
        "nan_fraction": round(nan_frac, 4),
        "raw_bytes": len(raw),
        "levels": {},
    }

    for level in GZIP_LEVELS:
        ct, blob = best_of(lambda: gzip.compress(raw, compresslevel=level))
        dt, back = best_of(lambda: gzip.decompress(blob))
        assert back == raw, "gzip round-trip changed the bytes"
        row["levels"][str(level)] = {
            "gzip_bytes": len(blob),
            "ratio": round(len(raw) / len(blob), 2),
            "compress_ms": round(ct * 1000, 2),
            "decompress_ms": round(dt * 1000, 2),
            "download_ms": {
                str(bw): round(len(blob) * 8 / (bw * 1e6) * 1000, 1)
                for bw in BANDWIDTHS_MBPS
            },
        }

    # The bytes node will actually load, written once at the level the endpoint
    # would serve (6 -- nginx/uvicorn default, and the ratio table below shows 9
    # buys almost nothing for several times the CPU).
    blob6 = gzip.compress(raw, compresslevel=6)
    base = os.path.join(out_dir, f"{label}_{n_frames}f_{cells}c")
    with open(base + ".bin", "wb") as fh:
        fh.write(raw)
    with open(base + ".bin.gz", "wb") as fh:
        fh.write(blob6)
    row["bin_path"] = base + ".bin"
    row["gz_path"] = base + ".bin.gz"

    # A NaN-heavy float32 field is the easy case for any compressor; note what a
    # naive alternative would cost so the choice of plain gzip is evidenced
    # rather than assumed.
    row["raw_float64_bytes"] = len(raw) * 2
    row["zlib_bytes"] = len(zlib.compress(raw, 6))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", default="/tmp/t59_frames")
    ap.add_argument("--out-dir", default="/tmp/t59_blobs")
    ap.add_argument("--cells", type=int, nargs="+", default=[20_000, 8_000])
    ap.add_argument("--kind", default="", help="filename infix, e.g. 'gran' for per-granule")
    ap.add_argument(
        "--only", nargs="*", default=None,
        help="restrict the pool to bundles whose filename contains one of these. "
             "The two coverage regimes must be measured separately: a full-domain "
             "TEMPO stack is ~66%% NaN and a regional subset ~16-22%%, and gzip on a "
             "float32 field is dominated by exactly that. Pooling them would report "
             "one ratio that describes neither.",
    )
    ap.add_argument("--label", default=None, help="override the label in output filenames")
    ap.add_argument("--json-out", default="/tmp/t59_storage.json")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = []

    for target in args.cells:
        stack, members = load_pool(args.pool_dir, target, args.kind, args.only)
        if stack.size == 0:
            print(f"\n  no harvested stacks for target {target}; skipping")
            continue
        label = f"{args.label or args.kind or 'bucket'}{target}"
        print(f"\n{'=' * 78}\n  budget {target:,} cells  ->  pool {stack.shape} "
              f"from {len(members)} bundle(s)")
        print(f"  pooled NaN fraction: {100 * (1 - np.isfinite(stack).sum() / stack.size):.1f}%")
        for n in FRAME_COUNTS:
            if stack.shape[0] < n:
                print(f"  N={n:<4} POOL SHORT: only {stack.shape[0]} real frames "
                      f"available; measured at {stack.shape[0]} and marked short "
                      f"(never padded -- a repeated frame gzips to ~0 and would "
                      f"flatter every number here)")
            row = measure(stack, n, args.out_dir, label)
            results.append(row)
            lv = row["levels"]["6"]
            print(f"  N={row['frames_actual']:<4} raw {row['raw_bytes'] / 1e6:>6.2f} MB   "
                  f"gz6 {lv['gzip_bytes'] / 1e6:>6.3f} MB  x{lv['ratio']:<5.2f}  "
                  f"comp {lv['compress_ms']:>7.1f} ms  decomp {lv['decompress_ms']:>6.1f} ms  "
                  f"dl@25Mbps {lv['download_ms']['25']:>7.1f} ms")
            if row["frames_actual"] < n:
                break  # every larger N would repeat this same measurement

    with open(args.json_out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  wrote {args.json_out} and blobs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
