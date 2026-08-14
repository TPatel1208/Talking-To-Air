"""T59 Phase 8 -- what the two shipped arrays say about each other.

Reads a live chart payload (the jsonb row as the frontend receives it) and the
gzip-inflated frame blob, and measures the D16 headline between the frames the
user scrubs and the frame-0 plane drawn above them. Both arrays are the ones
that ship; nothing here re-derives anything from the bundle.

Reproduce with (from anywhere with numpy, against a chart you own)::

    docker exec tta-db psql -U postgres -d talking_to_air_memory -A -t \
        -c "select payload from agent_charts where id='<chart_id>';" > payload.json
    curl -s -H "Authorization: Bearer $TOKEN" --compressed \
        "http://localhost:8000/chart/<chart_id>/frames.f32.gz" -o frames.bin
    python probe_t59_phase8_arrays.py payload.json frames.bin
"""
import json
import sys

import numpy as np


def d16(f, m, lats):
    """D16's metric, spelled as characterize_d4 and probe_t59_frame_stack spell
    it: a ratio of cos(lat)-weighted sums, never a mean of per-pixel ratios."""
    w = np.broadcast_to(np.cos(np.deg2rad(lats))[:, None], m.shape)
    ok = np.isfinite(f) & np.isfinite(m)
    den = float((np.abs(m)[ok] * w[ok]).sum())
    return (100.0 * float((np.abs(f - m)[ok] * w[ok]).sum()) / den) if den else float("nan")


def main(payload_path, blob_path):
    payload = json.load(open(payload_path))
    block = payload["frames"]
    shape = block["shape"]
    stack = np.frombuffer(open(blob_path, "rb").read(), dtype="<f4").reshape(shape).astype("float64")
    period = stack[block["period_index"]]
    frames = np.delete(stack, block["period_index"], axis=0)
    lats = np.array(block["lats"])

    print(f"tier={block['tier']} cadence={block['cadence']} "
          f"n_frames={len(block['frames'])} buckets/frame={block['buckets_per_frame']} "
          f"k={block['coarsen_k']} cells/frame={block['cells_per_frame']}")
    print(f"declared delta (measured at NATIVE resolution): {block['delta']}")

    mean_of_frames = np.nanmean(frames, axis=0)
    ok = np.isfinite(mean_of_frames) & np.isfinite(period)
    headline = d16(mean_of_frames, period, lats)
    rel = (np.abs(mean_of_frames - period) / np.abs(period))[ok]
    print("\nmean(frames) vs frame 0, ON THE SHIPPED ARRAYS")
    print(f"  D16 headline           : {headline:.4f} %")
    print(f"  max |F-M|              : {np.abs(mean_of_frames - period)[ok].max():.4e} {payload.get('units','')}")
    print(f"  cells compared         : {int(ok.sum())} of {period.size}")
    for q in (50, 90, 98, 100):
        print(f"  p{q:<3d} per-cell rel diff : {100 * np.percentile(rel, q):.3f} %")
    print(f"  share of cells >1%     : {100 * (rel > 0.01).mean():.1f} %")
    print(f"  share of cells >10%    : {100 * (rel > 0.10).mean():.1f} %")

    ramp_map = payload["vmax"] - payload["vmin"]
    lo, hi = block["value_range"]
    print("\nRisk 5, the two ramps for one field")
    print(f"  Map tab  {payload['vmin']:.4e} .. {payload['vmax']:.4e}  (width {ramp_map:.4e})")
    print(f"  scrubber {lo:.4e} .. {hi:.4e}  (width {hi - lo:.4e})")
    print(f"  scrubber ramp is {(hi - lo) / ramp_map:.2f}x the Map tab's")

    entries = block["frames"]
    qa = sum(1 for e in entries if e["n_granules"] == 0 and e["qa_pass_rate"] is not None)
    nr = sum(1 for e in entries if e["n_granules"] == 0 and e["qa_pass_rate"] is None)
    print(f"\nempty frames: {qa} qa-rejected + {nr} nothing-retrieved of {len(entries)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
