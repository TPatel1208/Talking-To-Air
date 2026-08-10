# T59 Phase 2 — frame storage benchmark + Risk 2 verification

**Measured 2026-08-10** on real materialized TEMPO NO2 L3 Harmony bundles, through the
production open + mask path (`_open_netcdf_bundle` → `AggregationService._resolve_and_mask`),
inside the `tta-backend` container (8 cores, 3.9 GB RAM).

Reproduce with:

```bash
docker exec tta-backend sh -c 'cd /app && python scripts/probe_t59_lazy_coarsen.py /data/harmony/job_d175709729a518f2/result.nc.zip --arm fused'
```

| script | what it answers |
|---|---|
| `Backend/scripts/probe_t59_lazy_coarsen.py` | Risk 2 — does the block mean stay inside the lazy graph |
| `Backend/scripts/probe_t59_coarsen_edge.py` | is `boundary="pad"` eating the region edge |
| `Backend/scripts/harvest_t59_frames.py` | produces real coarsened frame stacks |
| `Backend/scripts/bench_t59_frame_storage.py` | raw/gzip size, compress + decompress time |
| `Backend/scripts/bench_t59_frame_heap.mjs` | `Float32Array` heap cost, in plain node |
| `Backend/scripts/bench_t59_cell_budget.py` | what each cell budget costs in fidelity |

Bundles used: `job_52a95bb4cb79e2ee` (34 granules, regional 535×658),
`job_c1122dfd051c15ee` (49, regional), `job_d175709729a518f2` /
`job_fe7dda06d55c5a60` / `job_28df9807db84b511` (36/36/35, full-domain 2950×5771).

### Re-verified against a freshly rebuilt image

Every number below was first produced with the scripts `docker cp`'d into the running
container, then **re-run in full after `docker compose build backend && docker compose up -d
backend`**, with the scripts baked in by the Dockerfile's `COPY . .` rather than copied in.
All six scripts hash-match the repo inside the image, and the recreated container starts with
an empty `/tmp`, so nothing carried over.

Reproduction was exact where it should be and noisy only where it must be:

- **Harvested frame arrays are byte-identical** — `job_52a95bb4cb79e2ee_20000.npy` re-harvested
  on the fresh image is md5 `647a1692…`, matching the pre-rebuild run, and the 8,000 stack
  matches too. The open → mask → groupby → coarsen path is deterministic.
- **Every size, ratio, retention and coverage figure is unchanged to the last digit**, across
  all three storage pools, both fidelity regimes, and the node heap arm.
- **Only wall-clock timings moved** (compress 244 → 225 ms, etc.), which is host scheduler
  noise, and the one RSS caveat noted in §1.

---

## 1. Risk 2 — the lazy graph. **VERIFIED. Phase 3 proceeds as designed.**

`groupby(bucket).mean().coarsen(k, boundary="pad").mean()` stays lazy and materializes at
the coarse shape. It does **not** materialize at native resolution, so the memory budget is
not wrong by ~250× and Phase 3 does not need a chunked rewrite.

Full-domain bundle, 36 granules → 32 hourly buckets, native 2950×5771 = 17,024,450 cells:

| | value |
|---|---|
| lazy before `.compute()` | **yes** — `.chunks` present, 32,042 graph tasks |
| materialized output | **(32, 99, 193) = 4.9 MB** |
| `N × native` (the shape that must not materialize) | **2,179 MB** float32 / 4,358 MB float64 |
| peak RSS during `frames.compute()` | **1,075 / 1,164 MB** total across two runs |
| wall time | 150.6 s / 148.8 s |

> The *delta over the pre-compute baseline* was +392 MB on the first run and +683 MB on the
> rebuild-verification run, because it depends on when glibc returns the mask stage's arenas —
> the pre-compute baseline itself read 683 MB and 481 MB respectively. **Use the total peak
> (~1.1 GB, stable to 8 % across runs), not the delta.** Either way it is well under the
> 2,179 MB the native stack would need *on top of* the same open+mask working set.

Paired control on the small bundle, where the native arm actually fits in RAM:

| arm | peak RSS | materialized |
|---|---|---|
| fused `groupby().mean().coarsen().mean()` | **261 MB** | 2.9 MB |
| control `groupby().mean()` then coarsen | **454 MB** | **73.2 MB** intermediate |

25× more materialization at 352k native cells, rising to ~445× at the full TEMPO domain.

### `boundary="pad"` is safe — verified, not assumed

The fused run reported the trailing coarse **row** as all-NaN, which is the exact signature
of the pad not being skipped. It is not that:

- Synthetic control: `[1,2,3,4,5].coarsen(x=3, boundary="pad").mean(skipna=True)` → `[2.0, 4.5]`.
  The pad is skipped. With `skipna=False` it gives `[2.0, nan]` — that is what not skipping costs.
- An all-NaN trailing block correctly stays NaN rather than becoming 0.
- On the real bundle, the 10 native rows under that trailing block hold **0 finite cells**
  (latitude ~73 °N, past TEMPO's coverage). NaN is the only honest answer there.
- The trailing **column** survives the coarsen: 189,288 native finite cells → 626 of 3,168
  coarse cells finite.

**Keep `boundary="pad"`.** `"trim"` would have silently dropped up to `k−1` = 29 rows and
columns off the region edge.

### Two things Phase 3 must carry

1. **The pipeline is float64 end to end.** `groupby().mean().coarsen().mean()` returns
   float64, so the frame blob's float32 is a deliberate narrowing Phase 4 must perform
   explicitly. It halves storage and is lossless in any sense that matters: observed values
   span −9.9e14 … 4.8e17 molecules/cm², and float32's ~7 significant digits put its
   resolution around 1e10 at the top of that range — roughly five orders of magnitude finer
   than TEMPO's own retrieval uncertainty (~1e15).
2. **The reduction is not the expensive part.** `_resolve_and_mask` on the full-domain bundle
   costs **146 s and peaks at 1,060 MB** — slower and larger than the entire frame
   computation after it. Pre-existing and not introduced by T59, but it qualifies D2's
   premise: frames are nearly free *given a computed map*, while open + mask remains the
   dominant cost of the request as a whole.

---

## 2. Storage and compression

Frames were harvested from real bundles, never synthesised and **never repeated to reach a
target N** — a duplicated frame gzips to nothing and would flatter every number here. Where a
pool ran short the row is reported short.

Two coverage regimes were measured separately, because gzip on a float32 geophysical field is
dominated by the NaN fraction and pooling them would report a ratio describing neither:

| regime | source | NaN | cells/frame at 20k target |
|---|---|---|---|
| **regional** (dense) | 2 subset bundles | **19.0 %** | 14,124 |
| **full-domain** (sparse) | 3 full-domain bundles | **66–67 %** | 19,107 |

### 20,000-cell budget, gzip level 6

| regime | N | raw | gzip | ratio | compress | decompress | download @25 Mbps |
|---|---|---|---|---|---|---|---|
| full-domain | 10 | 0.76 MB | 0.208 MB | ×3.68 | 15.6 ms | 2.8 ms | 66 ms |
| full-domain | 24 | 1.83 MB | 0.510 MB | ×3.60 | 43.6 ms | 7.8 ms | 163 ms |
| full-domain | 60 | 4.59 MB | 1.410 MB | ×3.25 | 117.2 ms | 20.7 ms | 451 ms |
| full-domain | 100 † | 7.64 MB | 2.256 MB | ×3.39 | 177.4 ms | 37.9 ms | 722 ms |
| **regional** | 10 | 0.56 MB | **0.400 MB** | **×1.41** | 33.4 ms | 3.8 ms | 128 ms |
| **regional** | 24 | 1.36 MB | **0.996 MB** | **×1.36** | 78.9 ms | 9.8 ms | 319 ms |
| **regional** | 53 ‡ | 2.99 MB | **2.141 MB** | **×1.40** | 244.0 ms | 24.4 ms | 685 ms |

† from the per-granule pool (107 real distinct frames); the cadence-bucket pool holds only 95.
‡ the regional pool holds 53 real frames; N=60 was not fabricated.

### 8,000-cell budget (D5 fallback)

| regime | N | raw | gzip | ratio |
|---|---|---|---|---|
| full-domain | 60 | 1.86 MB | 0.587 MB | ×3.17 |
| full-domain | 100 † | 3.10 MB | 0.939 MB | ×3.30 |
| regional | 24 | 0.69 MB | 0.512 MB | ×1.36 |
| regional | 53 ‡ | 1.53 MB | 1.099 MB | ×1.40 |

**Download times are modeled, not measured.** Nothing in this stack can measure a real
network; a loopback transfer would report the speed of `memcpy` and read as though the blob
were free. `bytes × 8 / bandwidth`, at 5 / 25 / 100 Mbps in the JSON.

### Compression level: use 6

| level | regional N=53 | full-domain N=60 |
|---|---|---|
| 1 | 2.163 MB, 172 ms | 1.319 MB, 86 ms |
| **6** | **2.141 MB, 244 ms** | **1.298 MB, 102 ms** |
| 9 | 2.141 MB, 259 ms | 1.298 MB, 189 ms |

Level 9 buys **0.0–0.2 %** for 1.1–1.9× the CPU. Level 1 costs 1.0–1.6 % for 9–29 % less CPU.
Level 6 (the nginx/uvicorn default) is correct and needs no override.

### Heap cost in node (v24.14.1, plain node, no jsdom)

| stack | gunzip | `arrayBuffers` | `heapUsed` |
|---|---|---|---|
| 100 frames × 19,107 | 41.7 ms | **+7.64 MB** | +0.07 MB |
| 60 frames × 19,107 | 22.0 ms | **+4.59 MB** | +0.01 MB |
| 24 frames × 14,124 | 15.3 ms | **+1.36 MB** | +0.00 MB |

**The measurement trap, confirmed:** a typed array's storage is not in `heapUsed`. Reading
only `heapUsed` reports a 100-frame stack as costing 0.07 MB and concludes frames are free.
Budget from `arrayBuffers`, which equals the raw byte count exactly.

**The reused-buffer decision (Phase 6) is worth exactly one stack.** Per-frame `subarray`
views cost **+0.00 MB and 0.01–0.15 ms for 100 views**; per-frame `slice` copies cost
**+7.64 MB**, a full duplicate. (A few rows read +0.00 MB for the copies where GC collected
them before the sample — the majority show the full duplicate, and that is the number to
plan against.)

Round-trip verified: node's finite fractions (34.0 %, 82.8 %) match Python's NaN fractions
(66.0 %, 19.0 %) on the same blobs.

---

## 3. Recommendations

### `FRAME_STORE_MAX_BYTES = 1 GiB` (fixed default, `_int_env` override)

Per-entry size is **bounded by construction**: D7's gate caps N at 60 and D5 caps cells at
20,000, so no entry can exceed `60 × 20,000 × 4 = 4.58 MiB` raw, or ~3.5 MB gzipped at the
worst level-6 ratio measured (×1.36, dense field). At 1 GiB the store holds:

| case | per chart | charts at 1 GiB |
|---|---|---|
| worst (dense field, N=60, full 20,000 cells) | ~3.5 MB | **~300** |
| measured typical (full-domain, N=60) | 1.30–1.41 MB | ~760 |
| short scrub (N=24) | 0.46–1.00 MB | 1,000+ |

A quarter of `cube_store`'s 4 GiB, which is the right proportion: a frame stack is a
rendering convenience that D8 already refuses to regenerate, while a cube is minutes of
recompute. Fixed bytes, **not** a percentage of free space — that is how `docker_data.vhdx`
reached 296 GB, and `settings.py` already carries that lesson.

**Account by real bytes on disk, storing gzipped.** `cube_write_max_bytes` was measured
against `ds.nbytes` while the store accounted by disk bytes, and that mismatch charged cubes
up to 5× what they cost. Frames should not repeat it.

**No per-entry cap is needed** — a difference from `cube_store` worth stating explicitly.
`cube_write_max_store_fraction` exists to prevent thrash from an entry big enough to evict the
whole store; at ≤4.58 MiB against 1 GiB that cannot happen, so the frame store needs an LRU
sweeper and nothing else.

### Cell budget: **keep 20,000**, with 8,000 a live fallback

Two facts first.

**Integer coarsening quantizes the realized budget.** `k = ceil(sqrt(total/target))` lands
you at 70–96 % of the target, not at it: a regional 535×658 grid gives 14,124 cells at k=5
(k=4 would give 22,110, over budget), and the full domain gives 19,107 at k=30. "20,000
cells" is a ceiling, and the payload should carry the realized count rather than the target.

**Fidelity per byte is weak above 8,000.** Measured against the same frames at native
resolution:

| | regional, p98 retained | full-domain crop, p98 retained |
|---|---|---|
| 20,000-cell block mean | 96.97 % | 84.21 % |
| 8,000-cell block mean | 96.73 % | 81.34 % |

2.5× the bytes buys 0.2–2.9 percentage points of p98 retention.

The case for 20,000 is therefore **not** statistical — it is that 8,000 cells over the full
TEMPO domain is a 63×123 map, visibly blocky for something a user scrubs and pans. At the
measured worst case (N=60, dense) 20,000 costs 3.4 MB gzipped and 0.7 s to download at
25 Mbps, which is affordable for a one-time fetch. Keep it, and treat 8,000 as the lever to
pull if the store budget or a slow-network complaint ever makes it the better trade — it
costs ~3 pp of p98 and nothing else.

---

## 4. Two findings that correct the PRD

### D5's stated rationale is not what the data shows

D5 says *"A stride deletes plumes; a block mean preserves their mass (finding 4)."* Measured
against native resolution on both regimes, at both budgets, the block mean is **worse than
the incumbent stride on every percentile** and only marginally better on the mean.

> The fidelity arm needs the native stack resident to compute a per-frame percentile, so the
> full-domain bundle was cropped to a CONUS-west box (−120…−90, 30…45 → 750×1500, k=8) rather
> than run at 2950×5771, where 32 buckets of float64 is 4.4 GB in a 3.9 GB container — the
> same materialization §1's control arm refused. Retention is a property of the field's
> spatial structure, not the domain's extent, so the crop does not bias the comparison; it
> does mean `k` here is 8, not the 30 the storage arm used.

| full-domain crop, 20,000-cell budget (k=8) | block mean | stride |
|---|---|---|
| single-cell max retained (p50) | 29.8 % | **50.6 %** |
| p99 retained (p50) | 83.8 % | **99.5 %** |
| p98 retained (p50) | 84.2 % | **99.7 %** |
| spatial mean retained (p50) | **100.45 %** | 99.16 % |

The mechanism is straightforward in hindsight: a stride *samples* native values, so whatever
it lands on it reproduces exactly; a block mean averages every peak down against its `k²−1`
neighbours. The stride's weakness is variance, not attenuation — its max retention runs from
p10 = 10.8 % (missed the peak entirely) to p90 = 74.0 % (landed on it).

**This does not overturn D5.** The block mean is still the right choice: it preserves the
spatial mean, it degrades predictably rather than by lottery, and it does not alias. But its
justification should be restated as *predictable attenuation and mass preservation*, not
*peak preservation* — and the practical consequence is that **PRD finding 5 binds frames too**:
per-frame statistics must be computed pre-downsample, because at k=8 the frame grid has
already lost 16 % of its own p98 and 70 % of its max.

### Block-mean coarsening inflates apparent coverage — a D10 hazard the PRD did not anticipate

| median finite fraction | native | block mean | stride |
|---|---|---|---|
| regional, 20,000 (k=5) | 0.937 | **0.965** | 0.938 |
| full-domain crop, 20,000 (k=8) | 0.947 | **0.996** | 0.949 |
| full-domain crop, 8,000 (k=12) | 0.947 | **0.998** | 0.949 |

A block is finite if *any* cell in it is finite, so block-meaning a sparsely observed hour
produces a frame that looks 99.6 % covered when the truth is 94.7 % — and it gets worse as the
budget shrinks, because bigger blocks are likelier to catch one observation.

This is precisely the failure D10 exists to prevent, arriving through the downsample rather
than through masking. **`valid_fraction` must be computed at native resolution on the analyzed
region and carried into the frame payload — never derived from the frame grid**, which would
report near-full coverage for exactly the sparse intervals a scrubbing user most needs warned
about.

---

## 5. Decisions carried into the PRD

Recorded in [prd-t59-timescale-viewer.md](prds/prd-t59-timescale-viewer.md) so Phase 3 does not
have to re-derive them:

| | decision |
|---|---|
| Architectural Constraint | Keep the lazy `groupby().mean().coarsen().mean()` composition. **Do not hand-roll a chunked reduction** — it would add real complexity to solve a measured non-problem. Risk 2 retired. |
| D5 | 20,000 is a **ceiling**; the payload records the **realized** cell count. Rationale restated as deterministic aggregation with predictable degradation, **peak attenuation explicitly accepted**. |
| D5a (new) | **Every scientific quantity is computed at native resolution before the reduction; the float32 frame array is for rendering and storage only and nothing is derived from it.** |
| D10 | `valid_fraction` is a **correctness requirement**, computed natively pre-downsample — never from the frame grid. |
| Phase 4 | `FRAME_STORE_MAX_BYTES = 1 GiB`, fixed bytes; account by disk bytes storing gzipped at level 6; **no per-entry cap** (entry size is bounded by construction). |
| Phase 6 | Zero-copy `subarray` views, never `slice`; budget from `arrayBuffers`, never `heapUsed`. |

## 6. Open, not settled here

- **60 frames as the slider budget (Risk 3).** Storage does not constrain it: N=100 is
  2.26 MB gzipped and 42 ms to gunzip. Whether 60 stops is usable is a UX judgment no
  benchmark settles.
- **`overlay_store`'s missing eviction policy** (PRD finding 9) remains real and tracked
  separately.
