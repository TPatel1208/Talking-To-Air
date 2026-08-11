# T59 Phase 3 — the bucketed reduction, verified on real bundles

**Measured 2026-08-10** on materialized TEMPO NO2 L3 Harmony bundles, through the production
open + mask path (`_open_netcdf_bundle` → `AggregationService._resolve_and_mask`), inside the
`tta-backend` container. Source `docker cp`'d in and md5-verified on both sides before any
number below was taken.

Reproduce with:

```bash
docker exec tta-backend sh -c 'cd /app && python scripts/probe_t59_frame_stack.py /data/harmony/job_52a95bb4cb79e2ee/result.nc.zip'
```

| bundle | granules | valid after masking | grid | cadence |
|---|---|---|---|---|
| `job_52a95bb4cb79e2ee` | 34 | 28 | regional 535×658 | hourly |
| `job_c1122dfd051c15ee` | 49 | 43 | regional 535×658 | hourly |

---

## 1. The headline finding: the coarsened tier's D16 delta is 5–22%, not sub-1%

D16's discussion in the PRD illustrates the delta with *"0.4% is fine for one analysis and
material for another"*. On real TEMPO retrievals, forced into the coarsened tier at
`max_frames=12`:

| bundle | buckets/frame | **D16 headline** | max \|F−M\| |
|---|---|---|---|
| `job_52a95bb4cb79e2ee` | 5 | **22.27 %** | 9.58e16 molecules/cm² |
| `job_c1122dfd051c15ee` | 4 | **5.43 %** | 3.39e15 molecules/cm² |

**The mechanism is unequal observation per cadence bucket, and it is structural to TEMPO.**
Per-frame coverage on one bundle runs 4.6%, 34.4%, 100.0%, 100.0%, 99.99% across consecutive
frames — a scan that clipped the region's corner sits on the axis beside one that covered all
of it. Grouping buckets gives each *group* equal weight while the groups hold wildly different
amounts of real observation, so the mean of the coarse means is a materially different
aggregation from the period map.

This is exactly what D14 predicted in principle and it is much larger in practice than the
PRD's illustration implies. Two consequences:

- **D14's decision to compute the map independently in tier two is load-bearing**, not
  defensive. Deriving the map from coarse frames would have moved a real TEMPO map by 22%.
- **The disclosure is not a formality.** Risk 4 ("the delta becoming a badge nobody reads") is
  the live risk here, and Phase 7 should treat a double-digit delta as something the reader has
  to be shown rather than something available on request.

## 2. The coarsened tier is the COMMON case, not the exception

A **2.2-day** TEMPO retrieval at hourly cadence produces **54 frames**. The 60-frame budget is
therefore reached at roughly 2.5 days, and every longer scrub coarsens.

That sharpens Risk 3 (*"60 frames may be the wrong budget"*) into a concrete question: raising
the budget keeps more retrievals in the exact tier, and storage does not constrain it
(N=100 is 2.26 MB gzipped). This is still a UX judgment no benchmark settles, but it is now a
judgment with a threshold attached.

## 3. Tier one really is exact

| | `job_52a95bb4cb79e2ee` |
|---|---|
| frame 0 vs `_cadence_weighted_mean` | **0.000002 %** |
| `mean(frames)` vs frame 0 | **0.000002 %** |

Both residuals are float32 storage noise — the frames are stored narrowed, and that is the
floor. D4's guarantee holds on a real coverage pattern with real swath tiling: the period map
is the average of what the user scrubs, on the two arrays that ship.

## 4. Finding 3 is not hypothetical: 28 of 54 buckets are empty

TEMPO only scans in daylight, so on a padded 2.2-day span **28 of 54 hourly buckets hold no
contributing granule**. Under a `groupby` over the valid timesteps those 28 stops would have
vanished and the scrubber would have compressed two days into 26 stops with no gap visible
anywhere — the failure that "looks right".

The two empty states stay distinguishable on real data, which is what D10 asked for:

```
2025-06-14T00:00:00  n= 1  valid=0.3444  qa=0.3505  max=9.849e+17
2025-06-14T01:00:00  n= 0  valid=0.0000  qa=0.0000  max=--     <- observed, QA rejected all of it
2025-06-14T02:00:00  n= 0  valid=0.0000  qa=--      max=--     <- nothing retrieved
```

## 5. D9's pooled scale versus a union of per-frame clips

| bundle | pooled 2–98 | union of per-frame clips | union ramp is |
|---|---|---|---|
| `job_52a95bb4cb79e2ee` | −3.14e14 … 6.21e15 | −6.49e14 … 2.46e16 | **3.86×** as wide |
| `job_c1122dfd051c15ee` | 2.18e14 … 3.32e15 | −3.03e14 … 7.41e15 | **2.48×** as wide |

A union ramp 2.5–3.9× too wide would render every ordinary hour in the bottom third of the
colormap. Reusing the frontend's `computeSharedColorScale` here would have been a visible
defect, not a subtle one.

## 6. D10's coverage inflation, reproduced

| | native (shipped) | frame grid | inflation |
|---|---|---|---|
| `job_52a95bb4cb79e2ee`, k=(5,5) | 0.9380 | 0.9645 | **2.65 pp** |

Phase 2 measured 0.937 native against 0.965 block-meaned on the same regional regime. Phase 3's
shipped `valid_fraction` reproduces the native figure to the fourth decimal, confirming it is
computed before the block mean rather than after it.

## 7. Memory and laziness

| | `job_52a95bb4cb79e2ee`, 54 frames |
|---|---|
| `build_frame_stack` wall time | 9.2 s (against 12–15 s open+mask) |
| peak RSS | 385 MB |
| materialized frame stack | **3.05 MB** float32 |
| `N × native` float64 (must not appear) | 152 MB |
| realized cells/frame | 14,124 of the 20,000 ceiling (70.6 %) |

14,124 is the same realized count Phase 2 measured on this grid — `k = ceil(sqrt(total/target))`
quantizes, and the payload carries what was realized.

**One graph walk.** Frames, per-frame native statistics, coverage, the QA roll-up, the D16
delta and the pooled histogram all ride a single `compute()`, asserted in
`test_the_whole_stack_costs_one_pass_over_the_bundle`. The pooled scale is what would otherwise
have cost a second full I/O pass — a quantile is not a streaming reduction, so its histogram
bins cannot be chosen until the field's extremes are known. `apply_quality_mask` now measures
those extremes on the walk it was already doing (`value_min`/`value_max` in its counters), the
same shortcut `_fused_valid_flags` takes and for the same reason. Without them the stack costs
two passes, asserted separately so the trade stays visible.

### The full TEMPO domain does NOT fit, and the cause is not isolated

**Measured, and it is a real limit, not a probe artifact.** A single production-shaped
`build_frame_stack` (target 20,000 cells) on `job_d175709729a518f2` — 2950×5771, 54 hourly
buckets, `N × native` float64 = 7,355 MB — was **OOM-killed by the kernel** in the 3.9 GB
container, twice (cgroup `memory.events` `oom_kill`). Phase 2's bare
`groupby().mean().coarsen().mean()` survives the same bundle at **1,363 MB** peak, materializing
8.25 MB. So Phase 3's additions are what push it over.

**Which addition is unknown.** Stacking the consumers cheapest-first on a CONUS crop
(1250×3000, same 54 buckets, `--bbox -125 25 -65 50`) gives non-monotonic peaks:

| arm | peak RSS | baseline before the arm |
|---|---|---|
| 1 frames only (Phase 2's shape) | 572 MB | 451 MB |
| 2 + granule counts + coverage | **1,870 MB** | 532 MB |
| 3 + per-frame statistics | 923 MB | 894 MB |
| 4 + pooled histogram | 1,359 MB | 902 MB |
| 5 + frame 0, a cross-bucket reduction (= production) | 1,342 MB | 1,019 MB |

Arm 2 peaks highest and the complete production shape sits *below* arms 2 and 4, so adding
consumers does not monotonically raise the peak. The baselines climb 451 → 1,019 MB across
sequential computes in one process: glibc arena retention, the same effect Phase 2 recorded when
it said to use the total peak rather than the delta. **These peaks are allocator timing, not a
consumer signal.**

An earlier reading of this — that frame 0's cross-bucket reduction was the expensive term —
came from a first pass that ran that arm *second*, on top of arm 1's retained baseline. The
reorder above refutes it. Isolating the term properly needs one arm per process, which is
~40 minutes of full-domain I/O and repeatedly OOMs the shared container; it was not done.

**What this means for Phase 5.** D7's gate (`2 ≤ N ≤ 60`) is a frame-count gate and does not
bound spatial extent. A full-domain request currently reaches a reduction that will be killed.
Either the gate gains an extent limit refusing above some cell count via D3's existing
`CATEGORY_TOO_LARGE` backstop, or the reduction gets a cheaper chunking. Two levers exist and
neither is measured: rechunking the grouped intermediate spatially before the fused consumers
(each chunk is a full 2950×5771×8 = 136 MB slab today), and unfusing the pooled histogram back
into its own pass. Note also that `build_frame_stack` currently **recomputes** the period map
that `aggregate()` already produced — Phase 1b made `_cadence_weighted_mean` algebraically the
mean of the per-bucket means, verified here at 0.000002%, so accepting it from the caller would
remove duplicated work regardless of whether it helps the peak.

### RESOLVED in Phase 5, by the extent gate — `MAX_FRAME_NATIVE_CELLS = 4,000,000`

**Route taken: the gate, not a cheaper reduction.** The two levers above are still unmeasured,
and measuring them honestly needs one arm per process — ~40 minutes of full-domain I/O each, in
a container that OOMs on the attempt and has wedged Docker Desktop twice. Buying an unbounded
scrub with an unmeasured claim is exactly the trade §7 warns against, so Phase 5 refuses instead
and says so.

The constant is the **largest extent measured to complete**, rounded up, not a guess with a
safety factor: the CONUS crop (1250×3000 = 3.75 M cells) finished at 1,342 MB and the next
measured point up (17.0 M cells) was OOM-killed. It is enforced on the **narrowed** field, where
`_profile_scale_guard` takes its reading and for the same reason.

**What it costs:** the full-domain scrub, outright. A TEMPO request that is not narrowed to a
region gets its map and an explicit `extent_too_large` refusal instead of a slider. Either lever
above can raise the constant later — with a number attached.

A second backstop, on **span**, was added beside it: `MAX_BUCKETS_PER_FRAME = 24`, so
`MAX_SPAN_BUCKETS = 1,440`. D14's coarsening has no ceiling of its own and will fold a decade of
hourly buckets into 60 stops; expressing the limit as buckets *per frame* rather than as a
duration keeps a stop an interval a reader asked for (hourly reaches 60 days, daily ~4 years).

The period-map duplication was **not** removed. It is a change to `build_frame_stack`'s contract
and its 48 tests, taken for a memory saving nobody has measured, on a path the gate now bounds.
It stays available as the third lever.

---

## 8. Decisions taken in Phase 3

| | decision |
|---|---|
| Module | A new `preprocessing/frame_stack.py`, **not** an extension of `reduce_keeping_axes`. That function reduces every dim EXCEPT `keep`, so frames would mean `keep=("time","lat","lon")` — collapsing nothing. Grouping time and block-meaning lat/lon are additions to it, not uses of it. |
| Reuse | `reduce_keeping_axes(keep=("bucket",))` computes the per-frame REGIONAL statistics, which *is* its job. The cos-lat-weighted-mean / unweighted-everything-else split stays in one place, so a frame's mean cannot drift from the map's. |
| Pooled percentile | A 2^16-bin histogram over the pooled native distribution, one walk. An exact order-statistic quantile needs either N × native resident (forbidden) or two walks. At 2^16 bins the quantization is ~1.5e-5 of the field's range. |
| Coarse frames | A frame is the mean of its cadence buckets' MEANS, not of its granules — D4's rule one level up, so the tier that exists to fit long spans does not reintroduce granule weighting where sampling is most uneven. |
| Unknown cadence | Refused by name. A Harmony bundle member carries no `short_name`, so an unregistered product resolves to `"unknown"` — the same condition under which `_cadence_weighted_mean` degrades to a plain unweighted mean. Inventing a bucket width would label a scrubber with intervals the product never published. |
| `region_area` | Recorded at `mask_data_by_geometry` alongside `region_cells`, cos-lat weighted. A count cannot denominate an area-weighted numerator without describing two different fields at once. |

## 9. Open, not settled here

- ~~**The full-domain memory limit** (§7).~~ **CLOSED in Phase 5** by the extent gate
  (`MAX_FRAME_NATIVE_CELLS = 4,000,000`), not by isolating the term. The responsible consumer is
  still unknown and the two cheaper-reduction levers are still unmeasured; what changed is that
  a request above the largest measured-safe extent is now refused out loud instead of reaching a
  reduction the kernel kills. See §7.
- **Whether 60 is the right frame budget** (Risk 3), now with §2's threshold attached.
- **How prominently a double-digit D16 delta must be shown** (Phase 7, Risk 4).
- **`plot_singular` wiring, the gate and the refusal** — Phase 5, deliberately untouched.
