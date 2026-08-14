# T59 Phase 12 — the reduction gets a second and third plane

**Built 2026-08-13.** The first implementation phase Phase 11's GO unblocks, and deliberately
the same scope Phase 3 had for the mean: `build_frame_stack` can now produce block-max and
block-min planes beside the existing block-mean one, proven correct and proven cheap, and
**nothing else in the codebase changed**. No store entry, no endpoint, no payload field, no
frontend, no `methods.md` prose. Those are Phase 13 and later.

## The interface

```python
build_frame_stack(..., statistics=("mean", "max", "min"))
```

Defaults to `("mean",)`, which is exactly today's behaviour — `FrameStack.planes` is `{}` and
every top-level field means what it has always meant. Anything else asked for lands in
`FrameStack.planes`, keyed by statistic, as a `StatisticPlane`:

| field | what it is |
|---|---|
| `values` | the float32 frames, block-**maxed**/**minned** (not block-meaned) |
| `period_values` | stop 0: the period max/min on the same grid by the same method (D6a decision 3) |
| `frame_grid_delta` | the agreement figure, reduced across frames by **this plane's own** statistic |
| `value_range` | this plane's own pooled 2–98 clip (D9) |
| `extent_overstatement` | D6a decision 9 — **`"max"` only**, `None` everywhere else |

`"mean"` is never a key. It stays in the top-level fields, unrenamed, so decision 5's *"the mean
entry keeps its exact current shape, URL and cost"* holds at the Python level and not only at
the wire level — a test asserts the mean tier's `values`, `period_values`, `delta`,
`frame_grid_delta`, `value_range` and per-frame statistics are identical with and without the
extra planes requested.

## Two things the phase prompt assumed that turned out to be false

**1. `_frame_grid_delta` was not statistic-agnostic.** It takes arbitrary arrays, but it
hardcodes the **mean** as the across-frame combiner (`total / contributing`). Applied verbatim
to a max plane it would compare *mean-of-frames* against the *period max* — a large non-zero
number, not the identity D6a decision 8 claims. G5's identity A is
`max-over-buckets(block-max)` vs `block-max(period native max)`, a **max** combiner. So the
helper gained a keyword:

```python
_frame_grid_delta(values, period_values, weights, *, combine="mean")
```

The mean path is unchanged to the bit and `FRAME_GRID_DELTA_BASIS` is untouched. Each plane
carries its own basis string from `PLANE_AGREEMENT_BASIS`, differing in exactly the word that
carries the claim — "max of the stored frame planes" where the mean's says "mean of". A shared
basis would have been a particularly quiet falsehood here: the max plane's number is `0.0` under
its own basis and would be large under the mean's, so the printed value would not even hint at
the mismatch.

**2. `_FRAME_STATS` and the new planes are not the same number, on a fixture that shows it.**
`_FRAME_STATS = ("mean", "min", "max")` is the per-frame *regional scalar*, reduced from the
frame's **mean** field. On the test fixture where one cell holds 100 at 00:00 and 1 at 12:00 of
the same day, frame 0's `statistics["max"]` is **50.5** — the largest value on the averaged
field — while the max plane's peak is **100.0**. A readout quietly re-pointed at the plane would
print 100 where it has always printed 50.5. The new constant is `PLANE_STATISTICS`, and its
docstring says in so many words that it is not `_FRAME_STATS` and why.

## The three quantities the max plane discloses, and the one it does not

`delta` is untouched and stays mean-only — Phase 9 verified it to 1e-12 and this phase reuses
`_frame_grid_delta` rather than editing it. What a selection plane carries instead:

- **`frame_grid_delta`, measured and exactly `0.0`.** Computed rather than asserted, because a
  promised zero and a measured one are different claims — the same reason `frame_grid_delta`
  exists at all. The test asserts it with `assertEqual`, not `assertAlmostEqual`: a tolerance
  here would hide a bug rather than accommodate float noise that does not exist, since `max`
  *selects* a value rather than accumulating one.
- **`extent_overstatement`, the tier's one real cost.** New, with its own name and its own
  `EXTENT_OVERSTATEMENT_BASIS` beside `DELTA_BASIS` and `FRAME_GRID_DELTA_BASIS` — three
  quantities, three accounts of themselves. It measures a different thing from either delta:
  *area painted* versus *area observed*, not value agreement. Max only; a minimum has no
  analogous overstatement story and measuring one would be inventing a number to be symmetric
  with.

Computed with `coarsen(...).construct(...)` rather than a second reduction, because each native
cell has to be compared against **its own block's** max — the block max has to come back down to
native resolution to be compared, and `construct` is the reshape that puts both in one lazy
expression. It is a view, not a materialization, so it rides the walk the frames were already
paying for.

## What was measured

Unit tests: 24 new, all through the public interface, every fixture a single hot cell against a
cold or empty block and every one at k=(2,2) or larger — never k=(1,1), where every block
reduction is the identity function and a pass proves only that `coarsen` was not called.

**Real bundles**, through `Backend/scripts/probe_t59_phase12_planes.py`, calling the generalized
`build_frame_stack` directly at Phase 8–11's bbox:

| | cadence bundle `job_a5c9813780a9300b` | Phase 11's gate |
|---|---|---|
| tier / k | cadence, group=1, k=(5,5), 14,124 cells | same |
| block max retention (p10/p50/p90/worst) | **100.00 / 100.00 / 100.00 / 100.00** | 100.0 everywhere |
| the shipped mean plane, for contrast | 15.29 / 41.22 / 73.62 / 6.70 | 15.29 / 41.22 / 73.62 / 12.23 |
| extent overstatement, pooled | **24.6985×** (worst frame 24.8599×) | 24.699× (per-frame max 24.860×) |
| `frame_grid_delta`, max and min planes | **`0.0` / `0.0`** | 0.0 |
| `frame_grid_delta`, mean plane | 0.0207 | — |

| | coarsened bundle `job_983d861c6a66eac1` | |
|---|---|---|
| tier | coarsened, group=3, 48 frames | |
| identity B, max plane | **exact on 366,215 finite cells, NaN pattern matches, max abs diff `0.0`** | 0.0 |
| identity B, min plane | **exact on 366,215 finite cells, max abs diff `0.0`** | — |
| `frame_grid_delta`, max and min | **`0.0` / `0.0`** | 0.0 |
| extent overstatement, pooled | 24.717× (worst frame 24.8605×) | 24.747× |
| `±inf` cells, either plane, frames or period | **0** | 0 |

### Three numbers that differ from the gate, and why each is expected

- **The mean row's "worst frame" (6.70% vs the gate's 12.23%).** Different quantities. The gate's
  "block mean" row is `block-mean(per-bucket MAX)`; this row is the **shipped** mean plane,
  `block-mean(per-bucket MEAN)`. They coincide at p10/p50/p90 to four figures because 24 of the
  26 cadence buckets hold exactly one granule, where a bucket's mean *is* its max — and they
  part company at the minimum, which is one of the two buckets that hold two.
- **The coarsened bundle's overstatement (24.717× vs 24.747×).** The gate measured on the
  per-bucket native max; this measures on `frames_native`, the temporally-grouped field that is
  actually rendered. The overstatement is a property of the rendered plane, so the grouped field
  is the right one to measure. 0.12% apart.
- **The coarsened bundle's group (3, not the gate's 2).** `_axis_starts` spans the whole
  *requested* range including the intervals nothing was retrieved for (finding 3), so the axis
  carries 144 buckets where the gate's `groupby` over survivors counted 79. Documented
  behaviour, not drift.

## Explicitly not done

`frame_store.py`, the `/chart/{id}/frames*.gz` endpoint, `_attach_frames`/`_wire_frames_url`,
`_chart_model_summary`, any frontend file, `methods.md`/`methods_export_service.py`,
playback, `MAX_FRAMES`, variable switching, per-frame export, comparison-panel scrubbers, frame
regeneration, the extent gate, `overlay_store` eviction. The sentence decision 8 promises and
the disclosure G4's number argues for are Phase 13's or later — this phase's job was only that
the numbers, computed, are correct.

## Reproducing

```bash
docker exec tta-backend sh -c 'cd /app && python scripts/probe_t59_phase12_planes.py \
    /data/harmony/job_a5c9813780a9300b/result.nc.zip \
    --bbox -106.6458459 25.83706 -93.5078217 36.5004529'
```
