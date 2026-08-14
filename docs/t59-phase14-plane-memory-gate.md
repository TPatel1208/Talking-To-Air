# T59 Phase 14 — the wiring, behind a second extent gate

**Measured 2026-08-13/14, then built. The measurement changed the phase's shape.**

Phase 14 was to turn the planes on: `_attach_frames` asking `build_frame_stack` for
`("mean","max","min")`, three store entries per chart, three URLs in the payload, and the toggle
named to the agent. Its own prompt made that conditional on one measurement, and the measurement
came back against it. **`build_frame_stack` with three statistics is OOM-killed by the kernel at
every extent above ≈1.4 M native cells per interval**, including the exact extent
`MAX_FRAME_NATIVE_CELLS` was derived from.

The unconditional wiring the prompt described was therefore not built, and the decision went back
with the numbers below. **The answer was the statistic-aware gate**, and that is what shipped: a
second constant `MAX_PLANE_NATIVE_CELLS = 1,000,000` and a `plane_gate` beside `frame_gate`, so a
chart above the line **keeps exactly the mean scrubber it has today** and loses only the toggle —
with a disclosure saying which limit it hit. Lowering the existing constant instead would have
refused every chart between 1 M and 4 M cells outright, taking away a scrubber that works to pay
for one that does not exist yet.

§"What was built" is at the bottom; the measurement comes first because it is what decided the
shape.

## What was measured

`Backend/scripts/probe_t59_phase14_plane_memory.py`, in the live 3.9 GB `tta-backend` container,
**one arm per process** (peak RSS is `VmHWM`, a high-water mark that never falls, so two arms
sharing a heap measure the larger one twice — Phase 11's own harness note at :263). It calls the
real `build_frame_stack` with the real `statistics=` argument rather than rebuilding the
composition by hand, so what it measures is the code that shipped, not a model of it.

The bundle is the full-domain TEMPO NO2 open `job_d175709729a518f2` (2950×5771), centre-cropped to
the extent under test, 48 hourly buckets, `frame_gate` passing on every arm.

| native cells / interval | 1 statistic | 3 statistics | result |
|---|---|---|---|
| 1,050,000 (700×1500) | **681.8 MB**, 28.6 s | **1,308 MB**, 84.2 s | both complete — **1.92× peak RSS** |
| 1,400,000 (800×1750) | — | **OOM-killed**, exit 137 | dies |
| 1,800,000 (900×2000) | — | **OOM-killed**, exit 137 | dies |
| 3,750,000 (1250×3000) | **1,343.5 MB**, 92.2 s | **OOM-killed**, `oom_kill` | 1 statistic only |
| **4,000,000** = `MAX_FRAME_NATIVE_CELLS` | **never measured, at any arm count** | — | — |

Every kill is a cgroup `oom_kill`, confirmed in `/sys/fs/cgroup/memory.events` (0 → 3 over the
session), presenting exactly as §7's original blocker did: no traceback, no result line, a missing
process. `_attach_frames`' catch-all could not have caught any of them. The container and the
backend app survived each one — the kernel picked the probe, as the largest RSS.

## The three things this says

**1. Phase 11's ratio is right, and it was never the question.** 1,308 / 681.8 = **1.919×**,
against Phase 11's 1.91× measured on a regional bundle a third the size. The ratio reproduces
almost exactly at 3× the extent; D6a decision 6's single fused compute is behaving precisely as
G1 measured it. **What fails is headroom, not fusion.** 1.919 × 1,343.5 MB = **2,578 MB** needed at
the gate's anchor extent, in a 3,916 MB container whose backend app already holds ~2,100 MB
resident. There is no 2.6 GB to be had, and no amount of confirming G1 creates one.

**2. The constant was already at the edge, and had nothing left to spend.** This run reproduces
Phase 5's anchor measurement to within 0.1% — **1,343.5 MB against its recorded 1,342 MB** at the
same 1250×3000 — which is what makes the rest of the table trustworthy. But read what that number
*is*: Phase 5 set `MAX_FRAME_NATIVE_CELLS` at the largest extent it measured to complete, and that
extent completes by consuming essentially all the container's spare memory. A constant chosen as
"the last thing that survived" has **zero margin by construction**, so *any* increase in the
reduction's cost pushes the admitted maximum over — 1.9× was never needed to break it.

**And 4,000,000 itself has never been measured, at one statistic or three.** Phase 5 measured
3.75 M and rounded *up* by 6.7% to a round number. Today's gate therefore admits a band
(3.75 M–4.0 M) that has never been demonstrated to survive even the mean-only build it was written
for. That is a pre-existing finding, not one Phase 14 created, and it is small next to the rest.

**3. The survivable extent for three planes is ≈26% of the constant.** The bracket is
**[1,050,000 survives, 1,400,000 dies]**. A three-statistic build at 1.05 M costs about what a
one-statistic build at 3.75 M does — the same wall, reached at a quarter of the ground.

Wall-clock ratio at 1.05 M was **2.94×** (84.2 s vs 28.6 s), well above Phase 11's 1.48–1.64×.
**Treat that as soft**: the two arms ran at different points in the session and the one-statistic
arm ran last, with the bundle's pages warm, so I/O caching confounds it. The RSS ratio is not
affected by page cache and is the number to rely on.

## What this does *not* say

- **Nothing about the reduction's correctness.** Phase 12's numbers were not re-run and cannot
  have moved; `_plane_terms`, `_planes`, `_block_reduce` and every other line of the reduction are
  untouched — the only edit to `frame_stack.py` is a new constant and a new gate function. Block-max
  retention, the 24.6985× overstatement and both bit-exact identities stand as measured.
- **Nothing was changed in `frame_store.py`.** Phase 13 finished it, and it needed nothing: it
  already writes every plane, protects the mean from its own planes' evictions, and degrades one
  statistic at a time. Its `STORABLE_STATISTICS` and `PLANE_STATISTICS` mirror still holds.
- **Nothing against D6a.** Decision 6 ("the build stays eager, one compute") is *confirmed* by
  this run, not challenged: the fused three-statistic graph costs 1.92×, not 3×. Making the build
  lazy was explicitly not attempted — D2/D8 forbid regeneration, so laziness is a transport
  decision and never a compute one, and re-litigating that here because it would be convenient is
  how a PRD stops meaning anything.
- **Nothing about the charts people actually plot.** Phase 8 and Phase 11's live regional TEMPO
  bundles are 535×658 = **352,181 cells**, a third of the largest measured survivor. Phase 11
  measured three statistics there at +32.5 MB and +84.4 MB. The dead band is 1.4 M–4.0 M — the
  large-region tail, not the common case.
- **The measurement is not a laboratory one, deliberately.** The probe runs as a second process
  beside the live backend, which is how Phase 5's own 1,342 MB and its full-domain OOM verdict
  were reached, so the comparison is consistent with the constant's provenance. It does cost a
  separate interpreter and xarray import (~280–420 MB of `rss_before`) that a production build
  would not add on top of the app's own. A production build has that much *more* room — and it is
  nowhere near the ~1.2 GB shortfall at 3.75 M.
- **1,050,000 is a survivor, not a safe constant.** It survived once, at 1,308 MB, with roughly
  300 MB spare in a container whose app baseline moved between 2,045 MB and 2,447 MB during this
  session. A constant should sit below it, not at it.

## The decision this handed back, and how it went

Two options, and the cost of each was a number rather than a guess.

**A — lower `MAX_FRAME_NATIVE_CELLS` to ~1,000,000.** One constant, no new concept, and the
refusal text already explains itself. **It charges existing users for a feature nobody has yet:**
a chart at 3.75 M native cells gets a working mean scrubber today, measured at 1,343.5 MB, and
would begin being refused outright. Everything from 1.0 M to 4.0 M loses the scrubber it has.

**B — a statistic-aware extent gate.** Keep 4,000,000 for the mean; admit the extra planes only
below ~1,000,000. No existing chart loses anything, and every regional chart measured in this
project (352 k cells) is comfortably inside the plane band. The cost is conceptual: a second
threshold beside `frame_gate` and a two-tiered auto-upgrade in `_attach_frames`.

**B was chosen.** A is a regression to today's behaviour paid to enable tomorrow's, and the band
it would sacrifice is exactly the band the extent gate was written to preserve.

## What was built

`MAX_PLANE_NATIVE_CELLS = 1,000,000` and `plane_gate(da, *, time_dim)` in `frame_stack.py`, beside
the constant and the gate they parallel, with the table above recorded at the constant. **Chosen
below the largest measured survivor rather than at it** — 1,050,000 survived once at 1,308 MB with
roughly 300 MB spare, in a container whose own baseline moved by 400 MB during the session that
measured it.

`plane_gate` is a **separate function, not a parameter on `frame_gate`**, because the two answer
different questions about the same field: one decides whether there is a scrubber at all, the
other how many statistics it can offer. A field can pass the first and fail the second, and the
right outcome then is the mean scrubber it already had — never a refusal. It assumes its caller
has already run `frame_gate`, so the preconditions common to both (a real time axis, a bucketable
cadence, a 2-D field) keep one home rather than two.

In `plot_tools.py`, all of it additive (D15):

- `_attach_frames` consults `plane_gate` and asks `build_frame_stack` for `PLANE_STATISTICS` or
  `("mean",)`. The outer `try`/`except` is **unchanged and deliberately so**: a failed *build*
  still costs the whole scrubber, which is correct, because the planes come off one fused compute
  — if it died there is no mean either. Nothing was added for the store path, and the prompt's
  claim that `store_frame_stack` already degrades per plane and cannot raise was **verified by
  reading it**, not assumed: `write_frames` catches everything and returns `None`, and the block
  carries a `_key` only where one landed.
- `block["planes_unavailable"]` when the plane gate refuses — beside the axis rather than beside
  `frames_unavailable`, because this chart *has* a scrubber. Absent entirely when every plane
  built.
- `_wire_frames_url` mints `/chart/{id}/frames.{statistic}.f32.gz` per plane **that has a key**,
  and nothing for one that does not. The mean's url is untouched.
- `_frames_summary` gains `statistics`, and `_scrubbable_statistics` derives it from the keys that
  **landed** — never from what was requested, so the agent cannot offer a toggle that 404s. The
  mean leads the list although it is not a `planes` key, because a list naming only the extras
  reads as though the default were not among them.
- The export spec keeps `"statistic": "mean"` **exactly as it was** and gains a separate
  `"statistics": [...]`. Two facts, two keys: D12 says the export is the period aggregate and
  still is, while what the scrubber offers is a different fact about the same recipe. Repurposing
  the existing key would change the meaning of a field already on the wire and leave every
  archived row ambiguous about which sense it meant. `"exports": "period aggregate"` is unchanged.

### What three planes cost, so nobody re-runs the probe

| | 1 statistic | 3 statistics |
|---|---|---|
| peak RSS at 1.05 M cells | 681.8 MB | 1,308 MB (**1.92×**) |
| stored float32 arrays | 3.243 MB | 9.728 MB (**3.00×** — three entries of one size) |
| wall clock at 1.05 M cells | 28.6 s | 84.2 s (**2.94×**, and see the caveat above) |

The build is already charged to its own `phase_timer("frames", ...)`, so a chart that gains planes
gains that time visibly rather than smeared into `aggregate`. Three entries per chart means the
store reaches its cap three times as fast; the no-per-entry-cap argument is unaffected, because it
rests on the size of a single entry, which a plane does not change.

### A finding for Phase 16: `extent_overstatement.ceiling` is not an upper bound

Asserting D6a decision 9's figure on a real chart at k=(2,2) measured a pooled headline of
**4.0000014** against a `ceiling` of 4 — over it, not fractionally under it as Phase 11's two
bundles were (24.699 and 24.747 against 25).

Not a bug, and `_extent_overstatement`'s own docstring is already careful: it defines `ceiling` as
the value the figure *would* take "if exactly one native cell reached each block's max and no
block ran short of real cells", never as a maximum. The mechanism is in `_overstatement_terms` —
**both sums are cos(latitude)-weighted per cell**, so a block whose max happens to sit on its
lowest-weighted row contributes slightly more than k², and the pooled figure straddles k² rather
than approaching it from below. Phase 11's "fractionally under 25, not at it" was an observation
about bundles whose edge blocks held fewer than k² real cells, and it does not generalize.

**The consequence is entirely in the prose Phase 16 writes.** A `methods.md` sentence phrased as
*"up to k× native cells"* or *"at most 25×"* would be claiming a bound this quantity does not
have. "Roughly k²" or the measured figure itself is what the number supports. The same applies to
anything the Phase 15 scrubber prints beside the max mode.

### Tests

**Backend container suite: 1704 passed / 1 skipped / 0 failed** (from Phase 13's 1687), 25m24s.

**45 in `test_plot_frames_wiring.py`, from 28 — and all 28 pre-existing ones pass unchanged**,
which was the phase's first requirement and the regression it was most likely to cause. Two of the
16 new ones patch `MAX_PLANE_NATIVE_CELLS` down rather than building a million-cell fixture, so
they pay for the wiring and not for the reduction; the real constant is pinned separately in
`FrameGateTests` against the measured table.

One structural change to the file: the tool harness (`asyncSetUp`, `add_bundle`, `plot`,
`_uneven_coverage`) moved into a `_PlotHarness` **mixin**. A second class inheriting from
`PlotSingularFrameWiringTests` would have silently re-run all ~20 of its tool tests under a second
name, which is what the first attempt did. No test's assertions changed.

`PlotToPlaneEndpointTests` is the new class and the first thing in this project to cross all three
phases' seams: a chart plotted through the real tool, then fetched **over HTTP through the real
router** at the url the tool minted. Phase 13's endpoint tests drive the route from hand-made
stacks and the wiring tests drive the tool without a route; neither can see whether
`_wire_frames_url` mints a path this router resolves, or whether the key filed under a statistic is
the one the route hands `read_frames` for that statistic. A wrong plane still renders, which is why
that seam needed a test rather than a review.

## Reproducing

```bash
# The anchor, one statistic — expect ~1,343 MB, matching Phase 5's 1,342 MB
docker exec tta-backend sh -c 'cd /app && python -u \
    scripts/probe_t59_phase14_plane_memory.py \
    /data/harmony/job_d175709729a518f2/result.nc.zip --crop 1250 3000 --arm mean_only'

# The same extent, three statistics — expect a cgroup oom_kill and no result line
docker exec tta-backend sh -c 'cd /app && python -u \
    scripts/probe_t59_phase14_plane_memory.py \
    /data/harmony/job_d175709729a518f2/result.nc.zip --crop 1250 3000 --arm mean_max_min'

# The largest measured survivor for three statistics
docker exec tta-backend sh -c 'cd /app && python -u \
    scripts/probe_t59_phase14_plane_memory.py \
    /data/harmony/job_d175709729a518f2/result.nc.zip --crop 700 1500 --arm mean_max_min'

# A kill is a missing result, never a traceback. Read the counter, not the log:
docker exec tta-backend cat /sys/fs/cgroup/memory.events
```

**One arm per process, always**, and walk extents cheapest-first — the failure mode is a SIGKILL,
so an arm that dies takes the rest of the run with it.
