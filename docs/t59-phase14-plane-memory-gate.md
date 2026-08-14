# T59 Phase 14 — the wiring gate: three planes do not fit inside the extent gate

**Measured 2026-08-13/14. Verdict: NO-GO as specified. Nothing was wired.**

Phase 14 was to turn the planes on: `_attach_frames` asking `build_frame_stack` for
`("mean","max","min")`, three store entries per chart, three URLs in the payload, and the toggle
named to the agent. Its own prompt made that conditional on one measurement, and the measurement
came back against it. **`build_frame_stack` with three statistics is OOM-killed by the kernel at
every extent the current gate admits above ≈1.4 M native cells per interval**, including the exact
extent `MAX_FRAME_NATIVE_CELLS` was derived from. Per the prompt's own third outcome — *"it does
not fit → stop and report, do not wire it"* — `plot_tools.py`, `frame_stack.py`, `frame_store.py`
and every test are untouched. The only new file is the probe that produced the numbers.

## What was measured

`Backend/scripts/probe_t59_phase14_plane_memory.py`, in the live 3.9 GB `tta-backend` container,
**one arm per process** (peak RSS is `VmHWM`, a high-water mark that never falls, so two arms
sharing a heap measure the larger one twice — Phase 11's own harness note at :263). It calls the
real `build_frame_stack` with the real `statistics=` argument rather than rebuilding the
composition by hand, so what it measures is the code Phase 14 would have shipped.

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
  have moved; no reduction code was touched. Block-max retention, the 24.6985× overstatement and
  both bit-exact identities stand as measured.
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

## The decision this hands back

Two options, both requiring a change this phase's scope forbids, and the cost of each is now a
number rather than a guess.

**A — lower `MAX_FRAME_NATIVE_CELLS` to ~1,000,000.** One constant, no new concept, and the
refusal text already explains itself. **It charges existing users for a feature nobody has yet:**
a chart at 3.75 M native cells gets a working mean scrubber today, measured at 1,343.5 MB, and
would begin being refused outright. Everything from 1.0 M to 4.0 M loses the scrubber it has.

**B — a statistic-aware extent gate.** Keep 4,000,000 for the mean; admit the extra planes only
below ~1,000,000. No existing chart loses anything, and every regional chart measured in this
project (352 k cells) is comfortably inside the plane band. The cost is conceptual: `frame_gate`
grows a second threshold and `_attach_frames` has to decide what to ask for before it asks,
which makes the auto-upgrade two-tiered rather than one — and it is a `frame_stack.py` change,
which Phase 14 was explicitly told not to make.

**B is the recommendation.** A is a regression to today's behaviour paid to enable tomorrow's,
and the band it would sacrifice is exactly the band the extent gate was written to preserve.

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
