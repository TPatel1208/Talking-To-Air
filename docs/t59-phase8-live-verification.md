# T59 Phase 8 — the viewer, verified against a live frames payload

**Measured 2026-08-11.** Phases 1b–5 were measured on real bundles through the production
open + mask path; Phases 6 and 7 were built entirely against fixtures hand-shaped from
`frame_store._axis_block`. Until this session nothing had ever been *rendered* against a
live payload, so the frontend's field names, the grid orientation, the blob's byte layout
and `methods.md`'s numbers had never met each other.

Two charts were produced end to end through the real agent (`POST /chat` →
`search_datasets` → `safe_retrieve` → `await_retrieval` → `plot_singular`), one per tier,
and read back through the browser, the `frames.f32.gz` endpoint and `methods.md`:

| chart | span | timesteps valid after masking | tier | frames | k | cells/frame |
|---|---|---|---|---|---|---|
| `map_2ea3dd7b34cf` | 2025-06-14/15 (2 d) | 28 | `cadence` | 48 hourly | (5,5) | 14,124 |
| `map_1531e35a0e18` | 2025-06-14/19 (6 d) | 82 | `coarsened` | 48 × 3 buckets | (5,5) | 14,124 |

Both are TEMPO NO2 L3 `vertical_column_troposphere` over Texas, native grid 535×658.
The 6-day bundle was retrieved for this session (`job_983d861c6a66eac1`, 116 granules,
225 MB, 628 s wall) because every bundle already on disk spans two days and stays in tier
one — Phase 3 §2's ~2.5-day threshold is real, and reaching tier two needed new data.

Reproduce a chart with:

```bash
docker exec tta-backend sh -c 'cd /app && python scripts/probe_t59_phase8_live.py --bundle /data/harmony/job_983d861c6a66eac1/result.nc.zip --handle obs_cf4e6008b3c7aae9 --location Texas --user-id <uid> --thread t59-phase8 --json-out /tmp/p8.json'
```

---

## 1. The headline finding: tier one's identity is exact in the reduction and 1.88 % off on the arrays that ship

**Prediction 5 is falsified, and this is the deliverable.**

D5 says frame 0 sits *"on the same grid by the same method"* so that D4's identity is
**"verifiable on the two arrays a user can actually touch"**. Phase 5's decision 6 repeats
it: `mean(frames) == frame 0` *"holds on the two arrays that ship"*. Phase 3 §3 measured
`0.000002 %`. `methods.md` states it flatly, and so does the scrubber:

> Each frame is one hourly interval of this product, and the period map is the
> equally-weighted mean of the frames: every interval counts once, however many granules
> it holds.

On `map_2ea3dd7b34cf`'s shipped `(49, 107, 132)` float32 planes — the blob the browser
actually downloads, fetched back through `GET /chart/{id}/frames.f32.gz` — averaging the
48 frames and comparing to plane 0 gives:

| | shipped arrays | same bundle, native resolution |
|---|---|---|
| **D16 headline** | **1.8760 %** | **0.000002 %** |
| max \|F−M\| | **2.72e15 molecules/cm²** | float32 noise |
| median per-cell relative difference | 0.41 % | — |
| p90 / p98 / worst cell | 4.35 % / 13.50 % / 170.0 % | — |
| cells differing by >1 % / >10 % | 33.9 % / 3.1 % | — |

2.72e15 is not a rounding artifact: the Map tab's own 2–98 clip for this figure runs
5.7e14 … 3.3e15, so the worst disagreeing cell is off by most of the colour ramp.

Reproduce the left column with `Backend/scripts/probe_t59_phase8_arrays.py` (its docstring
carries the two commands that fetch the payload and the blob), and the right column with
the Phase 3 probe at the production span:

```bash
docker exec tta-backend sh -c 'cd /app && python scripts/probe_t59_frame_stack.py /data/harmony/job_52a95bb4cb79e2ee/result.nc.zip --bbox -106.6458459 25.83706 -93.5078217 36.5004529 --pad-span-hours 0'
```

**The cause is that the block mean and the across-frame mean do not commute under partial
coverage.** Both skip NaN. `period_values` is `block_mean(mean over buckets)`; the frames
are `mean over buckets of block_mean(...)`. Inside a 5×5 block whose native cells are
covered in different hours, the first weights each *native cell* by how many hours saw it
and the second weights each *hour* equally regardless of how much of the block it filled.
At the worst cell (row 47, col 24) 21 of the 48 frames are finite; twenty of them sit
between −0.07e15 and 4.4e15 and one is **67.3e15**. Averaging the frames carries that
frame at weight 1/21 (giving 4.32e15); the period map, which averages the buckets at each
*native* cell before the block mean, gives 1.60e15.

This is the same mechanism as Phase 1b's finding, one level down: a mean of means over
unequal support is not the support-weighted mean. Phase 1b fixed it in time; it is
unfixed in space, and the space version is introduced by the rendering downsample.

**The science is unaffected.** D5a's rule holds — every scientific quantity is still
computed at native resolution, the map's numbers are unchanged, and `valid_fraction`,
`qa_pass_rate`, the per-frame statistics and the pooled scale are all reduced before the
block mean. What is wrong is the *claim*: the one identity the design promised a reader
could check for themselves is the one that does not survive on the arrays they can touch.

### Why no test caught it

Both identity tests exercise the claim only where it cannot fail:

- `test_bucketed_frame_stack.py::test_the_period_map_is_the_average_of_the_frames` runs on
  a fixture grid far under the 20,000-cell ceiling.
- `test_plot_frames_wiring.py`'s end-to-end version uses an 8×8 bundle and **asserts
  `frames["coarsen_k"] == [1, 1]`** before comparing — it pins the regime in which the
  block mean is the identity function.

A real regional TEMPO retrieval is 352,181 native cells and lands at k=(5,5). The tested
regime is not a regime any real chart is in.

### What to do about it (Phase 9, not done here)

The delta is free to measure: both arrays are already materialized side by side in
`build_frame_stack` before `_axis_block` writes them. Applying D16's own metric to the
**shipped** planes and disclosing it in *both* tiers turns an unverifiable claim into a
measured one, and costs no I/O. That also repairs a second gap: the coarsened tier's
`delta` is measured `at native resolution` (its own basis string says so), so it does not
describe the arrays on screen either — see §2.

The alternative, scoping the sentence ("…at native resolution; the frames you see are
block-meaned to the rendering grid, which the map is not"), is cheaper and weaker: it
tells the reader the check will not work rather than telling them the answer.

---

## 2. The coarsened tier: the disclosed delta and the shipped delta are different numbers

`map_1531e35a0e18`, 3 hourly buckets per frame:

| | value |
|---|---|
| disclosed `delta.headline` (native resolution) | **3.74 %** |
| `mean(frames)` vs plane 0, **on the shipped arrays** | **3.28 %** |
| disclosed `max_abs` | 3.127e15 molecules/cm² |
| max \|F−M\| on the shipped arrays | 1.953e15 molecules/cm² |
| median / p90 / p98 per-cell relative difference | 2.51 % / 6.29 % / 11.89 % |
| cells differing by >1 % | 81.2 % |

The two are not additive and the shipped one is *smaller* here — the block mean partly
cancels the time-coarsening term rather than compounding it. That is precisely why it has
to be measured rather than reasoned about: neither number bounds the other.

Everything the tier discloses is otherwise correct and consistent. `methods.md`:

> - Disagreement: **3.7%**, largest absolute difference 3.127e+15 molecules/cm^2.

and the scrubber, in the same instant:

> Each frame averages 3 hourly intervals, so the frames are a different temporal
> aggregation from the map above. They differ from it by 3.7%, up to 3.127e+15
> molecules/cm^2 at the worst pixel.

3.7 % is below `_DELTA_HIGH`, so the lead is unbolded in both places — the real production
case, not a hand-picked one. Phase 3's 22.27 % remains the case that earns the bold.

---

## 3. Prediction 1 — grid orientation: HOLDS

`_axis_block` ships `lats` **ascending** (25.890003 … 36.470001) and `lons` ascending
(−106.590004 … −93.520004); the payload's own map grid ascends in both too.
`canvasCornersFromArrays` follows the array's own row direction rather than assuming
north-up, so an ascending-lat array is placed on ascending-lat corners and renders the
right way up.

Checked observationally rather than by reading the code: frame 0 sampled onto the map's
grid by nearest neighbour, then correlated against the drawn field over the 3,399 cells
finite in both.

| arrangement | correlation with the drawn map |
|---|---|
| as shipped | **0.907** |
| latitude flipped | 0.078 |
| longitude flipped | 0.254 |

T56's index-fallback bug — the atmosphere drawn upside down, looking fine until someone
read the axis — does not recur here.

## 4. Prediction 2 — `period_index`: HOLDS

`period_index` is `0` in the live payload and `frameAxis.js` reads the planes off it. On
screen, stop 0 is "Period aggregate" and stops 1…48 carry the frames' own `t_start`s
(`14 Jun 00:00 UTC`, `14 Jun 01:00 UTC`, …), labelled UTC explicitly. The slider's range
is `min=0 max=48` — 49 stops for 1 + 48 planes.

## 5. Prediction 3 — the pooled scale does not jump when the bytes land: HOLDS

Sampled on a **cold** blob (a chart never opened before), three distinct states inside
110 ms:

| t (ms) | legend ticks | caption | slider |
|---|---|---|---|
| 32067 | 8.0e14 … 2.7e15 | `Color scale clipped at 2nd–98th percentile` | absent (Map tab) |
| 32105 | **4.3e13 … 4.1e15** | `2nd–98th pct, pooled across frames` | **disabled**, "Loading frame values — the time axis is ready, the pixels are still arriving." |
| 32176 | 4.3e13 … 4.1e15 | `2nd–98th pct, pooled across frames` | **enabled** |

The recolour happens on the mode click, before any byte of the stack has arrived, and
nothing changes when it does. D9's reasoning is confirmed on live data: because
`POOLED_SCALE_BASIS` pools the period mean *with* the frames, the aggregate already is its
own pooled colour. Decision 2's disabled-and-parked state is also real and legible rather
than theoretical.

What *does* change when the bytes land is which downsample is drawn — the payload's
stride-thinned 77×94 grid (3,399 finite cells, p98 3.18e15, max 6.71e15) gives way to the
block-meaned 107×132 frame grid (6,878 finite cells, p98 3.33e15, max 1.15e16). Note the
direction: here the block mean's peak is *higher* than the stride's, because the stride
missed that peak entirely — D5a's "the stride's weakness is variance, not attenuation",
observed.

## 6. Prediction 4 — the two empty states: HOLDS, and the document agrees with the screen

Both kinds render distinctly, off the live payload:

```
stop 2  14 Jun 01:00 UTC   0 granules · 0.0% of the region covered · QA pass rate 0.0%
                           Observed — QA rejected every pixel in this interval.
stop 3  14 Jun 02:00 UTC   0 granules · 0.0% of the region covered · QA pass rate not applied
                           Nothing retrieved for this interval. QA never ran on it.
```

and the counts match `methods.md` exactly, in both tiers:

| chart | screen | `methods.md` |
|---|---|---|
| cadence | 2 qa-rejected + 20 nothing-retrieved of 48 | "Empty intervals: 22 of 48 — 2 observed but rejected entirely by quality screening, and 20 with nothing retrieved." |
| coarsened | 1 + 12 of 48 | "Empty frames: 13 of 48 — 1 observed but rejected entirely by quality screening, and 12 with nothing retrieved." |

They agree because they cannot disagree: `_gap_rule` reads `n_granules == 0` and
`frameAxis.bucketState` reads `statistics.count <= 0`, and `n_granules` is the count of
timesteps that contributed a finite value **anywhere**, so one is zero exactly when the
other is. Checked rather than argued: across all **96** live frames in the two payloads,
`n_granules == 0` and `statistics.count <= 0` disagree **zero** times. The two definitions
are different expressions of the same fact rather than two facts, which is why the tally
survived first contact.

Note also the tier-two noun: `methods.md` says "Empty **frames**" for the coarsened chart
and "Empty **intervals**" for the cadence one, so three empty 3-hour frames are never
reported as three empty hours.

---

## 7. Prediction 6 — the full-domain refusal: HOLDS. A map, an explicit `extent_too_large`, no OOM

An un-narrowed TEMPO request — the full-domain bundle `job_d175709729a518f2` (2950×5771,
2025-06-08/09) plotted at `location="North America"`, bbox −171.73 … −12.21, 7.22 … 83.65:

```
plot_singular returned in 352.3s
  render_type heatmap, grid_dims [63, 123], vmin 1.06e14 vmax 3.83e15   <- the map, drawn
  frames_unavailable {"reason": "extent_too_large",
    "detail": "A frame axis over this region would reduce 17,024,450 cells per interval,
               above the 4,000,000-cell limit a frame stack is built within."}
```

- **The map is produced.** D15's additive posture holds: the figure is exactly what it
  would have been before T59.
- **The refusal is explicit and reaches the agent.** `frames_unavailable` is in the payload
  *and* in `_chart_model_summary`, so the model can relay it (Phase 5 decision 2), *and* in
  `export["frames"]["unavailable"]`, so the jsonb row records that frames were refused
  rather than never attempted.
- **No OOM.** Peak RSS 1.78 GiB in a 3.9 GB container, sampled every 45 s across the run;
  `memory.events` reports `oom_kill 0`. 17,024,450 is the extent Phase 3 §7 measured being
  OOM-killed, and the gate now names that number in the sentence it refuses with.
- **No spinner.** 352 s is the map's own cost on 17 M cells — the `frames` phase never
  starts, because `frame_gate` reads the narrowed field's size before the reduction.

`methods.md` quotes the gate verbatim rather than paraphrasing:

> ### Temporal frames
>
> No frame-by-frame time axis was built for this figure. A frame axis over this region
> would reduce 17,024,450 cells per interval, above the 4,000,000-cell limit a frame stack
> is built within.

and the real block through the frontend's own utils resolves to the refused state, with
the backend's sentence relayed rather than restated:

```
mode: refused   offered: false   sliderEnabled: false
reason: extent_too_large
detail: A frame axis over this region would reduce 17,024,450 cells per interval, …
delta: null   scrubScale: null
```

Two scope notes on this one, stated rather than glossed. This chart was produced through
`probe_t59_phase8_live.py` (which stubs only the MCP's `export_result`) rather than through
the agent, because re-retrieving a 1.6 GB full-domain bundle into a second workspace was
not worth the wall clock; and its `methods.md` was rendered by calling
`build_methods_markdown` directly, because the bundle's handle belongs to another user's
workspace and the endpoint's lineage lookup refuses it (see §11). Neither substitution
touches the gate, the payload, the document's frames section or the frontend's reading of
it. The refused box itself was not seen rendered, for the reason in §11.

---

## 8. Risk 3 — is 60 the right frame budget? **Keep 60.**

Now a judgment with evidence, after scrubbing a real 49-stop axis and a real coarsened one.

1. **The slider's real estate binds before storage does.** Measured in the running app at a
   1280×720 viewport: the output panel is 360 px wide and the `input[type=range]` is
   **284 px**, so 49 stops is **5.8 px per stop** — a six-pixel drag moves an hour, and only
   the two end labels are printed. Doubling the budget halves that. Storage is nowhere near
   the constraint (Phase 3: N=100 is 2.26 MB gzipped, 42 ms to gunzip), and that is exactly
   why the budget is not a storage decision.
2. **Extra budget buys mostly empty stops.** 22 of 48 stops on the real 2-day TEMPO scrub
   hold nothing, because TEMPO's gaps are diurnal. Frames are spent at the product's
   cadence and the cadence spends them on night. A 60-frame budget over hourly TEMPO is
   ~2.5 days of which ~28 stops are blank; a 120-frame budget is ~5 days of which ~60 are.
3. **The cost at the boundary is small and it is disclosed twice.** The first coarsening a
   real request meets is group=3, measured here at 3.74 %, stated in `methods.md` and in
   the scrubber. Phase 3's 22.27 % was group=5.
4. **The number that actually governs the cost is the group, not the frame count**, and it
   already has its own ceiling — `MAX_BUCKETS_PER_FRAME = 24`. Raising `MAX_FRAMES` would
   move the tier boundary without changing what a bad frame costs.

The lever worth pulling is not the budget; it is the control. Tick marks or a hatch for
empty stops, and a wider scrubber, would make 49 stops usable — and would make 120
tolerable if anyone later wants them. That is a UI change, not a constant change.

## 9. Risk 5 — two colour scales for one field: **the mitigation holds**, with one sentence to fix

Measured on real data, the divergence the mode switch has to carry:

| chart | Map tab ramp | scrubber pooled ramp | scrubber / Map tab |
|---|---|---|---|
| cadence | 5.7e14 … 3.3e15 | −3.1e14 … 5.0e15 | **1.99×** |
| coarsened | 8.0e14 … 2.7e15 | 4.3e13 … 4.1e15 | **2.09×** |

So entering scrubber mode roughly halves the apparent intensity of everything on the map.
That is a large change, and it is carried by three things that all fire on the same click:
the legend's numbers change, its caption changes from
`Color scale clipped at 2nd–98th percentile` to `2nd–98th pct, pooled across frames`, and
the button's label flips to `Show period aggregate`. Nothing changes mid-scrub. On the
evidence, that is enough — the alternative (deriving the map's scale from the stack) would
let a storage gate change a map's appearance with no disclosure surface at all, which is
strictly worse.

The one thing to fix is the anchor sentence at stop 0:

> The period aggregate — the same field the Map tab shows, on the frame grid.

It is the sentence a reader uses to re-orient, and it is carrying two unstated caveats: a
different colour ramp (this section) and a 1.88 % disagreement with the average of the
frames beside it (§1). "The same field" is doing more work than it can support.

## 10. The `_DELTA_HIGH` duplication — resolved, not accepted

`_DELTA_HIGH = 0.10` in `methods_export_service.py` and `severityOf`'s `>= 0.1` in
`frameDelta.js` are one number in two languages. The claim that "a test pins the value and
names the JS source, so drift fails loudly" is **false, and was measured to be false**:
editing the JS edge to `0.15` left all **339** frontend tests and all **24** tests in
`test_methods_export_service.py` green. A comment naming the other file is not a check.

Resolved rather than accepted, because the repo already has the idiom twice over:
`backend-test` bind-mounts repo-root files that live outside `./Backend`'s build context
(`docker-compose.yml`, the two `init_agent_*.sql` files, `.coveragerc`), and
`test_jobs_service.py::FinishedRowStatusContractTests` already reads `jobCard.js`'s
`TERMINAL_STATUSES` literal out of the JS source for exactly this reason. So: one mount,
and a test that reads the JS edge and compares it.

- `docker-compose.yml`: `./Frontend/src/utils:/Frontend/src/utils:ro`
- `test_methods_export_service.py::SharedDeltaThresholdTests` — two tests: the constants
  match, **and** the compose file declares the mount, so the first can never quietly skip
  for want of a file.

Red/green confirmed: with the JS at 0.15 the test fails with `0.15 != 0.1`; reverted, it
passes.

### The companion test earned itself immediately

The full container suite came back **1 failed, 1626 passed** — and the failure was not
mine:

```
FAILED tests/test_jobs_service.py::FinishedRowStatusContractTests::test_frontend_terminal_statuses_match_the_backend_finished_row_set
E   AssertionError: False is not true : jobCard.js not found at /Frontend/src/utils/jobCard.js
```

The repo's one pre-existing Python-reads-JS contract test resolves its path relative to
`Backend/tests`, which lands on the real file **on the host** and on a non-existent
`/Frontend/...` **in the container** — and `docker-compose.yml` had never mounted it
(`git log -S jobCard -- docker-compose.yml` is empty). It has never once run in the
container suite, on any commit. It fails loudly rather than skipping, so it was not
silent; it was simply never green there, which is presumably why the suite is usually run
on the host.

That is precisely the failure the companion test exists to name, met on its first run. The
fix folds both into one convention: mount the whole `Frontend/src/utils` directory at the
absolute path the repo-relative lookup already produces, and let both tests use that one
resolution on host and in container alike. `SharedDeltaThresholdTests` dropped its own
ad-hoc `/frontend/frameDelta.js` path to use it — two conventions for one thing is the
drift this section is about.

---

## 11. Operational traps this session hit

- **The `backend-test` image is not rebuilt by `docker compose build backend`.** They are
  different Dockerfile targets (`devtools` vs the runtime stage). A stale devtools image
  ran **3** tests from a file that has **24** and reported them all passing. Always
  `docker compose --profile test build backend-test` before trusting a test count.
- **The host suite and the container suite are not the same suite.** Any test resolving a
  path relative to `Backend/tests` reaches the rest of the repo on the host and reaches
  nothing in the container, where only `./Backend` is in the build context. §10 has the
  case that was live at the start of this session.
- **Both deployed images were stale at session start** — the backend image predated
  `479d206` (so Phases 3, 4, 5 and 7 were all absent from the running container) and the
  frontend image predated `0296fc0`. Neither is visible from `docker ps`; compare
  `docker image inspect --format '{{.Created}}'` against `git log`.
- **`{ cmd; echo "EXIT=$?"; } > file` really is required.** `cmd > file; echo "EXIT=$?"`
  reported `EXIT=0` for a run that died on an import error, and the harness's own
  completion notification said "exit code 0" for the same failure.
- **Handles are workspace-scoped.** Plotting a bundle whose handle belongs to another
  user's workspace works — the plot path never asks the MCP anything once the export
  resolves — but `methods.md` fails with a generic contract 500, because `get_lineage`/
  `get_citations` refuse a handle the caller does not own. The message in the logs is
  `handle '…' is not owned by workspace 'user-…'`; the response body is not.
- **The in-app browser pane could not composite**, so MapLibre's render loop never ran, its
  style never finished loading, and no `overlay-canvas` source was ever added. Every DOM,
  payload, network and control assertion in this document was made in the live page;
  none of the pixel-level ones were, which is why §3 checks orientation by correlation
  rather than by eye. The basemap is also an external dependency
  (`basemaps.cartocdn.com`) — worth knowing that an unreachable tile CDN takes the data
  overlay down with it, though that is MapLibre coupling and not T59's.
