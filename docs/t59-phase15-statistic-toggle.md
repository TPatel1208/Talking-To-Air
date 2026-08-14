# T59 Phase 15 — the toggle, and the four sentences that stop being true when you flip it

**Built 2026-08-14.** Phase 14 shipped three planes per chart and nothing in the frontend read any
of them. This phase reads them: a `Show mean / max / min` control inside scrubber mode, each
statistic fetched from its own url, drawn on its own pooled clip, with its own legend caption, its
own anchor sentence, its own agreement disclosure, and — for max — its own measured extent
overstatement.

**Frontend: 430 passed, from 379.** 51 new tests. **Every one of the 379 that existed before this
phase passes unchanged**; none needed editing, which was deliverable 1 and the regression this
phase was most likely to cause. The backend was not touched, and
`test_methods_export_service.py` — which parses `frameDelta.js` by regex through the bind mount —
is green at 32 passed / 1 skipped.

The mechanical half was an afternoon, as predicted. The rest of this document is the disclosure.

## The four sentences, and what each became

Every one is composed whole in a util where a test reads it. Phase 8 §1's falsified sentence
survived two phases because it lived in JSX, and this phase added three more chances to repeat
that.

**1. The anchor (`frameDelta.aggregateAnchor`) inverts rather than qualifies.** D11's explicit
exception is not a caveat on the old sentence — it is its negation. `plot_singular` has no
statistic parameter, so the Map tab is always a period mean and stop 0 in max mode is
*emphatically not* the field above it. `selectionAnchor` says so outright: *"This is NOT the field
the Map tab shows: that stays the period mean whichever statistic this toggle is on."*

It also stops calling a block max a block mean. D6a decision 4 makes **block max** the spatial
reducer for a selection plane, so `block-meaned onto the frame grid` names a mechanism the plane
does not have — the same defect `shippedSentence` was rewritten to fix one tier along. The
selection anchor reads `block-maxed onto the frame grid, so each rendered cell holds a value some
native cell in that block actually had`, which is the one genuinely *reassuring* thing about block
max and had nowhere else to go.

**2. The delta became a stated identity, not a formatted zero.** `resolveFrameDelta` takes a
statistic and branches before it touches either figure on the block, because **both of them are
the mean's**: `delta` is the mean's temporal disagreement and `frame_grid_delta` its shipped-array
one, and a selection plane has neither. Passing the mean's 1.876% through would attribute a
mean-only disagreement to a plane that does not have it; passing the plane's own `0.0` through
`formatPct` would print **"under 0.1%"**, which D6a names as *"exactly the failure `_DELTA_FLOOR`
exists to prevent"*.

What it says instead: *"Taking the highest of the frames you can download reproduces the period
plane at stop 0 exactly — a maximum selects one of the values it is given rather than combining
them, so there is no disagreement here to measure."*

**The identity is asserted from the payload, never from the operation's name.** If a plane ever
ships a non-zero `frame_grid_delta`, the sentence reports the figure and tells the reader to read
it as a defect rather than a caveat — because for this statistic it would be one. That branch has
its own test. Claiming "exactly" because max is associative *in theory* would be the same class of
falsehood Phase 8 caught here: true of the mechanism, false of the arrays.

**3. The cadence/coarsened summary is statistic-aware, not suppressed.** *"the map above is their
equally-weighted average"* is not a relationship a max plane has, and deleting it would leave the
max tier silent about something the mean tier explains at length. `selectionDelta` keeps the tier
distinction (`one hourly interval` / `the highest value across 3 hourly intervals`) and replaces
the averaging clause with the associativity that is *why* the figure is zero. A test asserts no
selection summary contains the word "average" in either tier.

**4. The legend caption names the statistic — except the mean's, deliberately.**
`SCRUBBER_LEGEND_NOTE` stays byte-identical and the planes get
`2nd–98th pct, pooled across the max frames` / `…the min frames`. D6a decision 5 keeps the mean
entry exactly as it was, and a chart that gained planes must not have its default mode reworded by
modes the reader has not opened. Risk 5's mitigation — the legend's numbers, its caption and the
button all changing on one click — now covers four ramps on one field instead of two.

## Design tension 1: scoping the label

**Resolved by scoping, and narrowly.** `statsForStop` gains a `fieldNote` — *"of the mean field —
not the max plane the map is showing"* — rendered as the subtitle of the **Mean, Max and Min cards
only**.

`count`, `validPct`, `qaPassRate`, the granule count and the empty state are facts about the
**interval** and are equally true under any plane. Caveating them too would be noise that hides
the one row that needs it. A test asserts the non-scoped fields are `deepEqual` across statistics.

The other two options are rejected in the code with reasons, so neither is quietly reinvented:
withholding empties the tab's most useful row exactly when someone has gone looking for a peak, and
recomputing from the rendered array is what **D5a forbids outright** — at k=8 a frame has already
lost 16% of its own p98 and 70% of its max, so it would understate precisely the peak the scrubber
exists to find.

**`fieldNote` is absent, not null, in mean mode.** The Statistics tab's first test destructures
`statsForStop`'s result and compares the remainder to `computeChartStats`, so an always-present key
would have broken a passing test — which would have been a finding about the change, not a reason
to edit the test. Absence is also the honest shape: the mean has nothing to disclose.

*Could a reader in max mode copy a number off this screen into a paper and be wrong?* Not without
reading past a line that says which field it is of. Phase 12 measured the two at **50.5 against
100.0** on its own fixture.

## Design tension 3: what is on screen between the click and the bytes

**`selected` and `rendered` are two facts, and everything that speaks keys off `rendered`.**

`useFrameStack` holds one result keyed by one url, so during a switch there is no stack at all and
the canvas falls back to the chart's own period aggregate — which **is** the mean. Saying "max"
over it is exactly the lie Phase 13 decision 4's manifest guard refuses to serve. So the scale, the
legend, the anchor, the delta, the overstatement and the Statistics tab's scoping all read
`statistic.rendered`, which stays `mean` until `loadState === 'loaded'`. The label, the ramp, the
caption, the sentences and the pixels change on one event.

The *fetch* keys off `selected`. Fetching `rendered` would be circular — it can never leave the
mean, so the plane is never requested at all.

**The stop survives; the display parks.** Decided both ways at once, because the two goals are not
in conflict once they are separated:

- The remembered index lives in `scrubChoice` and **`selectStatistic` does not touch it**. Finding
  a peak is the entire reason the max plane exists, so someone who scrubbed to hour 17 and asked
  for its maximum has done precisely the thing that must not be punished.
- `resolveScrubStop` parks the **display** on the aggregate while `sliderEnabled` is false, which
  is Phase 6 decision 2 unchanged: the aggregate is the only stop whose pixels are actually on
  screen, and a readout naming hour 17 over the period map *"would read as a quiet two days to
  someone scrubbing for an event."*
- When the bytes land the slider re-enables and the index comes back on its own. No effect, no
  setState cascade, nothing for the react-hooks lint to reject.

**The degradation sentences name the plane.** `resolveFrameState` gained a third parameter that
changes wording only. *"The frame values for this map have expired"* is false of a chart whose mean
is sitting there intact — `store_frame_stack` protects the mean from its own planes' evictions and
degrades one statistic at a time — so a failed plane says which one failed, says the map below is
the period mean, and points at the toggle. The mean's three strings are unchanged, asserted
against each other in a test.

`selectStop` had to start preserving the statistic. Dropping it would have sent the map back to the
mean on the first drag with the max button still lit — the exact failure this phase is written
against, arriving through the one path nobody would think to check.

## Design tension 4: three stacks, measured

**Decision: hold one, which is what `useFrameStack` keyed by url already does.** No cache was
added; the policy was already there and now has a number.

`Frontend/scripts/measure_frame_stack_residency.mjs`, 101 planes × 100×200 float32 (the 20,000-cell
rendering ceiling, 100 frames plus the aggregate):

| | `arrayBuffers` |
|---|---|
| 1 statistic resident | **+7.71 MB** |
| 3 statistics resident | **+23.12 MB** (3.00×) |
| evicted back to 1 | **+7.71 MB** |
| 100 `planeView()` views | **+0.00 MB** (`views[0].buffer === stack.values.buffer`) |

Re-decode of an already-fetched blob: **0.096 ms**. `_FRAME_CACHE_CONTROL` is
`private, immutable, max-age=31536000`, so coming back to a plane the reader has already seen is an
HTTP cache hit and costs that decode, not a download. **15.4 MB of resident typed arrays to save
0.1 ms is not a trade worth making**, and the simple thing and the cheap thing are the same thing
here.

**A measurement trap worth recording: one `gc()` is not enough.** The first run reported
**+15.41 MB** after dropping two of three stacks — one whole uncollected stack still accounted —
which would have made "evict all but the active statistic" look half as effective as it is. The
script calls `global.gc()` twice and says why. And, as Phase 6 established, this is all invisible to
`heapUsed`: budget from `arrayBuffers` or the frame store reads as free.

## Design tension 5: the overstatement, worded inside Phase 14's constraint

`extentOverstatementNote` renders the chart's **own measured `headline`** beside the max-mode
scrubber, as Phase 11 G4 asked (`methods.md` is Phase 16's):

> Every cell of a block is drawn at that block's highest value, so the ground shown at peak level
> is about 24.7× the ground the peak was actually measured over — pooled across this chart's own
> blocks, cos(latitude)-weighted. Its single worst frame stretched one further, to 24.9×.

**`ceiling` is not rendered at all.** Phase 14 measured a pooled headline of **4.0000014 against a
ceiling of 4** — both sums in `_overstatement_terms` are cos(latitude)-weighted per cell, so a
block whose max sits on its lowest-weighted row exceeds k², and the pooled figure straddles k²
rather than approaching it from below. Nothing here says "up to k×" or "at most 25×", and a test
asserts it against a fixture built from Phase 14's own straddling measurement.

## Deliverable 7: a chart above the plane ceiling

`frames.planes_unavailable` produces an `offered` toggle whose single `Mean` button is **disabled**,
with the backend's own sentence beside it — the same posture the three disclosed frame refusals are
relayed in. Phase 5 decision 2's rule, one level in: a reader who finds no toggle and no reason is
exactly the person the disclosure exists to keep out of that position.

A chart that simply never had planes gets **neither** a toggle nor an explanation. `offered` is
false and `StatisticChoice` renders nothing, so its scrubber is byte-identical to what shipped —
which is why Phase 13 omitted `planes` rather than emitting it empty in the first place.

## Two findings

**1. `frameScale.test.mjs`'s "the aggregate is drawn on the pooled scale too" test has been vacuous
since Phase 6, and still is.** It calls `resolveScrubberScale(chart, true, 0)` and
`resolveScrubberScale(chart, true, 2)` — a stop index passed to a function that had two parameters
— and asserts the two are equal. Before this phase the third argument was ignored, so it compared
a call to itself. After it, `0` and `2` are unrecognized statistics and both return `null`, so
`deepEqual(null, null)` passes. **It has never once exercised what its name claims.**

It is left unedited, per deliverable 1's rule that a pre-existing test needing changes is a finding
to state rather than a test to adjust. The property it names is now structural — the function has
no stop parameter at all — and the four new scale tests cover what actually varies. **Phase 16 or a
later sweep should delete or re-point it**; it is currently a green line that means nothing.

**2. Nothing in this phase could pin the toggle's most important seams from a util test.** Five
decisions live only in JSX — which statistic each util is handed, that `selectStop` preserves the
statistic, that `selectStatistic` preserves the stop, that the Map tab is handed neither, that the
plane refusal reaches the screen. They are asserted against the component source, the idiom Phase
10 established for `ScrubTrack` and `SharedDeltaThresholdTests` for `_DELTA_HIGH`. Each regex was
verified to **bite** by mutating the source in memory and confirming it fails — a contract test
that cannot fail is worse than none, and this repo has already measured that exact thing happening.

## What was not done

No backend file. `methods.md` / `methods_export_service.py` — D6a decision 8's identity sentence
and G4's disclosure **in the document** are Phase 16's, and they will need to agree with the
sentences above. The Map tab following the statistic (D11's exception is explicit: it does not, and
a test pins that `MapLibreHeatmapPanel` is handed no statistic). `MAX_PLANE_NATIVE_CELLS`.
Playback. Per-statistic export (D12: the export is the period aggregate and stays so). Comparison
panels (D7). Variable switching (D1). Frame regeneration (D8).

**Not verified live.** Everything here is measured against fixtures and the payload contract
`test_plot_frames_wiring.py` pins; a plane has never been *watched* rendering. The frontend docker
image is a static bundle, so seeing this in the running app needs
`docker compose build frontend` first.

## Reproducing

```bash
cd Frontend && npm test
```

```bash
cd Frontend && node --expose-gc scripts/measure_frame_stack_residency.mjs
```

```bash
cd Backend && python -m pytest tests/test_plot_frames_wiring.py -q -k "carries_a_max_and_a_min or url_of_its_own"
```
