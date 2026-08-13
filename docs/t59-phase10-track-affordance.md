# T59 Phase 10 — making the track say what the axis already knows

**Built and measured 2026-08-13.** Phases 1b–9 made the viewer correct and its disclosure
honest. The one control a reader actually touches had never been designed, only measured:
284 px over 49 stops, two printed labels, and nothing on the track saying which stops hold
anything.

---

## 0. The gate: both empty states draw as the same nothing

Nobody had ever watched an empty stop. The browser pane still cannot composite (`document.hidden`
is true, so MapLibre's render loop never runs and `readPixels`/`toDataURL` return an all-zero
buffer, exactly as in Phase 8 §11 and Phase 9 §8). The question was answered on the **arrays**
instead, which turns out to be stronger than a capture: a capture shows two blank maps looked
alike on one machine; this shows they cannot differ on any.

Fetched from the live blob, `map_2ea3dd7b34cf`, 49 planes × 14,124 cells:

| plane | stop | state | finite cells |
|---|---|---|---|
| 0 | aggregate | — | 6,878 (48.70 %) |
| 1 | 1 | observed | 2,156 (15.26 %) |
| **2** | **2** | **qa-rejected** | **0** |
| **3** | **3** | **not-retrieved** | **0** |
| 12 | 12 | observed | **54 (0.38 %)** |

Both empty kinds are entirely NaN, and `buildCanvasFallbackFrame` allocates a fresh
zero-filled `Uint8ClampedArray` per frame and leaves every non-finite cell at alpha 0. So both
draw as a **byte-identical fully transparent overlay** — basemap only, no data polygon at all.

**The drawn field carries none of D10's distinction.** That settles what the marking is
carrying: all of it. It also means the labels were not the better spend.

### One boundary the marking deliberately does not cover

Stops 12 and 36 hold 54 and 62 finite cells — 0.38 % and 0.44 % — and are correctly
`observed`, so they are **not** marked. They will still draw as ~50 scattered pixels on an
otherwise blank map. That is dawn, the counterpart to the dusk QA rejection below.

The track marks what the **axis** has, not what will **look** blank, and there is no
defensible threshold for the second. Stated rather than papered over.

---

## 1. The distribution, which decided the design

Read off the live axis rather than assumed:

```
stop  1  2  3-11      12-18   19 20  21-25  26  27-35     36-48
      O  Q  NNNNNNNNN OOOOOOO N  N   OOOOO  Q   NNNNNNNNN OOOOOOOOOOOOO
```

tally `observed 26, qa-rejected 2, not-retrieved 20` — 22 empty of 48, matching `methods.md`.

**`qa-rejected` is at stops 2 and 26, each the single LEADING stop of a ten-stop run.** That
is dusk: the last scan of the day is attempted and QA rejects all of it, then night follows
with nothing retrieved. The two kinds are not interleaved; one is a 1-stop edge on the other.

Empty runs: **10, 2, 10 stops = 59, 12, 59 px** at the measured 5.92 px/stop. Runs survive the
density that kills per-stop ticks, which is why the marks are by run — measured, not assumed.

---

## 2. The four tensions, resolved

**1. Gaps made visible, never skipped.** `_axis_starts`' docstring owns this and it is
untouched: the axis stays uniform in time and the marks are decoration over the existing stop
numbering. Nothing renumbers.

**2. ONE empty state on the track, two in the readout.** A per-kind split renders the
QA-rejected distinction as a **5.9 px sliver** on the edge of a 59 px run — reintroducing, at
6 px, precisely the density failure that rules out per-stop ticks at 5.9 px. D10's own wording
places the distinction in the readout (*"the readout says which blank this is"*), which names
both kinds at the stop the reader is on; the track did not exist when D10 was written. The
track answers *"is there anything here"*, the readout answers *"why not"*. Pinned by
`a run spans both kinds of empty rather than splitting on the reason`.

**3. A marks layer behind a transparent track, not a re-implemented control.** The element
stays a real `<input type="range">`, so **keyboard interaction is untouched** — arrow keys,
Home/End, PageUp/PageDown are the same as before, because styling cannot reach keyboard
semantics. Webkit only honours `::-webkit-slider-runnable-track` once `appearance: none` is
set, and that also removes the native thumb, so **the thumb is ours on webkit and the
browser's on Firefox**. The focus ring is made **explicit** (`:focus-visible`) rather than
inherited, because a control whose track we have taken over should not lean on a UA default we
may have disturbed. What is genuinely lost is the OS-drawn fill — which was never ours and
differed per machine (`accent-color: auto`, measured); the progress fill replaces it with one
that is the same everywhere.

**4. A structural floor, and no untestable guard.** `min-width: 260px` on the scrubber box, so
the layout **cannot produce** a 0 px track rather than detecting one afterwards. Its ancestor
already carries `overflow-x: auto`, so the panel scrolls instead of crushing the control. 260
is today's 284 rounded down to where density is unchanged (5.3 px/stop at 49 stops) and the
smallest run real data produces — two stops — is still ~11 px. Measured at Phase 9's own
repro, a 556 px viewport with all three panels open: **the track is 232 px, not 0**.

### Phase 9's 0 px finding, re-attributed

Phase 9 recorded *"the slider renders at 0 px wide while still reporting `disabled: false`"*.
Measured at the same viewport, the truth is larger and not the scrubber's: **the entire output
panel is 0 px wide and pushed off-screen at `left: 634`** — `.maplibregl-map` measures 0 px
too. The map collapses with it. The slider was where it was noticed, not what was wrong.

Giving the output panel its own floor is an app-layout change affecting every artifact type,
so it is **not** made here. What is fixed is the narrower claim tension 4 actually stated: an
interactive control can no longer report zero width while claiming to be enabled.

---

## 3. Shipped

`Frontend/src/utils/frameTrack.js` — three pure functions, tested:

| export | what it answers |
|---|---|
| `buildTrackMarks(axis)` | which stretches of the track hold nothing, as `{startStop, endStop, start, end}` runs |
| `buildDayBoundaries(axis)` | which day each stretch belongs to |
| `trackLegend(axis)` | what the hatching means, in the axis's own numbers |

**Semantic runs, not a finished gradient string.** A string is testable only by matching the
thing you just wrote — the exact shape of Phase 9's `aggregateAnchor` failure, whose test
asserted `/block-meaned/` and held a false sentence in place. Runs are data: a test asks which
stops they cover and gets an answer keyed to the fixture's own resolved state.

**Day boundaries, and why they earn their place.** The two printed labels on a real chart are
`Period aggregate` and `15 Jun 23:00 UTC` — `stops[0]` **is** the aggregate, so the day the
axis *starts* on appeared nowhere on it. Without days, two marked nights are two identical
grey bands. Live: `14 Jun / 15 Jun` on the cadence chart, `14–19 Jun` on the coarsened one.

**A legend, because hatching with no key is grey stripes a reader has to guess at** — this
phase's own failure one level along. It states the tally and stops there: the *reason* belongs
to the readout and to `methods.md`, and a third home for one fact is how homes start
disagreeing.

`utcDayLabel` is exported from `frameAxis.js` rather than reimplemented. The
naive-stamp-is-UTC rule is exactly the kind of fact that goes wrong quietly with two homes, and
a track labelled in local time would disagree with the stop labels directly above it by the
viewer's own offset.

---

## 4. A bug the second distribution caught, before it shipped

The legend read **"13 of the 48 intervals hold nothing"** on `map_1531e35a0e18`, whose every
stop averages **three** hourly intervals. There are 144 intervals on that chart and 48 frames.
The backend keeps the two nouns apart deliberately — Phase 8 §6: `methods.md` says "Empty
**frames**" in the coarsened tier and "Empty **intervals**" in the cadence one, so that three
empty 3-hour frames are never reported as three empty hours.

Same class as Phase 9's `block-meaned`: prose asserting a property the data does not have, on
the tier where it is wrong. Found by rendering the *second* real distribution, not by the
first, and not by any unit test written before it.

The noun now follows `axis.coarsened`, which `buildScrubAxis` publishes rather than letting
each consumer re-derive — its two witnesses (`tier === 'coarsened'` **or**
`buckets_per_frame > 1`) exist because deciding on one field surviving understates a frame's
width by the coarsening factor. Three tests pin it, including the missing-`tier` payload.

Live after the fix: `13 of the 48 frames` / `22 of the 48 intervals`.

---

## 5. Tests

**19 new**, all in `Frontend/tests/frameTrack.test.mjs`. Frontend suite **373 passed, 0
failed** (354 before). ESLint clean. No backend change: the only Frontend files `backend-test`
reads are `frameDelta.js` and `jobCard.js`, neither touched.

The fixtures are the real distributions — the 48-stop TEMPO axis with its 22 empty stops in
both kinds, and a fully populated 13-stop axis for the control case — and every axis is built
through the real `buildScrubAxis`, so a drift in state resolution fails these too.

Two Phase 9 lessons applied rather than restated:

- **Properties keyed to the fixture's own state, never the literal output.** The coverage test
  computes the expected empty stops by walking the axis with a differently-written reference;
  the legend tests assert the axis's own counts appear, not the sentence that happens to be
  written.
- **The call site gets a source-reading contract test.** `MapScrubber` called
  `aggregateAnchor()` with no argument for a whole build while its util test stayed green.
  Four assertions now read the JSX: that `ScrubTrack` is given `state.axis`, that all three
  utils are passed it, that the marks layer reads each mark's `start`/`end`, and that the
  legend is actually rendered. Plus one that reads `index.css`: **without `.frame-scrub` the
  input keeps its opaque OS track and paints over every mark, so the whole feature renders as
  nothing while all util tests stay green** — the failure mode with no symptom.

Red/green confirmed by mutation: keying `isEmpty` off state alone rather than off `kind` fails
all three of the first tests.

---

## 6. Live verification

Deployed bundle, signed in as `t59phase8`, 1280×720.

| | cadence `map_2ea3dd7b34cf` | coarsened `map_1531e35a0e18` |
|---|---|---|
| track | 284 px, `appearance: none`, transparent | 284 px |
| marks | **3** — 59, 12, 59 px, hatched | **6** — 12×4, 18, 12 px |
| day labels | `14 Jun`, `15 Jun` | `14 Jun` … `19 Jun` |
| legend | `22 of the 48 intervals hold nothing` | `13 of the 48 frames hold nothing` |
| scrub | stop 2 → fill 4.17 %, `QA rejected every pixel` | — |

Mark geometry matches the prediction from the 5.92 px/stop measurement exactly.

**Still not seen: an empty stop drawn.** The pane cannot composite, so no pixel-level
assertion was made this phase either. §0 makes that a much smaller gap than it was — the two
empty planes are provably all-NaN and provably render as one transparent overlay — but nobody
has watched the track's hatching, the day labels or the thumb with their own eyes. A
human-supplied capture is the only way, as in Phases 8 and 9.

---

## 7. Deliberately left out

- **The `valid_fraction` / `qa_pass_rate` coincidence** (Phase 9 §9). Cheap to disambiguate,
  and left alone on purpose: it is a readout change in a phase about the control, and folding
  it in makes the diff about two things — the same reasoning Phase 9 used to defer the slider.
- **A thumb-following label.** The header already names the current stop and updates; at
  5.9 px/stop a label at the thumb collides constantly.
- **The output panel's own collapse** — real, re-attributed in §2, and an app-layout change.
- Everything in the PRD's Non-Goals, unchanged: no playback, no skipping empty stops, no
  change to `MAX_FRAMES`, no backend change or new payload field, no per-frame export, no
  variable switching, no statistic but mean, no scrubbers on `plot_multiple`, no
  `overlay_store` eviction. `frame_grid_delta` and the delta sentences are untouched.
