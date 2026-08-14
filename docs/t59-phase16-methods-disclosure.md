# T59 Phase 16 — the document, and the claim you can pin when the sentence is prose

**Built 2026-08-14.** Phase 15 gave the max plane four sentences on screen. `methods.md` said
nothing about any of it. This phase adds a `### Browsable statistics` section: what the axis can be
browsed as, what identity each plane satisfies, what each identity was checked against, and — for
max — how much ground a rendered peak claims.

This is the last phase of T59 and the one where the disclosure leaves the screen. A `methods.md`
goes into supplementary material and is read by someone who was not sitting at the scrubber and
cannot be asked what they clicked.

**27 new tests, all in `test_methods_export_service.py`** — 33 items → 60 (32 passed / 1 skipped →
58 passed / 2 skipped on the host). **All 33 that existed before this phase pass unchanged**; none
needed editing, which was deliverable 1. Frontend stays at **430 passed** — one test re-pointed,
none added, none removed. `mypy` clean on the changed module.

**The suite numbers, and why they are two numbers.** The 1704 baseline this phase's prompt quotes
is a *container* measurement, and the container suite is not the host suite (Phase 8's own
finding): 12 tests skip on the host and pass under `docker compose backend-test`. Both reconcile
exactly against one collection count.

| | selected | passed | skipped |
|---|---|---|---|
| baseline, container (prompt's figure) | 1,705 | 1,704 | 1 |
| baseline, host (derived) | 1,705 | 1,692 | 13 |
| **after this phase, host (measured)** | **1,732** | **1,718** | **14** |
| after this phase, container (predicted, NOT run) | 1,732 | 1,731 | 1 |

`pytest --collect-only` reports **1,732 selected / 33 deselected** at this commit, against 1,705
before — +27, which is exactly the tests added, in the one file they were added to. The host's
extra skip is this phase's own compose-mount assertion, which passes in the container. **The
container row is arithmetic, not a measurement**: the ~36-minute host run is what was actually
executed.

## Tension 1 — the section, and why it is a section

**Resolved: its own `### Browsable statistics`, parallel to `### Frame–map agreement`.**

The deciding question the prompt names answers itself: *if a reader cites this figure having
browsed it in max mode, which of these documents leaves them able to describe what they saw?* Only
the disclosing one. A capability line alone would leave G4's ≈24.7× unstated in the one artifact
that outlives the session, and the two figures above it got their own section for exactly the
reason this fact needs one — it qualifies what the reader is about to cite.

D12 is unchanged and is stated twice over. `_frames_section`'s export bullet **survives verbatim**
— a test asserts it whole, asserts it appears exactly once in the document, and asserts the new
section sits after `### Temporal frames` and before `### References`. The new section's own lead
says the same thing in its own words: *"They are viewing modes: exports and downloads are
unchanged."*

## Tension 2 — pin the claims, not the wording

`_DELTA_HIGH` ↔ `severityOf`'s high edge and `_DELTA_FLOOR` ↔ `formatPct`'s floor are constants,
and a constant can be read out of the other language and compared. **Prose cannot.** A test
asserting the document and the screen hold equal strings would be a test forbidding the document
from reading like a document, and `_figure_lines`' rule — *quoted, never paraphrased* — is about a
string that IS quoted, which is exactly what these sentences are not.

`SharedPlaneClaimTests` pins the claim inside each sentence, from both sides, through the bind
mount that already serves `frameDelta.js`. `frameStatistic.js` is inside the same mount and needed
no compose change; the mount itself is asserted against `docker-compose.yml`, because a skip when
the file is absent would let it disappear silently.

| the claim | the screen | the document |
|---|---|---|
| no percentage for a plane's agreement | the exact branch of `selectionDelta` contains no `pct` | no `_pct`, no `"under 0.1%"`, no `"0.0%"` |
| no bound on the overstatement | `extentOverstatementNote`'s rendered string has no `up to` / `at most` / `ceiling` | same three, plus the ceiling's own value absent from the section |
| the figure is measured, not asserted | reads `extent_overstatement` and `headline.toFixed(1)`, no `24.7` literal | two fixtures produce two documents |

**Each pattern was verified to bite**, by mutating the JS in memory and confirming the assertion
fails: `up to` reintroduced into the rendered sentence, the ceiling rendered as a bound, the
measured figure replaced by G4's literal, and the identity turned into `${grid.pct}`. **4/4
caught.** A contract test that cannot fail is worse than none, and this repo has already measured
that exact thing happening — Phase 8 edited the JS edge to 0.15 and watched every test in both
languages stay green.

**The shared-basis alternative was not taken**, deliberately. It is stronger and it is what this
project reaches for elsewhere, but Phase 15 does not quote `extent_overstatement.basis` on screen,
so taking it would have meant editing `frameStatistic.js` — which this phase's scope puts
off-limits except as a read-only check. It stays available: the payload already carries the string.

What the document *does* quote, because these are quotable, is a basis under every claim:
`PLANE_AGREEMENT_BASIS[statistic]` beneath each identity and `EXTENT_OVERSTATEMENT_BASIS` beneath
the overstatement. **The third basis string in `frame_stack` had never been printed anywhere until
now.** Each plane gets its own, never the mean's: a reader shown *"mean of the stored frame
planes"* beside an identity produced by taking their maximum has been told something false, and
quietly, because the max plane's number is zero under its own basis and would be large under the
mean's.

## Tension 3 — which block, and the fact I did not widen

`statistics` is spec-only, the plane disclosures are render-only, and **neither is a shared field,
so `_SHARED_RECIPE_FIELDS` is untouched.** `_recipe` already merges the whole spec, so the recipe's
`statistics` arrives for free; the per-plane disclosures are read off `render` directly, which is
`_delta_section`'s own precedent for a render-only fact.

`_browsable_statistics` prefers the recipe's list — it is `_scrubbable_statistics`' own output,
derived at plot time from the keys that got a `_key` — and falls back to the render block's planes
**read off `url`**, the same witness the frontend uses. A row whose spec says `["mean", "max"]`
discloses one plane even with two blocks present, and a row with no spec at all falls back without
losing the section.

## Tension 4 — three states, and the middle one is new

- **planes present** → the section above.
- **`planes_unavailable`** → *"This figure's time axis is browsable as a period mean only."* plus
  the gate's own sentence, quoted. `_refusal_section`'s posture one level in: this figure HAS a
  scrubber, so what is missing is the toggle rather than the axis. A refusal carrying no detail
  still names what is missing.
- **no `planes` key** → **silence**, and a document byte-identical to what it was. Phase 13 omitted
  the key rather than emitting it empty precisely so this state stays distinguishable from the one
  above, and every chart plotted before Phase 14 is in it. A test asserts the section is absent and
  that the word "maximum" appears nowhere in such a document.

## Tension 5 — the noun trap, hit for the third time

`_gap_rule` and `_weighting_rule` already branch on `_is_coarsened` for this. A max plane's frame
in tier two holds the highest value **across** the intervals it spans, not the maximum **of** one —
and the wrong noun is wrong in the direction that makes the plane sound more precise than it is,
which is the direction *"13 of the 48 intervals"* was wrong in. Asserted on a coarsened fixture,
including the row that lost `buckets_per_frame` and must read *"across the hourly intervals it
spans"* rather than *"across the 0 hourly intervals"*.

Phase 15's `selectionDelta` was checked first, as the prompt asked, and its wording is right.

## One thing this phase found and fixed that was not on the list

**The document has a misattribution hazard the screen does not have.** On screen the toggle
*replaces* the delta disclosure — `resolveFrameDelta` branches on the statistic before it touches
either of the mean's figures — so the mean's 1.876% is never on the page beside a max plane. In
this document both sections are always present, and `**1.9%**` sits three lines above bullets about
a different statistic with nothing saying whose it is.

One clause closes it, and it lives in the **new** section rather than the old one, so the figures
above keep their single account of themselves:

> The frame–map agreement figures above are the period mean's; each plane's own is stated with it.

Emitted only when there is an agreement section to point at.

## The finding Phase 15 left, closed

`frameScale.test.mjs`'s *"the aggregate is drawn on the pooled scale too"* has been **vacuous since
Phase 6**: it passed a stop index to a function that has never had a stop parameter. Before Phase
15 the third argument was ignored, so it compared a call to itself; after Phase 15 gave that
position a `statistic`, `0` and `2` became unrecognized statistics and both returned `null`, so it
asserted `deepEqual(null, null)`.

**Re-pointed rather than deleted.** Its named property is now structural — there is no stop to vary
— so it asserts the half that remains checkable: the pooled clip is taken whole and brackets the
aggregate, and the basis says so. Two lines were added pinning what actually made it vacuous: the
third parameter is a statistic, `'mean'` is its default, and an index in that position selects no
plane at all. Frontend stays at 430; one green line stopped meaning nothing.

## A guard added on the way out

`_SELECTION_NOUNS` is not a label table. The identity sentence beside each noun asserts that the
statistic **selects one of the values it is given**, which is true of a maximum and a minimum and
false of a mean, a median over an even count, or a percentile. So a statistic added to
`PLANE_STATISTICS` must not be given prose here by default — and must not vanish from the document
in silence either. A test pins the vocabulary against `PLANE_STATISTICS`, so adding one fails
loudly with a message saying what decision is owed.

That is the fourth pairing of this kind in T59, after `_DELTA_HIGH`, `_DELTA_FLOOR` and
`STATISTIC_ORDER` ↔ `PLANE_STATISTICS`.

## What was not done

No frontend source file. No change to what is exported (D12). `plot_tools.py`, `frame_stack.py`,
`frame_store.py`, the endpoint. `MAX_PLANE_NATIVE_CELLS`. The Map tab following the statistic
(D11's exception is explicit: it does not). Playback. Comparison-panel scrubbers (D7). Variable
switching (D1). Frame regeneration (D8).

**A finding stated rather than fixed:** `### Frame–map agreement` does not name the statistic its
figures belong to. Now that three exist, that section is implicitly mean-scoped and says so only by
position. Scoping it in its own words would edit assertions in `FrameMapAgreementTests`, which are
pre-existing tests — a finding to state, not a test to adjust. The clause added to the new section
covers the reader; the older section's own wording is a later sweep's call.

**Not verified live.** Everything here is measured against fixtures shaped as `_attach_frames` and
`store_frame_stack` write them, and against the payload contract `test_plot_frames_wiring.py` pins.
No real chart's `methods.md` has been generated with a plane in it.

## Reproducing

```bash
cd Backend && python -m pytest tests/test_methods_export_service.py -q
```

```bash
cd Frontend && npm test
```

The cross-language patterns, proven able to fail:

```bash
cd Backend && python -m pytest tests/test_methods_export_service.py -q -k SharedPlaneClaim
```
