# T59 Phase 13 — one store entry per plane, and a URL to fetch it by

**Built 2026-08-13.** The transport for what Phase 12 computed, and deliberately the same scope
Phase 4 had for the mean: a `FrameStack` carrying `planes={"max": ..., "min": ...}` now stores
one independently-addressable, independently-evictable entry per plane, each servable by its own
URL with its own ETag — and **nothing calls it yet**. `_attach_frames` still asks for the mean
alone, so no chart produces a plane, and every payload on the wire is byte-identical to the one
that shipped before this. Phase 14 is the wiring, Phase 15 the frontend toggle, Phase 16
`methods.md`.

## The interface

```
GET /chart/{id}/frames.f32.gz              # the mean — untouched, byte for byte
GET /chart/{id}/frames.{statistic}.f32.gz  # max, min
```

```python
write_frames(source, *, pipeline_version, statistic="mean", protect=())
read_frames(key, *, pipeline_version, statistic="mean")
evict_to_fit(incoming_bytes, *, protect=())
```

`store_frame_stack` writes the mean exactly as before and then one entry per plane, listing each
under `block["planes"][statistic]` with `_key`, `value_range`, `frame_grid_delta` and (max only)
`extent_overstatement` — `StatisticPlane` minus the arrays. `"mean"` is not a key in it, for the
same reason it is not a key in `FrameStack.planes`.

**`"planes"` is omitted entirely when there are none**, rather than emitted empty. Nothing
upstream asks for a plane this phase, so an always-present `"planes": {}` would have been a wire
change to every chart ever plotted, in exchange for saving one `?? {}` in a consumer that has to
handle the key's absence on old rows regardless.

## The four design decisions, and why each went the way it did

**1. A path per statistic, not a query parameter.** `?statistic=max` makes one URL serve several
bodies, so every cache between the browser and the app has to be told with `Vary`, and
`_FRAME_CACHE_CONTROL`'s `immutable` becomes a claim about a URL whose content is no longer
fixed. A distinct path is a distinct cache entry needing no coordination at all, and it leaves
the mean's URL untouched *outright* rather than untouched-if-nobody-passes-the-parameter — which
is what D6a decision 5's "the mean entry keeps its exact shape, URL and cost" actually asks for.
The segment is constrained to `STORABLE_STATISTICS` at the route, so it can never become an
arbitrary key lookup into the chart's own block.

**2. The per-plane block carries only what differs per plane.** `t_start`/`t_end`/`n_granules`/
`valid_fraction`/`qa_pass_rate` describe the *interval*, and `statistics` is the per-frame
regional scalar off the **mean** field — which Phase 12 pinned as a measurably different number
from the max plane's peak (50.5 against 100.0 on its fixture). `shape`/`dtype`/`period_index`/
`lats`/`lons`/`cells_per_frame`/`coarsen_k`/`cadence`/`tier`/`buckets_per_frame` are shared by
construction: every plane of a chart is the same blocks of the same grid over the same axis.
Copying any of them per plane would give a quantity that has one account a second one.

**3. A chart's planes cannot evict its own mean.** `write_frames` evicts to fit before every
write, so three sequential writes against one cap let a chart's **max** entry evict that chart's
own **mean** — silently, and correctly by LRU's own rules. The result is a chart that scrubs in
max mode and is dead in the default one, which is the worst available way for this to fail
because mean is what every existing consumer reads. `evict_to_fit` gained an explicit `protect`
set (defaulting to empty, so `EvictionTests` passes unchanged); `store_frame_stack` writes the
mean first and protects it, and each landed plane, from the writes that follow. Protection is
per-call and never persisted — nothing in the store is permanently unevictable, and a separate
test pins that the next chart evicts the last one exactly as before. A write left with nothing
evictable still proceeds and the store sits briefly over the cap, which is the same shape of
bound `evict_to_fit` already documents for an entry larger than the store.

**4. The manifest records which statistic it holds, and a mismatch is refused.** Every plane of
one chart has the same shape, dtype, grid and pipeline version, so serving the max blob where the
mean was asked for passes every check `read_frames` made and renders a believable field with the
wrong numbers in it. `_intact` asks whether these are the bytes that were *written*; nothing
asked whether they are the bytes that were *asked for*. This is T54's "content_digest ALONE is an
unsafe cube key" lesson arriving at a third axis.

### Where this phase departed from its own prompt

**The prompt specified "the same drop-not-just-miss posture the `pipeline_version` guard takes"
for a statistic mismatch. That is wrong, and the guard refuses without dropping.** The version
guard drops because the version only moves forward, so *no future request can ever match that
entry* and keeping it charges the cap for something unreachable. The premise inverts here: an
entry refused for the wrong statistic is perfectly servable to the right one. A mismatch means
the **request** is wrong, not the entry — and dropping would let a mis-keyed read of the mean
permanently destroy a healthy max, turning one degraded statistic into two on a path whose entire
job is to degrade narrowly. It is logged under the existing `frame_read_failed` event either way,
so the mixup is still visible in logs rather than only in a wrong picture.

**A manifest with no `statistic` reads as the mean.** Not a fudge: every entry written before this
phase *is* a mean entry, because no other kind could be produced. Treating the absence as unknown
would have stripped the scrubber off every chart already sitting in a deployed store, for a
version bump that never happened.

## What was measured

**Backend container suite: 1687 passed / 1 skipped / 0 failed** (from 1665), 14m46s. 22 new
tests:

- 13 in `test_frame_store.py` (12 → 25) across four new classes and one extended — one per
  guarantee the existing five classes already pin for the mean.
- 9 endpoint tests in `test_chat_endpoint.py`, beside the existing five rather than editing them.
- The mean's five existing endpoint tests, `EvictionTests`, `SelfHealTests` and
  `ChartRowCarriesTheAxisTests` all pass **unchanged**.

The first test written was `test_attaching_planes_moves_nothing_about_the_mean_entry`: the mean
blob's bytes, its ETag, and every key of its block are identical with and without planes
attached, with `"planes"` the only difference. It is first for the same reason Phase 12's
equivalent was — it is the regression this phase was most likely to cause and least likely to
notice.

**Two tests passed on their first run, which is itself the finding.** `sweep_store` reclaims
plane entries on a superseded version, including the half-deployed chart whose planes straddle
the boundary, with no code change at all: the sweep judges each entry directory on its own
manifest and a plane entry is an ordinary entry to it. The same held for per-plane eviction.

Nothing in the reduction was touched, so Phase 12's live numbers — retention 100.00% at every
percentile, overstatement 24.6985×, both identities exactly `0.0` — are unchanged by
construction and were not re-run.

## Deployment

No change. nginx proxies `/api/` wholesale, so the new path needs no location block. The
`frame_store` named volume, its ownership and its exclusion from the public output tree are
unaffected — but `test_frame_store_persistence.py`'s claim that "frames are served only through
`/chart/{id}/frames.f32.gz`" became false the moment a second route existed, and now names both.
A chart with all three statistics is three entries rather than one, so it reaches the cap three
times as fast; the no-per-entry-cap argument is unaffected, because it rests on the size of a
single entry, which a plane does not change.

## Explicitly not done

`plot_tools.py` in any form — `_attach_frames` still calls `build_frame_stack` with no
`statistics` argument, `_wire_frames_url` still mints one url, `_frames_summary` names no toggle,
and `payload["export"]["frames"]["spec"]["statistic"]` is still `"mean"`. `frame_stack.py`. Any
frontend file. `methods.md`/`methods_export_service.py` — the sentence D6a decision 8 promises and
the disclosure G4's ≈24.7× argues for are Phase 16's, and writing them here would document a
capability no user can reach. `_pack`'s layout, `_PERIOD_INDEX`, the gzip level, the minted-key
rule and the integrity check are all untouched: a plane is a fourth entry speaking the existing
protocol, not a reason to revisit the protocol.
