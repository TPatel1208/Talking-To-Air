import { computeChartStats } from './chartStats.js'
import { resolveMasking } from './maskingProvenance.js'

// D11, in one function: the chart describes what the slider points at.
//
// Statistics and QA disclosure follow the slider; provenance, citations and
// methods describe the artifact and must NOT flicker per frame -- that would
// imply the source changed. Parking at the aggregate reproduces today exactly,
// because frame 0 IS the period aggregate, which is why this delegates rather
// than reimplements there.
//
// Every per-frame number comes off the payload, never off the rendered frame
// grid (D5a): at k=8 a frame has already lost 16% of its own p98 and 70% of
// its max, so recomputing a maximum from the pixels on screen would understate
// exactly the peak the scrubber exists to find.
export function statsForStop(chart, stop) {
  if (!stop || stop.kind === 'aggregate') {
    const today = computeChartStats(chart)
    // Null stays null: the Statistics tab's "no numeric values" branch is what
    // shows masking disclosure on an all-masked scene, which is exactly when
    // it matters most. Parking at the aggregate must not manufacture a card.
    if (!today) return null
    return {
      ...today,
      qaPassRate: resolveMasking(chart)?.qaPassRate ?? null,
      scope: 'Period aggregate',
      state: 'observed',
      empty: false,
    }
  }

  const s = stop.statistics || {}
  const count = Number.isFinite(s.count) ? s.count : 0
  const empty = count <= 0

  return {
    // Absent, not zero. `{count: 0}` alone is what a frame carries when
    // nothing survived, and a maximum of zero is a measurement nobody made.
    mean: empty ? null : numberOrNull(s.mean),
    max: empty ? null : numberOrNull(s.max),
    min: empty ? null : numberOrNull(s.min),
    units: chart?.units,
    basis: 'analyzed-region',
    count,
    validPct: (stop.validFraction ?? 0) * 100,
    qaPassRate: stop.qaPassRate,
    scope: stop.label,
    state: stop.state,
    empty,
  }
}

// A frame's QA pass rate as a display percentage, floored at both ends so
// rounding cannot manufacture a verdict the pixels do not support: one
// rejected pixel in 15,000 rounds to 100.0%, and reading that as a clean
// interval is the same error the map-level rate is already floored against.
// The frame payload carries a rate and no pixel counts, so the floor keys off
// the rate itself -- a genuine 1 or 0 still renders verbatim.
export function formatFrameQaRate(rate) {
  if (rate == null || !Number.isFinite(rate)) return null
  let pct = rate * 100
  if (rate < 1 && pct > 99.9) pct = 99.9
  if (rate > 0 && pct < 0.1) pct = 0.1
  return `${pct.toFixed(1)}%`
}

function numberOrNull(value) {
  return Number.isFinite(value) ? value : null
}
