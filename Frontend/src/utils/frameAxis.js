// Turns a heatmap payload's `frames` block (T59 D13 -- the axis lives in the
// chart's jsonb row, the values live in a separate blob) into the scrub axis
// the Map tab's slider is drawn from.
//
// Pure, and deliberately the FIRST thing built: if the axis mis-reads an
// empty bucket or a null QA rate, every pixel drawn after it is describing
// the wrong hour.

// One stop per plane in the [1+N, ny, nx] blob: the period aggregate plus one
// per cadence bucket. The aggregate leads because parking there reproduces the
// Map tab exactly (D5/D11), not because it is plane 0 -- `period_index` says
// which plane it is, and the buckets take the planes it does not.
export function buildScrubAxis(chart) {
  const block = chart?.frames
  if (!block || !Array.isArray(block.frames) || !block.frames.length) return null

  const periodPlane = Number.isInteger(block.period_index) ? block.period_index : 0
  const bucketPlanes = []
  for (let plane = 0; bucketPlanes.length < block.frames.length; plane++) {
    if (plane !== periodPlane) bucketPlanes.push(plane)
  }

  const cadence = block.cadence || 'unknown'
  // Two witnesses, the same pair `_is_coarsened` uses in the export service:
  // what this decides is whether a stop may be labelled as one interval, and
  // that is a category error when it is wrong -- understating a frame's width
  // by the coarsening factor -- so it must not turn on one field surviving.
  const coarsened = block.tier === 'coarsened'
    || (Number.isFinite(block.buckets_per_frame) && block.buckets_per_frame > 1)
  const stops = [
    { kind: 'aggregate', index: 0, plane: periodPlane, label: 'Period aggregate' },
    ...block.frames.map((frame, i) => ({
      kind: 'interval',
      index: i + 1,
      plane: bucketPlanes[i],
      tStart: frame.t_start,
      tEnd: frame.t_end,
      nGranules: frame.n_granules,
      validFraction: numberOrNull(frame.valid_fraction),
      // Absent stays absent. `null` means QA never ran on this interval, which
      // is a different fact from a real 0% -- defaulting it to 0 would report
      // "QA rejected everything" for a bucket nothing was retrieved for.
      qaPassRate: numberOrNull(frame.qa_pass_rate),
      statistics: frame.statistics || null,
      state: bucketState(frame),
      label: intervalLabel(frame.t_start, frame.t_end, cadence, coarsened),
    })),
  ]

  return {
    stops,
    cadence,
    tier: block.tier || null,
    cellsPerFrame: Number.isFinite(block.cells_per_frame) ? block.cells_per_frame : null,
  }
}

// D10's three-way read of a bucket. `{count: 0}` alone is what a frame carries
// when nothing survived -- absent statistics, not a maximum of zero -- so
// emptiness is read off the surviving count, and the two empty kinds split on
// whether QA ever ran.
function bucketState(frame) {
  const count = frame?.statistics?.count
  const empty = Number.isFinite(count) ? count <= 0 : !(frame?.valid_fraction > 0)
  if (!empty) return 'observed'
  return frame?.qa_pass_rate == null ? 'not-retrieved' : 'qa-rejected'
}

// The scrubber's whole render state in one read: whether it is offered at all,
// whether the slider can move, and what to say when it cannot. Five modes,
// because the three ways this degrades (deliverable 8) plus the two ways a
// stack that WILL work looks before it does are all different sentences.
//
// `loadState` is the blob fetch: 'idle' | 'loading' | 'loaded' | 'expired'
// (a 404 -- evicted, or superseded pipeline version) | 'failed'.
export function resolveFrameState(chart, loadState = 'idle') {
  const axis = buildScrubAxis(chart)

  if (!axis) {
    const refusal = chart?.frames_unavailable
    // Three refusals are deliberately silent backend-side and produce no key
    // at all; a missing block with nothing beside it is the ordinary
    // single-granule map, not a failure to report.
    if (!refusal) return { mode: 'none', offered: false, sliderEnabled: false, reason: null, detail: null, axis: null }
    return {
      mode: 'refused',
      offered: false,
      sliderEnabled: false,
      reason: refusal.reason || null,
      detail: refusal.detail || null,
      axis: null,
    }
  }

  const base = { offered: true, axis, reason: null }

  // D8: no regeneration, ever. Both of these are terminal, and they are
  // different facts -- one stack was never stored, the other was and aged out.
  if (!chart.frames.url) {
    return {
      ...base, mode: 'axis-only', sliderEnabled: false, reason: 'not-stored',
      detail: 'The frame values for this map were not stored, so the time axis is labelled but cannot be scrubbed. Frames are never rebuilt after the fact.',
    }
  }
  if (loadState === 'expired') {
    return {
      ...base, mode: 'axis-only', sliderEnabled: false, reason: 'expired',
      detail: 'The frame values for this map have expired and are no longer stored, so the time axis is labelled but cannot be scrubbed. Frames are never rebuilt after the fact.',
    }
  }
  if (loadState === 'failed') {
    return {
      ...base, mode: 'axis-only', sliderEnabled: false, reason: 'fetch-failed',
      detail: 'The frame values could not be loaded, so the time axis is labelled but cannot be scrubbed.',
    }
  }
  if (loadState === 'loaded') {
    return { ...base, mode: 'ready', sliderEnabled: true, detail: null }
  }
  // Decision 2: disabled, parked on the aggregate. An enabled slider that
  // showed the aggregate at every stop would read as a quiet two days to
  // someone scrubbing for an event.
  return {
    ...base, mode: 'pending', sliderEnabled: false, reason: 'loading',
    detail: 'Loading frame values — the time axis is ready, the pixels are still arriving.',
  }
}

function numberOrNull(value) {
  return Number.isFinite(value) ? value : null
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

// The backend writes naive ISO stamps off a UTC time axis, and `new Date()`
// reads a naive stamp as LOCAL. Labelling one that way silently re-times every
// interval by the viewer's offset -- a reader in New York would be told
// TEMPO's 01:00 UTC scan is 21:00 the previous day, with nothing on screen
// saying so. Parsed as UTC and labelled as UTC, both explicitly.
// In tier one a frame IS one cadence bucket, so its start names it exactly and
// the weighting disclosure already states the width. In tier two a frame
// averages several, and `_frame_intervals` ships `t_end` for a stated reason:
// "a stop labeled with one day's timestamp while showing three days averaged
// is a lie the picture itself cannot correct". This is the axis honouring
// that -- before Phase 9 it read `t_start` and dropped the end, leaving the
// delta sentence as the only thing on screen carrying a frame's width.
function intervalLabel(start, end, cadence, coarsened) {
  const at = parseUtc(start)
  if (!at) return String(start ?? '')
  const clock = cadence === 'hourly' || cadence === 'sub_hourly' || cadence === 'sub-hourly'
  const until = coarsened ? parseUtc(end) : null
  if (!until) return clock ? `${dayOf(at)} ${timeOf(at)} UTC` : `${dayOf(at)} ${at.getUTCFullYear()}`

  if (clock) {
    // `t_end` is exclusive, and for a clock range that is exactly how a reader
    // reads it: 03:00-06:00 is three hours. Compact only inside one day --
    // TEMPO's gaps are diurnal, so a 3-hour frame across midnight is ordinary.
    return dayOf(at) === dayOf(until)
      ? `${dayOf(at)} ${timeOf(at)}–${timeOf(until)} UTC`
      : `${dayOf(at)} ${timeOf(at)} – ${dayOf(until)} ${timeOf(until)} UTC`
  }
  // For whole days the exclusive end has to come back a day, or a 3-day frame
  // reads as four. Which is the same lie, one unit up.
  const last = new Date(until.getTime() - 86_400_000)
  return dayOf(at) === dayOf(last)
    ? `${dayOf(at)} ${at.getUTCFullYear()}`
    : `${dayOf(at)} – ${dayOf(last)} ${last.getUTCFullYear()}`
}

function dayOf(at) {
  return `${String(at.getUTCDate()).padStart(2, '0')} ${MONTHS[at.getUTCMonth()]}`
}

function timeOf(at) {
  return `${String(at.getUTCHours()).padStart(2, '0')}:${String(at.getUTCMinutes()).padStart(2, '0')}`
}

function parseUtc(stamp) {
  if (typeof stamp !== 'string' || !stamp) return null
  // Already zoned (trailing Z or a +hh:mm offset) -> trust it; naive -> UTC.
  const zoned = /(?:Z|[+-]\d{2}:?\d{2})$/.test(stamp) ? stamp : `${stamp}Z`
  const at = new Date(zoned)
  return Number.isNaN(at.getTime()) ? null : at
}
