/**
 * MapScrubber.jsx
 * ---------------
 * T59 Phase 6: the time axis of a multi-granule map, made browsable, inside
 * the Map tab. `payload.type` stays "heatmap" and CHART_TABS is unchanged
 * (D15) -- this is an additive control below the map, not a new artifact type.
 *
 * The scrubber is a MODE, not a slider position (D9). Entering it recolours
 * the map once onto the stack's pooled 2-98 and redraws the legend to say so;
 * every stop including the aggregate is drawn on that one scale, so colours
 * never jump mid-scrub. Leaving it hands the Map tab back exactly as it was.
 *
 * All interpretation lives in pure utils with their own tests (there is no
 * jsdom here): frameAxis for the stops and the degradation states, frameScale
 * for the pooled scale, frameDelta for the tier disclosure, frameStats for the
 * per-stop readout, frameStack for the zero-copy layout.
 */
import { aggregateAnchor } from '../utils/frameDelta.js'
import { formatFrameQaRate } from '../utils/frameStats.js'
import { buildDayBoundaries, buildTrackMarks, trackLegend } from '../utils/frameTrack.js'

// The floor exists so the layout CANNOT produce the 0 px case, rather than so
// something can detect it afterwards. Measured live: the output panel carries
// `min-width: 0`, which defeats the automatic minimum content size, so with the
// sessions, chat and jobs panels open at a 556 px viewport it squeezes to 28 px
// and `width: 100%` on the input resolves to 0 -- an interactive control of zero
// width still reporting `disabled: false`. Its own parent already has
// `overflow-x: auto`, so giving the scrubber a floor makes that ancestor scroll
// instead of crushing the control.
//
// A width is a DOM measurement no util can see and there is no jsdom here to
// fake one, so the alternative -- measure the track and disable below a
// threshold -- would be a guard no test in this repo could reach. 260 px is
// today's 284 px rounded down to where the density is unchanged (5.3 px per
// stop at 49 stops) and the smallest run real data produces, two stops, is
// still ~11 px wide. Below that the track cannot answer honestly, and the fix
// is to never get there.
const boxStyle = {
  border: '1px solid var(--border)', borderRadius: '10px',
  background: 'var(--bg-card)', padding: '11px 13px',
  display: 'flex', flexDirection: 'column', gap: '9px',
  minWidth: '260px',
}

const toggleStyle = (active) => ({
  fontSize: '12px', fontWeight: 700, cursor: 'pointer',
  borderRadius: '7px', padding: '5px 11px',
  color: active ? 'var(--teal-text)' : 'var(--text-secondary)',
  background: active ? 'var(--teal-light)' : 'var(--bg-primary)',
  border: `1px solid ${active ? 'var(--teal)' : 'var(--border)'}`,
})

const noteStyle = { fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.45 }

// D10: an empty bucket is a first-class rendered state, and the two kinds of
// empty are different measurements. A blank map reads as zero, and zero is a
// measurement -- so the readout says which blank this is.
const EMPTY_STATE_TEXT = {
  'qa-rejected': 'Observed — QA rejected every pixel in this interval.',
  'not-retrieved': 'Nothing retrieved for this interval. QA never ran on it.',
}

export default function MapScrubber({
  state, delta, stop, stats, scrubbing, onToggle, onSelect,
}) {
  // Nothing to say: the ordinary single-granule map (no frames block, no
  // refusal beside it).
  if (!state || state.mode === 'none') return null

  // One of the three disclosed refusals, relayed in the backend's own words.
  if (state.mode === 'refused') {
    return (
      <div style={boxStyle}>
        <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)' }}>Time axis unavailable</div>
        <div style={noteStyle}>{state.detail}</div>
      </div>
    )
  }

  const stops = state.axis.stops

  if (!scrubbing) {
    return (
      <div style={boxStyle}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '10px' }}>
          <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
            This map covers {stops.length - 1} {state.axis.cadence || ''} intervals.
          </div>
          <button type="button" onClick={onToggle} style={toggleStyle(false)}>Browse time</button>
        </div>
      </div>
    )
  }

  return (
    <div style={boxStyle}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '10px' }}>
        <div style={{ fontSize: '12.5px', fontWeight: 700, color: 'var(--text-primary)' }}>{stop?.label}</div>
        <button type="button" onClick={onToggle} style={toggleStyle(true)}>Show period aggregate</button>
      </div>

      <ScrubTrack
        axis={state.axis}
        stops={stops}
        stop={stop}
        enabled={state.sliderEnabled}
        onSelect={onSelect}
      />

      {/* Decision 2: while the pixels are in flight, and after they have been
          evicted, the slider is disabled and parked on the aggregate -- the
          only stop whose pixels are actually on screen. An enabled slider
          showing the aggregate at every stop would read as a quiet two days
          to someone scrubbing for an event. */}
      {state.detail && <div style={noteStyle}>{state.detail}</div>}

      {/* `delta` reaches BOTH: the anchor reads `downsampled` off it, and
          these two sentences render one above the other. Passing it to only
          one of them let the anchor claim a block mean on a k=(1,1) chart
          while the line beneath it said the frames were not downsampled --
          adjacent, contradictory, and invisible to a util test that hands the
          argument over by hand. */}
      <StopReadout stop={stop} stats={stats} delta={delta} />
      <DeltaDisclosure delta={delta} />
    </div>
  )
}

// The control a reader actually touches. Phase 8 §8 measured it at 284 px over
// 49 stops -- 5.9 px each, two printed labels, and nothing on the track saying
// which stops hold anything. A reader dragging across twelve consecutive empty
// hours got twelve blank maps with no indication that the blankness was the
// axis rather than the air.
//
// Measured on the design target's own blob: both empty kinds are all-NaN, and
// `buildCanvasFallbackFrame` leaves every non-finite cell at alpha 0 in a fresh
// buffer, so a QA-rejected hour and a nothing-retrieved hour draw as the SAME
// fully transparent overlay. The picture cannot carry this distinction at all.
// That is why it goes on the track and in the readout rather than being left to
// the map.
//
// The marks sit BEHIND the input (which `.frame-scrub` makes transparent)
// rather than being drawn into the track's own pseudo-elements, so the element
// stays a real range input: same keyboard behaviour, same focus handling, one
// thumb to drag.
function ScrubTrack({ axis, stops, stop, enabled, onSelect }) {
  const marks = buildTrackMarks(axis)
  const days = buildDayBoundaries(axis)
  const legend = trackLegend(axis)
  const last = stops.length - 1
  const progress = last > 0 ? (stop?.index ?? 0) / last : 0

  return (
    <>
      <div style={{ position: 'relative', height: '18px' }}>
        {/* aria-hidden: the input already carries the label and the readout
            below already states, in words, what these marks show. A screen
            reader gains nothing from the geometry and would have to walk it. */}
        <div aria-hidden="true" style={{
          position: 'absolute', left: 0, right: 0, top: '50%',
          transform: 'translateY(-50%)', height: '6px', borderRadius: '3px',
          background: 'var(--bg-primary)', border: '1px solid var(--border)',
          overflow: 'hidden',
        }}>
          <div style={{
            position: 'absolute', top: 0, bottom: 0, left: 0,
            width: `${progress * 100}%`, background: 'var(--teal-light)',
          }} />
          {marks.map((mark) => (
            <div key={mark.startStop} style={{
              position: 'absolute', top: 0, bottom: 0,
              left: `${mark.start * 100}%`,
              width: `${(mark.end - mark.start) * 100}%`,
              // Hatched rather than filled: a solid band reads as a value, and
              // "no data" is the one thing this must not look like a value of.
              backgroundImage: 'repeating-linear-gradient(45deg, var(--text-muted) 0 1px, transparent 1px 4px)',
              opacity: 0.55,
            }} />
          ))}
        </div>
        <input
          className="frame-scrub"
          type="range"
          min={0}
          max={last}
          step={1}
          value={stop?.index ?? 0}
          disabled={!enabled}
          onChange={(e) => onSelect(Number(e.target.value))}
          aria-label="Time interval"
          style={{ cursor: enabled ? 'pointer' : 'not-allowed' }}
        />
      </div>

      {/* Day boundaries. Without them two marked nights are two identical grey
          bands, and the axis never prints the day it STARTS on anywhere --
          `stops[0]` is the aggregate, so the left-hand label below reads
          "Period aggregate" rather than a date. */}
      {days.length > 0 && (
        <div aria-hidden="true" style={{ position: 'relative', height: '11px' }}>
          {days.map((day) => (
            <span key={day.stop} style={{
              position: 'absolute', left: `${day.fraction * 100}%`,
              transform: dayLabelShift(day.fraction),
              fontSize: '9.5px', color: 'var(--text-muted)', whiteSpace: 'nowrap',
            }}>{day.label}</span>
          ))}
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-muted)' }}>
        <span>{stops[0].label}</span>
        <span>{stops[last].label}</span>
      </div>

      {legend && <div style={noteStyle}>{legend}</div>}
    </>
  )
}

// A label centred on a boundary at either extreme hangs off the end of the
// track, so the first and last are aligned inward instead.
function dayLabelShift(fraction) {
  if (fraction < 0.08) return 'translateX(0)'
  if (fraction > 0.92) return 'translateX(-100%)'
  return 'translateX(-50%)'
}

function StopReadout({ stop, stats, delta }) {
  // The anchor sentence lives in frameDelta.js, not here. With no jsdom, a
  // sentence written in a component is a sentence no test can read -- which is
  // how "the same field the Map tab shows" went two phases carrying an
  // unstated block mean and an unstated 2x colour ramp (Phase 8 §1, §9).
  if (!stop || stop.kind === 'aggregate') {
    return <div style={noteStyle}>{aggregateAnchor(delta)}</div>
  }

  const qa = formatFrameQaRate(stats?.qaPassRate)

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 14px', fontSize: '11.5px', color: 'var(--text-secondary)' }}>
      <span>{stop.nGranules} granule{stop.nGranules === 1 ? '' : 's'}</span>
      {/* A 90%-masked interval renders as a clean low uniform field, which is
          indistinguishable from a calm one. Coverage travels with the frame
          rather than being inferable from the picture (D10). */}
      <span>{(stats?.validPct ?? 0).toFixed(1)}% of the region covered</span>
      <span>QA pass rate {qa ?? 'not applied'}</span>
      {stats?.empty && (
        <span style={{ width: '100%', color: 'var(--text-muted)' }}>
          {EMPTY_STATE_TEXT[stop.state] || 'Nothing survived masking in this interval.'}
        </span>
      )}
    </div>
  )
}

// D16/D14, shown rather than offered. Phase 3 measured the coarsened tier's
// disagreement at 5.4% and 22.3% on two real TEMPO bundles -- not the sub-1%
// D16's own illustration implies -- and the coarsened tier is the COMMON case,
// reached at roughly 2.5 days. Risk 4 is this becoming a badge nobody reads,
// so a double-digit delta gets the warning treatment rather than a footnote.
function DeltaDisclosure({ delta }) {
  if (!delta) return null

  // Presentation follows SEVERITY, not tier. Phase 8 found the cadence tier --
  // the one this used to render as a plain unbordered note, because its
  // temporal relationship is an exact identity -- disagreeing by 1.876% on the
  // arrays it ships. A tier cannot be trusted to imply how loud its own
  // caveat should be; the largest measured disagreement can, and `severity`
  // is that, over both figures.
  if (delta.severity === 'none') {
    return <div style={noteStyle}>{delta.summary}</div>
  }

  const loud = delta.severity === 'high'
  return (
    <div style={{
      ...noteStyle,
      color: loud ? 'var(--text-primary)' : 'var(--text-muted)',
      fontWeight: loud ? 600 : 400,
      borderLeft: `3px solid ${loud ? 'var(--amber, #b45309)' : 'var(--border)'}`,
      paddingLeft: '8px',
    }}>
      {/* Composed in frameDelta.js, whole. Assembling the tail here is what
          kept it out of reach of every test in this repo. */}
      {delta.summary}
    </div>
  )
}
