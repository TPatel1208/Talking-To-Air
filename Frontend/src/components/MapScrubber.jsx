/**
 * MapScrubber.jsx
 * ---------------
 * The time axis of a multi-granule map, made browsable, inside the Map tab.
 * `payload.type` stays "heatmap" and CHART_TABS is unchanged -- this is an additive control below the map, not a new artifact type.
 *
 * The scrubber is a MODE, not a slider position. Entering it recolours
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
import { STATISTIC_LABELS } from '../utils/frameStatistic.js'
import { buildDayBoundaries, buildTrackMarks, trackLegend } from '../utils/frameTrack.js'
import { SCRUB_TRACK_MIN_WIDTH } from '../utils/panelLayout.js'
import SciText from './SciText.jsx'

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
// threshold -- would be a guard no test in this repo could reach.
//
// The number is shared with the panel that has to CONTAIN this, and lives
// there: a floor the scrubber picks for itself and a floor the output panel
// picks for itself are two numbers that quietly stop fitting inside each other.
const boxStyle = {
  border: '1px solid var(--border)', borderRadius: '10px',
  background: 'var(--bg-card)', padding: '11px 13px',
  display: 'flex', flexDirection: 'column', gap: '9px',
  minWidth: `${SCRUB_TRACK_MIN_WIDTH}px`,
}

const toggleStyle = (active) => ({
  fontSize: '12px', fontWeight: 700, cursor: 'pointer',
  borderRadius: '7px', padding: '5px 11px',
  color: active ? 'var(--teal-text)' : 'var(--text-secondary)',
  background: active ? 'var(--teal-light)' : 'var(--bg-primary)',
  border: `1px solid ${active ? 'var(--teal)' : 'var(--border)'}`,
})

const noteStyle = { fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.45 }

// The statistic toggle (mean/max/min). Smaller than the mode
// button beside it, because entering scrubber mode is the decision and this is
// a choice made inside it.
const statisticStyle = (active, enabled) => ({
  fontSize: '11px', fontWeight: 700,
  cursor: enabled ? 'pointer' : 'not-allowed',
  borderRadius: '6px', padding: '3px 9px',
  color: active ? 'var(--teal-text)' : 'var(--text-secondary)',
  background: active ? 'var(--teal-light)' : 'var(--bg-primary)',
  border: `1px solid ${active ? 'var(--teal)' : 'var(--border)'}`,
  opacity: enabled ? 1 : 0.6,
})

// An empty bucket is a first-class rendered state, and the two kinds of
// empty are different measurements. A blank map reads as zero, and zero is a
// measurement -- so the readout says which blank this is.
const EMPTY_STATE_TEXT = {
  'qa-rejected': 'Observed — QA rejected every pixel in this interval.',
  'not-retrieved': 'Nothing retrieved for this interval. QA never ran on it.',
}

export default function MapScrubber({
  state, delta, stop, stats, scrubbing, onToggle, onSelect,
  choice, overstatement, onSelectStatistic,
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

      {/* Every string here is composed in a util. The toggle changes four
          sentences at once (the anchor, the delta, the legend caption and the
          Statistics tab's scoping), and composing them in a util keeps every
          one testable. */}
      <StatisticChoice choice={choice} onSelect={onSelectStatistic} />

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
      {/* The max plane's extent-overstatement note, beside the scrubber.
          Resolved from the chart's own measured `headline` upstream, so this
          renders nothing when there is none. */}
      {overstatement && <div style={noteStyle}>{overstatement}</div>}
    </div>
  )
}

// The statistic toggle. A chart above `MAX_PLANE_NATIVE_CELLS` gets it
// DISABLED with the backend's own reason beside it, never silently missing.
//
// A chart that never had planes gets neither -- `offered` is false and this
// renders nothing.
function StatisticChoice({ choice, onSelect }) {
  if (!choice || !choice.offered) return null
  const enabled = choice.options.length > 1

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
      <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Show</span>
      {choice.options.map((name) => (
        <button
          key={name}
          type="button"
          disabled={!enabled}
          aria-pressed={choice.selected === name}
          onClick={() => onSelect(name)}
          style={statisticStyle(choice.selected === name, enabled)}
        >
          {STATISTIC_LABELS[name] || name}
        </button>
      ))}
      {/* The backend's own words, relayed -- the same posture the three
          disclosed frame refusals are relayed in. */}
      {choice.refusal?.detail && (
        <span style={{ ...noteStyle, flexBasis: '100%' }}>{choice.refusal.detail}</span>
      )}
    </div>
  )
}

// The control a reader actually touches. Stops can be ~6 px wide, so the track
// marks which ones hold data; otherwise dragging across empty hours just shows
// blank maps with no explanation.
//
// A QA-rejected hour and a nothing-retrieved hour draw as the SAME transparent
// overlay, so the distinction goes on the track and in the readout.
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
  // The anchor sentence lives in frameDelta.js, not here: with no jsdom, a
  // sentence written in a component is a sentence no test can read.
  if (!stop || stop.kind === 'aggregate') {
    return <div style={noteStyle}>{aggregateAnchor(delta)}</div>
  }

  const qa = formatFrameQaRate(stats?.qaPassRate)

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 14px', fontSize: '11.5px', color: 'var(--text-secondary)' }}>
      <span>{stop.nGranules} granule{stop.nGranules === 1 ? '' : 's'}</span>
      {/* A 90%-masked interval renders as a clean low uniform field, which is
          indistinguishable from a calm one. Coverage travels with the frame
          rather than being inferable from the picture. */}
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

// The frame/map disagreement, shown rather than offered. On real TEMPO bundles
// the coarsened tier (the common case, from ~2.5 days) disagrees by 5-22%, so
// a double-digit delta gets the warning treatment rather than a footnote.
function DeltaDisclosure({ delta }) {
  if (!delta) return null

  // Presentation follows SEVERITY, not tier: even the cadence tier (an exact
  // temporal identity) can disagree on the arrays it ships. `severity` is the
  // largest measured disagreement over both figures.
  if (delta.severity === 'none') {
    return <div style={noteStyle}><SciText>{delta.summary}</SciText></div>
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
      <SciText>{delta.summary}</SciText>
    </div>
  )
}
