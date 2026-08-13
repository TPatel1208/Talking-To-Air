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

const boxStyle = {
  border: '1px solid var(--border)', borderRadius: '10px',
  background: 'var(--bg-card)', padding: '11px 13px',
  display: 'flex', flexDirection: 'column', gap: '9px',
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

      <input
        type="range"
        min={0}
        max={stops.length - 1}
        step={1}
        value={stop?.index ?? 0}
        disabled={!state.sliderEnabled}
        onChange={(e) => onSelect(Number(e.target.value))}
        aria-label="Time interval"
        style={{ width: '100%', cursor: state.sliderEnabled ? 'pointer' : 'not-allowed' }}
      />
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-muted)' }}>
        <span>{stops[0].label}</span>
        <span>{stops[stops.length - 1].label}</span>
      </div>

      {/* Decision 2: while the pixels are in flight, and after they have been
          evicted, the slider is disabled and parked on the aggregate -- the
          only stop whose pixels are actually on screen. An enabled slider
          showing the aggregate at every stop would read as a quiet two days
          to someone scrubbing for an event. */}
      {state.detail && <div style={noteStyle}>{state.detail}</div>}

      <StopReadout stop={stop} stats={stats} />
      <DeltaDisclosure delta={delta} />
    </div>
  )
}

function StopReadout({ stop, stats }) {
  // The anchor sentence lives in frameDelta.js, not here. With no jsdom, a
  // sentence written in a component is a sentence no test can read -- which is
  // how "the same field the Map tab shows" went two phases carrying an
  // unstated block mean and an unstated 2x colour ramp (Phase 8 §1, §9).
  if (!stop || stop.kind === 'aggregate') {
    return <div style={noteStyle}>{aggregateAnchor()}</div>
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
