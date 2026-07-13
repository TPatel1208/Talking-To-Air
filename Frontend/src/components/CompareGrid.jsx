/**
 * CompareGrid.jsx
 * ----------------
 * Compare mode's comparison grid. Two independent kinds share the same
 * slot/badge/cap shell (App.jsx, compareMode.js) but render differently:
 *
 * - heatmap (T28): one independent, fully interactive MapLibreHeatmapPanel
 *   per filled slot -- own pan/zoom/hover, no synced viewport. Grid is
 *   repeat(min(n,2)) columns -- 2 gives one row of 2, 3 gives a 2+1 layout,
 *   4 gives a proper 2x2. When every filled slot shares the same variable/units, all
 *   panels are recolored onto one shared vmin/vmax/colormap and render one
 *   shared legend here instead of their own; otherwise each panel keeps its
 *   own natively computed scale and legend, with an inline note explaining
 *   why.
 * - timeseries (T29): when every filled slot shares the same units and every
 *   pair of time ranges overlaps, one overlaid Plotly figure (one trace/
 *   legend entry per slot) replaces the grid entirely. On any mismatch, it
 *   falls back to the same small-multiple grid layout, mounting one
 *   independent TimeSeriesPanel per filled slot instead of MapLibreHeatmapPanel.
 */
import { useMemo } from 'react'
import MapLibreHeatmapPanel from './MapLibreHeatmapPanel.jsx'
import { TimeSeriesPanel, TimeSeriesOverlayPanel } from './ChartMessage.jsx'
import { colorbarGeometry } from '../utils/colorbarGeometry.js'
import { computeSharedColorScale } from '../utils/sharedColorScale.js'
import { filledCharts, activeCompareKind } from '../utils/compareMode.js'
import { timeseriesOverlayCompatible, toOverlaySeries } from '../utils/timeseriesCompare.js'

function SharedLegend({ vmin, vmax, colormap, units }) {
  const { gradientStops, ticks } = colorbarGeometry({ vmin, vmax, lut: colormap?.lut, tickCount: 5 })
  if (!gradientStops.length) return null
  const gradientId = 'tta-compare-shared-gradient'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
      <svg width={200} height={14} style={{ display: 'block', flexShrink: 0 }}>
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="1" y2="0">
            {gradientStops.map((stop, i) => (
              <stop key={i} offset={stop.offset} stopColor={stop.color} />
            ))}
          </linearGradient>
        </defs>
        <rect x={0} y={0} width={200} height={14} fill={`url(#${gradientId})`} rx={2} />
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', width: 200, fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
        {ticks.map((tick, i) => <span key={i}>{tick.value.toExponential(1)}</span>)}
      </div>
      {units && <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{units}</span>}
    </div>
  )
}

function chartLabel(chart) {
  return chart.title || chart.provenance?.variable || chart.variable || 'Output'
}

function SlotPlaceholder({ height, hint }) {
  return (
    <div
      style={{
        height, display: 'flex', alignItems: 'center', justifyContent: 'center',
        textAlign: 'center', padding: '0 16px',
        border: '1px dashed var(--border)', borderRadius: '8px',
        color: 'var(--text-muted)', fontSize: '12.5px',
      }}
    >
      {hint}
    </div>
  )
}

// Small-multiple grid shared by both kinds: one card per slot, `renderChart`
// supplies the kind-specific panel for filled slots.
function SlotGrid({ compareCount, compareSelection, height, hint, renderChart }) {
  return (
    <div
      style={{
        flex: 1,
        minHeight: 0,
        overflow: 'auto',
        display: 'grid',
        gridTemplateColumns: `repeat(${Math.min(compareCount, 2)}, 1fr)`,
        gap: '14px',
        alignContent: 'start',
      }}
    >
      {compareSelection.map((chart, i) => (
        <div
          key={i}
          style={{
            border: '1px solid var(--border)', borderRadius: '10px',
            padding: '10px', background: 'var(--bg-card)', minWidth: 0,
          }}
        >
          <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '6px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            Slot {i + 1}{chart ? ` — ${chartLabel(chart)}` : ''}
          </div>
          {chart ? renderChart(chart, i) : <SlotPlaceholder height={height} hint={hint} />}
        </div>
      ))}
    </div>
  )
}

// heatmap kind (T28): independent live MapLibreHeatmapPanel per slot, plus
// the shared-color-scale legend/toggle when every filled slot's variable and
// units match.
function HeatmapCompareBody({ compareCount, compareSelection, accessToken, autoScaleEach, onToggleAutoScale, height }) {
  const filled = useMemo(() => filledCharts(compareSelection), [compareSelection])
  const shared = useMemo(() => computeSharedColorScale(filled), [filled])
  const useShared = shared.available && !autoScaleEach

  const colorScaleOverride = useMemo(() => (
    useShared ? { vmin: shared.vmin, vmax: shared.vmax, colormap: shared.colormap } : null
  ), [useShared, shared.vmin, shared.vmax, shared.colormap])

  const units = filled[0]?.units || filled[0]?.provenance?.units || ''

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', flex: 1, minHeight: 0 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap', minHeight: '22px' }}>
        {shared.available && (
          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12.5px', color: 'var(--text-secondary)', cursor: 'pointer', userSelect: 'none' }}>
            <input
              type="checkbox"
              checked={autoScaleEach}
              onChange={e => onToggleAutoScale(e.target.checked)}
            />
            Auto-scale each panel
          </label>
        )}
        {!shared.available && shared.reason && (
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', fontStyle: 'italic' }}>{shared.reason}</div>
        )}
        {useShared && (
          <SharedLegend vmin={shared.vmin} vmax={shared.vmax} colormap={shared.colormap} units={units} />
        )}
      </div>

      <SlotGrid
        compareCount={compareCount}
        compareSelection={compareSelection}
        height={height}
        hint="Click a map in chat to add"
        renderChart={(chart) => (
          <MapLibreHeatmapPanel
            payload={chart}
            height={height}
            accessToken={accessToken}
            colorScaleOverride={colorScaleOverride}
            hideLegend={useShared}
          />
        )}
      />
    </div>
  )
}

// timeseries kind (T29): one overlaid Plotly figure when every filled slot
// shares units and its time range overlaps every other slot's; otherwise the
// same small-multiple grid as the heatmap path, mounting independent
// TimeSeriesPanel instances instead.
function TimeSeriesCompareBody({ compareCount, compareSelection, height }) {
  const filled = useMemo(() => filledCharts(compareSelection), [compareSelection])
  const { compatible, reason } = useMemo(() => timeseriesOverlayCompatible(filled), [filled])
  const series = useMemo(() => (compatible ? toOverlaySeries(filled) : []), [compatible, filled])

  if (compatible) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', flex: 1, minHeight: 0 }}>
        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
          {filled.length} series overlaid — matching units, overlapping time ranges.
        </div>
        <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
          <TimeSeriesOverlayPanel series={series} height={height + 20} />
        </div>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', flex: 1, minHeight: 0 }}>
      {filled.length >= 2 && reason && (
        <div style={{ fontSize: '12px', color: 'var(--text-muted)', fontStyle: 'italic' }}>{reason}</div>
      )}
      <SlotGrid
        compareCount={compareCount}
        compareSelection={compareSelection}
        height={height}
        hint="Click a chart in chat to add"
        renderChart={(chart) => <TimeSeriesPanel payload={chart} />}
      />
    </div>
  )
}

export default function CompareGrid({ compareCount, compareSelection, accessToken, autoScaleEach, onToggleAutoScale, height = 420 }) {
  const kind = activeCompareKind(compareSelection)

  if (kind === 'timeseries') {
    return (
      <TimeSeriesCompareBody
        compareCount={compareCount}
        compareSelection={compareSelection}
        height={height}
      />
    )
  }

  return (
    <HeatmapCompareBody
      compareCount={compareCount}
      compareSelection={compareSelection}
      accessToken={accessToken}
      autoScaleEach={autoScaleEach}
      onToggleAutoScale={onToggleAutoScale}
      height={height}
    />
  )
}
