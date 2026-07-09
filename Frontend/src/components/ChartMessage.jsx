/**
 * ChartMessage.jsx
 * ----------------
 * Dispatches a chart payload emitted by the backend to its renderer:
 * MapLibreHeatmapPanel/HeatmapMultiPanel for geo charts (T23), Plotly for
 * time series. Also renders the shared toolbar (query/CSV/PNG export) and
 * provenance block around whichever panel is chosen.
 */
import { useState, useEffect, useRef } from 'react'
import Plotly from 'plotly.js-dist-min'
import _createPlotlyComponent from 'react-plotly.js/factory'
import { flattenPayload } from '../utils/flattenPayload.js'
import MapLibreHeatmapPanel from './MapLibreHeatmapPanel.jsx'
import HeatmapMultiPanel from './HeatmapMultiPanel.jsx'

const createPlotlyComponent =
  typeof _createPlotlyComponent === 'function'
    ? _createPlotlyComponent
    : _createPlotlyComponent.default

const Plot = createPlotlyComponent(Plotly)

// ── Shared config ─────────────────────────────────────────────────────────────
const BASE_CONFIG = {
  displayModeBar: true,
  modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
  displaylogo:    false,
  responsive:     true,
  toImageButtonOptions: { format: 'png', scale: 2 },
}

function sanitizeFilename(value, fallback = 'chart') {
  return String(value || fallback)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80) || fallback
}

function csvEscape(value) {
  if (value == null) return ''
  const text = String(value)
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
}

function rowsToCsv(rows) {
  if (!rows.length) return ''
  const headers = Object.keys(rows[0])
  return [
    headers.map(csvEscape).join(','),
    ...rows.map(row => headers.map(header => csvEscape(row[header])).join(',')),
  ].join('\n')
}

function heatmapRows(payload, panelName = '') {
  const { variable, units } = payload
  const { lat, lon, val } = flattenPayload(payload)
  return val.map((value, i) => ({
    ...(panelName ? { panel: panelName } : {}),
    variable,
    latitude: lat[i],
    longitude: lon[i],
    value,
    units,
  }))
}

function chartRows(chart) {
  if (chart.type === 'heatmap') return heatmapRows(chart)
  if (chart.type === 'heatmap_multi') {
    return (chart.panels || []).flatMap(panel => heatmapRows(panel, panel.title || panel.provenance?.region_name || 'panel'))
  }
  if (chart.type === 'timeseries') {
    return (chart.times || []).map((time, i) => ({
      variable: chart.variable,
      time,
      stat: chart.stat,
      value: chart.values?.[i],
      units: chart.units,
    }))
  }
  return []
}

function downloadText(filename, content, type) {
  const blob = new Blob([content], { type })
  downloadBlob(filename, blob)
}

function downloadBlob(filename, blob) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}

function filenameFromDisposition(disposition, fallback) {
  const match = /filename="?([^";]+)"?/i.exec(disposition || '')
  return match?.[1]?.trim() || fallback
}

async function downloadFromUrl(url, fallbackFilename, accessToken) {
  const response = await fetch(url, {
    headers: accessToken ? { Authorization: `Bearer ${accessToken}` } : {},
  })
  if (!response.ok) {
    let detail = ''
    try {
      const body = await response.json()
      detail = body?.detail || ''
    } catch {
      detail = await response.text().catch(() => '')
    }
    throw new Error(detail || `Export failed with status ${response.status}`)
  }
  const blob = await response.blob()
  const filename = filenameFromDisposition(response.headers.get('content-disposition'), fallbackFilename)
  downloadBlob(filename, blob)
}

function compactDate(value) {
  if (!value) return ''
  return String(value).replace('T00:00:00', '').replace('T23:59:59', '').replace(/Z$/, '')
}

function formatBBox(bbox) {
  if (!Array.isArray(bbox)) return bbox || ''
  return bbox.map(value => Number.isFinite(value) ? value.toFixed(4) : value).join(', ')
}

function GranuleList({ meta, provenance }) {
  const [open, setOpen] = useState(false)
  const dates = meta?.granule_dates || provenance?.granule_dates || []
  const nGranules = meta?.n_granules || provenance?.n_granules
  const cadence = meta?.cadence || provenance?.cadence || ''
  if (!nGranules && !dates.length) return null

  const label = `${nGranules || dates.length} ${cadence ? `${cadence} ` : ''}granule${(nGranules || dates.length) === 1 ? '' : 's'}`
  return (
    <div style={{ minWidth: 0 }}>
      <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
        Granules
      </div>
      <button
        type="button"
        onClick={() => setOpen(value => !value)}
        style={{
          border: 0,
          background: 'transparent',
          padding: 0,
          color: 'var(--text-secondary)',
          fontSize: '11px',
          fontFamily: 'var(--font)',
          cursor: dates.length ? 'pointer' : 'default',
          textAlign: 'left',
        }}
        disabled={!dates.length}
      >
        {label}{dates.length ? (open ? ' ^' : ' v') : ''}
      </button>
      {open && dates.length > 0 && (
        <div style={{
          marginTop: '6px',
          maxHeight: '120px',
          overflow: 'auto',
          fontSize: '11px',
          lineHeight: 1.45,
          color: 'var(--text-secondary)',
        }}>
          {dates.map((date, index) => (
            <div key={`${date}-${index}`}>{date}</div>
          ))}
        </div>
      )}
    </div>
  )
}

export function ProvenanceBlock({ provenance, aggregationMeta }) {
  if (!provenance || typeof provenance !== 'object') return null
  const items = [
    ['Dataset', [provenance.dataset, provenance.variable].filter(Boolean).join(' / ')],
    ['Date Range', [compactDate(provenance.start_date), compactDate(provenance.end_date)].filter(Boolean).join(' to ')],
    ['Region', provenance.region_name || formatBBox(provenance.bbox)],
    ['Aggregation', aggregationMeta?.aggregation_label || provenance.aggregation],
    ['Source', provenance.source || provenance.endpoint],
  ].filter(([, value]) => value)

  const hasGranules = aggregationMeta?.n_granules || provenance.n_granules || aggregationMeta?.granule_dates?.length || provenance.granule_dates?.length
  if (!items.length && !hasGranules) return null

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
      gap: '8px',
      padding: '10px',
      borderTop: '1px solid var(--border)',
      background: 'var(--bg-secondary)',
    }}>
      {items.map(([label, value]) => (
        <div key={label} style={{ minWidth: 0 }}>
          <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            {label}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-secondary)', overflowWrap: 'anywhere', lineHeight: 1.45 }}>
            {value}
          </div>
        </div>
      ))}
      <GranuleList meta={aggregationMeta} provenance={provenance} />
    </div>
  )
}

export function ChartToolbar({ chart, plotRootRef, accessToken }) {
  const [copyState, setCopyState] = useState('')
  const [exportState, setExportState] = useState({ status: '', message: '' })
  const fileBase = sanitizeFilename(chart.title || chart.metadata?.name || chart.type)
  const query = chart.query || chart.provenance || {}

  const handleCopyQuery = async () => {
    const text = JSON.stringify(query, null, 2)
    try {
      await navigator.clipboard.writeText(text)
      setCopyState('Copied')
    } catch {
      downloadText(`${fileBase}-query.json`, text, 'application/json;charset=utf-8')
      setCopyState('Saved')
    }
    window.setTimeout(() => setCopyState(''), 1600)
  }

  const handleCsv = async () => {
    setExportState({ status: 'preparing', message: 'Preparing export' })
    if (chart.chart_id && chart.export) {
      try {
        setExportState({ status: 'progress', message: 'Export in progress' })
        await downloadFromUrl(`/api/chart/${chart.chart_id}/export.csv`, `${fileBase}.csv`, accessToken)
        setExportState({ status: 'complete', message: 'Export complete' })
        window.setTimeout(() => setExportState({ status: '', message: '' }), 2200)
      } catch (error) {
        setExportState({ status: 'failed', message: error.message || 'Export failed' })
      }
      return
    }

    const rows = chartRows(chart)
    if (!rows.length) {
      setExportState({ status: 'failed', message: 'No chart rows available to export' })
      return
    }
    downloadText(`${fileBase}.csv`, rowsToCsv(rows), 'text/csv;charset=utf-8')
    setExportState({ status: 'complete', message: 'Export complete' })
    window.setTimeout(() => setExportState({ status: '', message: '' }), 2200)
  }

  const handlePng = async () => {
    setExportState({ status: 'preparing', message: 'Preparing export' })
    if (chart.chart_id && chart.export) {
      try {
        setExportState({ status: 'progress', message: 'Export in progress' })
        await downloadFromUrl(`/api/chart/${chart.chart_id}/export.png`, `${fileBase}.png`, accessToken)
        setExportState({ status: 'complete', message: 'Export complete' })
        window.setTimeout(() => setExportState({ status: '', message: '' }), 2200)
      } catch (error) {
        setExportState({ status: 'failed', message: error.message || 'Export failed' })
      }
      return
    }

    const plotDivs = Array.from(plotRootRef.current?.querySelectorAll?.('.js-plotly-plot') || [])
    if (!plotDivs.length) {
      setExportState({ status: 'failed', message: 'No chart image available to export' })
      return
    }
    plotDivs.forEach((plotDiv, index) => {
      const width = Math.max(plotDiv.clientWidth || 900, 900)
      const height = Math.max(plotDiv.clientHeight || 500, 500)
      Plotly.downloadImage(plotDiv, {
        format: 'png',
        filename: plotDivs.length > 1 ? `${fileBase}-${index + 1}` : fileBase,
        width,
        height,
        scale: 2,
      })
    })
    setExportState({ status: 'complete', message: 'Export complete' })
    window.setTimeout(() => setExportState({ status: '', message: '' }), 2200)
  }

  const exportBusy = exportState.status === 'preparing' || exportState.status === 'progress'

  const buttonStyle = {
    border: '1px solid var(--border)',
    background: 'var(--bg-card)',
    color: 'var(--text-secondary)',
    borderRadius: '7px',
    padding: '5px 9px',
    fontSize: '11px',
    fontFamily: 'var(--font)',
    cursor: 'pointer',
  }

  return (
    <div style={{ padding: '2px 2px 8px' }}>
      <div style={{
        display: 'flex',
        gap: '6px',
        flexWrap: 'wrap',
        justifyContent: 'flex-end',
      }}>
        <button type="button" onClick={handleCopyQuery} style={buttonStyle}>
          {copyState || 'Copy Query JSON'}
        </button>
        <button
          type="button"
          onClick={handleCsv}
          style={{ ...buttonStyle, opacity: exportBusy ? 0.65 : 1, cursor: exportBusy ? 'wait' : 'pointer' }}
          disabled={exportBusy}
        >
          Export CSV
        </button>
        <button
          type="button"
          onClick={handlePng}
          style={{ ...buttonStyle, opacity: exportBusy ? 0.65 : 1, cursor: exportBusy ? 'wait' : 'pointer' }}
          disabled={exportBusy}
        >
          Export PNG
        </button>
      </div>
      {exportState.message && (
        <div style={{
          marginTop: '6px',
          textAlign: 'right',
          color: exportState.status === 'failed' ? 'var(--error, #b42318)' : 'var(--text-muted)',
          fontSize: '11px',
          lineHeight: 1.4,
        }}>
          {exportState.message}
        </div>
      )}
    </div>
  )
}

// ── Time-series ───────────────────────────────────────────────────────────────
export function TimeSeriesPanel({ payload }) {
  const { title, units, stat, times, values } = payload

  const [revision, setRevision] = useState(0)
  const mounted = useRef(false)
  useEffect(() => {
    if (!mounted.current) { mounted.current = true; requestAnimationFrame(() => setRevision(r => r + 1)) }
  }, [])

  const data = [{
    type: 'scatter', mode: 'lines+markers',
    x: times, y: values,
    line:          { color: '#1D9E75', width: 2 },
    marker:        { color: '#1D9E75', size: 5 },
    fill:          'tozeroy', fillcolor: 'rgba(29,158,117,0.08)',
    hovertemplate: `%{x|%Y-%m-%d %H:%M}<br>${stat}: %{y:.3e} ${units}<extra></extra>`,
  }]

  const layout = {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font:   { family: "'Manrope', system-ui, sans-serif", size: 12, color: '#333333' },
    margin: { t: 40, r: 16, b: 40, l: 16 },
    title:  { text: title, font: { size: 13, weight: 500 }, x: 0.5, xanchor: 'center' },
    height: 300,
    xaxis:  { title: 'Time', showgrid: true, gridcolor: 'rgba(0,0,0,0.06)', zeroline: false, tickfont: { size: 10 } },
    yaxis:  { title: `${stat} (${units})`, showgrid: true, gridcolor: 'rgba(0,0,0,0.06)', zeroline: false, tickfont: { size: 10 }, exponentformat: 'e' },
    datarevision: revision,
  }

  return (
    <Plot data={data} layout={layout} config={BASE_CONFIG} revision={revision}
      style={{ width: '100%' }} useResizeHandler />
  )
}

// ── Public component ──────────────────────────────────────────────────────────
export default function ChartMessage({ chart, accessToken }) {
  const plotRootRef = useRef(null)
  if (!chart || typeof chart !== 'object' || !chart.type) return null

  const inner = (() => {
    switch (chart.type) {
      case 'heatmap':       return <MapLibreHeatmapPanel payload={chart} accessToken={accessToken} />
      case 'heatmap_multi': return <HeatmapMultiPanel payload={chart} accessToken={accessToken} />
      case 'timeseries':    return <TimeSeriesPanel payload={chart} />
      default:
        return <div style={{ fontSize: '12px', color: 'var(--text-muted)', padding: '8px' }}>Unknown chart type: {chart.type}</div>
    }
  })()

  return (
    <div style={{
      margin: '8px 0', background: 'var(--bg-card)',
      border: '1px solid var(--border)', borderRadius: '10px',
      overflow: 'hidden', padding: '8px',
    }}>
      <ChartToolbar chart={chart} plotRootRef={plotRootRef} accessToken={accessToken} />
      <div ref={plotRootRef}>
        {inner}
      </div>
      <ProvenanceBlock provenance={chart.provenance} aggregationMeta={chart.aggregation_meta} />
    </div>
  )
}
