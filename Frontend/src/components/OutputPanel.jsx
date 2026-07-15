import { useMemo, useState, useEffect, useRef } from 'react'
import { TimeSeriesPanel, ChartToolbar } from './ChartMessage'
import MapLibreHeatmapPanel from './MapLibreHeatmapPanel.jsx'
import HeatmapMultiPanel from './HeatmapMultiPanel.jsx'
import CompareGrid from './CompareGrid.jsx'
import ArtifactMessage, { TableArtifactMessage } from './ArtifactMessage'
import { computeChartStats, computeHistogram } from '../utils/chartStats'
import { resolveMasking } from '../utils/maskingProvenance'
import { filledCharts } from '../utils/compareMode'
import { focusChartPayload } from '../utils/compareSlotOverview'
import { shouldShowCollapseHint } from '../utils/compareLayoutHint'
import {
  NOT_AVAILABLE, fmt, dateRangeLabel, granuleSummary, maskingStatusColor,
  citationString, datasetLandingUrl, spatialFields, temporalFields,
  qaMethodologyFields, variableDefinitionFields, reproducibilityFields,
  reproducibilityQuery, rawMetadataJson,
} from '../utils/metadataDisplay'
import {
  tableOverviewFields, tableDetailsFields,
  groundValidationOverviewFields, groundValidationDetailsFields,
  rawArtifactMetadataJson,
} from '../utils/artifactMetadataDisplay'

function compactDate(value) {
  if (!value) return ''
  return String(value).replace('T00:00:00', '').replace('T23:59:59', '').replace(/Z$/, '')
}

function chartTitle(chart) {
  return chart.title || chart.provenance?.variable || chart.variable || 'Output'
}

function chartMetaChips(chart) {
  const p = chart.provenance || {}
  const chips = []
  if (p.dataset) chips.push(p.dataset)
  const units = chart.units || p.units
  if (units) chips.push(units)
  const dateRange = [compactDate(p.start_date), compactDate(p.end_date)].filter(Boolean).join(' → ')
  if (dateRange) chips.push(dateRange)
  if (p.region_name) chips.push(p.region_name)
  return chips
}

function artifactMetaChips(artifact) {
  const meta = artifact.metadata || {}
  const chips = []
  if (meta.variable) chips.push(meta.variable)
  if (meta.units) chips.push(meta.units)
  if (Array.isArray(meta.bbox)) chips.push(meta.bbox.map(v => Number(v).toFixed(2)).join(', '))
  return chips
}

function TabButton({ label, active, onClick }) {
  return (
    <div
      onClick={onClick}
      style={{
        fontSize: '13px', fontWeight: 700,
        color: active ? 'var(--text-primary)' : 'var(--text-muted)',
        padding: '8px 14px 12px', cursor: 'pointer',
        borderBottom: `2px solid ${active ? 'var(--teal)' : 'transparent'}`,
      }}
    >
      {label}
    </div>
  )
}

function MetaChip({ children }) {
  return (
    <div style={{
      fontSize: '11.5px', fontWeight: 600, color: 'var(--text-secondary)',
      background: 'var(--bg-secondary)', border: '1px solid var(--border)',
      borderRadius: '7px', padding: '5px 10px',
    }}>
      {children}
    </div>
  )
}

function StatCard({ label, value }) {
  return (
    <div style={{
      background: 'var(--bg-card)', border: '1px solid var(--border)',
      borderRadius: '10px', padding: '11px 13px',
    }}>
      <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '5px' }}>{label}</div>
      <div style={{ fontSize: '15px', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{value}</div>
    </div>
  )
}

// Read-only QA-flag masking provenance. Lets the user see whether QA masking
// actually ran on the plotted data (and by which tier) rather than inferring
// it from the valid-pixel count above. Renders nothing when the payload
// carries no masking record.
function MaskingDisclosure({ chart }) {
  const masking = useMemo(() => resolveMasking(chart), [chart])
  if (!masking) return null
  return (
    <div style={{
      background: 'var(--bg-secondary)', border: '1px solid var(--border)',
      borderRadius: '10px', padding: '11px 13px',
    }}>
      <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '5px' }}>
        QA-flag masking
      </div>
      <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>
        {masking.qaStatus}
      </div>
      {masking.qaSource && (
        <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)', marginTop: '4px' }}>
          Source: {masking.qaSource}
        </div>
      )}
      {masking.qaNote && (
        <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)', marginTop: '4px', lineHeight: 1.45 }}>
          {masking.qaNote}
        </div>
      )}
    </div>
  )
}

function StatisticsTab({ chart }) {
  const stats = useMemo(() => computeChartStats(chart), [chart])
  if (!stats) {
    return <div style={{ padding: '24px', color: 'var(--text-muted)', fontSize: '13px' }}>No numeric values available for this output.</div>
  }
  const fmt = (n) => Number.isFinite(n) ? n.toExponential(3) : '—'
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '10px' }}>
        <StatCard label="Mean" value={`${fmt(stats.mean)} ${stats.units || ''}`} />
        <StatCard label="Max" value={`${fmt(stats.max)} ${stats.units || ''}`} />
        <StatCard label="Min" value={`${fmt(stats.min)} ${stats.units || ''}`} />
        <StatCard label="Valid values" value={`${stats.validPct.toFixed(1)}%`} />
        <StatCard label="Sample count" value={stats.count.toLocaleString()} />
      </div>
      <MaskingDisclosure chart={chart} />
    </div>
  )
}

function HistogramTab({ chart }) {
  const histogram = useMemo(() => computeHistogram(chart), [chart])
  if (!histogram) {
    return <div style={{ padding: '24px', color: 'var(--text-muted)', fontSize: '13px' }}>No numeric values available for this output.</div>
  }
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
      <div style={{ flex: 1, display: 'flex', alignItems: 'flex-end', gap: '6px', borderBottom: '1px solid var(--border)', paddingBottom: '2px', minHeight: '160px' }}>
        {histogram.buckets.map((bucket, i) => (
          <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-end', height: '100%' }}>
            <div style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', marginBottom: '4px' }}>{bucket.count}</div>
            <div style={{
              width: '100%', height: `${Math.max(bucket.pct, 2)}%`,
              background: 'linear-gradient(180deg, var(--teal), var(--teal-hover))',
              borderRadius: '4px 4px 0 0',
            }} />
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: '8px' }}>
        <span>{histogram.min.toExponential(2)}</span>
        <span>{histogram.max.toExponential(2)}</span>
      </div>
    </div>
  )
}

// ── Metadata tab (T32): Overview (fully expanded) / Details (accordion) ────
// Field derivation (which fact goes in which section, "Not available"
// fallbacks, citation/query string-building) lives in utils/metadataDisplay
// -- pure and unit-tested there; this file is just the JSX shell around it.

function MetaField({ label, value }) {
  return (
    <div style={{ minWidth: 0 }}>
      <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
        {label}
      </div>
      <div style={{ fontSize: '12px', color: 'var(--text-secondary)', overflowWrap: 'anywhere', lineHeight: 1.45 }}>
        {fmt(value)}
      </div>
    </div>
  )
}

const smallButtonStyle = {
  border: '1px solid var(--border)', background: 'var(--bg-card)',
  color: 'var(--text-secondary)', borderRadius: '7px', padding: '4px 9px',
  fontSize: '11px', fontFamily: 'var(--font)', cursor: 'pointer',
}

async function copyToClipboard(text, setState) {
  try {
    await navigator.clipboard.writeText(text)
    setState('Copied')
  } catch {
    setState('Copy failed')
  }
  window.setTimeout(() => setState(''), 1600)
}

export function MetadataOverview({ chart, onViewStatistics, onViewFullMetadata, note }) {
  const provenance = chart.provenance || {}
  const masking = useMemo(() => resolveMasking(chart), [chart])
  const [copyState, setCopyState] = useState('')
  const landingUrl = datasetLandingUrl(provenance.collection_id)
  const dateRange = dateRangeLabel(provenance)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
      <div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '10px', marginBottom: '8px' }}>
          <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)' }}>
            This view
          </div>
          {onViewFullMetadata && (
            <button type="button" onClick={onViewFullMetadata} style={{
              border: 0, background: 'transparent', color: 'var(--teal-text)',
              fontSize: '11px', fontWeight: 700, cursor: 'pointer', padding: 0, flexShrink: 0,
            }}>
              View full metadata →
            </button>
          )}
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '10px' }}>
          <MetaField label="Dataset" value={provenance.dataset} />
          <MetaField label="Variable" value={provenance.variable} />
          <MetaField label="Date Range" value={dateRange} />
          <MetaField label="Region" value={provenance.region_name} />
          <MetaField label="Aggregation" value={chart.aggregation_meta?.aggregation_label || provenance.aggregation} />
          <MetaField label="Granules" value={granuleSummary(chart)} />
        </div>
      </div>

      <div style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: '10px', padding: '11px 13px' }}>
        <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '5px' }}>
          Data quality
        </div>
        {masking ? (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: '7px' }}>
              <span aria-hidden style={{
                width: '8px', height: '8px', borderRadius: '50%', flexShrink: 0,
                background: maskingStatusColor(masking.qaStatus),
              }} />
              <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>
                {masking.qaStatus}
              </span>
            </div>
            {masking.qaNote && (
              <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)', marginTop: '4px', lineHeight: 1.45 }}>
                {masking.qaNote}
              </div>
            )}
            {onViewStatistics && (
              <button type="button" onClick={onViewStatistics} style={{
                marginTop: '6px', border: 0, background: 'transparent',
                color: 'var(--teal-text)', fontSize: '11px', fontWeight: 700,
                cursor: 'pointer', padding: 0,
              }}>
                See Statistics tab for details →
              </button>
            )}
          </>
        ) : (
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{NOT_AVAILABLE}</div>
        )}
      </div>

      <div style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: '10px', padding: '11px 13px' }}>
        <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '5px', display: 'flex', flexWrap: 'wrap', alignItems: 'baseline', gap: '8px' }}>
          <span>Source dataset</span>
          {note && <span style={{ fontWeight: 400, fontStyle: 'italic', color: 'var(--text-muted)' }}>{note}</span>}
        </div>
        <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>{fmt(provenance.dataset)}</div>
        <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)', marginTop: '4px' }}>{fmt(provenance.dataset_description)}</div>
        <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)', marginTop: '4px' }}>
          Version {fmt(provenance.dataset_version)} · {fmt(provenance.source)}
        </div>
        <div style={{ display: 'flex', gap: '10px', marginTop: '8px', alignItems: 'center' }}>
          {landingUrl && (
            <a href={landingUrl} target="_blank" rel="noreferrer" style={{ fontSize: '11px', color: 'var(--teal-text)', fontWeight: 700 }}>
              View source dataset ↗
            </a>
          )}
          <button type="button" onClick={() => copyToClipboard(citationString(provenance), setCopyState)} style={smallButtonStyle}>
            {copyState || 'Copy citation'}
          </button>
        </div>
      </div>
    </div>
  )
}

function DetailsSection({ title, children }) {
  return (
    <details style={{ border: '1px solid var(--border)', borderRadius: '10px', background: 'var(--bg-card)' }}>
      <summary style={{ cursor: 'pointer', padding: '10px 13px', fontSize: '12.5px', fontWeight: 700, color: 'var(--text-primary)' }}>
        {title}
      </summary>
      <div style={{ padding: '0 13px 13px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {children}
      </div>
    </details>
  )
}

function SpatialSection({ chart }) {
  const { regionName, bbox } = spatialFields(chart)
  return (
    <>
      <MetaField label="Region" value={regionName} />
      <MetaField label="Bounding box" value={bbox} />
    </>
  )
}

function TemporalSection({ chart }) {
  const { dateRange, cadence, dates } = temporalFields(chart)
  return (
    <>
      <MetaField label="Date Range" value={dateRange} />
      <MetaField label="Cadence" value={cadence} />
      <div>
        <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '4px' }}>
          Granule dates
        </div>
        {dates.length ? (
          <div style={{ maxHeight: '140px', overflow: 'auto', fontSize: '11px', lineHeight: 1.5, color: 'var(--text-secondary)' }}>
            {dates.map((date, i) => <div key={`${date}-${i}`}>{date}</div>)}
          </div>
        ) : (
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{NOT_AVAILABLE}</div>
        )}
      </div>
    </>
  )
}

function ProvenanceSection({ chart }) {
  const { qualityFlagVar, qaGoodValues, qaBadValues, fillValueSource, validRangeSource } = qaMethodologyFields(chart)
  return (
    <>
      <MetaField label="Quality flag variable" value={qualityFlagVar} />
      <MetaField label="QA good values" value={qaGoodValues ? JSON.stringify(qaGoodValues) : null} />
      <MetaField label="QA bad values" value={qaBadValues ? JSON.stringify(qaBadValues) : null} />
      <MetaField label="Fill-value source" value={fillValueSource} />
      <MetaField label="Valid-range source" value={validRangeSource} />
    </>
  )
}

function VariableDefinitionSection({ chart }) {
  const { longName, units, advisoryNotes, validRange, maskNote } = variableDefinitionFields(chart)
  return (
    <>
      <MetaField label="Long name" value={longName} />
      <MetaField label="Units" value={units} />
      <MetaField label="Advisory notes" value={advisoryNotes} />
      <MetaField label="Valid range" value={validRange} />
      <MetaField label="Mask note" value={maskNote} />
    </>
  )
}

function ReproducibilitySection({ chart }) {
  const [copyState, setCopyState] = useState('')
  const { dataset, startDate, endDate, bbox, aggregation, sourceHandles } = reproducibilityFields(chart)
  const handleCopy = () => {
    copyToClipboard(JSON.stringify(reproducibilityQuery(chart), null, 2), setCopyState)
  }
  return (
    <>
      <MetaField label="Dataset" value={dataset} />
      <MetaField label="Start date" value={startDate} />
      <MetaField label="End date" value={endDate} />
      <MetaField label="Bounding box" value={bbox} />
      <MetaField label="Aggregation" value={aggregation} />
      <MetaField label="Source handles" value={sourceHandles} />
      <button type="button" onClick={handleCopy} style={{ ...smallButtonStyle, alignSelf: 'flex-start' }}>
        {copyState || 'Copy query'}
      </button>
    </>
  )
}

function RawJsonToggle({ chart }) {
  return (
    <DetailsSection title="Raw JSON">
      <pre style={{
        fontSize: '10.5px', overflow: 'auto', maxHeight: '260px',
        background: 'var(--bg-secondary)', padding: '8px', borderRadius: '6px', margin: 0,
      }}>
        {JSON.stringify(rawMetadataJson(chart), null, 2)}
      </pre>
    </DetailsSection>
  )
}

function MetadataDetails({ chart }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      <DetailsSection title="Spatial"><SpatialSection chart={chart} /></DetailsSection>
      <DetailsSection title="Temporal"><TemporalSection chart={chart} /></DetailsSection>
      <DetailsSection title="Provenance"><ProvenanceSection chart={chart} /></DetailsSection>
      <DetailsSection title="Variable Definition"><VariableDefinitionSection chart={chart} /></DetailsSection>
      <DetailsSection title="Reproducibility"><ReproducibilitySection chart={chart} /></DetailsSection>
      <RawJsonToggle chart={chart} />
    </div>
  )
}

const metaViewButtonStyle = (active) => ({
  fontSize: '12px', fontWeight: 700,
  color: active ? 'var(--text-primary)' : 'var(--text-muted)',
  background: active ? 'var(--bg-secondary)' : 'transparent',
  border: '1px solid var(--border)', borderRadius: '7px',
  padding: '5px 11px', cursor: 'pointer',
})

// ── Artifact-shaped Metadata tab (T33): table and ground-validation
// timeseries artifacts (map/comparison/chart-backed timeseries are
// deliberately excluded -- see utils/artifactReachability.js). A sibling to
// MetadataOverview/MetadataDetails above, reading ArtifactReference.metadata
// shape via utils/artifactMetadataDisplay.js instead of chart provenance.

// Table artifacts' ArtifactReference is a lightweight stub (id/title/
// row_count/metadata) -- the full column list only exists on the paginated
// `/api/artifacts/{id}` response, so Overview/Details fetch a minimal
// (limit=1) page themselves to learn it, same endpoint TableArtifactMessage
// already pages through for the grid.
function useTableColumnsPreview(artifact, accessToken) {
  const [state, setState] = useState({ page: null, status: 'loading' })
  useEffect(() => {
    if (!artifact?.id || artifact.type !== 'table') return undefined
    let cancelled = false
    fetch(`/api/artifacts/${artifact.id}?offset=0&limit=1`, {
      headers: accessToken ? { Authorization: `Bearer ${accessToken}` } : {},
    })
      .then(response => (response.ok ? response.json() : Promise.reject(new Error(`HTTP ${response.status}`))))
      .then(page => { if (!cancelled) setState({ page, status: 'ready' }) })
      .catch(() => { if (!cancelled) setState({ page: null, status: 'failed' }) })
    return () => { cancelled = true }
  }, [artifact?.id, artifact?.type, accessToken])
  return state
}

function TableMetadataOverview({ artifact, accessToken }) {
  const { page, status } = useTableColumnsPreview(artifact, accessToken)
  const fields = tableOverviewFields(artifact, page)
  const loadingOr = (value) => (status === 'loading' ? '…' : fmt(value))
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
      <div>
        <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '8px' }}>
          This table
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '10px' }}>
          <StatCard label="Rows" value={loadingOr(fields.rowCount?.toLocaleString?.() ?? fields.rowCount)} />
          <StatCard label="Columns" value={loadingOr(fields.columnCount)} />
        </div>
      </div>
      <div style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: '10px', padding: '11px 13px' }}>
        <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '5px' }}>
          Source dataset
        </div>
        <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>{fmt(fields.sourceDataset)}</div>
      </div>
    </div>
  )
}

function TableCsvExport({ artifact, accessToken }) {
  const [state, setState] = useState('')
  async function downloadCsv() {
    setState('downloading')
    try {
      const response = await fetch(`/api/artifacts/${artifact.id}/csv`, {
        headers: accessToken ? { Authorization: `Bearer ${accessToken}` } : {},
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const blob = await response.blob()
      const disposition = response.headers.get('content-disposition')
      const match = /filename="?([^";]+)"?/i.exec(disposition || '')
      const filename = match?.[1]?.trim() || `${artifact.title || artifact.id}.csv`
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(url)
      setState('')
    } catch (err) {
      setState(err.message || 'Export failed')
    }
  }
  return (
    <button type="button" onClick={downloadCsv} disabled={state === 'downloading'} style={smallButtonStyle}>
      {state === 'downloading' ? 'Exporting…' : (state || 'Download CSV')}
    </button>
  )
}

function ArtifactRawJsonToggle({ artifact }) {
  return (
    <DetailsSection title="Raw JSON">
      <pre style={{
        fontSize: '10.5px', overflow: 'auto', maxHeight: '260px',
        background: 'var(--bg-secondary)', padding: '8px', borderRadius: '6px', margin: 0,
      }}>
        {JSON.stringify(rawArtifactMetadataJson(artifact), null, 2)}
      </pre>
    </DetailsSection>
  )
}

function TableMetadataDetails({ artifact, accessToken }) {
  const { page, status } = useTableColumnsPreview(artifact, accessToken)
  const { columns } = tableDetailsFields(artifact, page)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      <DetailsSection title="Columns">
        {status === 'loading' ? (
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Loading…</div>
        ) : columns.length ? (
          <div style={{ maxHeight: '160px', overflow: 'auto', fontSize: '11px', lineHeight: 1.5, color: 'var(--text-secondary)' }}>
            {columns.map(column => <div key={column}>{column}</div>)}
          </div>
        ) : (
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{NOT_AVAILABLE}</div>
        )}
      </DetailsSection>
      <DetailsSection title="Export">
        <TableCsvExport artifact={artifact} accessToken={accessToken} />
      </DetailsSection>
      <ArtifactRawJsonToggle artifact={artifact} />
    </div>
  )
}

function GroundValidationOverview({ artifact }) {
  const fields = groundValidationOverviewFields(artifact)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
      <div>
        <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '8px' }}>
          This comparison
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '10px' }}>
          <MetaField label="Satellite variable" value={fields.satelliteVariable} />
          <MetaField label="Ground station(s)" value={fields.groundStations} />
        </div>
      </div>
      <div style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: '10px', padding: '11px 13px' }}>
        <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '5px' }}>
          Correlation / coverage
        </div>
        <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>{fmt(fields.correlationSummary)}</div>
      </div>
      <div style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: '10px', padding: '11px 13px' }}>
        <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '5px' }}>
          Data quality
        </div>
        <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)' }}>{fmt(fields.qaStatus)}</div>
      </div>
    </div>
  )
}

function GroundValidationDetails({ artifact }) {
  const fields = groundValidationDetailsFields(artifact)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      <DetailsSection title="Series">
        {fields.series.length ? fields.series.map((s, i) => (
          <MetaField key={i} label={s.source_kind} value={s.station_id ? `${s.label} (${s.station_id})` : s.label} />
        )) : <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{NOT_AVAILABLE}</div>}
      </DetailsSection>
      {fields.exceedanceDates?.length ? (
        <DetailsSection title="Exceedance dates">
          <div style={{ maxHeight: '140px', overflow: 'auto', fontSize: '11px', lineHeight: 1.5, color: 'var(--text-secondary)' }}>
            {fields.exceedanceDates.map(date => <div key={date}>{date}</div>)}
          </div>
        </DetailsSection>
      ) : fields.stats ? (
        <DetailsSection title="Correlation methodology">
          <MetaField label="Pearson r" value={fields.stats.r != null ? fields.stats.r.toFixed(3) : null} />
          <MetaField label="N paired days" value={fields.stats.n} />
          <MetaField label="Coverage fraction" value={fields.stats.coverage_fraction != null ? `${(fields.stats.coverage_fraction * 100).toFixed(1)}%` : null} />
        </DetailsSection>
      ) : null}
      <MetaField label="Source handles" value={fields.sourceHandles} />
      <ArtifactRawJsonToggle artifact={artifact} />
    </div>
  )
}

function MetadataTab({ chart, artifact, accessToken, onViewStatistics }) {
  const [view, setView] = useState('overview')
  if (chart) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        <div style={{ display: 'flex', gap: '6px' }}>
          <button type="button" onClick={() => setView('overview')} style={metaViewButtonStyle(view === 'overview')}>Overview</button>
          <button type="button" onClick={() => setView('details')} style={metaViewButtonStyle(view === 'details')}>Details</button>
        </div>
        {view === 'overview'
          ? <MetadataOverview chart={chart} onViewStatistics={onViewStatistics} />
          : <MetadataDetails chart={chart} />}
      </div>
    )
  }
  if (artifact?.type === 'table' || artifact?.type === 'timeseries') {
    const isTable = artifact.type === 'table'
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        <div style={{ display: 'flex', gap: '6px' }}>
          <button type="button" onClick={() => setView('overview')} style={metaViewButtonStyle(view === 'overview')}>Overview</button>
          <button type="button" onClick={() => setView('details')} style={metaViewButtonStyle(view === 'details')}>Details</button>
        </div>
        {isTable
          ? (view === 'overview'
            ? <TableMetadataOverview artifact={artifact} accessToken={accessToken} />
            : <TableMetadataDetails artifact={artifact} accessToken={accessToken} />)
          : (view === 'overview'
            ? <GroundValidationOverview artifact={artifact} />
            : <GroundValidationDetails artifact={artifact} />)}
      </div>
    )
  }
  if (artifact) {
    return <ArtifactMessage artifact={artifact} accessToken={undefined} />
  }
  return null
}

const CHART_TABS = {
  heatmap: ['map', 'statistics', 'histogram', 'metadata'],
  heatmap_multi: ['map', 'metadata'],
  timeseries: ['chart', 'statistics', 'histogram', 'metadata'],
}
// Table artifacts keep their existing grid alongside the new Metadata tab;
// ground-validation timeseries artifacts have no chartable series data of
// their own (T33 -- see PRD), so Metadata is their only tab.
const ARTIFACT_TABS = {
  table: ['table', 'metadata'],
  timeseries: ['metadata'],
}
const TAB_LABELS = { map: 'Map', chart: 'Chart', statistics: 'Statistics', histogram: 'Histogram', metadata: 'Metadata', table: 'Table' }

const PANEL_COUNTS = [2, 3, 4]

const compareButtonStyle = {
  fontSize: '12.5px', fontWeight: 700, color: 'var(--teal-text)',
  background: 'var(--teal-light)', border: '1px solid var(--teal)',
  borderRadius: '7px', padding: '7px 12px', cursor: 'pointer',
}
const countButtonStyle = {
  fontSize: '12.5px', fontWeight: 700, color: 'var(--text-primary)',
  background: 'var(--bg-card)', border: '1px solid var(--border)',
  borderRadius: '7px', padding: '6px 12px', cursor: 'pointer', minWidth: '32px',
}
const exitButtonStyle = {
  fontSize: '12.5px', fontWeight: 700, color: 'var(--text-secondary)',
  background: 'var(--bg-card)', border: '1px solid var(--border)',
  borderRadius: '7px', padding: '7px 12px', cursor: 'pointer',
}
const cancelChooserStyle = {
  fontSize: '14px', fontWeight: 700, color: 'var(--text-muted)',
  background: 'transparent', border: 'none', cursor: 'pointer', padding: '4px 6px',
}
const hintActionStyle = {
  fontSize: '11.5px', fontWeight: 700, color: 'var(--text-secondary)',
  background: 'var(--bg-primary)', border: '1px solid var(--border)',
  borderRadius: '6px', padding: '4px 9px', cursor: 'pointer', whiteSpace: 'nowrap',
}
const hintDismissStyle = {
  fontSize: '14px', fontWeight: 700, color: 'var(--text-muted)',
  background: 'transparent', border: 'none', cursor: 'pointer', padding: '2px 4px',
}

// Nudge (not an auto-collapse -- that jumped the layout around outside the
// user's control, see App.jsx) offering a one-click way to free up width
// when every side panel is still expanded during active compare.
function CollapseHint({ onCollapseSessions, onCollapseRightPanel, onDismiss }) {
  return (
    <div style={{
      margin: '10px 22px 0', padding: '8px 12px', borderRadius: '6px',
      background: 'var(--bg-card)', border: '1px solid var(--border)',
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      gap: '12px', fontSize: '12px', color: 'var(--text-secondary)', flexShrink: 0,
    }}>
      <span>Comparing side by side needs room — collapse a side panel for more space.</span>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0 }}>
        <button type="button" onClick={onCollapseSessions} style={hintActionStyle}>Hide sessions</button>
        <button type="button" onClick={onCollapseRightPanel} style={hintActionStyle}>Hide jobs &amp; discover</button>
        <button type="button" onClick={onDismiss} aria-label="Dismiss hint" style={hintDismissStyle}>×</button>
      </div>
    </div>
  )
}

// Compare control (T28): idle "Compare" button -> inline 2/3/4 chooser ->
// active status + exit. Lives in the output panel header regardless of
// which body is currently rendered (empty state, single output, or grid).
function CompareControl({ compareMode, compareCount, filledCount, onStart, onCancelChoosing, onPickCount, onExit }) {
  if (compareMode === 'choosing-count') {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <span style={{ fontSize: '12px', color: 'var(--text-muted)', marginRight: '2px' }}>Compare how many?</span>
        {PANEL_COUNTS.map(n => (
          <button key={n} type="button" onClick={() => onPickCount(n)} style={countButtonStyle}>{n}</button>
        ))}
        <button type="button" onClick={onCancelChoosing} style={cancelChooserStyle} aria-label="Cancel compare">×</button>
      </div>
    )
  }

  if (compareMode === 'active') {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <span style={{ fontSize: '12.5px', fontWeight: 700, color: 'var(--text-secondary)' }}>
          Comparing {filledCount} of {compareCount}
        </span>
        <button type="button" onClick={onExit} style={exitButtonStyle}>Exit compare</button>
      </div>
    )
  }

  return (
    <button type="button" onClick={onStart} style={compareButtonStyle}>
      Compare
    </button>
  )
}

// Table and ground-validation timeseries artifacts (T33) get an outer tab
// bar of their own -- table keeps its existing grid alongside a new
// Metadata tab; ground-validation timeseries has no chartable series data
// of its own, so Metadata is its only tab (see ARTIFACT_TABS). The caller
// keys this by artifact.id so switching artifacts remounts it fresh
// (activeTab reset) instead of needing an effect to resync local state.
function ArtifactTabsPanel({ artifact, accessToken, compareControlProps }) {
  const tabs = ARTIFACT_TABS[artifact.type] || ['metadata']
  const [activeTab, setActiveTab] = useState(tabs[0])

  return (
    <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', background: 'var(--bg-primary)', overflow: 'hidden' }}>
      <div style={{ padding: '14px 22px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 }}>
        <div style={{ fontSize: '16px', fontWeight: 800, color: 'var(--text-primary)' }}>{artifact.title || 'Output'}</div>
        <CompareControl {...compareControlProps} />
      </div>

      <div style={{ display: 'flex', gap: '4px', padding: '14px 22px 0', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
        {tabs.map(tab => (
          <TabButton key={tab} label={TAB_LABELS[tab]} active={activeTab === tab} onClick={() => setActiveTab(tab)} />
        ))}
      </div>

      <div style={{ flex: 1, minHeight: 0, overflow: 'auto', padding: '18px 22px' }}>
        {activeTab === 'table' && <TableArtifactMessage artifact={artifact} accessToken={accessToken} />}
        {activeTab === 'metadata' && <MetadataTab artifact={artifact} accessToken={accessToken} />}
      </div>
    </div>
  )
}

export default function OutputPanel({
  focusedOutput, accessToken, onFocusOutput,
  compareMode = 'off', compareCount = 2, compareSelection = [], compareSessionId = 0,
  onStartCompare, onCancelChooseCompare, onEnterCompare, onExitCompare,
  sessionsCollapsed = false, chatCollapsed = false, rightPanelCollapsed = false,
  onCollapseSessions, onCollapseRightPanel,
}) {
  const kind = focusedOutput?.kind
  const chart = kind === 'chart' ? focusedOutput.data : null
  const artifact = kind === 'artifact' ? focusedOutput.data : null
  const isMetadataTabbedArtifact = artifact?.type === 'table' || artifact?.type === 'timeseries'

  const availableTabs = chart ? (CHART_TABS[chart.type] || ['metadata']) : []
  const [activeTab, setActiveTab] = useState(availableTabs[0])
  const [autoScaleEach, setAutoScaleEach] = useState(true)
  // Keyed by compareSessionId (bumped once per enterCompare in App.jsx)
  // rather than a boolean, so a dismissal doesn't leak into the next fresh
  // compare session -- purely derived from props, no effect/ref needed.
  const [hintDismissedForSession, setHintDismissedForSession] = useState(null)
  const plotRootRef = useRef(null)

  useEffect(() => {
    setActiveTab(availableTabs[0])
  }, [focusedOutput, availableTabs.join(',')])

  const showCollapseHint = shouldShowCollapseHint({
    compareMode, sessionsCollapsed, chatCollapsed, rightPanelCollapsed,
  }) && hintDismissedForSession !== compareSessionId

  const compareControlProps = {
    compareMode, compareCount, filledCount: filledCharts(compareSelection).length,
    onStart: onStartCompare, onCancelChoosing: onCancelChooseCompare,
    onPickCount: onEnterCompare, onExit: onExitCompare,
  }

  if (compareMode === 'active') {
    return (
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', background: 'var(--bg-primary)', overflow: 'hidden' }}>
        <div style={{ padding: '14px 22px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 }}>
          <div style={{ fontSize: '16px', fontWeight: 800, color: 'var(--text-primary)' }}>Compare</div>
          <CompareControl {...compareControlProps} />
        </div>
        {showCollapseHint && (
          <CollapseHint
            onCollapseSessions={onCollapseSessions}
            onCollapseRightPanel={onCollapseRightPanel}
            onDismiss={() => setHintDismissedForSession(compareSessionId)}
          />
        )}
        <div style={{ flex: 1, minHeight: 0, overflow: 'auto', padding: '18px 22px', display: 'flex' }}>
          <CompareGrid
            compareCount={compareCount}
            compareSelection={compareSelection}
            accessToken={accessToken}
            autoScaleEach={autoScaleEach}
            onToggleAutoScale={setAutoScaleEach}
            onFocusChart={onFocusOutput ? (chart) => {
              onFocusOutput(focusChartPayload(chart))
              onExitCompare?.()
            } : undefined}
          />
        </div>
      </div>
    )
  }

  if (!focusedOutput) {
    return (
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', background: 'var(--bg-primary)' }}>
        <div style={{ padding: '14px 22px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', flexShrink: 0 }}>
          <CompareControl {...compareControlProps} />
        </div>
        <div style={{
          flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center', textAlign: 'center',
          color: 'var(--text-muted)', padding: '0 24px',
        }}>
          <div style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '6px' }}>
            Ask a question to get started
          </div>
          <div style={{ fontSize: '13px', lineHeight: 1.5, maxWidth: '360px' }}>
            Run an analysis in the chat, then click any output card to open it here — map, chart, statistics, and metadata all in one place.
          </div>
        </div>
      </div>
    )
  }

  if (isMetadataTabbedArtifact) {
    return <ArtifactTabsPanel key={artifact.id} artifact={artifact} accessToken={accessToken} compareControlProps={compareControlProps} />
  }

  if (artifact && !chart) {
    return (
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', background: 'var(--bg-primary)', overflow: 'hidden' }}>
        <div style={{ padding: '14px 22px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', flexShrink: 0 }}>
          <CompareControl {...compareControlProps} />
        </div>
        <div style={{ flex: 1, minHeight: 0, overflow: 'auto', padding: '18px 22px' }}>
          <div style={{ fontSize: '16px', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '4px' }}>{artifact.title || 'Output'}</div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', margin: '10px 0 16px' }}>
            {artifactMetaChips(artifact).map((chip, i) => <MetaChip key={i}>{chip}</MetaChip>)}
          </div>
          <ArtifactMessage artifact={artifact} accessToken={accessToken} />
        </div>
      </div>
    )
  }

  const title = chartTitle(chart)
  const metaChips = chartMetaChips(chart)

  return (
    <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', background: 'var(--bg-primary)', overflow: 'hidden' }}>
      <div style={{ padding: '14px 22px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 }}>
        <div style={{ fontSize: '16px', fontWeight: 800, color: 'var(--text-primary)' }}>{title}</div>
        <CompareControl {...compareControlProps} />
      </div>

      {metaChips.length > 0 && (
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', padding: '12px 22px 0' }}>
          {metaChips.map((chip, i) => <MetaChip key={i}>{chip}</MetaChip>)}
        </div>
      )}

      <div style={{ display: 'flex', gap: '4px', padding: '14px 22px 0', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
        {availableTabs.map(tab => (
          <TabButton key={tab} label={TAB_LABELS[tab]} active={activeTab === tab} onClick={() => setActiveTab(tab)} />
        ))}
      </div>

      <div ref={plotRootRef} style={{ flex: 1, minHeight: 0, overflow: 'auto', padding: '18px 22px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
        {activeTab === 'map' && chart.type === 'heatmap' && <MapLibreHeatmapPanel payload={chart} height={480} accessToken={accessToken} />}
        {activeTab === 'map' && chart.type === 'heatmap_multi' && <HeatmapMultiPanel payload={chart} accessToken={accessToken} />}
        {activeTab === 'chart' && chart.type === 'timeseries' && <TimeSeriesPanel payload={chart} />}
        {activeTab === 'statistics' && <StatisticsTab chart={chart} />}
        {activeTab === 'histogram' && <HistogramTab chart={chart} />}
        {activeTab === 'metadata' && (
          <MetadataTab
            chart={chart}
            onViewStatistics={availableTabs.includes('statistics') ? () => setActiveTab('statistics') : undefined}
          />
        )}
      </div>

      {(activeTab === 'map' || activeTab === 'chart') && (
        <div style={{ padding: '0 22px 14px', flexShrink: 0 }}>
          <ChartToolbar chart={chart} plotRootRef={plotRootRef} accessToken={accessToken} />
        </div>
      )}
    </div>
  )
}
