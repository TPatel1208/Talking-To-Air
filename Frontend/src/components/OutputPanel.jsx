import { useMemo, useState, useEffect, useRef } from 'react'
import { HeatmapPanel, HeatmapMultiPanel, TimeSeriesPanel, ChartToolbar, ProvenanceBlock } from './ChartMessage'
import ArtifactMessage, { TableArtifactMessage } from './ArtifactMessage'
import { computeChartStats, computeHistogram } from '../utils/chartStats'

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

function StatisticsTab({ chart }) {
  const stats = useMemo(() => computeChartStats(chart), [chart])
  if (!stats) {
    return <div style={{ padding: '24px', color: 'var(--text-muted)', fontSize: '13px' }}>No numeric values available for this output.</div>
  }
  const fmt = (n) => Number.isFinite(n) ? n.toExponential(3) : '—'
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '10px' }}>
      <StatCard label="Mean" value={`${fmt(stats.mean)} ${stats.units || ''}`} />
      <StatCard label="Max" value={`${fmt(stats.max)} ${stats.units || ''}`} />
      <StatCard label="Min" value={`${fmt(stats.min)} ${stats.units || ''}`} />
      <StatCard label="Valid values" value={`${stats.validPct.toFixed(1)}%`} />
      <StatCard label="Sample count" value={stats.count.toLocaleString()} />
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

function MetadataTab({ chart, artifact }) {
  if (chart) {
    return <ProvenanceBlock provenance={chart.provenance} aggregationMeta={chart.aggregation_meta} />
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
const TAB_LABELS = { map: 'Map', chart: 'Chart', statistics: 'Statistics', histogram: 'Histogram', metadata: 'Metadata' }

export default function OutputPanel({ focusedOutput, accessToken }) {
  const kind = focusedOutput?.kind
  const chart = kind === 'chart' ? focusedOutput.data : null
  const artifact = kind === 'artifact' ? focusedOutput.data : null
  const isTableArtifact = artifact?.type === 'table'

  const availableTabs = chart ? (CHART_TABS[chart.type] || ['metadata']) : []
  const [activeTab, setActiveTab] = useState(availableTabs[0])
  const plotRootRef = useRef(null)

  useEffect(() => {
    setActiveTab(availableTabs[0])
  }, [focusedOutput, availableTabs.join(',')])

  if (!focusedOutput) {
    return (
      <div style={{
        flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center', textAlign: 'center',
        background: 'var(--bg-primary)', color: 'var(--text-muted)', padding: '0 24px',
      }}>
        <div style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '6px' }}>
          Ask a question to get started
        </div>
        <div style={{ fontSize: '13px', lineHeight: 1.5, maxWidth: '360px' }}>
          Run an analysis in the chat, then click any output card to open it here — map, chart, statistics, and metadata all in one place.
        </div>
      </div>
    )
  }

  if (isTableArtifact) {
    return (
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', background: 'var(--bg-primary)', overflow: 'auto', padding: '18px 22px' }}>
        <TableArtifactMessage artifact={artifact} accessToken={accessToken} />
      </div>
    )
  }

  if (artifact && !chart) {
    return (
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', background: 'var(--bg-primary)', overflow: 'auto', padding: '18px 22px' }}>
        <div style={{ fontSize: '16px', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '4px' }}>{artifact.title || 'Output'}</div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', margin: '10px 0 16px' }}>
          {artifactMetaChips(artifact).map((chip, i) => <MetaChip key={i}>{chip}</MetaChip>)}
        </div>
        <ArtifactMessage artifact={artifact} accessToken={accessToken} />
      </div>
    )
  }

  const title = chartTitle(chart)
  const metaChips = chartMetaChips(chart)

  return (
    <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', background: 'var(--bg-primary)', overflow: 'hidden' }}>
      <div style={{ padding: '14px 22px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 }}>
        <div style={{ fontSize: '16px', fontWeight: 800, color: 'var(--text-primary)' }}>{title}</div>
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
        {activeTab === 'map' && chart.type === 'heatmap' && <HeatmapPanel payload={chart} height={480} />}
        {activeTab === 'map' && chart.type === 'heatmap_multi' && <HeatmapMultiPanel payload={chart} />}
        {activeTab === 'chart' && chart.type === 'timeseries' && <TimeSeriesPanel payload={chart} />}
        {activeTab === 'statistics' && <StatisticsTab chart={chart} />}
        {activeTab === 'histogram' && <HistogramTab chart={chart} />}
        {activeTab === 'metadata' && <MetadataTab chart={chart} />}
      </div>

      {(activeTab === 'map' || activeTab === 'chart') && (
        <div style={{ padding: '0 22px 14px', flexShrink: 0 }}>
          <ChartToolbar chart={chart} plotRootRef={plotRootRef} accessToken={accessToken} />
        </div>
      )}
    </div>
  )
}
