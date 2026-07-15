/**
 * components/MetadataOverview.jsx
 * ----------------------------------
 * The chart Metadata tab's Overview panel (T32), split into its own module
 * (T34 review fix) so CompareGrid.jsx's per-slot metadata overview toggle
 * can import it without creating a circular dependency: OutputPanel.jsx
 * imports CompareGrid.jsx (for the Compare tab), and CompareGrid.jsx used
 * to import MetadataOverview back from OutputPanel.jsx.
 */
import { useMemo, useState } from 'react'
import { resolveMasking } from '../utils/maskingProvenance'
import {
  NOT_AVAILABLE, fmt, dateRangeLabel, granuleSummary, maskingStatusColor,
  citationString, datasetLandingUrl, regionLabel, hasProvenance,
} from '../utils/metadataDisplay'
import { MetaField } from './metadataPrimitives.jsx'
import { smallButtonStyle, copyToClipboard } from '../utils/metadataUiHelpers.js'

export function MetadataOverview({ chart, onViewStatistics, onViewFullMetadata, note }) {
  const provenance = chart.provenance || {}
  const masking = useMemo(() => resolveMasking(chart), [chart])
  const [copyState, setCopyState] = useState('')
  const landingUrl = datasetLandingUrl(provenance.collection_id)
  const dateRange = dateRangeLabel(provenance)

  if (!hasProvenance(chart)) {
    return (
      <div style={{ fontSize: '12px', color: 'var(--text-muted)', padding: '10px 0' }}>
        No metadata is available for this view.
      </div>
    )
  }

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
          <MetaField label="Region" value={regionLabel(chart)} />
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
