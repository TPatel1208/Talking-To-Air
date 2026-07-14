import { useEffect, useMemo, useState } from 'react'
import { sortArtifactRows } from '../utils/artifactTable'

const API_BASE = '/api'
const PAGE_SIZE = 100

function authHeaders(accessToken) {
  return accessToken ? { Authorization: `Bearer ${accessToken}` } : {}
}

function filenameFromDisposition(disposition, fallback) {
  const match = /filename="?([^";]+)"?/i.exec(disposition || '')
  return match?.[1]?.trim() || fallback
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

function formatCell(value) {
  if (value === null || value === undefined) return ''
  if (Array.isArray(value)) return value.join(', ')
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function sanitizeFilename(value) {
  return String(value || 'artifact')
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80) || 'artifact'
}

const TYPE_LABEL = { map: 'Map', comparison: 'Comparison', timeseries: 'Time series' }

function TypeBadge({ type }) {
  return (
    <span style={{
      fontSize: '10px', fontWeight: 600, color: 'var(--text-muted)',
      textTransform: 'uppercase', letterSpacing: '0.04em',
      border: '1px solid var(--border)', borderRadius: '4px', padding: '2px 6px',
    }}>
      {TYPE_LABEL[type] || type}
    </span>
  )
}

function ExportButtons({ artifactId, title, accessToken }) {
  const [state, setState] = useState('')

  async function download(format) {
    setState('downloading')
    try {
      const response = await fetch(`${API_BASE}/chart/${artifactId}/export.${format}`, {
        headers: authHeaders(accessToken),
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const blob = await response.blob()
      const filename = filenameFromDisposition(
        response.headers.get('content-disposition'),
        `${sanitizeFilename(title)}.${format}`,
      )
      downloadBlob(filename, blob)
      setState('')
    } catch (err) {
      setState(err.message || 'Export failed')
    }
  }

  const buttonStyle = {
    border: '1px solid var(--border)', borderRadius: '6px', background: 'var(--bg-secondary)',
    color: 'var(--text-secondary)', padding: '5px 9px', fontSize: '11px', cursor: 'pointer',
  }

  return (
    <div style={{ display: 'flex', gap: '6px', flexShrink: 0 }}>
      <button type="button" onClick={() => download('csv')} disabled={state === 'downloading'} style={buttonStyle}>CSV</button>
      <button type="button" onClick={() => download('png')} disabled={state === 'downloading'} style={buttonStyle}>PNG</button>
      {state && state !== 'downloading' && (
        <span style={{ fontSize: '11px', color: 'var(--danger, #b42318)' }}>{state}</span>
      )}
    </div>
  )
}

function CardShell({ artifact, accessToken, children }) {
  return (
    <div style={{
      border: '1px solid var(--border)', borderRadius: '8px', margin: '12px 0',
      background: 'var(--bg-primary)', overflow: 'hidden',
    }}>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px',
        padding: '10px 12px', borderBottom: '1px solid var(--border)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: 0 }}>
          <TypeBadge type={artifact.type} />
          <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)', overflowWrap: 'anywhere' }}>
            {artifact.title || 'Untitled artifact'}
          </div>
        </div>
        <ExportButtons artifactId={artifact.id} title={artifact.title} accessToken={accessToken} />
      </div>
      <div style={{ padding: '10px 12px', fontSize: '12px', color: 'var(--text-secondary)' }}>
        {children}
      </div>
    </div>
  )
}

export function MetaRow({ label, value }) {
  if (value === null || value === undefined || value === '') return null
  return (
    <div style={{ display: 'flex', gap: '6px', marginBottom: '4px' }}>
      <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{label}:</span>
      <span style={{ overflowWrap: 'anywhere' }}>{value}</span>
    </div>
  )
}

function MapArtifactCard({ artifact, accessToken }) {
  const meta = artifact.metadata || {}
  return (
    <CardShell artifact={artifact} accessToken={accessToken}>
      <MetaRow label="Variable" value={meta.variable} />
      <MetaRow label="Units" value={meta.units} />
      <MetaRow label="Bounding box" value={meta.bbox?.map(v => Number(v).toFixed(2)).join(', ')} />
      <MetaRow label="Color range" value={meta.colorbar ? `${meta.colorbar.vmin} to ${meta.colorbar.vmax}` : null} />
      <MetaRow label="Source handles" value={meta.source_handles?.join(', ')} />
    </CardShell>
  )
}

function ComparisonArtifactCard({ artifact, accessToken }) {
  const meta = artifact.metadata || {}
  return (
    <CardShell artifact={artifact} accessToken={accessToken}>
      <MetaRow label="Mode" value={meta.mode} />
      <MetaRow
        label="Panels"
        value={(meta.panels || []).map(p => p.title || p.handle).join(', ')}
      />
      <MetaRow label="Source handles" value={meta.source_handles?.join(', ')} />
    </CardShell>
  )
}

function TimeseriesArtifactCard({ artifact, accessToken }) {
  const meta = artifact.metadata || {}
  return (
    <CardShell artifact={artifact} accessToken={accessToken}>
      <MetaRow
        label="Series"
        value={(meta.series || [])
          .map(s => s.station_id ? `${s.label} (${s.source_kind}, ${s.station_id})` : `${s.label} (${s.source_kind})`)
          .join('; ')}
      />
      <MetaRow label="Source handles" value={meta.source_handles?.join(', ')} />
    </CardShell>
  )
}

export function TableArtifactMessage({ artifact, accessToken }) {
  const [page, setPage] = useState({ columns: [], rows: [], total_rows: artifact?.row_count || 0, offset: 0, limit: PAGE_SIZE })
  const [offset, setOffset] = useState(0)
  const [sort, setSort] = useState({ column: null, direction: 'asc' })
  const [status, setStatus] = useState('loading')
  const [error, setError] = useState('')
  const [exportState, setExportState] = useState('')

  useEffect(() => {
    if (!artifact?.id || artifact.type !== 'table') return undefined
    let cancelled = false
    async function loadPage() {
      setStatus('loading')
      setError('')
      try {
        const response = await fetch(`${API_BASE}/artifacts/${artifact.id}?offset=${offset}&limit=${PAGE_SIZE}`, {
          headers: authHeaders(accessToken),
        })
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const data = await response.json()
        if (!cancelled) {
          setPage(data)
          setStatus('ready')
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message || 'Unable to load table')
          setStatus('failed')
        }
      }
    }
    loadPage()
    return () => {
      cancelled = true
    }
  }, [artifact?.id, artifact?.type, accessToken, offset])

  const visibleColumns = useMemo(() => (
    page.columns?.length
      ? page.columns
      : Object.keys(page.rows?.[0] || {})
  ), [page.columns, page.rows])

  const visibleRows = useMemo(() => {
    const rows = [...(page.rows || [])]
    if (!sort.column) return rows
    return sortArtifactRows(rows, sort.column, sort.direction)
  }, [page.rows, sort])

  const totalRows = page.total_rows ?? artifact.row_count ?? 0
  const pageStart = totalRows ? offset + 1 : 0
  const pageEnd = Math.min(offset + (page.rows?.length || 0), totalRows)
  const canPrev = offset > 0
  const canNext = offset + PAGE_SIZE < totalRows

  async function downloadCsv() {
    setExportState('downloading')
    try {
      const response = await fetch(`${API_BASE}/artifacts/${artifact.id}/csv`, {
        headers: authHeaders(accessToken),
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const blob = await response.blob()
      const filename = filenameFromDisposition(
        response.headers.get('content-disposition'),
        `${sanitizeFilename(artifact.title || artifact.id)}.csv`,
      )
      downloadBlob(filename, blob)
      setExportState('')
    } catch (err) {
      setExportState(err.message || 'Export failed')
    }
  }

  if (!artifact || artifact.type !== 'table') return null

  function toggleSort(column) {
    setSort(current => (
      current.column === column
        ? { column, direction: current.direction === 'asc' ? 'desc' : 'asc' }
        : { column, direction: 'asc' }
    ))
  }

  return (
    <div style={{
      border: '1px solid var(--border)',
      borderRadius: '8px',
      overflow: 'hidden',
      margin: '12px 0',
      background: 'var(--bg-primary)',
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '12px',
        padding: '10px 12px',
        borderBottom: '1px solid var(--border)',
      }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
            {artifact.title || 'Table artifact'}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
            {totalRows.toLocaleString()} rows
          </div>
        </div>
        <button
          type="button"
          onClick={downloadCsv}
          disabled={exportState === 'downloading'}
          style={{
            border: '1px solid var(--border)',
            borderRadius: '6px',
            background: 'var(--bg-secondary)',
            color: 'var(--text-secondary)',
            padding: '6px 10px',
            fontSize: '12px',
            cursor: 'pointer',
            flexShrink: 0,
          }}
        >
          {exportState === 'downloading' ? 'Exporting' : 'CSV'}
        </button>
      </div>

      {status === 'failed' ? (
        <div style={{ padding: '12px', fontSize: '12px', color: 'var(--danger, #b42318)' }}>
          {error}
        </div>
      ) : (
        <>
          <div style={{ overflow: 'auto', maxHeight: '420px' }}>
            <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: '12px' }}>
              <thead>
                <tr>
                  {visibleColumns.map(column => (
                    <th key={column} onClick={() => toggleSort(column)} style={{
                      position: 'sticky',
                      top: 0,
                      background: 'var(--bg-secondary)',
                      borderBottom: '1px solid var(--border)',
                      color: 'var(--text-secondary)',
                      fontWeight: 600,
                      padding: '7px 10px',
                      textAlign: 'left',
                      whiteSpace: 'nowrap',
                      cursor: 'pointer',
                      userSelect: 'none',
                    }}>
                      {column}{sort.column === column ? ` ${sort.direction}` : ''}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {status === 'loading' ? (
                  <tr>
                    <td colSpan={Math.max(visibleColumns.length, 1)} style={{ padding: '12px', color: 'var(--text-muted)' }}>
                      Loading table...
                    </td>
                  </tr>
                ) : visibleRows.map((row, rowIndex) => (
                  <tr key={`${offset}-${rowIndex}`}>
                    {visibleColumns.map(column => (
                      <td key={column} style={{
                        borderBottom: '1px solid var(--border)',
                        padding: '7px 10px',
                        color: 'var(--text-primary)',
                        whiteSpace: 'nowrap',
                      }}>
                        {formatCell(row[column])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '8px',
            padding: '9px 12px',
            borderTop: '1px solid var(--border)',
            fontSize: '12px',
            color: 'var(--text-muted)',
          }}>
            <span>{pageStart.toLocaleString()}-{pageEnd.toLocaleString()} of {totalRows.toLocaleString()}</span>
            <div style={{ display: 'flex', gap: '6px' }}>
              <button type="button" disabled={!canPrev} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))} style={pagerStyle(canPrev)}>
                Prev
              </button>
              <button type="button" disabled={!canNext} onClick={() => setOffset(offset + PAGE_SIZE)} style={pagerStyle(canNext)}>
                Next
              </button>
            </div>
          </div>
        </>
      )}
      {exportState && exportState !== 'downloading' ? (
        <div style={{ padding: '0 12px 10px', fontSize: '11px', color: 'var(--danger, #b42318)' }}>
          {exportState}
        </div>
      ) : null}
    </div>
  )
}

function pagerStyle(enabled) {
  return {
    border: '1px solid var(--border)',
    borderRadius: '6px',
    background: enabled ? 'var(--bg-secondary)' : 'transparent',
    color: enabled ? 'var(--text-secondary)' : 'var(--text-muted)',
    padding: '5px 9px',
    fontSize: '12px',
    cursor: enabled ? 'pointer' : 'default',
  }
}

export default function ArtifactMessage({ artifact, accessToken }) {
  if (!artifact) return null
  switch (artifact.type) {
    case 'table': return <TableArtifactMessage artifact={artifact} accessToken={accessToken} />
    case 'map': return <MapArtifactCard artifact={artifact} accessToken={accessToken} />
    case 'comparison': return <ComparisonArtifactCard artifact={artifact} accessToken={accessToken} />
    case 'timeseries': return <TimeseriesArtifactCard artifact={artifact} accessToken={accessToken} />
    default: return null
  }
}
