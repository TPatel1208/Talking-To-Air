function GranulesTable({ granules }) {
  return (
    <table style={{ width: '100%', fontSize: '11px', borderCollapse: 'collapse', color: 'var(--text-secondary)' }}>
      <thead>
        <tr style={{ textAlign: 'left', color: 'var(--text-hint)' }}>
          <th style={{ fontWeight: 500, paddingBottom: '2px' }}>Timestamp</th>
          <th style={{ fontWeight: 500, paddingBottom: '2px' }}>Size</th>
        </tr>
      </thead>
      <tbody>
        {granules.map((granule, index) => (
          <tr key={granule.granule_ur || index}>
            <td style={{ padding: '2px 6px 2px 0' }}>{granule.time_start || '—'}</td>
            <td style={{ padding: '2px 0' }}>
              {typeof granule.size_mb === 'number' ? `${granule.size_mb.toFixed(1)} MB` : '—'}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function DatasetCard({ dataset, location, timeRange, preview, coverage, granules, onPreview, onCoverage, onGranules, onRetrieve }) {
  const handle = dataset.dataset_handle
  // Retrieve turns into the same chat message the agent parses for its time
  // range (App.jsx::handleRetrieve) — an empty window silently drops the
  // date instead of validating it, which is exactly how a retrieval ends up
  // defaulting to "now". Gate it the same way checkCoverage/inspectGranules
  // already require an area + window before calling out.
  const canRetrieve = location.trim() && timeRange.trim()

  return (
    <div style={{
      padding:       '10px 12px',
      borderRadius:  '10px',
      background:    'var(--bg-card)',
      boxShadow:     'var(--shadow-sm)',
      display:       'flex',
      flexDirection: 'column',
      gap:           '6px',
    }}>
      <div style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text-primary)' }}>
        {dataset.summary || handle}
      </div>

      {Array.isArray(dataset.variables) && dataset.variables.length > 0 && (
        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
          Variables: {dataset.variables.join(', ')}
        </div>
      )}
      <div style={{ fontSize: '11px', color: 'var(--text-hint)', display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
        {dataset.short_name && <span>{dataset.short_name}</span>}
        {dataset.processing_level && <span>L{dataset.processing_level}</span>}
        {dataset.version && <span>v{dataset.version}</span>}
        {dataset.temporal_extent && <span>{dataset.temporal_extent}</span>}
        {dataset.provider && <span>{dataset.provider}</span>}
      </div>

      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
        <button
          onClick={() => onPreview(handle)}
          style={{
            fontSize: '11px', padding: '4px 10px', borderRadius: '6px',
            border: '1px solid var(--border)', background: 'transparent',
            color: 'var(--text-secondary)', cursor: 'pointer',
          }}
        >
          Quick-look
        </button>
        <button
          onClick={() => onCoverage(handle)}
          style={{
            fontSize: '11px', padding: '4px 10px', borderRadius: '6px',
            border: '1px solid var(--border)', background: 'transparent',
            color: 'var(--text-secondary)', cursor: 'pointer',
          }}
        >
          Check coverage
        </button>
        <button
          onClick={() => onGranules(handle)}
          style={{
            fontSize: '11px', padding: '4px 10px', borderRadius: '6px',
            border: '1px solid var(--border)', background: 'transparent',
            color: 'var(--text-secondary)', cursor: 'pointer',
          }}
        >
          Granules
        </button>
        <button
          onClick={() => onRetrieve(dataset, location, timeRange)}
          disabled={!canRetrieve}
          title={canRetrieve ? undefined : 'Set an area and time window above first.'}
          style={{
            fontSize: '11px', padding: '4px 10px', borderRadius: '6px',
            border: 'none', background: canRetrieve ? 'var(--teal)' : 'var(--border)',
            color: canRetrieve ? 'white' : 'var(--text-hint)',
            cursor: canRetrieve ? 'pointer' : 'not-allowed',
          }}
        >
          Retrieve
        </button>
      </div>

      {preview?.loading && (
        <div style={{ fontSize: '11px', color: 'var(--text-hint)' }}>Loading quick-look…</div>
      )}
      {preview?.error && (
        <div style={{ fontSize: '11px', color: 'var(--error)' }}>{preview.error}</div>
      )}
      {preview && !preview.loading && !preview.error && preview.has_gibs_layer === false && (
        <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontStyle: 'italic' }}>
          {preview.message || 'No browse layer available for this dataset.'}
        </div>
      )}
      {preview && !preview.loading && !preview.error && preview.gibs_url && (
        <img
          src={preview.gibs_url}
          alt={`GIBS quick-look for ${handle}`}
          style={{ width: '100%', borderRadius: '6px', border: '1px solid var(--border)' }}
        />
      )}

      {coverage?.loading && (
        <div style={{ fontSize: '11px', color: 'var(--text-hint)' }}>Checking coverage…</div>
      )}
      {coverage?.error && (
        <div style={{ fontSize: '11px', color: 'var(--error)' }}>{coverage.error}</div>
      )}
      {coverage && !coverage.loading && !coverage.error && coverage.covered !== undefined && (
        <div style={{ fontSize: '11px', color: coverage.covered ? 'var(--teal-text)' : 'var(--error)' }}>
          {coverage.covered
            ? `Data available${typeof coverage.granule_count === 'number' ? ` — ${coverage.granule_count} granules` : ''}`
            : 'No data for this area/window'}
        </div>
      )}

      {granules?.loading && (
        <div style={{ fontSize: '11px', color: 'var(--text-hint)' }}>Listing granules…</div>
      )}
      {granules?.error && (
        <div style={{ fontSize: '11px', color: 'var(--error)' }}>{granules.error}</div>
      )}
      {granules && !granules.loading && !granules.error && granules.note && (
        <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontStyle: 'italic' }}>
          {granules.note.message}
        </div>
      )}
      {granules && !granules.loading && !granules.error && Array.isArray(granules.granules) && granules.granules.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
            First {granules.limit_applied} — {granules.count} granule{granules.count === 1 ? '' : 's'}, {granules.total_size_mb.toFixed(1)} MB total
          </div>
          <GranulesTable granules={granules.granules} />
        </div>
      )}
    </div>
  )
}

export default function DiscoveryPane({
  query, setQuery,
  location, setLocation,
  timeRange, setTimeRange,
  results, loading, error,
  previews, coverages, granules,
  onSearch, onPreview, onCoverage, onGranules, onRetrieve,
}) {
  return (
    <div style={{
      flex:          1,
      minHeight:     0,
      display:       'flex',
      flexDirection: 'column',
      overflow:      'hidden',
    }}>
      <div style={{ padding: '0 0 12px' }}>
        <div style={{ fontSize: '15px', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '2px' }}>
          Discover datasets
        </div>
        <div style={{ fontSize: '11.5px', color: 'var(--text-muted)' }}>
          Find the right data for your analysis.
        </div>

        <form
          onSubmit={event => { event.preventDefault(); onSearch() }}
          style={{ display: 'flex', gap: '6px', marginTop: '12px' }}
        >
          <input
            value={query}
            onChange={event => setQuery(event.target.value)}
            placeholder="soil moisture, formaldehyde…"
            style={{
              flex: 1, height: '32px', fontSize: '12px', padding: '0 10px',
              border: '1px solid var(--border)', borderRadius: '9px',
              background: 'var(--bg-secondary)', color: 'var(--text-primary)',
            }}
          />
          <button
            type="submit"
            style={{
              height: '32px', padding: '0 14px', fontSize: '12px', fontWeight: 700,
              border: 'none', borderRadius: '9px', background: 'var(--teal)',
              color: 'white', cursor: 'pointer',
            }}
          >
            Search
          </button>
        </form>

        <div style={{ display: 'flex', gap: '6px', marginTop: '8px' }}>
          <input
            value={location}
            onChange={event => setLocation(event.target.value)}
            placeholder="Area (e.g. Raritan basin)"
            style={{
              flex: 1, height: '28px', fontSize: '11px', padding: '0 8px',
              border: '1px solid var(--border)', borderRadius: '7px',
              background: 'var(--bg-secondary)', color: 'var(--text-primary)',
            }}
          />
          <input
            value={timeRange}
            onChange={event => setTimeRange(event.target.value)}
            placeholder="Window (e.g. 2026-06-01/2026-06-30)"
            style={{
              flex: 1, height: '28px', fontSize: '11px', padding: '0 8px',
              border: '1px solid var(--border)', borderRadius: '7px',
              background: 'var(--bg-secondary)', color: 'var(--text-primary)',
            }}
          />
        </div>
        <div style={{ fontSize: '10px', color: 'var(--text-hint)', marginTop: '4px' }}>
          Quick-look, coverage checks, and Retrieve below use this area/window.
        </div>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '2px 0 10px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {loading && (
          <div style={{ padding: '4px 2px', color: 'var(--text-hint)', fontSize: '12px' }}>Searching…</div>
        )}
        {error && (
          <div style={{ color: 'var(--error)', fontSize: '12px', padding: '4px 2px' }}>{error}</div>
        )}
        {!loading && !error && results.length === 0 && (
          <div style={{ padding: '12px 6px', color: 'var(--text-hint)', fontSize: '12px', fontStyle: 'italic' }}>
            Search a phenomenon to browse datasets
          </div>
        )}
        {results.map(dataset => (
          <DatasetCard
            key={dataset.dataset_handle}
            dataset={dataset}
            location={location}
            timeRange={timeRange}
            preview={previews[dataset.dataset_handle]}
            coverage={coverages[dataset.dataset_handle]}
            granules={granules[dataset.dataset_handle]}
            onPreview={onPreview}
            onCoverage={onCoverage}
            onGranules={onGranules}
            onRetrieve={onRetrieve}
          />
        ))}
      </div>
    </div>
  )
}
