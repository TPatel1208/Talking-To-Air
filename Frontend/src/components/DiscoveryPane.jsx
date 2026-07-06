function DatasetCard({ dataset, location, timeRange, preview, coverage, onPreview, onCoverage, onRetrieve }) {
  const handle = dataset.dataset_handle

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
          onClick={() => onRetrieve(dataset, location, timeRange)}
          style={{
            fontSize: '11px', padding: '4px 10px', borderRadius: '6px',
            border: 'none', background: 'var(--teal)',
            color: 'white', cursor: 'pointer',
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
      {coverage && !coverage.loading && !coverage.error && coverage.has_data !== undefined && (
        <div style={{ fontSize: '11px', color: coverage.has_data ? 'var(--teal-text)' : 'var(--error)' }}>
          {coverage.has_data
            ? `Data available${typeof coverage.granule_count === 'number' ? ` — ${coverage.granule_count} granules` : ''}`
            : 'No data for this area/window'}
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
  previews, coverages,
  onSearch, onPreview, onCoverage, onRetrieve,
}) {
  return (
    <div style={{
      width:         '300px',
      flexShrink:    0,
      background:    'var(--bg-secondary)',
      borderLeft:    '1px solid var(--border)',
      display:       'flex',
      flexDirection: 'column',
      overflow:      'hidden',
    }}>
      <div style={{
        padding:      '20px 14px 12px',
        borderBottom: '1px solid var(--border)',
      }}>
        <span style={{
          fontSize:      '10px',
          fontWeight:    '500',
          color:         'var(--text-muted)',
          textTransform: 'uppercase',
          letterSpacing: '0.09em',
        }}>
          Discover
        </span>

        <form
          onSubmit={event => { event.preventDefault(); onSearch() }}
          style={{ display: 'flex', gap: '6px', marginTop: '10px' }}
        >
          <input
            value={query}
            onChange={event => setQuery(event.target.value)}
            placeholder="soil moisture, formaldehyde…"
            style={{
              flex: 1, height: '30px', fontSize: '12px', padding: '0 8px',
              border: '1px solid var(--border)', borderRadius: '6px',
              background: 'var(--bg-primary)', color: 'var(--text-primary)',
            }}
          />
          <button
            type="submit"
            style={{
              height: '30px', padding: '0 10px', fontSize: '12px',
              border: 'none', borderRadius: '6px', background: 'var(--teal)',
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
              border: '1px solid var(--border)', borderRadius: '6px',
              background: 'var(--bg-primary)', color: 'var(--text-primary)',
            }}
          />
          <input
            value={timeRange}
            onChange={event => setTimeRange(event.target.value)}
            placeholder="Window (e.g. 2026-06-01/2026-06-30)"
            style={{
              flex: 1, height: '28px', fontSize: '11px', padding: '0 8px',
              border: '1px solid var(--border)', borderRadius: '6px',
              background: 'var(--bg-primary)', color: 'var(--text-primary)',
            }}
          />
        </div>
        <div style={{ fontSize: '10px', color: 'var(--text-hint)', marginTop: '4px' }}>
          Quick-look and coverage checks below use this area/window.
        </div>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '10px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
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
            onPreview={onPreview}
            onCoverage={onCoverage}
            onRetrieve={onRetrieve}
          />
        ))}
      </div>
    </div>
  )
}
