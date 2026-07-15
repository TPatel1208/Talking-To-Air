/**
 * utils/artifactMetadataDisplay.js
 * ----------------------------------
 * Pure field-derivation helpers behind the artifact-shaped Metadata tab
 * Overview/Details (T33) -- a sibling to utils/metadataDisplay.js's chart
 * version, reading ArtifactReference.metadata shape (table's fetched page,
 * or TimeseriesArtifactMetadata for ground-validation composites) instead
 * of chart provenance. Kept separate from the JSX for the same reason as
 * T32: this repo's frontend test runner is plain `node --test`, no
 * jsdom/RTL, so display logic needs to be testable without rendering.
 */

// ── Table (table-producing tools: EPA AQS summaries, etc.) ─────────────────

export function tableOverviewFields(artifact, page) {
  return {
    rowCount: page?.total_rows ?? artifact?.row_count ?? null,
    columnCount: page?.columns?.length ?? null,
    sourceDataset: artifact?.metadata?.dataset ?? null,
  }
}

export function tableDetailsFields(artifact, page) {
  return {
    columns: page?.columns ?? [],
  }
}

// ── Ground-validation timeseries (validate_against_ground / exceedance_overlay) ─

function seriesByKind(series, kind) {
  return (series || []).filter(s => s.source_kind === kind)
}

function seriesLabel(s) {
  return s.station_id ? `${s.label} (${s.station_id})` : s.label
}

// r + n + coverage% when stats are available (validate_against_ground);
// falls back to a coverage-only line when only coverage is known
// (exceedance_overlay never computes a correlation).
function correlationSummary(stats, coverage) {
  if (stats && typeof stats.r === 'number') {
    const pct = stats.coverage_fraction != null ? `, ${Math.round(stats.coverage_fraction * 100)}% coverage` : ''
    return `r = ${stats.r.toFixed(2)} (n=${stats.n}${pct})`
  }
  if (coverage?.coverage_fraction != null) {
    return `${Math.round(coverage.coverage_fraction * 100)}% valid coverage`
  }
  return null
}

export function groundValidationOverviewFields(artifact) {
  const meta = artifact?.metadata || {}
  const satellite = seriesByKind(meta.series, 'satellite')[0]
  const groundStations = seriesByKind(meta.series, 'ground')

  return {
    satelliteVariable: satellite ? seriesLabel(satellite) : null,
    groundStations: groundStations.length ? groundStations.map(seriesLabel).join(', ') : null,
    correlationSummary: correlationSummary(meta.stats, meta.coverage),
    qaStatus: meta.masking?.qa_status ?? null,
  }
}

export function groundValidationDetailsFields(artifact) {
  const meta = artifact?.metadata || {}
  return {
    series: meta.series || [],
    exceedanceDates: meta.exceedance_dates ?? null,
    sourceHandles: meta.source_handles?.length ? meta.source_handles.join(', ') : null,
    stats: meta.stats ?? null,
  }
}

// ── Raw JSON toggle (shared shape for both artifact kinds) ─────────────────

export function rawArtifactMetadataJson(artifact) {
  return artifact?.metadata ?? null
}
