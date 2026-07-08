// Normalize the earthdata-retrieval MCP's search_datasets response into the
// flat view-model the discovery pane renders.
//
// The real MCP returns:
//   { datasets: [{ handle, summary: { entry_title, short_name, version,
//                                     processing_level, concept_id, ... } }],
//     count }
// (live-verified 2026-07-08; see Backend/tests/live_smoke/test_mcp_contract.py,
// which blesses `datasets`/`handle` as the real contract keys).
//
// An earlier fake fixture returned a different, fictional shape
//   { results: [{ dataset_handle, summary: "<string>", provider, ... }] }
// which the pane was originally coded against — so the pane read `data.results`
// and blanked when the real MCP sent `data.datasets`. This normalizer is the
// single place that tolerates BOTH shapes, so a future contract shift degrades
// to a plainer card instead of a blank pane.
export function normalizeSearchResults(data) {
  const rows = data?.datasets || data?.results || []
  if (!Array.isArray(rows)) return []

  return rows.map((row) => {
    const handle = row.handle || row.dataset_handle
    const summary = row.summary
    const meta = summary && typeof summary === 'object' ? summary : {}
    const title =
      (typeof summary === 'string' ? summary : meta.entry_title) || handle

    return {
      // Keep `dataset_handle`/`summary`(string) so DatasetCard's buttons and
      // App.handleRetrieve keep working against one internal shape.
      dataset_handle: handle,
      summary: title,
      short_name: meta.short_name || undefined,
      version: meta.version && meta.version !== 'NA' ? meta.version : undefined,
      processing_level:
        meta.processing_level && meta.processing_level !== 'NA'
          ? meta.processing_level
          : undefined,
      // Present only when the MCP enriches (e.g. describe); absent at search
      // time, so the card hides these rather than showing empties.
      variables: Array.isArray(row.variables) ? row.variables : undefined,
      temporal_extent: row.temporal_extent || meta.temporal_extent,
      provider: row.provider,
    }
  })
}
