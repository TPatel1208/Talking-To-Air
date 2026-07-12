// QA-flag masking provenance rides along on chart payloads from the backend
// (Backend/preprocessing/aggregation_service.resolve_and_mask ->
// datasets/qa_flags.resolve_qa_info). Its qa_status is one of the vocabulary
// values in qa_flags.py: "verified", "cf-deterministic", "inferred, not
// verified", "ambiguous — awaiting classification", "not applied — semantics
// unknown". It lands in a different spot per chart type: top-level `masking`
// for timeseries (plot_tools sets ts_payload["masking"]), and inside
// `provenance.masking` / `aggregation_meta.masking` for heatmaps (plot_tools
// _provenance copies agg_meta["masking"] there). Check all three so the
// Statistics tab can show whether QA masking actually ran, instead of leaving
// the user to infer it from the valid-pixel count.
export function resolveMasking(chart) {
  if (!chart || typeof chart !== 'object') return null

  const masking =
    chart.masking ||
    chart.provenance?.masking ||
    chart.aggregation_meta?.masking ||
    null
  if (!masking || typeof masking !== 'object') return null

  const qaStatus = masking.qa_status
  if (!qaStatus) return null

  return {
    qaStatus,
    qaSource: masking.qa_source || '',
    qaNote: masking.qa_note || '',
  }
}
