// Which artifacts get their own clickable card (T33).
//
// `table` artifacts have no other rendering, so they're always reachable.
// Chart-backed artifact types (map, comparison, and a chart-backed
// `timeseries`) duplicate what the parallel entry in msg.charts already
// shows -- deliberately excluded (Chat.jsx's original comment). Ground-
// validation `timeseries` artifacts (validate_against_ground/
// exceedance_overlay, Backend/tools/satellite_tools/validation_tools.py)
// mint their own chart_id and never call emit_chart, so nothing in
// msg.charts shares that id -- they have no duplicate to avoid and need
// their own card too.
function hasMatchingChart(artifact, charts) {
  return (charts || []).some(chart => {
    const chartId = chart?.chart_id ?? chart?.id
    return chartId != null && chartId === artifact.id
  })
}

export function isReachableArtifact(artifact, charts) {
  if (!artifact) return false
  if (artifact.type === 'table') return true
  if (artifact.type === 'timeseries') return !hasMatchingChart(artifact, charts)
  return false
}

export function reachableArtifacts(msg) {
  return (msg?.artifacts || []).filter(artifact => isReachableArtifact(artifact, msg?.charts))
}
