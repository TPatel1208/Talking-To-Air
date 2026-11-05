// A ground-monitor table artifact whose rows carry coordinates can also be
// shown on a map (TableMapPanel). It stays a "table" artifact -- "map" is the
// heatmap artifact type -- so this is the one place that decides it.
export function isGroundMapArtifact(artifact) {
  if (!artifact || artifact.type !== 'table') return false
  if (artifact.metadata?.has_coordinates) return true
  return Array.isArray(artifact.rows)
    && artifact.rows.some(row => row && row.latitude != null && row.longitude != null)
}
