export function compareArtifactValues(left, right) {
  if (left === right) return 0
  if (left === null || left === undefined || left === '') return 1
  if (right === null || right === undefined || right === '') return -1
  const leftNumber = Number(left)
  const rightNumber = Number(right)
  if (Number.isFinite(leftNumber) && Number.isFinite(rightNumber)) {
    return leftNumber - rightNumber
  }
  return String(left).localeCompare(String(right), undefined, { numeric: true, sensitivity: 'base' })
}

export function sortArtifactRows(rows, column, direction = 'asc') {
  if (!column) return [...(rows || [])]
  const multiplier = direction === 'desc' ? -1 : 1
  return [...(rows || [])].sort(
    (left, right) => compareArtifactValues(left[column], right[column]) * multiplier,
  )
}
