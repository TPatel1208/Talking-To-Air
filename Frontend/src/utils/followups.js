// T22: pulls the additive suggested_followups key off a chat 'done' event
// payload. Missing/malformed input normalizes to [] so callers can render
// with a plain length check instead of null-guarding everywhere.
export function extractSuggestedFollowups(doneData) {
  const suggestions = doneData?.suggested_followups
  if (!Array.isArray(suggestions)) return []
  return suggestions.filter(s => typeof s === 'string')
}
