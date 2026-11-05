// T64 Phase 4: the title of the collapsible tool card above an answer. A
// supervisor turn's tool calls are dispatch calls, titled by the requests
// they carried; a fast-path turn's are the sub-agent's own steps.

function plural(count, word) {
  return `${count} ${word}${count === 1 ? '' : 's'}`
}

export function toolStepsTitle(toolCalls, error) {
  const dispatches = (toolCalls || []).filter(tc => tc?.name === 'dispatch')
  if (dispatches.length) {
    if (error) return 'dispatch · error'
    const requests = dispatches.reduce((total, tc) => total + (tc.args?.requests?.length || 0), 0)
    return `dispatch · ${plural(requests, 'request')}`
  }
  return error ? 'error' : plural((toolCalls || []).length, 'step')
}
