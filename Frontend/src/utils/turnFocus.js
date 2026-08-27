// Which output the central panel focuses when a chat turn finishes.
//
// A turn "settles" the moment `loading` goes true -> false. At that instant
// the newest chart of the final assistant message takes focus, or failing
// that its first reachable artifact (T33's card rules).
//
// `null` means "leave focus where it is", and it is load-bearing in three
// cases: mid-turn, a turn that ends on something other than an assistant
// message, and a turn that produced no output at all. None of those may
// clear whatever the user is currently looking at.
import { reachableArtifacts } from './artifactReachability.js'

export function turnCompletionFocus(wasLoading, loading, messages) {
  if (!wasLoading || loading) return null

  const list = messages || []
  const last = list[list.length - 1]
  if (last?.role !== 'assistant') return null

  if (last.charts?.length) {
    return { kind: 'chart', data: last.charts[last.charts.length - 1] }
  }

  const artifact = reachableArtifacts(last)[0]
  return artifact ? { kind: 'artifact', data: artifact } : null
}
