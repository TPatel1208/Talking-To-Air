import { resultExportUrl } from './resultMode.js'

function filenameFromDisposition(disposition, fallback) {
  const match = /filename="?([^";]+)"?/i.exec(disposition || '')
  return match?.[1]?.trim() || fallback
}

// Fetches a result's archive through `fetcher` (apiFetch, so the request is
// authenticated) and returns its blob and filename. Throws on a failed fetch.
export async function fetchResultArchive({ apiBase, runId, fallbackName, fetcher }) {
  const response = await fetcher(resultExportUrl(apiBase, runId))
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(body?.detail || `Download failed with status ${response.status}`)
  }
  const blob = await response.blob()
  return { blob, filename: filenameFromDisposition(response.headers?.get?.('content-disposition'), fallbackName) }
}
