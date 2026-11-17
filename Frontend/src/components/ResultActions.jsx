import { useState } from 'react'
import { apiFetch } from '../utils/apiFetch.js'
import { API_BASE } from '../config.js'
import { connectorNotices, downloadableResults } from '../utils/resultMode.js'
import { fetchResultArchive } from '../utils/resultDownload.js'
import ModeBadge from './ModeBadge'

const linkStyle = {
  border: 0, background: 'transparent', color: 'var(--teal-text)', fontSize: '11.5px',
  fontWeight: 700, cursor: 'pointer', padding: 0, fontFamily: 'var(--font)', textDecoration: 'underline',
}

function saveBlob(filename, blob) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}

function DownloadResult({ item, many }) {
  const [state, setState] = useState('')

  async function download() {
    setState('downloading')
    try {
      const { blob, filename } = await fetchResultArchive({
        apiBase: API_BASE, runId: item.runId, fallbackName: item.filename, fetcher: apiFetch,
      })
      saveBlob(filename, blob)
      setState('')
    } catch (error) {
      setState(error.message || 'Download failed')
    }
  }

  return (
    <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
      <button type="button" onClick={download} disabled={state === 'downloading'} style={linkStyle}>
        {state === 'downloading' ? 'Preparing download' : 'Download result'}
      </button>
      {many && <ModeBadge badge={item.badge} />}
      {state && state !== 'downloading' && (
        <span style={{ fontSize: '11px', color: 'var(--error, #b42318)' }}>{state}</span>
      )}
    </div>
  )
}

/* Under an answer: why a run stopped on a missing connector, with a way to
 * connect it, and a download of each finished result. */
export default function ResultActions({ results, onOpenConnectors }) {
  const notices = connectorNotices(results)
  const downloads = downloadableResults(results)
  if (!notices.length && !downloads.length) return null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', padding: '2px 2px 0' }}>
      {notices.map((notice, i) => (
        <div
          key={notice.runId || i}
          style={{
            fontSize: '12px', lineHeight: 1.5, color: 'var(--text-secondary)',
            background: 'var(--amber-bg, #fff8e6)', border: '1px solid var(--amber, #d99100)',
            borderRadius: '8px', padding: '8px 10px',
          }}
        >
          <span>{notice.reason}</span>{' '}
          {onOpenConnectors && (
            <button type="button" onClick={() => onOpenConnectors(notice.connector)} style={linkStyle}>
              Connect {notice.connectorName}
            </button>
          )}
        </div>
      ))}
      {downloads.map(item => (
        <DownloadResult key={item.runId} item={item} many={downloads.length > 1} />
      ))}
    </div>
  )
}
