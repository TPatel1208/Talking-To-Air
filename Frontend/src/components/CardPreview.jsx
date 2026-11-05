import { useEffect, useState } from 'react'
import { cardPreview } from '../utils/cardPreview.js'
import { loadThumbnail } from '../utils/thumbnailSource.js'
import { apiFetch } from '../utils/apiFetch.js'
import { API_BASE } from '../config.js'

// The square on a chat output card: a miniature of what the output is. See
// utils/cardPreview.js for what each output type shows.

const SIZE = 48

const tileStyle = {
  width: SIZE, height: SIZE, borderRadius: '8px', flexShrink: 0, overflow: 'hidden',
  background: 'var(--bg-secondary)', border: '1px solid var(--border)',
  display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
}

const ICON_PATHS = {
  map: (
    <>
      <path d="M9 4 3 6v14l6-2 6 2 6-2V4l-6 2-6-2Z" />
      <path d="M9 4v14M15 6v14" />
    </>
  ),
  timeseries: <path d="M3 3v18h18M7 15l4-5 3 3 5-7" />,
  profile: <path d="M4 3v18M4 21h16M10 20c1-4 6-5 5-9s-5-5-3-8" />,
  table: (
    <>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="M3 10h18M3 15h18M9 4v16" />
    </>
  ),
  generic: <path d="M4 4h16v16H4zM8 9h8M8 13h8M8 17h5" />,
}

function Icon({ name, size = 22 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="var(--teal)" strokeWidth="1.7"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {ICON_PATHS[name] || ICON_PATHS.generic}
    </svg>
  )
}

function Curve({ path }) {
  return (
    <svg width={SIZE} height={SIZE} viewBox="0 0 40 40" aria-hidden="true">
      <path d={path} fill="none" stroke="var(--teal)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

// The map's server-rendered image, fetched with auth into a blob: an <img>
// cannot send the bearer header. The map icon stands in until it loads, and
// stays if it never does.
function MapPreview({ overlayUrl }) {
  const [loaded, setLoaded] = useState(null)
  useEffect(() => loadThumbnail(`${API_BASE}${overlayUrl}`, {
    fetcher: apiFetch,
    onLoad: (objectUrl) => setLoaded({ url: overlayUrl, objectUrl }),
  }), [overlayUrl])
  if (loaded?.url !== overlayUrl) return <Icon name="map" />
  return <img src={loaded.objectUrl} alt="" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
}

export default function CardPreview({ item }) {
  const preview = cardPreview(item)
  return (
    <div style={tileStyle}>
      {preview.kind === 'map' && <MapPreview overlayUrl={preview.overlayUrl} />}
      {(preview.kind === 'sparkline' || preview.kind === 'profile') && <Curve path={preview.path} />}
      {preview.kind === 'table' && (
        <>
          <Icon name="table" size={preview.rowLabel ? 18 : 22} />
          {preview.rowLabel && (
            <span style={{ fontSize: '10px', fontWeight: 700, color: 'var(--teal-text)', marginTop: '2px' }}>
              {preview.rowLabel}
            </span>
          )}
        </>
      )}
      {preview.kind === 'icon' && <Icon name={preview.icon} />}
    </div>
  )
}
