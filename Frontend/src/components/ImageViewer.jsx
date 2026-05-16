import { useState } from 'react'

const toUrl = (path) => {
  if (!path) return null
  // If already absolute, return as-is
  if (path.startsWith('http')) return path
  // Ensure it goes through /api/outputs/ so nginx proxies to backend
  if (path.startsWith('/outputs/')) return `/api${path}`
  return path
}

export default function ImageViewer({ images }) {
  const [selected, setSelected] = useState(0)

  if (!images?.length) {
    return (
      <div style={{
        flex:           1,
        display:        'flex',
        flexDirection:  'column',
        alignItems:     'center',
        justifyContent: 'center',
        color:          'var(--text-secondary)',
        gap:            '12px',
      }}>
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
          <circle cx="12" cy="12" r="10"/>
          <path d="M12 8v4l3 3"/>
        </svg>
        <p style={{ fontSize: '15px' }}>Ask a question to see maps and charts here</p>
        <p style={{ fontSize: '12px', opacity: 0.6 }}>
          Try: "Plot NO2 in Texas on April 8 2024"
        </p>
      </div>
    )
  }

  const currentUrl = toUrl(images[selected])

  return (
    <div style={{
      flex:          1,
      display:       'flex',
      flexDirection: 'column',
      overflow:      'hidden',
      padding:       '16px',
      gap:           '12px',
    }}>
      {/* Main image */}
      <div style={{
        flex:           1,
        display:        'flex',
        alignItems:     'center',
        justifyContent: 'center',
        background:     'var(--bg-secondary)',
        borderRadius:   '12px',
        border:         '1px solid var(--border)',
        overflow:       'hidden',
      }}>
        {currentUrl ? (
          <img
            src={currentUrl}
            alt={`Output ${selected + 1}`}
            style={{
              maxWidth:  '100%',
              maxHeight: '100%',
              objectFit: 'contain',
            }}
          />
        ) : (
          <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>Loading image...</p>
        )}
      </div>

      {/* Thumbnails — only show if more than one image */}
      {images.length > 1 && (
        <div style={{
          display:        'flex',
          gap:            '8px',
          justifyContent: 'center',
          flexShrink:     0,
        }}>
          {images.map((url, i) => {
            const thumbUrl = toUrl(url)
            return (
              <button
                key={i}
                onClick={() => setSelected(i)}
                style={{
                  width:        '64px',
                  height:       '48px',
                  borderRadius: '8px',
                  border:       i === selected
                    ? '2px solid var(--accent)'
                    : '2px solid var(--border)',
                  overflow:     'hidden',
                  cursor:       'pointer',
                  padding:      0,
                  background:   'var(--bg-tertiary)',
                  flexShrink:   0,
                }}
              >
                {thumbUrl && (
                  <img
                    src={thumbUrl}
                    alt={`thumb ${i + 1}`}
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  />
                )}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}