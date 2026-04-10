import { useState } from 'react'

const API_BASE = 'http://localhost:8000'

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
        <img
          src={`${API_BASE}${images[selected]}`}
          alt={`Output ${selected + 1}`}
          style={{
            maxWidth:   '100%',
            maxHeight:  '100%',
            objectFit:  'contain',
          }}
        />
      </div>

      {/* Thumbnails — only show if more than one image */}
      {images.length > 1 && (
        <div style={{
          display:        'flex',
          gap:            '8px',
          justifyContent: 'center',
          flexShrink:     0,
        }}>
          {images.map((url, i) => (
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
              <img
                src={`${API_BASE}${url}`}
                alt={`thumb ${i + 1}`}
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              />
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
