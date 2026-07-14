/**
 * HeatmapMultiPanel.jsx
 * ----------------------
 * Comparison charts (T08/T23): region mode shows each panel as a static
 * small-multiple (its own server-rendered overlay PNG, or a canvas-fallback
 * thumbnail) over a lightweight neutral background -- click one to expand
 * it into the single interactive MapLibreHeatmapPanel. Period mode's
 * diverging difference map IS the composite view, so it renders directly
 * as one interactive map. At most one live WebGL context at a time.
 */
import { useEffect, useRef, useState } from 'react'
import MapLibreHeatmapPanel from './MapLibreHeatmapPanel.jsx'
import { buildCanvasFallbackFrame } from '../utils/canvasFallback.js'

function ThumbnailCanvas({ lats, lons, values, vmin, vmax, lut }) {
  const ref = useRef(null)
  useEffect(() => {
    if (!ref.current || !Array.isArray(lats) || !Array.isArray(lons) || !Array.isArray(values)) return
    const frame = buildCanvasFallbackFrame({ lats, lons, values, vmin, vmax, lut })
    if (!frame.width || !frame.height) return
    ref.current.width = frame.width
    ref.current.height = frame.height
    ref.current.getContext('2d').putImageData(new ImageData(frame.pixels, frame.width, frame.height), 0, 0)
  }, [lats, lons, values, vmin, vmax, lut])
  return <canvas ref={ref} style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
}

function PanelThumbnail({ panel, onClick }) {
  const { title, overlay, lats, lons, values, vmin, vmax, colormap, bounds } = panel
  const [minx, miny, maxx, maxy] = bounds || overlay?.bounds || [0, 0, 1, 1]
  const aspect = (maxx - minx) > 0 && (maxy - miny) > 0 ? (maxx - minx) / (maxy - miny) : 1

  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        display: 'block', width: '100%', padding: 0, border: '1px solid var(--border)',
        borderRadius: '10px', overflow: 'hidden', background: 'var(--bg-card)', cursor: 'pointer', textAlign: 'left',
      }}
    >
      <div style={{ position: 'relative', width: '100%', aspectRatio: aspect, background: '#e4e1d8' }}>
        {overlay?.url ? (
          <img
            src={`/api${overlay.url}`}
            alt={title || 'comparison panel'}
            style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
          />
        ) : (
          <ThumbnailCanvas lats={lats} lons={lons} values={values} vmin={vmin} vmax={vmax} lut={colormap?.lut} />
        )}
      </div>
      {title && (
        <div style={{ padding: '6px 8px', fontSize: '11px', color: 'var(--text-secondary)' }}>{title}</div>
      )}
    </button>
  )
}

const backButtonStyle = {
  border: '1px solid var(--border)',
  background: 'var(--bg-card)',
  color: 'var(--text-secondary)',
  borderRadius: '7px',
  padding: '5px 10px',
  fontSize: '11px',
  fontFamily: 'var(--font)',
  cursor: 'pointer',
  marginBottom: '8px',
}

export default function HeatmapMultiPanel({ payload, accessToken }) {
  const { title, mode, panels, difference } = payload
  const [expanded, setExpanded] = useState(null)

  // Period mode: the single diverging difference map is already the
  // composite view -- nothing to compare side by side.
  if (mode === 'difference' && difference && Array.isArray(difference.lats)) {
    return (
      <MapLibreHeatmapPanel
        payload={{ ...difference, title: difference.title || title }}
        accessToken={accessToken}
      />
    )
  }

  if (!panels?.length) return null

  if (expanded !== null && panels[expanded]) {
    return (
      <div>
        <button type="button" onClick={() => setExpanded(null)} style={backButtonStyle}>
          ← Back to comparison
        </button>
        <MapLibreHeatmapPanel payload={panels[expanded]} accessToken={accessToken} />
      </div>
    )
  }

  return (
    <div>
      {title && (
        <div style={{ fontWeight: 500, fontSize: '13px', marginBottom: '8px', color: 'var(--text-primary)' }}>
          {title}
        </div>
      )}
      <div style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${Math.min(panels.length, 3)}, 1fr)`,
        gap: '12px',
      }}>
        {panels.map((panel, i) => (
          <PanelThumbnail key={i} panel={panel} onClick={() => setExpanded(i)} />
        ))}
      </div>
    </div>
  )
}
