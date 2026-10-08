/**
 * HeatmapMultiPanel.jsx
 * ----------------------
 * Comparison charts (T08/T23): region mode shows each panel as a static
 * small-multiple (its own server-rendered overlay PNG, or a canvas-fallback
 * thumbnail) with state outlines drawn over it and a colorbar -- one shared
 * bar when every panel was drawn on the same scale -- and no basemap; click
 * one to expand it into the single interactive MapLibreHeatmapPanel. Period mode's
 * diverging difference map IS the composite view, so it renders directly
 * as one interactive map. At most one live WebGL context at a time.
 */
import { useEffect, useId, useMemo, useRef, useState } from 'react'
import MapLibreHeatmapPanel from './MapLibreHeatmapPanel.jsx'
import { buildCanvasFallbackFrame } from '../utils/canvasFallback.js'
import { colorbarGeometry } from '../utils/colorbarGeometry.js'
import { resolveThumbnailMode } from '../utils/overlayMode.js'
import { BORDER_LINE, fetchUsStatesGeoJSON, isConusBounds } from '../utils/regionBorders.js'
import { thumbnailBorders, thumbnailFrame } from '../utils/thumbnailGeometry.js'
import { thumbnailLegends } from '../utils/thumbnailLegend.js'
import { fitThumbnails } from '../utils/thumbnailLayout.js'
import { loadThumbnail } from '../utils/thumbnailSource.js'
import { apiFetch } from '../utils/apiFetch.js'
import { API_BASE } from '../config.js'

// The canvas paints rows and columns in the payload's own order (lats ascend
// in every stored payload), so the frame's flips are what make it north-up.
function ThumbnailCanvas({ lats, lons, values, vmin, vmax, lut, flipX, flipY }) {
  const ref = useRef(null)
  useEffect(() => {
    if (!ref.current || !Array.isArray(lats) || !Array.isArray(lons) || !Array.isArray(values)) return
    const frame = buildCanvasFallbackFrame({ lats, lons, values, vmin, vmax, lut })
    if (!frame.width || !frame.height) return
    ref.current.width = frame.width
    ref.current.height = frame.height
    ref.current.getContext('2d').putImageData(new ImageData(frame.pixels, frame.width, frame.height), 0, 0)
  }, [lats, lons, values, vmin, vmax, lut])
  return (
    <canvas
      ref={ref}
      style={{
        width: '100%', height: '100%', display: 'block',
        transform: `scale(${flipX ? -1 : 1}, ${flipY ? -1 : 1})`,
      }}
    />
  )
}

// US state outlines, fetched once per session (regionBorders.js) and only when
// some panel is over CONUS. Null until it arrives, and if the fetch fails.
function useStateBorders(enabled) {
  const [geojson, setGeojson] = useState(null)
  useEffect(() => {
    if (!enabled) return undefined
    let cancelled = false
    fetchUsStatesGeoJSON().then((data) => { if (!cancelled) setGeojson(data) })
    return () => { cancelled = true }
  }, [enabled])
  return enabled ? geojson : null
}

// Drawn over the tile in the same 0..100 box the image fills, styled like the
// expanded map's border layer. non-scaling-stroke keeps the width in screen
// pixels, as MapLibre's line-width is, however the box is stretched.
function BorderOverlay({ d }) {
  if (!d) return null
  return (
    <svg
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
    >
      <path d={d} fill="none" stroke={BORDER_LINE.haloColor} strokeWidth={BORDER_LINE.haloWidth} vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
      <path d={d} fill="none" stroke={BORDER_LINE.color} strokeWidth={BORDER_LINE.width} vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
    </svg>
  )
}

function ThumbnailColorbar({ legend, compact = false }) {
  // useId keeps one tile's gradient from painting another's: a fixed id is
  // shared page-wide, and these legends can differ.
  const gradientId = `tta-thumb-gradient-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`
  const { gradientStops, ticks } = colorbarGeometry({ vmin: legend.vmin, vmax: legend.vmax, lut: legend.lut, tickCount: compact ? 3 : 5 })
  if (!gradientStops.length) return null
  const tickStyle = { fontSize: compact ? '9px' : '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }
  return (
    <div style={{ width: '100%', maxWidth: compact ? undefined : '320px' }}>
      <svg viewBox="0 0 100 10" preserveAspectRatio="none" style={{ display: 'block', width: '100%', height: compact ? 8 : 12 }}>
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="1" y2="0">
            {gradientStops.map((stop, i) => <stop key={i} offset={stop.offset} stopColor={stop.color} />)}
          </linearGradient>
        </defs>
        <rect x={0} y={0} width={100} height={10} fill={`url(#${gradientId})`} rx={1} />
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 2 }}>
        {ticks.map((tick, i) => <span key={i} style={tickStyle}>{tick.value.toExponential(1)}</span>)}
      </div>
      {(legend.units || legend.clipNote) && (
        <div style={{ ...tickStyle, fontFamily: 'var(--font)', marginTop: 1, whiteSpace: compact ? 'nowrap' : undefined, overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {[legend.units, legend.clipNote].filter(Boolean).join(' · ')}
        </div>
      )}
    </div>
  )
}

// The panel's server-rendered png as `{ url, objectUrl }` once loaded, else
// null. Fetched with auth into a blob because an <img> cannot send the bearer
// header; cleanup revokes the blob when the url changes or the panel unmounts.
function useThumbnailBlob(overlayUrl) {
  const [loaded, setLoaded] = useState(null)
  useEffect(() => {
    if (!overlayUrl) return undefined
    return loadThumbnail(`${API_BASE}${overlayUrl}`, {
      fetcher: apiFetch,
      onLoad: (objectUrl) => setLoaded({ url: overlayUrl, objectUrl }),
    })
  }, [overlayUrl])
  return loaded
}

// The caption block under each image has a fixed height so the layout can
// budget for it exactly; +2 for the tile's 1px border top and bottom.
const CAPTION = { plain: 28, withLegend: 74 }
const captionHeight = (hasLegend) => (hasLegend ? CAPTION.withLegend : CAPTION.plain) + 2
const TILE_GAP = 12

// The aspect the layout budgets for: the canvas's degree-space one, which is
// never narrower than the mercator png's, so a tile that falls back to the
// canvas cannot overflow its row.
const layoutAspect = (panel) => thumbnailFrame(panel, 'canvas').aspect

function PanelThumbnail({ panel, legend, statesGeojson, imageHeight, onClick }) {
  const { title, overlay, lats, lons, values, vmin, vmax, colormap } = panel
  const loaded = useThumbnailBlob(overlay?.url)
  const mode = resolveThumbnailMode(overlay?.url, loaded)
  // Sized to the image actually showing (a mercator png, or a degree-space
  // canvas), so the image fills the tile exactly and the outlines line up.
  const frame = useMemo(() => thumbnailFrame(panel, mode), [panel, mode])
  const borders = useMemo(() => thumbnailBorders(statesGeojson, frame), [statesGeojson, frame])

  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        display: 'block', width: imageHeight * frame.aspect + 2, flexShrink: 0, padding: 0, border: '1px solid var(--border)',
        borderRadius: '10px', overflow: 'hidden', background: 'var(--bg-card)', cursor: 'pointer', textAlign: 'left',
      }}
    >
      <div style={{ position: 'relative', width: '100%', height: imageHeight, background: '#e4e1d8', overflow: 'hidden' }}>
        {mode === 'native' ? (
          <img
            src={loaded.objectUrl}
            alt={title || 'comparison panel'}
            style={{ width: '100%', height: '100%', objectFit: 'fill', display: 'block' }}
          />
        ) : (
          <ThumbnailCanvas
            lats={lats} lons={lons} values={values} vmin={vmin} vmax={vmax} lut={colormap?.lut}
            flipX={frame.flipX} flipY={frame.flipY}
          />
        )}
        <BorderOverlay d={borders} />
      </div>
      <div style={{
        height: legend ? CAPTION.withLegend : CAPTION.plain, boxSizing: 'border-box', overflow: 'hidden',
        padding: '6px 8px', display: 'flex', flexDirection: 'column', gap: '4px',
      }}>
        {title && (
          <div style={{ fontSize: '11px', lineHeight: '16px', color: 'var(--text-secondary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {title}
          </div>
        )}
        {legend && <ThumbnailColorbar legend={legend} compact />}
      </div>
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

// The box the `fill` element -- the thumbnail grid, or a map -- may take: its
// own width, and the visible height of the scrolling output pane (the parent)
// less everything else this panel draws (title, legend, back button), so a
// comparison fits on one screen without scrolling. `view` names what is
// showing; a box measured for another view is never handed out. Null until
// measured.
function useFitBox(rootRef, fillRef, view) {
  const [box, setBox] = useState(null)
  useEffect(() => {
    const root = rootRef.current
    const fill = fillRef.current
    const pane = root?.parentElement
    if (!view || !root || !fill || !pane) return undefined
    const measure = () => {
      const style = getComputedStyle(pane)
      const paneHeight = pane.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom)
      const next = {
        view,
        width: fill.clientWidth,
        height: Math.floor(paneHeight - (root.offsetHeight - fill.offsetHeight)),
      }
      setBox((prev) => (prev && prev.view === view && prev.width === next.width && prev.height === next.height ? prev : next))
    }
    const observer = new ResizeObserver(measure)
    observer.observe(pane)
    observer.observe(root)
    return () => observer.disconnect()
  }, [rootRef, fillRef, view])
  return box?.view === view ? box : null
}

const MIN_MAP_HEIGHT = 240
const panelTitleStyle = { fontWeight: 500, fontSize: '13px', marginBottom: '8px', color: 'var(--text-primary)' }

// A single map sized to the space useFitBox measured. It mounts only once
// measured, so maplibre is built at its real size.
function FittedMap({ fillRef, box, payload }) {
  return (
    <div ref={fillRef} style={{ height: box ? Math.max(MIN_MAP_HEIGHT, box.height) : 0 }}>
      {box && <MapLibreHeatmapPanel payload={payload} height={Math.max(MIN_MAP_HEIGHT, box.height)} />}
    </div>
  )
}

export default function HeatmapMultiPanel({ payload }) {
  const { title, mode, panels, difference } = payload
  const [expanded, setExpanded] = useState(null)
  const legends = useMemo(() => thumbnailLegends(panels || []), [panels])
  const anyConus = (panels || []).some((panel) => {
    const b = panel.overlay?.bounds || panel.bounds
    return Array.isArray(b) && isConusBounds(...b)
  })
  const statesGeojson = useStateBorders(anyConus)
  const isDifference = mode === 'difference' && difference && Array.isArray(difference.lats)
  const expandedPanel = !isDifference && expanded !== null ? panels?.[expanded] ?? null : null
  // Titles are drawn here rather than by the map, so the measured fill is the
  // map alone. Memoized: the map rebuilds whenever its payload identity changes.
  const mapPayload = useMemo(() => {
    if (isDifference) return { ...difference, title: null }
    return expandedPanel ? { ...expandedPanel, title: null } : null
  }, [isDifference, difference, expandedPanel])
  const view = isDifference ? 'difference' : expandedPanel ? `panel-${expanded}` : panels?.length ? 'grid' : null
  const rootRef = useRef(null)
  const fillRef = useRef(null)
  const box = useFitBox(rootRef, fillRef, view)

  // Period mode: the single diverging difference map is already the
  // composite view -- nothing to compare side by side.
  if (isDifference) {
    const mapTitle = difference.title || title
    return (
      <div ref={rootRef}>
        {mapTitle && <div style={panelTitleStyle}>{mapTitle}</div>}
        <FittedMap fillRef={fillRef} box={box} payload={mapPayload} />
      </div>
    )
  }

  if (!panels?.length) return null

  if (expandedPanel) {
    return (
      <div ref={rootRef}>
        <button type="button" onClick={() => setExpanded(null)} style={backButtonStyle}>
          ← Back to comparison
        </button>
        {expandedPanel.title && <div style={panelTitleStyle}>{expandedPanel.title}</div>}
        <FittedMap fillRef={fillRef} box={box} payload={mapPayload} />
      </div>
    )
  }

  const layout = fitThumbnails({
    // Each tile's 1px border left and right, for the up-to-three in a row.
    width: (box?.width ?? 0) - 2 * Math.min(panels.length, 3),
    height: box?.height ?? 0,
    aspects: panels.map(layoutAspect),
    gap: TILE_GAP,
    labelHeight: captionHeight(legends.perPanel.some(Boolean)),
  })

  return (
    <div ref={rootRef}>
      {title && <div style={panelTitleStyle}>{title}</div>}
      {/* Hidden, not absent, until measured: the grid's width is what is
          measured, and a frame of floor-sized tiles would flash first. */}
      <div ref={fillRef} style={{ display: 'flex', flexDirection: 'column', gap: TILE_GAP, visibility: box ? 'visible' : 'hidden' }}>
        {layout.rows.map((row) => (
          <div key={row[0]} style={{ display: 'flex', gap: TILE_GAP, alignItems: 'flex-start' }}>
            {row.map((i) => (
              <PanelThumbnail
                key={i}
                panel={panels[i]}
                legend={legends.perPanel[i]}
                statesGeojson={statesGeojson}
                imageHeight={layout.tiles[i].height}
                onClick={() => setExpanded(i)}
              />
            ))}
          </div>
        ))}
      </div>
      {legends.shared && (
        <div style={{ marginTop: '10px' }}>
          <div style={{ fontSize: '10.5px', color: 'var(--text-muted)', marginBottom: '4px' }}>
            Shared color scale — colors are comparable across panels
          </div>
          <ThumbnailColorbar legend={legends.shared} />
        </div>
      )}
    </div>
  )
}
