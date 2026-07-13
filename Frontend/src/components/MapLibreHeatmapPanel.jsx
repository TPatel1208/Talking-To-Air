/**
 * MapLibreHeatmapPanel.jsx
 * -------------------------
 * Renders a single heatmap chart payload as a NASA-scientific interactive
 * map (T23): a shaded-relief terrain basemap, the data field as a
 * full-native-resolution server-rendered PNG overlay (GPU-bilinear
 * smoothed), region borders, per-cell hover, and a scientific colorbar.
 *
 * Visual fidelity (the overlay PNG) and interaction resolution (the
 * shipped lats/lons/values arrays) are deliberately decoupled -- hover and
 * stats read the arrays; only the picture comes from the server render.
 * Degrades instead of dying: overlay missing/failed -> client canvas from
 * the arrays; basemap/terrain tiles failed -> flat fill, overlay still
 * shows.
 */
import { useEffect, useRef, useState } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { nearestCell } from '../utils/heatmapHover.js'
import { colorbarGeometry } from '../utils/colorbarGeometry.js'
import { buildCanvasFallbackFrame } from '../utils/canvasFallback.js'
import { fetchUsStatesGeoJSON, isConusBounds } from '../utils/regionBorders.js'

const FALLBACK_TILE_CONFIG = {
  basemap_light_url: 'https://basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png',
  basemap_dark_url: 'https://basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}.png',
  terrain_dem_url: 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png',
  basemap_attribution: '© CARTO © OpenStreetMap contributors',
  terrain_attribution: 'Terrain tiles: AWS Terrain Tiles',
}

let _tileConfigPromise = null
function fetchTileConfig() {
  if (_tileConfigPromise) return _tileConfigPromise
  _tileConfigPromise = fetch('/api/config/map-tiles')
    .then(r => (r.ok ? r.json() : null))
    .then(config => ({ ...FALLBACK_TILE_CONFIG, ...(config || {}) }))
    .catch(() => FALLBACK_TILE_CONFIG)
  return _tileConfigPromise
}

function buildMapStyle(tileConfig) {
  return {
    version: 8,
    sources: {
      basemap: {
        type: 'raster',
        tiles: [tileConfig.basemap_light_url],
        tileSize: 256,
        attribution: tileConfig.basemap_attribution,
      },
      'terrain-dem': {
        type: 'raster-dem',
        tiles: [tileConfig.terrain_dem_url],
        tileSize: 256,
        encoding: 'terrarium',
        attribution: tileConfig.terrain_attribution,
      },
    },
    layers: [
      // Flat fill underneath everything -- the degrade-don't-die fallback
      // when basemap/terrain tiles fail to load (T17 posture).
      { id: 'background', type: 'background', paint: { 'background-color': '#e4e1d8' } },
      { id: 'basemap', type: 'raster', source: 'basemap', paint: { 'raster-fade-duration': 0 } },
      {
        id: 'hillshade',
        type: 'hillshade',
        source: 'terrain-dem',
        paint: { 'hillshade-shadow-color': '#3b3b3b', 'hillshade-exaggeration': 0.35 },
      },
    ],
  }
}

// A request is "ours" (needs the bearer token) only when it targets our own
// backend's chart overlay route -- never the third-party CARTO/terrarium
// tile hosts, which must never see the user's access token.
function isOwnOverlayRequest(url) {
  try {
    return new URL(url, window.location.origin).pathname.startsWith('/api/chart/')
  } catch {
    return false
  }
}

function overlayCornersFromBounds(bounds) {
  const [minx, miny, maxx, maxy] = bounds
  return [
    [minx, maxy], // top-left
    [maxx, maxy], // top-right
    [maxx, miny], // bottom-right
    [minx, miny], // bottom-left
  ]
}

// Canvas fallback coordinates follow the *array's own* row/column
// direction (lats[0]/lons[0] is whatever the source data's first row/col
// is -- not assumed north-up), unlike the server PNG which is always
// pre-flipped to north-up before reprojection.
function canvasCornersFromArrays(lats, lons) {
  const firstLat = lats[0], lastLat = lats[lats.length - 1]
  const firstLon = lons[0], lastLon = lons[lons.length - 1]
  return [
    [firstLon, firstLat],
    [lastLon, firstLat],
    [lastLon, lastLat],
    [firstLon, lastLat],
  ]
}

function addCanvasFallbackOverlay(map, payload) {
  const { lats, lons, values, vmin, vmax, colormap } = payload
  if (!Array.isArray(lats) || !Array.isArray(lons) || !Array.isArray(values)) return

  const frame = buildCanvasFallbackFrame({ lats, lons, values, vmin, vmax, lut: colormap?.lut })
  if (!frame.width || !frame.height) return

  const canvas = document.createElement('canvas')
  canvas.width = frame.width
  canvas.height = frame.height
  const ctx = canvas.getContext('2d')
  ctx.putImageData(new ImageData(frame.pixels, frame.width, frame.height), 0, 0)

  if (map.getLayer('overlay')) map.removeLayer('overlay')
  if (map.getSource('overlay')) map.removeSource('overlay')
  if (map.getSource('overlay-canvas')) map.removeSource('overlay-canvas')

  map.addSource('overlay-canvas', {
    type: 'canvas',
    canvas,
    coordinates: canvasCornersFromArrays(lats, lons),
    animate: false,
  })
  map.addLayer({
    id: 'overlay',
    type: 'raster',
    source: 'overlay-canvas',
    paint: { 'raster-resampling': 'nearest', 'raster-fade-duration': 0 },
  })
}

function addBorderLayer(map, geojson) {
  if (!geojson) return
  if (map.getSource('region-borders')) return
  map.addSource('region-borders', { type: 'geojson', data: geojson })
  map.addLayer({
    id: 'region-borders',
    type: 'line',
    source: 'region-borders',
    paint: { 'line-color': 'rgba(30,30,30,0.85)', 'line-width': 1.1 },
  })
}

export default function MapLibreHeatmapPanel({ payload, height = 420, accessToken, colorScaleOverride = null, hideLegend = false }) {
  const { title, variable, units, vmin, vmax, colormap, overlay, bounds, lats, lons } = payload
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const [hover, setHover] = useState(null)

  // An externally supplied vmin/vmax/colormap wins over the payload's own --
  // this is the hook compare mode's shared-scale logic (T28) uses to recolor
  // every panel onto one range. The server-rendered overlay PNG is baked
  // with the payload's native scale and can't be recolored client-side, so
  // an override always forces the canvas-fallback path (rendered from the
  // same lats/lons/values arrays hover already reads).
  const effectiveVmin = colorScaleOverride?.vmin ?? vmin
  const effectiveVmax = colorScaleOverride?.vmax ?? vmax
  const effectiveColormap = colorScaleOverride?.colormap ?? colormap

  const overrideRef = useRef(colorScaleOverride)
  useEffect(() => { overrideRef.current = colorScaleOverride }, [colorScaleOverride])

  const [minx, miny, maxx, maxy] = bounds || [
    Math.min(...(lons || [])), Math.min(...(lats || [])),
    Math.max(...(lons || [])), Math.max(...(lats || [])),
  ]

  useEffect(() => {
    let cancelled = false
    let map = null
    let basemapFailed = false
    let terrainFailed = false

    // Renders the data overlay using whatever scale is current at call
    // time (overrideRef, not the closed-over prop) -- either the server PNG
    // (native scale, no override active) or a canvas frame built from the
    // payload's arrays at the effective vmin/vmax/colormap.
    const drawOverlay = (map) => {
      const override = overrideRef.current
      if (!override && overlay?.url) {
        if (map.getLayer('overlay')) map.removeLayer('overlay')
        if (map.getSource('overlay-canvas')) map.removeSource('overlay-canvas')
        if (!map.getSource('overlay')) {
          map.addSource('overlay', {
            type: 'image',
            url: `/api${overlay.url}`,
            coordinates: overlayCornersFromBounds(overlay.bounds || bounds),
          })
        }
        if (!map.getLayer('overlay')) {
          map.addLayer({
            id: 'overlay',
            type: 'raster',
            source: 'overlay',
            paint: { 'raster-resampling': 'linear', 'raster-fade-duration': 0 },
          })
        }
        return
      }
      addCanvasFallbackOverlay(map, {
        ...payload,
        vmin: override?.vmin ?? vmin,
        vmax: override?.vmax ?? vmax,
        colormap: override?.colormap ?? colormap,
      })
    }

    fetchTileConfig().then(tileConfig => {
      if (cancelled || !containerRef.current) return

      map = new maplibregl.Map({
        container: containerRef.current,
        style: buildMapStyle(tileConfig),
        attributionControl: false,
        transformRequest: (url) => {
          if (isOwnOverlayRequest(url) && accessToken) {
            return { url, headers: { Authorization: `Bearer ${accessToken}` } }
          }
          return { url }
        },
      })
      mapRef.current = map

      map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-left')
      map.addControl(new maplibregl.AttributionControl({ compact: true }))

      map.on('error', (e) => {
        const sourceId = e?.sourceId
        if (sourceId === 'basemap' && !basemapFailed) {
          basemapFailed = true
          if (map.getLayer('basemap')) map.setLayoutProperty('basemap', 'visibility', 'none')
        } else if (sourceId === 'terrain-dem' && !terrainFailed) {
          terrainFailed = true
          if (map.getLayer('hillshade')) map.setLayoutProperty('hillshade', 'visibility', 'none')
        } else if (sourceId === 'overlay') {
          const override = overrideRef.current
          addCanvasFallbackOverlay(map, {
            ...payload,
            vmin: override?.vmin ?? vmin,
            vmax: override?.vmax ?? vmax,
            colormap: override?.colormap ?? colormap,
          })
        }
      })

      map.on('load', () => {
        if (cancelled) return

        // Fit bounds after load (not via constructor options) -- at
        // construction time the container may not have its final layout
        // size yet, which throws off the initial zoom level.
        map.resize()
        map.fitBounds([[minx, miny], [maxx, maxy]], { padding: 24, animate: false })

        drawOverlay(map)

        if (isConusBounds(minx, miny, maxx, maxy)) {
          fetchUsStatesGeoJSON().then(geojson => {
            if (!cancelled) addBorderLayer(map, geojson)
          })
        }
      })

      map.on('mousemove', (e) => {
        const cell = nearestCell(e.lngLat.lng, e.lngLat.lat, payload)
        if (cell) setHover({ ...cell, x: e.point.x, y: e.point.y })
      })
      map.on('mouseleave', () => setHover(null))
    })

    return () => {
      cancelled = true
      if (map) map.remove()
      mapRef.current = null
    }
    // Rebuild only when the payload identity actually changes -- bounds/
    // overlay/lats/lons all derive from `payload`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [payload])

  // Recolor in place when the override changes after the map has already
  // loaded (e.g. compare mode's "auto-scale each" toggle) -- no need to
  // rebuild the whole WebGL map just to switch the overlay's color scale.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !map.isStyleLoaded()) return
    const override = colorScaleOverride
    if (!override && overlay?.url) return // native PNG already showing, nothing to redraw
    addCanvasFallbackOverlay(map, {
      ...payload,
      vmin: override?.vmin ?? vmin,
      vmax: override?.vmax ?? vmax,
      colormap: override?.colormap ?? colormap,
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [colorScaleOverride])

  const { gradientStops, ticks } = colorbarGeometry({ vmin: effectiveVmin, vmax: effectiveVmax, lut: effectiveColormap?.lut, tickCount: 5 })

  return (
    <div style={{ width: '100%' }}>
      {title && (
        <div style={{ fontWeight: 500, fontSize: '13px', marginBottom: '8px', color: 'var(--text-primary)' }}>
          {title}
        </div>
      )}
      <div style={{ position: 'relative', width: '100%', height, borderRadius: '8px', overflow: 'hidden' }}>
        <div ref={containerRef} style={{ width: '100%', height: '100%' }} />

        {hover && (
          <div
            style={{
              position: 'absolute',
              left: Math.min(hover.x + 12, (containerRef.current?.clientWidth || 400) - 160),
              top: hover.y + 12,
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              borderRadius: '6px',
              padding: '6px 8px',
              fontSize: '11px',
              lineHeight: 1.4,
              color: 'var(--text-secondary)',
              pointerEvents: 'none',
              boxShadow: 'var(--shadow-sm)',
              zIndex: 2,
            }}
          >
            <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
              {variable}: {Number.isFinite(hover.value) ? hover.value.toExponential(3) : '—'} {units}
            </div>
            <div>Lat: {hover.lat.toFixed(3)}, Lon: {hover.lon.toFixed(3)}</div>
          </div>
        )}

        {!hideLegend && <MapColorbar gradientStops={gradientStops} ticks={ticks} units={units} />}
      </div>
    </div>
  )
}

function MapColorbar({ gradientStops, ticks, units }) {
  if (!gradientStops.length) return null
  const gradientId = 'tta-colorbar-gradient'
  return (
    <div
      style={{
        position: 'absolute',
        left: 12,
        bottom: 12,
        background: 'rgba(255,255,255,0.92)',
        borderRadius: '6px',
        padding: '6px 10px 8px',
        boxShadow: 'var(--shadow-sm)',
        zIndex: 1,
      }}
    >
      <svg width={180} height={14} style={{ display: 'block' }}>
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="1" y2="0">
            {gradientStops.map((stop, i) => (
              <stop key={i} offset={stop.offset} stopColor={stop.color} />
            ))}
          </linearGradient>
        </defs>
        <rect x={0} y={0} width={180} height={14} fill={`url(#${gradientId})`} rx={2} />
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', width: 180, marginTop: 3 }}>
        {ticks.map((tick, i) => (
          <span key={i} style={{ fontSize: '9px', color: '#333', fontFamily: 'var(--font)' }}>
            {tick.value.toExponential(1)}
          </span>
        ))}
      </div>
      {units && (
        <div style={{ fontSize: '9px', color: '#555', textAlign: 'center', marginTop: 2 }}>{units}</div>
      )}
    </div>
  )
}
