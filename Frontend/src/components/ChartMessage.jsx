/**
 * ChartMessage.jsx
 * ----------------
 * Renders interactive Plotly charts from chart payloads emitted by the backend.
 *
 * Heatmap architecture
 * --------------------
 * Uses a scattergeo trace with square markers colored by value. Each grid cell
 * becomes one marker point. This is the correct Plotly pattern for gridded
 * geo data — it renders on top of the basemap borders natively, supports
 * zoom/pan, and shows hover tooltips per cell.
 */
import { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import Plotly from 'plotly.js-dist-min'
import _createPlotlyComponent from 'react-plotly.js/factory'

const createPlotlyComponent =
  typeof _createPlotlyComponent === 'function'
    ? _createPlotlyComponent
    : _createPlotlyComponent.default

const Plot = createPlotlyComponent(Plotly)

// ── Colorscale helpers ────────────────────────────────────────────────────────
const CMAP_MAP = {
  Spectral_r: 'RdYlGn',
  Spectral:   'RdYlGn_r',
  viridis:    'Viridis',
  plasma:     'Plasma',
  YlOrRd:     'YlOrRd',
  RdYlBu_r:   'RdYlBu',
  hot:        'Hot',
  cool:       'Blues',
}
const toPlotlyCmap = (cmap) => CMAP_MAP[cmap] || 'RdYlGn'

// Colorscale stop tables — [position 0-1, [r, g, b]]
const SCALES = {
  RdYlGn:   [[0,[215,25,28]],[0.25,[253,174,97]],[0.5,[255,255,191]],[0.75,[166,217,106]],[1,[26,150,65]]],
  RdYlGn_r: [[0,[26,150,65]],[0.25,[166,217,106]],[0.5,[255,255,191]],[0.75,[253,174,97]],[1,[215,25,28]]],
  Viridis:  [[0,[68,1,84]],[0.25,[59,82,139]],[0.5,[33,145,140]],[0.75,[94,201,98]],[1,[253,231,37]]],
  Plasma:   [[0,[13,8,135]],[0.25,[126,3,168]],[0.5,[204,71,120]],[0.75,[248,149,64]],[1,[240,249,33]]],
  YlOrRd:   [[0,[255,255,178]],[0.25,[254,204,92]],[0.5,[253,141,60]],[0.75,[240,59,32]],[1,[189,0,38]]],
  RdYlBu:   [[0,[215,25,28]],[0.25,[253,174,97]],[0.5,[255,255,191]],[0.75,[171,217,233]],[1,[44,123,182]]],
  Hot:      [[0,[0,0,0]],[0.33,[255,0,0]],[0.66,[255,200,0]],[1,[255,255,255]]],
  Blues:    [[0,[247,251,255]],[0.5,[107,174,214]],[1,[8,48,107]]],
}

function getColorscale(scaleName) {
  const stops = SCALES[scaleName] || SCALES.RdYlGn
  return stops.map(([pos, [r, g, b]]) => [pos, `rgb(${r},${g},${b})`])
}

// ── Flatten 2D grid → parallel lat/lon/value arrays ──────────────────────────
// Skips null cells. Optionally downsamples to MAX_POINTS for performance.
const MAX_POINTS = 8000
function flattenGrid(lats, lons, values) {
  const flatLat = [], flatLon = [], flatVal = []
  for (let ri = 0; ri < lats.length; ri++) {
    for (let ci = 0; ci < lons.length; ci++) {
      const v = values[ri][ci]
      if (v != null) {
        flatLat.push(lats[ri])
        flatLon.push(lons[ci])
        flatVal.push(v)
      }
    }
  }
  // Downsample evenly if too many points
  if (flatLat.length > MAX_POINTS) {
    const step = Math.ceil(flatLat.length / MAX_POINTS)
    return {
      lat: flatLat.filter((_, i) => i % step === 0),
      lon: flatLon.filter((_, i) => i % step === 0),
      val: flatVal.filter((_, i) => i % step === 0),
    }
  }
  return { lat: flatLat, lon: flatLon, val: flatVal }
}

// Compute marker size so each square fills one grid cell at the *current* zoom.
//
// Arguments
//   cellDegLon  — angular width of one grid cell in degrees longitude
//   lonSpanShown — total degrees of longitude currently visible in the map
//   mapWidthPx   — pixel width of the geo subplot (container width minus margins)
//
// scattergeo marker 'size' is in screen pixels, fixed at layout time.  We
// recompute and restyle whenever the viewport lon range changes (zoom/pan) so
// the squares always cover exactly their cell — no gaps, no overlap.
// fill_factor 0.90 leaves a hairline between adjacent cells.
function computeMarkerSize(cellDegLon, lonSpanShown, mapWidthPx) {
  if (cellDegLon <= 0 || lonSpanShown <= 0 || mapWidthPx <= 0) return 4
  const pxPerDeg = mapWidthPx / lonSpanShown
  const raw      = cellDegLon * pxPerDeg * 0.90
  return Math.max(1, Math.min(60, raw))   // float — Plotly accepts it
}

// ── Shared config ─────────────────────────────────────────────────────────────
const BASE_CONFIG = {
  displayModeBar: true,
  modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
  displaylogo:    false,
  responsive:     true,
  toImageButtonOptions: { format: 'png', scale: 2 },
}

// ── Geo layout ────────────────────────────────────────────────────────────────
function makeGeoLayout(minx, miny, maxx, maxy, pad = 0.5) {
  return {
    scope:          'world',
    resolution:     50,
    projection:     { type: 'mercator' },
    lonaxis:        { range: [minx - pad, maxx + pad] },
    lataxis:        { range: [miny - pad, maxy + pad] },
    showland:       true,
    landcolor:      'rgba(220,216,208,1)',
    showocean:      true,
    oceancolor:     'rgba(210,228,248,1)',
    showlakes:      true,
    lakecolor:      'rgba(210,228,248,1)',
    showrivers:     false,
    showcoastlines: true,
    coastlinecolor: 'rgba(20,20,20,0.9)',
    coastlinewidth: 1.2,
    showcountries:  true,
    countrycolor:   'rgba(20,20,20,0.85)',
    countrywidth:   1.2,
    showsubunits:   false,   // we draw state lines ourselves as scattergeo traces on top
    bgcolor:        'rgba(0,0,0,0)',
    framewidth:     0,
  }
}

// ── Border overlay traces ─────────────────────────────────────────────────────
// Converts a GeoJSON FeatureCollection into an array of scattergeo line traces.
// Each polygon ring becomes a separate lat/lon path separated by nulls so
// Plotly draws them as disconnected segments in one trace (fewer DOM nodes).
function geojsonToScattergeo(geojson, color, width, name) {
  if (!geojson?.features?.length) return null
  const lats = [], lons = []

  for (const feature of geojson.features) {
    const geom = feature.geometry
    if (!geom) continue
    const polys = geom.type === 'Polygon'
      ? [geom.coordinates]
      : geom.type === 'MultiPolygon'
        ? geom.coordinates
        : []
    for (const poly of polys) {
      for (const ring of poly) {
        for (const [lon, lat] of ring) { lons.push(lon); lats.push(lat) }
        lons.push(null); lats.push(null)  // pen-up between rings
      }
    }
  }

  return {
    type:       'scattergeo',
    mode:       'lines',
    lat:        lats,
    lon:        lons,
    hoverinfo:  'none',
    showlegend: false,
    line:       { color, width },
    name,
  }
}

// Cached GeoJSON promise so we only fetch once per session
// Returns true when the bounding box is substantially over the continental US.
// Used to skip the state-border fetch for non-CONUS maps (global, Europe, etc.)
// and avoid a pointless cross-origin request that can cause a noticeable flash.
function isCONUS(minx, miny, maxx, maxy) {
  // CONUS rough bounds: lon -130..-65, lat 24..50
  const lonOverlap = Math.min(maxx, -65) - Math.max(minx, -130)
  const latOverlap = Math.min(maxy,  50) - Math.max(miny,  24)
  const mapArea    = (maxx - minx) * (maxy - miny)
  if (mapArea <= 0) return false
  const overlap = Math.max(0, lonOverlap) * Math.max(0, latOverlap)
  return overlap / mapArea > 0.3   // >30% of the map must be within CONUS
}

// Cached GeoJSON promise so we only fetch once per session
let _bordersPromise = null
function fetchBorders() {
  if (_bordersPromise) return _bordersPromise
  // US states from a reliable public CDN (Natural Earth via PublicaMundi)
  const statesUrl = 'https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json'
  // Fix #3: add a 5-second timeout so a slow/failed CDN response never hangs the chart
  const timeout = new Promise(resolve => setTimeout(() => resolve(null), 5000))
  _bordersPromise = Promise.race([
    fetch(statesUrl).then(r => r.ok ? r.json() : null).catch(() => null),
    timeout,
  ])
  return _bordersPromise
}

// Hook: returns border scattergeo traces once loaded, [] while loading or when
// the map extent is outside CONUS (fix #3 -- skip fetch for non-US maps).
function useBorderTraces(minx, miny, maxx, maxy) {
  const [traces, setTraces] = useState([])
  useEffect(() => {
    if (!isCONUS(minx, miny, maxx, maxy)) return   // not over the US -- skip
    fetchBorders().then(states => {
      const result = []
      const st = geojsonToScattergeo(states, 'rgba(30,30,30,0.85)', 1.1, 'states')
      if (st) result.push(st)
      setTraces(result)
    })
  }, [minx, miny, maxx, maxy])
  return traces
}

// ── Heatmap panel ─────────────────────────────────────────────────────────────
function HeatmapPanel({ payload, height = 420 }) {
  const { title, variable, units, lats, lons, values, vmin, vmax, cmap, bounds } = payload

  const [minx, miny, maxx, maxy] = bounds || [
    Math.min(...lons), Math.min(...lats),
    Math.max(...lons), Math.max(...lats),
  ]
  const pad = 0.5

  // ── Refs ──────────────────────────────────────────────────────────────────
  // plotDivRef  — the raw DOM <div> that Plotly owns (passed via onInitialized)
  // containerRef — our wrapper div, used only for ResizeObserver
  const plotDivRef   = useRef(null)
  const containerRef = useRef(null)

  // Pre-compute the cell angular width once (doesn't change with zoom)
  const cellDegLon = lons.length > 1
    ? Math.abs(lons[lons.length - 1] - lons[0]) / (lons.length - 1)
    : 1

  // ── Core: recompute + restyle marker size given current lon viewport ───────
  // Called on mount (after Plotly initialises) and on every relayout (zoom/pan).
  // Uses Plotly.restyle — O(1), no React re-render, no flicker.
  const MARGIN_PX = 90   // colorbar + right-margin pixel budget
  const restyleMarkers = useCallback((plotDiv, currentLonSpan) => {
    if (!plotDiv) return
    const mapPx = Math.max((plotDiv.clientWidth || 500) - MARGIN_PX, 80)
    const sz    = computeMarkerSize(cellDegLon, currentLonSpan, mapPx)
    Plotly.restyle(plotDiv, { 'marker.size': sz }, [0])
  }, [cellDegLon])

  // Current lon span — starts at the initial view, updated on relayout
  const lonSpanRef = useRef((maxx - minx) + 2 * pad)

  // ── After Plotly first renders, fire an initial restyle ───────────────────
  const handleInitialized = useCallback((figure, plotDiv) => {
    plotDivRef.current = plotDiv
    restyleMarkers(plotDiv, lonSpanRef.current)
  }, [restyleMarkers])

  // ── On every zoom / pan, extract new lon range and restyle ────────────────
  const handleRelayout = useCallback((eventData) => {
    const lo = eventData['geo.lonaxis.range[0]']
    const hi = eventData['geo.lonaxis.range[1]']
    if (lo != null && hi != null) {
      lonSpanRef.current = Math.abs(hi - lo)
    }
    // Also fires on autosize / reset — always restyle with current span
    restyleMarkers(plotDivRef.current, lonSpanRef.current)
  }, [restyleMarkers])

  // ── On container resize, restyle (map px width changed) ───────────────────
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      restyleMarkers(plotDivRef.current, lonSpanRef.current)
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [restyleMarkers])

  const [revision, setRevision] = useState(0)
  useEffect(() => {
    requestAnimationFrame(() => setRevision(r => r + 1))
  }, [])

  const borderTraces = useBorderTraces(minx, miny, maxx, maxy)

  const scaleName  = toPlotlyCmap(cmap)
  const colorscale = getColorscale(scaleName)
  const { lat, lon, val } = flattenGrid(lats, lons, values)

  // Initial marker size — will be immediately overridden by handleInitialized,
  // but set to something reasonable so the first paint isn't obviously broken.
  const initMarkerSize = computeMarkerSize(
    cellDegLon, (maxx - minx) + 2 * pad, 400
  )

  const hoverText = val.map((v, i) =>
    `${variable}: ${v.toExponential(3)} ${units}<br>Lat: ${lat[i].toFixed(3)}<br>Lon: ${lon[i].toFixed(3)}`
  )

  const data = [
    {
      type:       'scattergeo',
      mode:       'markers',
      lat,
      lon,
      text:       hoverText,
      hoverinfo:  'text',
      showlegend: false,
      marker: {
        symbol:     'square',
        size:       initMarkerSize,
        color:      val,
        cmin:       vmin,
        cmax:       vmax,
        colorscale,
        showscale:  true,
        colorbar: {
          title:          { text: units, side: 'right', font: { size: 10 } },
          thickness:      14,
          outlinewidth:   0,
          tickfont:       { size: 10 },
          exponentformat: 'e',
        },
        opacity: 0.92,
      },
    },
    ...borderTraces,
  ]

  const layout = {
    paper_bgcolor: 'transparent',
    font:          { family: "'DM Sans', system-ui, sans-serif", size: 12, color: '#2c2a28' },
    margin:        { t: 40, r: 80, b: 10, l: 10 },
    height,
    title:         { text: title, font: { size: 13, weight: 500 }, x: 0.5, xanchor: 'center' },
    geo:           makeGeoLayout(minx, miny, maxx, maxy, pad),
    datarevision:  revision,
  }

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <Plot
        data={data}
        layout={layout}
        config={BASE_CONFIG}
        revision={revision}
        style={{ width: '100%' }}
        useResizeHandler
        onInitialized={handleInitialized}
        onRelayout={handleRelayout}
      />
    </div>
  )
}

// ── Multi-panel heatmap ───────────────────────────────────────────────────────
function HeatmapMultiPanel({ payload }) {
  const { panels, title } = payload
  if (!panels?.length) return null
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
          <div key={i} style={{
            background: 'var(--bg-card)', border: '1px solid var(--border)',
            borderRadius: '10px', overflow: 'hidden', padding: '8px',
          }}>
            <HeatmapPanel payload={panel} height={300} />
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Time-series ───────────────────────────────────────────────────────────────
function TimeSeriesPanel({ payload }) {
  const { title, units, stat, times, values } = payload

  const [revision, setRevision] = useState(0)
  const mounted = useRef(false)
  useEffect(() => {
    if (!mounted.current) { mounted.current = true; requestAnimationFrame(() => setRevision(r => r + 1)) }
  }, [])

  const data = [{
    type: 'scatter', mode: 'lines+markers',
    x: times, y: values,
    line:          { color: '#1D9E75', width: 2 },
    marker:        { color: '#1D9E75', size: 5 },
    fill:          'tozeroy', fillcolor: 'rgba(29,158,117,0.08)',
    hovertemplate: `%{x|%Y-%m-%d %H:%M}<br>${stat}: %{y:.3e} ${units}<extra></extra>`,
  }]

  const layout = {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font:   { family: "'DM Sans', system-ui, sans-serif", size: 12, color: '#2c2a28' },
    margin: { t: 40, r: 16, b: 40, l: 16 },
    title:  { text: title, font: { size: 13, weight: 500 }, x: 0.5, xanchor: 'center' },
    height: 300,
    xaxis:  { title: 'Time', showgrid: true, gridcolor: 'rgba(0,0,0,0.06)', zeroline: false, tickfont: { size: 10 } },
    yaxis:  { title: `${stat} (${units})`, showgrid: true, gridcolor: 'rgba(0,0,0,0.06)', zeroline: false, tickfont: { size: 10 }, exponentformat: 'e' },
    datarevision: revision,
  }

  return (
    <Plot data={data} layout={layout} config={BASE_CONFIG} revision={revision}
      style={{ width: '100%' }} useResizeHandler />
  )
}

// ── Public component ──────────────────────────────────────────────────────────
export default function ChartMessage({ chart }) {
  if (!chart || typeof chart !== 'object' || !chart.type) return null

  const inner = (() => {
    switch (chart.type) {
      case 'heatmap':       return <HeatmapPanel payload={chart} />
      case 'heatmap_multi': return <HeatmapMultiPanel payload={chart} />
      case 'timeseries':    return <TimeSeriesPanel payload={chart} />
      default:
        return <div style={{ fontSize: '12px', color: 'var(--text-muted)', padding: '8px' }}>Unknown chart type: {chart.type}</div>
    }
  })()

  return (
    <div style={{
      margin: '8px 0', background: 'var(--bg-card)',
      border: '1px solid var(--border)', borderRadius: '10px',
      overflow: 'hidden', padding: '8px',
    }}>
      {inner}
    </div>
  )
}