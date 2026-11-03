import { useEffect, useRef, useState } from 'react'
import  * as maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { fitSmokeMap, smokeMapBounds, smokeMapStyle } from '../utils/hmsMap.js'
import { SMOKE_COLORS as COLORS, SMOKE_LEGEND, smokeMapPng } from '../utils/hmsMapExport.js'

export default function MapLibreSmokePanel({ payload }) {
  const container = useRef(null)
  const mapRef = useRef(null)
  const exportRef = useRef(null)
  const tileFailure = useRef(false)
  const [exportState, setExportState] = useState(null)
  const [failure, setFailure] = useState(null)
  useEffect(() => {
    let map
    let observer
    let loaded = false
    const bounds = smokeMapBounds(payload)
    function initialize() {
      try {
        map = new maplibregl.Map({
          container: container.current, center: [-100, 40], zoom: 2,
          renderWorldCopies: false,
          preserveDrawingBuffer: true,
          pixelRatio: Math.max(2, window.devicePixelRatio || 1),
          style: smokeMapStyle(),
        })
        mapRef.current = map
        tileFailure.current = false
        map.addControl(new maplibregl.NavigationControl())
        observer = new ResizeObserver(() => {
          map.resize()
          if (loaded) fitSmokeMap(map, bounds, container.current)
        })
        observer.observe(container.current)
        map.on('error', event => {
          if (event.sourceId === 'basemap') {
            tileFailure.current = true
            setFailure({ source: payload, message: 'Basemap unavailable. Reload the map before exporting PNG.' })
          }
        })
        // Do not wait for remote raster tiles before installing/fitting the smoke.
        map.once('style.load', () => {
          map.addSource('smoke', { type: 'geojson', data: payload.geojson })
          map.addLayer({ id: 'smoke-fill', type: 'fill', source: 'smoke',
            layout: { 'fill-sort-key': ['match', ['get', 'density'], 'light', 0, 'medium', 1, 'heavy', 2, 3] },
            paint: { 'fill-opacity': 0.5, 'fill-color': ['match', ['get', 'density'],
              'light', COLORS.light, 'medium', COLORS.medium, 'heavy', COLORS.heavy, COLORS.unknown] } })
          map.addLayer({ id: 'smoke-line', type: 'line', source: 'smoke',
            paint: { 'line-color': '#594c46', 'line-width': 0.6 } })
          loaded = true
          map.resize()
          fitSmokeMap(map, bounds, container.current)
          map.on('click', 'smoke-fill', event => {
            const properties = event.features?.[0]?.properties || {}
            new maplibregl.Popup().setLngLat(event.lngLat).setText(
              `${properties.density} smoke | ${properties.satellite} | ${properties.start} – ${properties.end}`,
            ).addTo(map)
          })
        })
      } catch (error) { setFailure({ source: payload, message: `Map unavailable: ${error.message}` }) }
    }
    initialize()
    return () => {
      exportRef.current?.abort()
      exportRef.current = null
      mapRef.current = null
      observer?.disconnect()
      map?.remove()
    }
  }, [payload])

  async function download() {
    const map = mapRef.current
    if (exportRef.current) return
    const controller = new AbortController()
    exportRef.current = controller
    setExportState({ source: payload, busy: true })
    try {
      if (!map) throw new Error('The map is not ready yet.')
      if (tileFailure.current) throw new Error('Basemap tiles failed to load. Reload the map before exporting PNG.')
      const blob = await smokeMapPng(map, { signal: controller.signal })
      if (controller.signal.aborted) return
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `hms-smoke-${payload.date}.png`
      link.click()
      setTimeout(() => URL.revokeObjectURL(url), 1000)
      setExportState({ source: payload, busy: false })
    } catch (error) {
      if (!controller.signal.aborted) setExportState({ source: payload, busy: false, error: error.message })
    } finally {
      if (exportRef.current === controller) exportRef.current = null
    }
  }

  return <section>
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, marginBottom: 12 }}>
      {SMOKE_LEGEND.map(({ key, label, color }) => <span key={key}>
        <span style={{ display: 'inline-block', width: 12, height: 12, background: color, marginRight: 5 }} />
        {label}
      </span>)}
      <button type="button" onClick={download} disabled={exportState?.source === payload && exportState.busy}>
        {exportState?.source === payload && exportState.busy ? 'Exporting PNG…' : 'Download PNG'}
      </button>
    </div>
    {failure?.source === payload && <p role="alert">{failure.message}</p>}
    {exportState?.source === payload && exportState.error && <p role="alert">{exportState.error}</p>}
    {!payload.feature_count && <p>No usable smoke polygons in this region. This does not mean smoke-free conditions.</p>}
    <div ref={container} aria-label="NOAA HMS smoke polygon map" style={{ height: 480, minHeight: 300, width: '100%', minWidth: 0 }} />
    <p>{payload.provenance?.caveat}</p>
    <p>Overlapping polygons may darken the display; use the category legend and click a polygon for its observation interval.</p>
    <details><summary>NOAA source and processing</summary>
      <p><a href={payload.provenance?.source_url} target="_blank" rel="noreferrer">Original NOAA daily ZIP</a></p>
      <pre style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{JSON.stringify(payload.provenance, null, 2)}</pre>
    </details>
  </section>
}
