import { useEffect, useRef, useState } from 'react'
import * as maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { fitSmokeMap, polygonMapLayers, smokeMapStyle } from '../utils/hmsMap.js'
import { API_BASE } from '../config.js'

const FILL = '#3b82f6'
const LINE = '#1e3a8a'
const POINT = '#111827'

/* A harness `polygon_map` that is not smoke: its polygons and its point
 * overlay on the plain basemap. */
export default function MapLibrePolygonPanel({ payload }) {
  const container = useRef(null)
  const [failure, setFailure] = useState(null)

  useEffect(() => {
    let map
    let observer
    let loaded = false
    let disposed = false
    const { polygons, points, bounds } = polygonMapLayers(payload)

    function initialize(tileConfig) {
      try {
        map = new maplibregl.Map({
          container: container.current, center: [-100, 40], zoom: 2,
          renderWorldCopies: false, style: smokeMapStyle(tileConfig),
        })
        map.addControl(new maplibregl.NavigationControl())
        observer = new ResizeObserver(() => {
          map.resize()
          if (loaded) fitSmokeMap(map, bounds, container.current)
        })
        observer.observe(container.current)
        map.once('style.load', () => {
          map.addSource('polygons', { type: 'geojson', data: polygons })
          map.addLayer({ id: 'polygons-fill', type: 'fill', source: 'polygons', paint: { 'fill-color': FILL, 'fill-opacity': 0.35 } })
          map.addLayer({ id: 'polygons-line', type: 'line', source: 'polygons', paint: { 'line-color': LINE, 'line-width': 0.8 } })
          if (points.features.length) {
            map.addSource('points', { type: 'geojson', data: points })
            map.addLayer({ id: 'points', type: 'circle', source: 'points',
              paint: { 'circle-radius': 4, 'circle-color': POINT, 'circle-stroke-color': '#ffffff', 'circle-stroke-width': 1 } })
            map.on('click', 'points', event => {
              const properties = event.features?.[0]?.properties || {}
              new maplibregl.Popup().setLngLat(event.lngLat)
                .setText([properties.name || properties.id, properties.value].filter(v => v != null && v !== '').join(' | '))
                .addTo(map)
            })
          }
          loaded = true
          map.resize()
          fitSmokeMap(map, bounds, container.current)
        })
      } catch (error) { setFailure({ source: payload, message: `Map unavailable: ${error.message}` }) }
    }

    async function loadAndInitialize() {
      try {
        const response = await fetch(`${API_BASE}/config/map-tiles`)
        const tileConfig = response.ok ? await response.json() : {}
        if (!disposed) initialize(tileConfig)
      } catch {
        if (!disposed) initialize({})
      }
    }

    loadAndInitialize()
    return () => {
      disposed = true
      observer?.disconnect()
      map?.remove()
    }
  }, [payload])

  return <section>
    {failure?.source === payload && <p role="alert">{failure.message}</p>}
    {!payload.feature_count && <p>This map has no polygons.</p>}
    <div ref={container} aria-label={payload.title || 'Polygon map'} style={{ height: 480, minHeight: 300, width: '100%', minWidth: 0 }} />
  </section>
}
