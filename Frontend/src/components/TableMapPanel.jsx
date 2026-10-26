import { useEffect, useMemo, useRef, useState } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { fetchTileConfig, buildMapStyle } from './MapLibreHeatmapPanel.jsx'
import { fetchUsStatesGeoJSON, isConusBounds } from '../utils/regionBorders.js'
import { apiFetch } from '../utils/apiFetch.js'

function getAQIColor(val, minVal, maxVal) {
  if (val == null || isNaN(val)) return '#94a3b8'
  const range = maxVal - minVal
  const ratio = range > 0 ? Math.max(0, Math.min(1, (val - minVal) / range)) : 0.5
  if (ratio < 0.25) return '#10b981' // Green / Good
  if (ratio < 0.5) return '#f59e0b'  // Amber / Moderate
  if (ratio < 0.75) return '#f97316' // Orange / Sensitive
  return '#ef4444'                  // Red / High
}

function StatCard({ label, value, subtitle }) {
  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border)',
      borderRadius: '10px',
      padding: '11px 13px',
    }}>
      <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '5px' }}>{label}</div>
      <div style={{ fontSize: '15px', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{value}</div>
      {subtitle && (
        <div style={{ fontSize: '10.5px', color: 'var(--text-muted)', marginTop: '4px', lineHeight: 1.35 }}>
          {subtitle}
        </div>
      )}
    </div>
  )
}

function MapColorbar({ minVal, maxVal, units, title = 'EPA Ground Monitors' }) {
  const gradientId = 'ground-colorbar-gradient'
  const gradientStops = [
    { offset: '0%', color: '#10b981' },
    { offset: '33%', color: '#f59e0b' },
    { offset: '66%', color: '#f97316' },
    { offset: '100%', color: '#ef4444' },
  ]

  const ticks = useMemo(() => {
    const steps = [0, 0.25, 0.5, 0.75, 1]
    return steps.map(ratio => {
      const v = minVal + ratio * (maxVal - minVal)
      const label = Number.isFinite(v) ? (v >= 10 ? v.toFixed(1) : v.toFixed(2)) : '—'
      return { ratio, value: v, label }
    })
  }, [minVal, maxVal])

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
        pointerEvents: 'auto',
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
            {tick.label}
          </span>
        ))}
      </div>
      {units && (
        <div style={{ fontSize: '9px', color: '#555', textAlign: 'center', marginTop: 2 }}>{units}</div>
      )}
      {title && (
        <div style={{ fontSize: '8px', color: '#777', textAlign: 'center', marginTop: 2, fontStyle: 'italic' }}>
          {title}
        </div>
      )}
    </div>
  )
}

export default function TableMapPanel({ artifact, height = 480, isCompare = false }) {
  const [rows, setRows] = useState(artifact?.rows || [])
  const [loading, setLoading] = useState(!artifact?.rows?.length)
  const [error, setError] = useState(null)
  const [selectedSiteId, setSelectedSiteId] = useState(null)

  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const activePopupRef = useRef(null)

  // 1. Fetch artifact rows if not already populated on the artifact object
  useEffect(() => {
    let cancelled = false
    if (artifact?.rows && artifact.rows.length) {
      setRows(artifact.rows)
      setLoading(false)
      return
    }
    if (!artifact?.id) return

    setLoading(true)
    apiFetch(`/api/artifacts/${artifact.id}?offset=0&limit=500`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then(data => {
        if (!cancelled) {
          setRows(data.rows || [])
          setLoading(false)
        }
      })
      .catch(err => {
        if (!cancelled) {
          setError(err.message || 'Failed to load table data')
          setLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [artifact?.id, artifact?.rows])

  // 2. Filter rows with valid coordinates
  const monitors = useMemo(() => {
    if (!rows || !rows.length) return []
    return rows
      .filter(r => {
        const lat = Number(r.latitude)
        const lon = Number(r.longitude)
        return !isNaN(lat) && !isNaN(lon) && lat !== 0 && lon !== 0
      })
      .map(r => {
        const lat = Number(r.latitude)
        const lon = Number(r.longitude)
        const meanVal = r.mean != null && !isNaN(Number(r.mean)) ? Number(r.mean) : null
        const maxVal = r.max != null && !isNaN(Number(r.max)) ? Number(r.max) : null
        const peakVal = r.peak?.value != null ? Number(r.peak.value) : null
        const primaryVal = meanVal ?? maxVal ?? peakVal
        return {
          ...r,
          lat,
          lon,
          primaryVal,
          site_id: r.site_id || `${lat.toFixed(3)},${lon.toFixed(3)}`,
          displayName: r.local_site_name || r.monitor_name || r.site_id || 'Monitor',
          units: r.units || '',
        }
      })
  }, [rows])

  // Summary statistics
  const { minVal, maxVal, spatialMean, peakMonitor, lowestMonitor, units, pollutant } = useMemo(() => {
    const validVals = monitors.map(m => m.primaryVal).filter(v => v != null)
    const min = validVals.length ? Math.min(...validVals) : 0
    const max = validVals.length ? Math.max(...validVals) : 1
    const mean = validVals.length ? (validVals.reduce((acc, v) => acc + v, 0) / validVals.length) : 0
    const peakMon = monitors.find(m => m.primaryVal === max) || null
    const lowMon = monitors.find(m => m.primaryVal === min) || null
    const u = monitors.find(m => m.units)?.units || ''
    const p = monitors.find(m => m.pollutant_standard)?.pollutant_standard || artifact?.title || 'Air Quality'
    return {
      minVal: min,
      maxVal: max,
      spatialMean: mean,
      peakMonitor: peakMon,
      lowestMonitor: lowMon,
      units: u,
      pollutant: p,
    }
  }, [monitors, artifact?.title])

  // Helper to open station popup
  const openStationPopup = (m, mapInstance = mapRef.current) => {
    if (!mapInstance) return
    if (activePopupRef.current) {
      activePopupRef.current.remove()
      activePopupRef.current = null
    }

    const html = `
      <div style="font-family: var(--font, sans-serif); padding: 4px; min-width: 180px;">
        <div style="font-weight: 800; font-size: 13.5px; color: #111; margin-bottom: 2px;">
          ${m.displayName}
        </div>
        <div style="font-size: 10.5px; font-family: monospace; color: #666; margin-bottom: 8px;">
          Site ${m.site_id}
        </div>
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
          <span style="font-size: 11.5px; color: #555;">Daily Mean:</span>
          <span style="font-weight: 800; font-size: 13px; color: #000;">
            ${m.mean != null ? Number(m.mean).toFixed(2) : '—'} ${m.units}
          </span>
        </div>
        ${m.max != null && !isNaN(Number(m.max)) ? `
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
          <span style="font-size: 11.5px; color: #555;">Daily Max:</span>
          <span style="font-weight: 800; font-size: 13px; color: #e11d48;">
            ${Number(m.max).toFixed(2)} ${m.units}
          </span>
        </div>` : ''}
        ${m.peak?.first_max_hour != null ? `
        <div style="font-size: 10px; color: #777; margin-top: 4px; border-top: 1px solid #eee; padding-top: 4px;">
          Peak at hour ${m.peak.first_max_hour}:00
        </div>` : ''}
      </div>
    `

    const popup = new maplibregl.Popup({ offset: 12, closeButton: true })
      .setLngLat([m.lon, m.lat])
      .setHTML(html)
      .addTo(mapInstance)

    activePopupRef.current = popup
  }

  // 3. Initialize and mount MapLibre map
  useEffect(() => {
    if (!containerRef.current || !monitors.length) return

    let cancelled = false
    let map = null

    // Compute bounding box
    const lons = monitors.map(m => m.lon)
    const lats = monitors.map(m => m.lat)
    let minx = Math.min(...lons)
    let maxx = Math.max(...lons)
    let miny = Math.min(...lats)
    let maxy = Math.max(...lats)

    // Expand bounding box if points are tightly clustered
    if (maxx - minx < 0.08) {
      minx -= 0.05
      maxx += 0.05
    }
    if (maxy - miny < 0.08) {
      miny -= 0.05
      maxy += 0.05
    }

    fetchTileConfig().then(tileConfig => {
      if (cancelled || !containerRef.current) return

      map = new maplibregl.Map({
        container: containerRef.current,
        style: buildMapStyle(tileConfig),
        attributionControl: false,
      })
      mapRef.current = map

      map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-left')

      map.on('load', () => {
        if (cancelled) return

        map.resize()
        map.fitBounds([[minx, miny], [maxx, maxy]], { padding: 40, maxZoom: 14, animate: false })

        // Add US state boundaries
        if (isConusBounds(minx, miny, maxx, maxy)) {
          fetchUsStatesGeoJSON().then(geojson => {
            if (!cancelled && map.getSource && !map.getSource('region-borders')) {
              map.addSource('region-borders', { type: 'geojson', data: geojson })
              map.addLayer({
                id: 'region-borders',
                type: 'line',
                source: 'region-borders',
                paint: { 'line-color': 'rgba(40,40,40,0.6)', 'line-width': 1.1 },
              })
            }
          })
        }

        // Build GeoJSON FeatureCollection
        const geojson = {
          type: 'FeatureCollection',
          features: monitors.map(m => ({
            type: 'Feature',
            geometry: { type: 'Point', coordinates: [m.lon, m.lat] },
            properties: {
              site_id: m.site_id,
              name: m.displayName,
              mean: m.mean != null ? Number(m.mean).toFixed(2) : '—',
              max: m.max != null ? Number(m.max).toFixed(2) : '—',
              peak_hour: m.peak?.first_max_hour != null ? `${m.peak.first_max_hour}:00` : '',
              units: m.units,
              standard: m.pollutant_standard || '',
              color: getAQIColor(m.primaryVal, minVal, maxVal),
            },
          })),
        }

        map.addSource('ground-monitors', {
          type: 'geojson',
          data: geojson,
        })

        // Outer halo circle
        map.addLayer({
          id: 'ground-monitors-halo',
          type: 'circle',
          source: 'ground-monitors',
          paint: {
            'circle-radius': 14,
            'circle-color': ['get', 'color'],
            'circle-opacity': 0.25,
          },
        })

        // Core marker circle
        map.addLayer({
          id: 'ground-monitors-pin',
          type: 'circle',
          source: 'ground-monitors',
          paint: {
            'circle-radius': 8,
            'circle-color': ['get', 'color'],
            'circle-stroke-width': 2.5,
            'circle-stroke-color': '#ffffff',
          },
        })

        // Cursor & interactive popup
        map.on('mouseenter', 'ground-monitors-pin', () => {
          map.getCanvas().style.cursor = 'pointer'
        })
        map.on('mouseleave', 'ground-monitors-pin', () => {
          map.getCanvas().style.cursor = ''
        })

        map.on('click', 'ground-monitors-pin', e => {
          const feature = e.features?.[0]
          if (!feature) return
          const props = feature.properties
          const matchingMon = monitors.find(m => m.site_id === props.site_id)
          if (matchingMon) {
            setSelectedSiteId(matchingMon.site_id)
            openStationPopup(matchingMon, map)
          }
        })
      })
    })

    return () => {
      cancelled = true
      if (activePopupRef.current) {
        activePopupRef.current.remove()
        activePopupRef.current = null
      }
      if (map) map.remove()
      mapRef.current = null
    }
  }, [monitors, minVal, maxVal])

  // Fly to monitor when clicked from the station selector
  const handleSelectMonitor = (m) => {
    setSelectedSiteId(m.site_id)
    if (!mapRef.current) return
    mapRef.current.flyTo({
      center: [m.lon, m.lat],
      zoom: 12,
      essential: true,
    })
    openStationPopup(m, mapRef.current)
  }

  if (loading) {
    return (
      <div style={{ padding: '36px', textAlign: 'center', color: 'var(--text-muted)' }}>
        Loading spatial monitor coordinates...
      </div>
    )
  }

  if (error) {
    return (
      <div style={{ padding: '24px', color: 'var(--error)' }}>
        Error loading monitor coordinates: {error}
      </div>
    )
  }

  if (!monitors.length) {
    return (
      <div style={{
        padding: '48px 24px', textAlign: 'center',
        background: 'var(--bg-secondary)', borderRadius: '12px',
        border: '1px dashed var(--border)',
      }}>
        <div style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '8px' }}>
          No Coordinate Data Found
        </div>
        <div style={{ fontSize: '12.5px', color: 'var(--text-muted)', maxWidth: '420px', margin: '0 auto' }}>
          This table does not contain valid latitude and longitude fields to plot on a map.
        </div>
      </div>
    )
  }

  const mapHeight = height ? (typeof height === 'number' ? `${height}px` : height) : (isCompare ? '360px' : '480px')

  return (
    <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '14px' }}>
      {/* MapLibre container matching MapLibreHeatmapPanel structure */}
      <div style={{
        position: 'relative',
        width: '100%',
        height: mapHeight,
        borderRadius: '8px',
        overflow: 'hidden',
        border: '1px solid var(--border)',
      }}>
        <div ref={containerRef} style={{ width: '100%', height: '100%' }} />

        {/* Scientific Overlay Colorbar */}
        <MapColorbar
          minVal={minVal}
          maxVal={maxVal}
          units={units}
          title="EPA Ground Monitors"
        />
      </div>

      {/* Supporting Information & Statistics Grid (rendered in single view) */}
      {!isCompare && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '10px' }}>
            <StatCard
              label="Spatial Mean"
              value={`${spatialMean.toFixed(2)} ${units}`}
              subtitle="average across active monitors"
            />
            {peakMonitor && (
              <StatCard
                label="Regional Peak"
                value={`${peakMonitor.primaryVal.toFixed(2)} ${units}`}
                subtitle={`${peakMonitor.displayName}${peakMonitor.peak?.first_max_hour != null ? ` · hr ${peakMonitor.peak.first_max_hour}:00` : ''}`}
              />
            )}
            {lowestMonitor && (
              <StatCard
                label="Lowest Reading"
                value={`${lowestMonitor.primaryVal.toFixed(2)} ${units}`}
                subtitle={lowestMonitor.displayName}
              />
            )}
            <StatCard
              label="Active Monitors"
              value={monitors.length.toLocaleString()}
              subtitle="reporting valid coordinates"
            />
          </div>

          {/* Interactive Monitor Stations List */}
          <div style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border)',
            borderRadius: '10px',
            padding: '12px 14px',
          }}>
            <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '8px' }}>
              Monitor stations — select to inspect
            </div>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {monitors.map(m => {
                const isSelected = selectedSiteId === m.site_id
                const color = getAQIColor(m.primaryVal, minVal, maxVal)
                return (
                  <button
                    key={m.site_id}
                    type="button"
                    onClick={() => handleSelectMonitor(m)}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '6px',
                      fontSize: '11.5px',
                      fontWeight: isSelected ? 700 : 500,
                      color: isSelected ? 'var(--text-primary)' : 'var(--text-secondary)',
                      background: isSelected ? 'var(--bg-card)' : 'var(--bg-primary)',
                      border: `1px solid ${isSelected ? 'var(--teal)' : 'var(--border)'}`,
                      borderRadius: '6px',
                      padding: '5px 10px',
                      cursor: 'pointer',
                      whiteSpace: 'nowrap',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: color }} />
                    <span>{m.displayName}</span>
                    {m.primaryVal != null && (
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10.5px', opacity: 0.85 }}>
                        {m.primaryVal.toFixed(1)}
                      </span>
                    )}
                  </button>
                )
              })}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
