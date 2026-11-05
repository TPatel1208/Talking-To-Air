// Basemap tile config and MapLibre base style, shared by the heatmap and
// ground-monitor map panels.
import { API_BASE } from '../config.js'

const FALLBACK_TILE_CONFIG = {
  basemap_light_url: 'https://basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png',
  basemap_dark_url: 'https://basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}.png',
  terrain_dem_url: 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png',
  basemap_attribution: '© CARTO © OpenStreetMap contributors',
  terrain_attribution: 'Terrain tiles: AWS Terrain Tiles',
}

let _tileConfigPromise = null
export function fetchTileConfig() {
  if (_tileConfigPromise) return _tileConfigPromise
  _tileConfigPromise = fetch(`${API_BASE}/config/map-tiles`)
    .then(r => (r.ok ? r.json() : null))
    .then(config => ({ ...FALLBACK_TILE_CONFIG, ...(config || {}) }))
    .catch(() => FALLBACK_TILE_CONFIG)
  return _tileConfigPromise
}

export function buildMapStyle(tileConfig) {
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
