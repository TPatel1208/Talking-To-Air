// One legend definition for the map UI and exported PNG.
export const SMOKE_COLORS = { light: '#e6bc46', medium: '#da7734', heavy: '#943b60', unknown: '#777777' }
export const SMOKE_LEGEND = Object.entries(SMOKE_COLORS).map(([key, color]) => ({
  key, color, label: key[0].toUpperCase() + key.slice(1),
}))

export function waitForSmokeMap(map, signal, timeoutMs = 20000) {
  return new Promise((resolve, reject) => {
    let timer
    function finish(error) {
      clearTimeout(timer)
      map.off('idle', ready)
      map.off('error', failed)
      signal?.removeEventListener('abort', aborted)
      if (error) reject(error)
      else resolve()
    }
    function ready() {
      if (map.loaded() && map.areTilesLoaded() && !map.isMoving()) finish()
    }
    function failed() { finish(new Error('Map tiles could not be loaded. Please reload the map before exporting.')) }
    function aborted() { finish(new DOMException('Export cancelled.', 'AbortError')) }
    map.on('idle', ready)
    map.on('error', failed)
    signal?.addEventListener('abort', aborted, { once: true })
    timer = setTimeout(() => finish(new Error('The map is still loading. Wait for the tiles to appear and try again.')), timeoutMs)
    if (signal?.aborted) aborted()
    else { ready(); map.triggerRepaint() }
  })
}

export async function smokeMapPng(map, { signal, createCanvas = () => document.createElement('canvas') } = {}) {
  await waitForSmokeMap(map, signal)
  if (signal?.aborted) throw new DOMException('Export cancelled.', 'AbortError')
  // Capture the existing framebuffer, not a second map or a stretched screenshot.
  // MapLibre's anonymous cross-origin tile requests preserve canvas readability
  // when the tile server supplies Access-Control-Allow-Origin (as OSM does).
  const source = map.getCanvas()
  const cssWidth = source.clientWidth
  if (!cssWidth || !source.width || !source.height) throw new Error('The map must be visible before exporting.')
  const scale = source.width / cssWidth
  const columns = Math.max(1, Math.min(4, Math.floor((cssWidth - 24) / 110)))
  const rows = Math.ceil(SMOKE_LEGEND.length / columns)
  const footerHeight = 24 + rows * 28 + 36
  const output = createCanvas()
  output.width = source.width
  output.height = source.height + Math.ceil(footerHeight * scale)
  const ctx = output.getContext('2d')
  if (!ctx) throw new Error('Your browser could not create the PNG canvas.')
  try {
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, output.width, output.height)
    ctx.drawImage(source, 0, 0)
    ctx.scale(scale, scale)
    const top = source.height / scale + 16
    ctx.font = '14px sans-serif'
    ctx.textBaseline = 'middle'
    SMOKE_LEGEND.forEach(({ label, color }, index) => {
      const x = 12 + (index % columns) * ((cssWidth - 24) / columns)
      const y = top + Math.floor(index / columns) * 28
      ctx.fillStyle = color
      ctx.fillRect(x, y, 14, 14)
      ctx.fillStyle = '#202020'
      ctx.fillText(label, x + 21, y + 7)
    })
    ctx.font = '11px sans-serif'
    ctx.fillText('© OpenStreetMap contributors', 12, top + rows * 28 + 6)
    ctx.fillText('Smoke data: NOAA HMS', 12, top + rows * 28 + 22)
    return await new Promise((resolve, reject) => {
      output.toBlob(blob => blob ? resolve(blob) : reject(new Error('PNG encoding failed. Please try again.')), 'image/png')
    })
  } catch (error) {
    if (error.name === 'SecurityError') {
      throw new Error('PNG export was blocked by cross-origin tile permissions. Reload the public basemap and try again; the tile server must allow CORS.', { cause: error })
    }
    throw error
  }
}
