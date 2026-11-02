// Colorbars for the comparison thumbnails (HeatmapMultiPanel). A thumbnail is
// an already-rendered image, so its legend can only describe the scale it was
// drawn with: one shared colorbar when every panel was drawn on the same one,
// otherwise one per panel. Unlike compare mode's computeSharedColorScale, this
// never invents a common scale -- nothing here can recolor a png.
import { scaleClipNote } from './colorbarGeometry.js'

function legendOf(panel) {
  const { vmin, vmax, units, colormap, scale } = panel || {}
  if (!Number.isFinite(vmin) || !Number.isFinite(vmax) || !Array.isArray(colormap?.lut)) return null
  return { vmin, vmax, lut: colormap.lut, units: units ?? null, clipNote: scaleClipNote(scale) }
}

function sameScale(a, b, panelA, panelB) {
  return a.vmin === b.vmin && a.vmax === b.vmax && a.units === b.units
    && a.clipNote === b.clipNote && panelA.colormap?.name === panelB.colormap?.name
}

// -> { shared: legend | null, perPanel: (legend | null)[] }
export function thumbnailLegends(panels) {
  const legends = panels.map(legendOf)
  const [first] = legends
  const shared = panels.length > 1 && first
    && legends.every((legend, i) => legend && sameScale(first, legend, panels[0], panels[i]))
  return shared
    ? { shared: first, perPanel: legends.map(() => null) }
    : { shared: null, perPanel: legends }
}
