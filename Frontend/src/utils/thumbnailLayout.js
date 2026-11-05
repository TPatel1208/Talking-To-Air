// Sizes the comparison thumbnails (HeatmapMultiPanel) to fit on one screen,
// as the single map does: the panels are compared by looking from one to the
// other, which scrolling breaks. Every tile shares one image height -- the
// largest that fits both the width and the height -- and is as wide as its
// own aspect asks. More than three panels wrap into rows of three.
//
// width/height: the space the grid has; labelHeight: the fixed caption block
// under each image; aspects: each tile's width/height (thumbnailFrame). Below
// minHeight a tile is unreadable, so a window too small for it scrolls.
const PER_ROW = 3

export function fitThumbnails({ width, height, aspects, gap, labelHeight, minHeight = 120 }) {
  const rows = []
  for (let i = 0; i < aspects.length; i += PER_ROW) {
    rows.push(aspects.slice(i, i + PER_ROW).map((_, j) => i + j))
  }
  const byHeight = (height - rows.length * labelHeight - (rows.length - 1) * gap) / rows.length
  const byWidth = Math.min(...rows.map(row =>
    (width - gap * (row.length - 1)) / row.reduce((sum, i) => sum + aspects[i], 0)))
  const imageHeight = Math.max(minHeight, Math.min(byHeight, byWidth))
  return {
    rows,
    tiles: aspects.map(aspect => ({ width: imageHeight * aspect, height: imageHeight })),
  }
}
