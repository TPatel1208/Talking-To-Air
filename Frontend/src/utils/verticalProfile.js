// Turning a `profile` chart payload into something Plotly can draw, kept out
// of the component so the one thing most likely to be wrong is testable
// without a DOM.
//
// That one thing is which end of the array is the sky. TEMPO_O3PROF stores
// layer 0 at the TOP of the atmosphere (~0.175 hPa / 60 km) and its last layer
// at the surface, so a chart that plots values against the layer INDEX draws
// the atmosphere upside down and looks entirely plausible doing it. Everything
// here plots against the physical axis instead: with pressures on y and the
// axis reversed, the surface lands at the bottom no matter what order the
// array arrived in. The payload's measured `layer_order` is disclosure for the
// reader, not an input to the rendering — which is why reversing a payload
// produces the same picture.

const AXIS_ORDER = ['pressure', 'altitude']

function axisBlock(chart, kind) {
  const block = chart?.vertical?.[kind]
  return Array.isArray(block?.values) && block.values.length ? block : null
}

// Which physical axes this payload can actually be drawn against, in the order
// the toggle should offer them. Empty for a product that publishes none — the
// chart then falls back to the bare layer index, which is honest about being
// an index rather than dressing it up as a height.
export function availableAxes(chart) {
  return AXIS_ORDER.filter(kind => axisBlock(chart, kind) !== null)
}

// The axis to draw against: the caller's choice when the payload can honor it,
// else the payload's own default, else whatever exists. Never throws — a
// stale toggle from a previously-focused chart must degrade to a correct
// picture, not a blank one.
export function profileAxis(chart, requested) {
  const available = availableAxes(chart)
  const kind =
    (available.includes(requested) && requested) ||
    (available.includes(chart?.default_axis) && chart.default_axis) ||
    available[0] ||
    null
  if (!kind) {
    // No physical axis: fall back to the bare layer index. This is the ONE
    // place `layer_order` is a rendering input rather than disclosure -- with
    // no pressures on the axis there is nothing else to say which end is the
    // sky, and an index ascending from 0 draws a top-down product upside down
    // while looking perfectly ordinary. An `unknown` order claims nothing.
    const layers = chart?.layers || (chart?.values || []).map((_, i) => i)
    const order = chart?.layer_order
    const label =
      order === 'top_down' ? 'layer (0 = top of atmosphere)'
        : order === 'bottom_up' ? 'layer (0 = surface)'
          : 'layer'
    return { kind: null, units: '', values: layers, spread: [], label, order }
  }
  const block = axisBlock(chart, kind)
  return {
    kind,
    units: block.units || '',
    values: block.values,
    spread: Array.isArray(block.spread) ? block.spread : [],
    label: block.units ? `${kind} (${block.units})` : kind,
  }
}

export function profileTraces(chart, requested) {
  const axis = profileAxis(chart, requested)
  const units = chart?.units || ''
  return [{
    type: 'scatter',
    mode: 'lines+markers',
    x: chart?.values || [],
    y: axis.values,
    line: { color: '#1D9E75', width: 2 },
    marker: { color: '#1D9E75', size: 5 },
    hovertemplate:
      `${chart?.variable || 'value'}: %{x:.3f} ${units}<br>` +
      `${axis.label}: %{y}<extra></extra>`,
  }]
}

export function profileLayout(chart, requested, { height = 420 } = {}) {
  const axis = profileAxis(chart, requested)
  // Pressure falls with height, so reversing the axis is what puts the sky at
  // the top. Log scaling is what makes the stratosphere legible at all: the
  // top layer sits three orders of magnitude below the surface, and on a
  // linear axis the entire upper atmosphere collapses onto the top pixel.
  const isPressure = axis.kind === 'pressure'
  const allPositive = axis.values.every(v => typeof v === 'number' && v > 0)
  const isLog = isPressure && allPositive
  // Reverse for pressure (it falls with height) or for a top-down index
  // fallback (index 0 is the sky). Altitude and a bottom-up index already
  // ascend correctly, and an unknown order gets no claim either way.
  const reversed = isPressure || (axis.kind === null && axis.order === 'top_down')
  return {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { family: "'Manrope', system-ui, sans-serif", size: 12, color: '#333333' },
    margin: { t: 12, r: 16, b: 46, l: 16 },
    // No title. The panel already renders the chart title in its header,
    // immediately above this plot, and a second copy is centred on the
    // PLOTTING AREA rather than the panel -- so a real title ("ozone_profile
    // vertical profile over New Jersey, United States") overflows and is
    // clipped at both ends. A duplicate that is also unreadable. The PNG
    // export keeps its own title: a downloaded file has no header to inherit
    // one from.
    //
    // Width is deliberately absent and autosize on. Given neither, Plotly
    // falls back to a fixed 700px and the chart renders at a fraction of the
    // panel with dead space beside it, no matter how wide the container is.
    // Height stays explicit, and autosize only fills in what is missing, so
    // just the width follows the container.
    autosize: true,
    height,
    xaxis: {
      title: `${chart?.stat || 'mean'} (${chart?.units || ''})`,
      // The fixed margins above keep the plot filling the panel; automargin is
      // what stops that thrift from clipping a label. Measured on the real
      // 4-decade pressure axis, "1000" rendered as "00" and the rotated axis
      // title sat on top of the tick labels.
      automargin: true,
      showgrid: true, gridcolor: 'rgba(0,0,0,0.06)', zeroline: false, tickfont: { size: 10 },
    },
    yaxis: {
      title: axis.label,
      // A log axis silently DROPS a zero or negative point rather than
      // failing, so a pressure axis that contains one stays linear -- a
      // missing layer is a worse lie than a squashed one.
      type: isLog ? 'log' : 'linear',
      // One tick per DECADE. Left to itself over four decades Plotly also
      // labels the minor ticks, and on a log axis it prints them as bare
      // mantissas -- "2, 5, 1, 2, 5, 100, 2, 5" reads as nonsense sitting next
      // to a real 100. Decade-only labelling is also how a pressure axis is
      // conventionally drawn; the minor gridlines stay, just unlabelled.
      // Meaningless (and, at ~900 ticks, ruinous) on a linear axis, where
      // dtick counts hPa rather than powers of ten.
      ...(isLog ? { dtick: 1 } : {}),
      ...(reversed ? { autorange: 'reversed' } : {}),
      automargin: true,
      showgrid: true, gridcolor: 'rgba(0,0,0,0.06)', zeroline: false, tickfont: { size: 10 },
    },
  }
}

// How much of an approximation the regional-mean axis is, per request.
//
// The vertical grid is fixed aloft and terrain-following only near the surface,
// so averaging it over a region is EXACT for the upper layers and approximate
// below. How far below depends entirely on the region -- a box over the
// Rockies looks nothing like one over New Jersey -- so this reads the measured
// spreads rather than restating a constant. Returns null when there is no
// physical axis to qualify.
export function spreadCaveat(chart, requested) {
  const axis = profileAxis(chart, requested)
  if (!axis.kind || !axis.spread.length) return null

  // A layer whose spread could not be measured emits null (_rounded turns a
  // non-finite value into one). Coercing that to 0 counted it as EXACTLY
  // constant across the region -- reporting a missing measurement as a perfect
  // one, the same absent-vs-exact conflation the level panel guards against.
  // Unmeasured layers are dropped from both the count and its denominator.
  const spreads = axis.spread.filter(v => typeof v === 'number' && Number.isFinite(v))
  if (!spreads.length) return null
  const exactLayers = spreads.filter(v => v === 0).length
  const maxSpread = Math.max(...spreads)
  const rounded = Number(maxSpread.toPrecision(3))
  // Three sentences, because counting exactly-constant layers only says
  // something useful when there ARE some. Measured on a real granule, pressure
  // is exactly constant for 17 of 24 layers (a strong statement worth making)
  // while altitude wobbles by ~30 m aloft and so is constant for none of them.
  // Reporting the latter as "identical for 0 of 24 layers" is true and badly
  // misleading -- it makes 30 metres out of a 60 km axis sound like the axis
  // is unusable. When nothing is exact, lead with the size of the variation.
  let text
  if (maxSpread === 0) {
    text = `Every layer's ${axis.kind} is identical across the analyzed region, so this axis is exact.`
  } else if (exactLayers === 0) {
    text = `${axis.kind} is a regional mean, varying across the analyzed region by up to ` +
      `${rounded} ${axis.units} at its most variable layer.`
  } else {
    text = `${axis.kind} is identical across the analyzed region for ${exactLayers} of ` +
      `${spreads.length} layers; elsewhere it is a regional mean varying by up to ` +
      `${rounded} ${axis.units}.`
  }
  return { kind: axis.kind, units: axis.units, exactLayers, maxSpread: rounded, spreads, text }
}
