import assert from 'node:assert/strict'
import test from 'node:test'
import {
  availableAxes,
  profileAxis,
  profileLayout,
  profileTraces,
  spreadCaveat,
} from '../src/utils/verticalProfile.js'

// A TEMPO_O3PROF-shaped payload: layer 0 is the TOP of the atmosphere, which
// is the fact every assertion here exists to pin.
const profile = {
  type: 'profile',
  title: 'ozone_profile vertical profile over New Jersey',
  variable: 'ozone_profile',
  units: 'DU',
  stat: 'mean',
  layers: [0, 1, 2, 3],
  values: [0.23, 2.25, 29.7, 10.75],
  default_axis: 'pressure',
  layer_order: 'top_down',
  valid_fraction: [0.25, 1, 1, 1],
  vertical: {
    pressure: {
      kind: 'pressure', units: 'hPa',
      values: [0.175, 1.5, 130, 902], spread: [0, 0, 56.6, 43.1],
      layer_order: 'top_down',
    },
    altitude: {
      kind: 'altitude', units: 'km',
      values: [60, 44, 13, 1], spread: [0.06, 0.03, 2.21, 0.42],
      layer_order: 'top_down',
    },
  },
}

test('the default axis is pressure and both axes are offered', () => {
  assert.deepEqual(availableAxes(profile), ['pressure', 'altitude'])
  assert.equal(profileAxis(profile).kind, 'pressure')
})

test('an unknown axis request falls back to the payload default', () => {
  assert.equal(profileAxis(profile, 'nonsense').kind, 'pressure')
  assert.equal(profileAxis(profile, 'altitude').kind, 'altitude')
})

test('the value is plotted against the physical axis, not the layer index', () => {
  // This is what makes the ordering bug impossible rather than merely tested:
  // if y carries pressures, an inverted axis puts the surface at the bottom
  // whichever end of the array it came from.
  const [trace] = profileTraces(profile, 'pressure')
  assert.deepEqual(trace.x, [0.23, 2.25, 29.7, 10.75])
  assert.deepEqual(trace.y, [0.175, 1.5, 130, 902])
})

test('pressure is drawn on a reversed log axis so the sky is at the top', () => {
  const layout = profileLayout(profile, 'pressure')
  assert.equal(layout.yaxis.type, 'log')
  assert.equal(layout.yaxis.autorange, 'reversed')
  assert.match(layout.yaxis.title, /hPa/)
})

test('altitude is drawn upward on a linear axis', () => {
  const layout = profileLayout(profile, 'altitude')
  assert.equal(layout.yaxis.type, 'linear')
  assert.notEqual(layout.yaxis.autorange, 'reversed')
  assert.match(layout.yaxis.title, /km/)
})

test('a bottom-up product renders the same way up', () => {
  // Same atmosphere, array reversed. The picture must not flip -- which it
  // cannot, because the axis carries physical values.
  const flipped = {
    ...profile,
    layers: [0, 1, 2, 3],
    values: [...profile.values].reverse(),
    layer_order: 'bottom_up',
    vertical: {
      pressure: {
        ...profile.vertical.pressure,
        values: [...profile.vertical.pressure.values].reverse(),
        spread: [...profile.vertical.pressure.spread].reverse(),
        layer_order: 'bottom_up',
      },
      altitude: {
        ...profile.vertical.altitude,
        values: [...profile.vertical.altitude.values].reverse(),
        layer_order: 'bottom_up',
      },
    },
  }
  const pairsOf = payload => {
    const [trace] = profileTraces(payload, 'pressure')
    return trace.x.map((v, i) => [v, trace.y[i]]).sort((a, b) => a[1] - b[1])
  }
  assert.deepEqual(pairsOf(flipped), pairsOf(profile))
  assert.equal(profileLayout(flipped, 'pressure').yaxis.autorange, 'reversed')
})

test('a pressure axis with a non-positive value stays linear', () => {
  // A log axis silently drops a zero or negative point; a linear one shows it.
  const withZero = {
    ...profile,
    vertical: { ...profile.vertical, pressure: { ...profile.vertical.pressure, values: [0, 1.5, 130, 902] } },
  }
  assert.equal(profileLayout(withZero, 'pressure').yaxis.type, 'linear')
})

test('the spread caveat names the layers where the axis is an approximation', () => {
  // Finding 4: the vertical grid is fixed aloft and terrain-following near the
  // surface, so a regional-mean axis is exact above and approximate below. A
  // reader who is not told that reads the axis as definite.
  const caveat = spreadCaveat(profile, 'pressure')
  assert.equal(caveat.exactLayers, 2)
  assert.equal(caveat.maxSpread, 56.6)
  assert.equal(caveat.units, 'hPa')
  assert.match(caveat.text, /56.6 hPa/)
})

test('an axis that varies everywhere leads with how much, not with a zero', () => {
  // Measured on the real granule: altitude is never EXACTLY constant (~0.03 km
  // of wobble aloft), so counting exact layers reads "identical for 0 of 24"
  // -- true, and it makes a 30-metre variation across a 60 km axis sound like
  // the axis is unusable. Say the size of the variation instead.
  const altitude = spreadCaveat(profile, 'altitude')
  assert.equal(altitude.exactLayers, 0)
  assert.doesNotMatch(altitude.text, /0 of 4/)
  assert.match(altitude.text, /2\.21 km/)
})

test('an axis that never varies across the region says so plainly', () => {
  const rigid = {
    ...profile,
    vertical: { ...profile.vertical, pressure: { ...profile.vertical.pressure, spread: [0, 0, 0, 0] } },
  }
  const caveat = spreadCaveat(rigid, 'pressure')
  assert.equal(caveat.maxSpread, 0)
  assert.match(caveat.text, /identical/i)
})

test('a payload with no physical axis at all degrades to the layer index', () => {
  const bare = { ...profile, vertical: {}, default_axis: null }
  const [trace] = profileTraces(bare, 'pressure')
  assert.deepEqual(trace.y, [0, 1, 2, 3])
  assert.equal(profileLayout(bare, 'pressure').yaxis.type, 'linear')
  assert.equal(spreadCaveat(bare, 'pressure'), null)
  assert.deepEqual(availableAxes(bare), [])
})

test('the pressure axis labels whole decades, not bare log mantissas', () => {
  // A 4-decade pressure axis (0.17 -> 900 hPa) makes Plotly label the minor
  // ticks too, which renders as "2, 5, 1, 2, 5, 100, 2, 5..." -- the mantissas
  // of each decade with no exponent, which reads as nonsense next to a real
  // "100". Labelling decades only is also how a pressure axis is conventionally
  // drawn; the minor gridlines stay, unlabelled.
  const layout = profileLayout(profile, 'pressure')
  assert.equal(layout.yaxis.type, 'log')
  assert.equal(layout.yaxis.dtick, 1)
})

test('both axes reserve room for their own labels', () => {
  // The layout ships a deliberately small fixed margin so the plot fills the
  // panel. On a log pressure axis the tick labels plus the rotated axis title
  // overflow it and get clipped -- "1000" rendered as "00". automargin lets
  // Plotly grow the margin to whatever the labels actually need.
  for (const kind of ['pressure', 'altitude']) {
    const layout = profileLayout(profile, kind)
    assert.equal(layout.yaxis.automargin, true, kind)
    assert.equal(layout.xaxis.automargin, true, kind)
  }
})

test('a linear pressure axis does not force decade ticks', () => {
  // dtick:1 means "one decade" on a log axis but "every 1 hPa" on a linear
  // one -- which would try to draw ~900 ticks.
  const withZero = {
    ...profile,
    vertical: { ...profile.vertical, pressure: { ...profile.vertical.pressure, values: [0, 1.5, 130, 902] } },
  }
  const layout = profileLayout(withZero, 'pressure')
  assert.equal(layout.yaxis.type, 'linear')
  assert.equal(layout.yaxis.dtick, undefined)
})

test('the plot sizes itself to its container', () => {
  // Plotly defaults to a fixed 700px width when the layout gives neither a
  // width nor autosize -- so the chart rendered at a fraction of the panel
  // with dead space beside it, however wide the container was. Height stays
  // explicit (autosize only fills in what is missing), so only the width
  // follows the container.
  const layout = profileLayout(profile, 'pressure')
  assert.equal(layout.autosize, true)
  assert.equal(layout.width, undefined)
  assert.equal(typeof layout.height, 'number')
})

test('the plot carries no title of its own', () => {
  // The panel renders the chart title in its header, directly above this plot.
  // A second copy inside the plot is centred on the plotting area rather than
  // the panel, so a long title ("ozone_profile vertical profile over New
  // Jersey, United States") overflows and gets clipped at BOTH ends -- a
  // duplicate that is also unreadable. The PNG export keeps its own title:
  // a downloaded file has no header to inherit one from.
  const layout = profileLayout(profile, 'pressure')
  assert.equal(layout.title, undefined)
})

// When a product publishes no physical vertical axis -- or a retrieval failed
// to bring one back -- the chart falls back to the bare layer index. That
// fallback is the ONE place `layer_order` is a rendering input rather than
// disclosure: with no pressures on the axis there is nothing else to say which
// end is the sky, and an index axis ascending from 0 draws a top-down product
// upside down while looking perfectly ordinary.
const indexOnly = { ...profile, vertical: {}, default_axis: null }

test('a top-down product falls back to an index axis that still puts the sky up', () => {
  const layout = profileLayout(indexOnly, 'pressure')
  assert.equal(layout.yaxis.autorange, 'reversed')
  assert.match(layout.yaxis.title, /top/i)
})

test('a bottom-up product falls back to an ordinary ascending index axis', () => {
  const layout = profileLayout({ ...indexOnly, layer_order: 'bottom_up' }, 'pressure')
  assert.notEqual(layout.yaxis.autorange, 'reversed')
  assert.match(layout.yaxis.title, /surface|bottom/i)
})

test('an unknown layer order does not pretend to know which way is up', () => {
  const layout = profileLayout({ ...indexOnly, layer_order: 'unknown' }, 'pressure')
  assert.notEqual(layout.yaxis.autorange, 'reversed')
  assert.equal(layout.yaxis.title, 'layer')
})
