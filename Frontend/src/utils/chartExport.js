// Decides what an export button can actually deliver for a chart, so the
// component never has to guess and the answer can be tested on its own.
//
// The distinction that matters is full-resolution vs. thinned. The payload
// shipped to the browser carries a grid downsampled to <=8000 cells (see
// _downsample_grid, Backend/tta_backend/tools/satellite_tools/plot_tools.py);
// the source data behind it is up to 22.3M cells. Anything built from the
// payload grid is therefore a sample, and must not be handed over as if it
// were the field.

import { API_BASE } from '../config.js'

// A chart a harness run drew carries its run in `metadata.harness` (the stored
// row) or `run_id` (the live frame). The backend writes such a chart's CSV from
// its payload, which holds the data in full only for these types
// (payload_figures.ROW_TYPES); for any other it answers 409.
const HARNESS_CSV_TYPES = new Set(['timeseries', 'profile'])
// A handle the backend derived, which the data service cannot convert to NetCDF
// (harness/handles/model.py _DERIVED_ID).
const DERIVED_HANDLE = /^(?:grid|series|points|polygons|table|scalar)_[0-9a-f]{32}$/

export function harnessRunOf(chart) {
  const linked = chart?.metadata?.harness?.run_id
  if (typeof linked === 'string' && linked) return linked
  return typeof chart?.run_id === 'string' && chart.run_id ? chart.run_id : null
}

function serverCsv(chart) {
  if (!chart?.chart_id || !chart?.export) return false
  return !harnessRunOf(chart) || HARNESS_CSV_TYPES.has(chart.type)
}

export function resolveCsvExport(chart) {
  if (serverCsv(chart)) {
    return { kind: 'server', url: `${API_BASE}/chart/${chart.chart_id}/export.csv` }
  }

  // A timeseries payload ships every time and value it has -- nothing about
  // it is sampled -- so building the CSV in the browser loses nothing. This
  // is the one chart type where that is true; heatmaps fall through to the
  // refusal below rather than hand over their thinned grid.
  if (chart?.type === 'timeseries') {
    if (Array.isArray(chart.series) && chart.series.length > 1) {
      const rows = chart.series.flatMap(line => (line.times || []).map((time, i) => ({
        variable: chart.variable,
        series: line.label,
        time,
        stat: chart.stat,
        value: line.values?.[i],
        units: line.units,
      })))
      if (rows.length) return { kind: 'client', rows }
    }
    const rows = (chart.times || []).map((time, i) => ({
      variable: chart.variable,
      time,
      stat: chart.stat,
      value: chart.values?.[i],
      units: chart.units,
    }))
    if (rows.length) return { kind: 'client', rows }
  }

  // A profile qualifies for the same reason and no other: its 24 numbers per
  // axis ARE the measurement, so nothing is lost by writing them here. It gets
  // its own columns rather than reusing the timeseries ones -- routed as a
  // timeseries these rows would carry hPa under a `time` header, which is
  // precisely why `profile` is a payload type of its own.
  if (chart?.type === 'profile') {
    const rows = (chart.layers || []).map((layer, i) => {
      const row = {
        variable: chart.variable,
        layer,
        value: chart.values?.[i],
        units: chart.units,
      }
      for (const kind of ['pressure', 'altitude']) {
        const axis = chart.vertical?.[kind]
        if (!axis) continue
        row[kind] = axis.values?.[i]
        row[`${kind}_units`] = axis.units
        row[`${kind}_spread`] = axis.spread?.[i]
      }
      row.valid_fraction = chart.valid_fraction?.[i]
      return row
    })
    if (rows.length) return { kind: 'client', rows }
  }

  const run = harnessRunOf(chart)
  if (run) {
    return {
      kind: 'unavailable',
      message: `A ${chart.type || 'chart'} chart drawn by a harness run has no CSV export. `
        + `The archive of the result that drew it, /result/${run}/export.zip, holds the data the result kept.`,
    }
  }
  return {
    kind: 'unavailable',
    message: 'This chart does not include full-resolution export metadata.',
  }
}

// NetCDF, not CSV, is the right artifact for a gridded field: it keeps the
// grid structure, CF metadata, units and attributes that a flat
// lat/lon/value table throws away, and it is what a researcher opens in
// xarray or Panoply. The endpoint is GET /chart/{id}/export.nc.
export function resolveNetcdfExport(chart) {
  // Resolved the way the backend resolves it (_chart_source_handles in
  // api.py): metadata.source_handles, falling back to provenance's.
  const handles =
    chart?.metadata?.source_handles?.length
      ? chart.metadata.source_handles
      : chart?.provenance?.source_handles

  if (!chart?.chart_id || !handles?.length) {
    return {
      kind: 'unavailable',
      message: 'This chart does not include a source handle to export.',
    }
  }

  // Exactly one handle, not "at least one". The endpoint converts
  // `source_handles[0]` and nothing else, so on a chart built from several
  // inputs it delivers one of them under the whole chart's filename: a
  // two-region comparison (metadata.source_handles = [a, b, aligned]) would
  // download region A named after the comparison, and nothing in the file or
  // its name would say which side the reader was looking at. That is the
  // shape of mistake this product refuses elsewhere -- a plausible artifact
  // sourced from somewhere other than where it claims. Offering NetCDF for
  // multi-source charts needs the endpoint to bundle every handle first.
  if (handles.length > 1) {
    return {
      kind: 'unavailable',
      message:
        `This chart is built from ${handles.length} sources; NetCDF export ` +
        'delivers a single one, so it would not be the chart you are looking at. ' +
        'Export each source chart on its own.',
    }
  }

  const run = harnessRunOf(chart)
  if (run && DERIVED_HANDLE.test(handles[0])) {
    return {
      kind: 'unavailable',
      message: 'This chart shows data the backend derived, which the data service cannot convert to NetCDF. '
        + `The archive of the result that drew it, /result/${run}/export.zip, holds its methods and data.`,
    }
  }

  return { kind: 'server', url: `${API_BASE}/chart/${chart.chart_id}/export.nc` }
}
