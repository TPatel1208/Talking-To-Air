// Decides what an export button can actually deliver for a chart, so the
// component never has to guess and the answer can be tested on its own.
//
// The distinction that matters is full-resolution vs. thinned. The payload
// shipped to the browser carries a grid downsampled to <=8000 cells (see
// _downsample_grid, Backend/tta_backend/tools/satellite_tools/plot_tools.py);
// the source data behind it is up to 22.3M cells. Anything built from the
// payload grid is therefore a sample, and must not be handed over as if it
// were the field.

export function resolveCsvExport(chart) {
  if (chart?.chart_id && chart?.export) {
    return { kind: 'server', url: `/api/chart/${chart.chart_id}/export.csv` }
  }

  // A timeseries payload ships every time and value it has -- nothing about
  // it is sampled -- so building the CSV in the browser loses nothing. This
  // is the one chart type where that is true; heatmaps fall through to the
  // refusal below rather than hand over their thinned grid.
  if (chart?.type === 'timeseries') {
    const rows = (chart.times || []).map((time, i) => ({
      variable: chart.variable,
      time,
      stat: chart.stat,
      value: chart.values?.[i],
      units: chart.units,
    }))
    if (rows.length) return { kind: 'client', rows }
  }

  return {
    kind: 'unavailable',
    message: 'This chart does not include full-resolution export metadata.',
  }
}

// NetCDF, not CSV, is the right artifact for a gridded field: it keeps the
// grid structure, CF metadata, units and attributes that a flat
// lat/lon/value table throws away, and it is what a researcher opens in
// xarray or Panoply. The endpoint (GET /chart/{id}/export.nc) has existed
// since T10 with conversion and chunk-streaming already in place.
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

  return { kind: 'server', url: `/api/chart/${chart.chart_id}/export.nc` }
}
