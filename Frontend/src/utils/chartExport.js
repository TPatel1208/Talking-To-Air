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
  // Mirror the backend's own gate (_chart_source_handles in api.py) so the
  // button is offered exactly when the download can succeed.
  const handles =
    chart?.metadata?.source_handles?.length
      ? chart.metadata.source_handles
      : chart?.provenance?.source_handles

  if (chart?.chart_id && handles?.length) {
    return { kind: 'server', url: `/api/chart/${chart.chart_id}/export.nc` }
  }
  return {
    kind: 'unavailable',
    message: 'This chart does not include a source handle to export.',
  }
}
