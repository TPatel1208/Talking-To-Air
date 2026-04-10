const TOOL_COLORS = {
  convert_date_to_iso:          '#4f7cff',
  convert_temporal_range_to_iso:'#4f7cff',
  geocode_location:             '#2dd4a0',
  fetch_environmental_data:     '#f5a623',
  plot_singular:                '#a78bfa',
  plot_multiple:                '#a78bfa',
  compute_statistic_tool:       '#fb7185',
  conduct_temporal_statistic:   '#fb7185',
  find_daily_peak:              '#fb7185',
  detect_anomalies:             '#f97316',
}

const TOOL_LABELS = {
  convert_date_to_iso:           'date → iso',
  convert_temporal_range_to_iso: 'range → iso',
  geocode_location:              'geocode',
  fetch_environmental_data:      'fetch data',
  plot_singular:                 'plot map',
  plot_multiple:                 'plot comparison',
  compute_statistic_tool:        'statistics',
  conduct_temporal_statistic:    'trend',
  find_daily_peak:               'peak',
  detect_anomalies:              'anomaly',
}

export default function ToolCallBadge({ toolCalls }) {
  if (!toolCalls?.length) return null

  return (
    <div style={{
      display:    'flex',
      flexWrap:   'wrap',
      gap:        '6px',
      padding:    '8px 16px',
      borderBottom: '1px solid var(--border)',
    }}>
      {toolCalls.map((tc, i) => {
        const color = TOOL_COLORS[tc.name] || '#9ea3c0'
        const label = TOOL_LABELS[tc.name] || tc.name
        return (
          <span key={i} title={tc.name} style={{
            display:      'inline-flex',
            alignItems:   'center',
            gap:          '5px',
            padding:      '2px 10px',
            borderRadius: '999px',
            fontSize:     '11px',
            fontWeight:   '500',
            color:        color,
            background:   `${color}18`,
            border:       `1px solid ${color}40`,
          }}>
            <span style={{
              width:        '6px',
              height:       '6px',
              borderRadius: '50%',
              background:   color,
              flexShrink:   0,
            }}/>
            {label}
          </span>
        )
      })}
    </div>
  )
}