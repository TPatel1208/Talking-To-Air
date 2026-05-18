const TOOL_COLORS = {
  convert_date_to_iso:           { color: '#2563eb', bg: '#eff6ff', border: '#bfdbfe' },
  convert_temporal_range_to_iso: { color: '#2563eb', bg: '#eff6ff', border: '#bfdbfe' },
  geocode_location:              { color: '#0F6E56', bg: '#E1F5EE', border: '#9FE1CB' },
  fetch_environmental_data:      { color: '#92400e', bg: '#fffbeb', border: '#fde68a' },
  plot_singular:                 { color: '#6d28d9', bg: '#f5f3ff', border: '#ddd6fe' },
  plot_multiple:                 { color: '#6d28d9', bg: '#f5f3ff', border: '#ddd6fe' },
  compute_statistic_tool:        { color: '#9f1239', bg: '#fff1f2', border: '#fecdd3' },
  conduct_temporal_statistic:    { color: '#9f1239', bg: '#fff1f2', border: '#fecdd3' },
  find_daily_peak:               { color: '#9f1239', bg: '#fff1f2', border: '#fecdd3' },
  detect_anomalies:              { color: '#c2410c', bg: '#fff7ed', border: '#fed7aa' },
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
      display:       'flex',
      flexWrap:      'wrap',
      gap:           '6px',
      padding:       '10px 16px',
      borderBottom:  '1px solid var(--border)',
      background:    'var(--bg-secondary)',
    }}>
      {toolCalls.map((tc, i) => {
        const theme = TOOL_COLORS[tc.name] || { color: '#5a5750', bg: '#f0ede8', border: '#dddbd5' }
        const label = TOOL_LABELS[tc.name] || tc.name
        return (
          <span key={i} title={tc.name} style={{
            display:      'inline-flex',
            alignItems:   'center',
            gap:          '5px',
            padding:      '3px 10px',
            borderRadius: '100px',
            fontSize:     '11px',
            fontWeight:   '500',
            color:        theme.color,
            background:   theme.bg,
            border:       `1px solid ${theme.border}`,
          }}>
            <span style={{
              width:        '5px',
              height:       '5px',
              borderRadius: '50%',
              background:   theme.color,
              flexShrink:   0,
            }}/>
            {label}
          </span>
        )
      })}
    </div>
  )
}
