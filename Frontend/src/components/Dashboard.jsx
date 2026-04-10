import ImageViewer from './ImageViewer'
import ToolCallBadge from './ToolCallBadge'

export default function Dashboard({ images, toolCalls }) {
  return (
    <div style={{
      display:       'flex',
      flexDirection: 'column',
      height:        '100%',
      overflow:      'hidden',
    }}>
      {/* Header */}
      <div style={{
        display:        'flex',
        alignItems:     'center',
        justifyContent: 'space-between',
        padding:        '12px 16px',
        borderBottom:   '1px solid var(--border)',
        flexShrink:     0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            width:        '8px',
            height:       '8px',
            borderRadius: '50%',
            background:   images?.length ? 'var(--success)' : 'var(--text-secondary)',
          }}/>
          <span style={{
            fontSize:   '12px',
            fontWeight: '500',
            color:      'var(--text-secondary)',
            letterSpacing: '0.05em',
            textTransform: 'uppercase',
          }}>
            {images?.length
              ? `${images.length} output${images.length > 1 ? 's' : ''}`
              : 'No outputs yet'}
          </span>
        </div>

        <span style={{
          fontSize: '11px',
          color:    'var(--text-secondary)',
          opacity:  0.6,
        }}>
          Talking to Air
        </span>
      </div>

      {/* Tool call badges */}
      <ToolCallBadge toolCalls={toolCalls} />

      {/* Image viewer — takes remaining space */}
      <ImageViewer images={images} />
    </div>
  )
}