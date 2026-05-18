import ImageViewer from './ImageViewer'
import ToolCallBadge from './ToolCallBadge'

export default function Dashboard({ images, toolCalls }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>

      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '12px 16px', borderBottom: '1px solid var(--border)',
        background: 'var(--bg-primary)', flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            width: '7px', height: '7px', borderRadius: '50%',
            background: images?.length ? 'var(--teal)' : 'var(--border)',
            transition: 'background 0.3s',
          }}/>
          <span style={{
            fontSize: '11px', fontWeight: '500', color: 'var(--text-muted)',
            letterSpacing: '0.06em', textTransform: 'uppercase',
          }}>
            {images?.length
              ? `${images.length} output${images.length > 1 ? 's' : ''}`
              : 'No outputs yet'}
          </span>
        </div>

        <span style={{
          fontFamily: 'var(--font-serif)', fontSize: '13px',
          color: 'var(--text-muted)', fontStyle: 'italic',
        }}>
          Talking to Air
        </span>
      </div>

      {/* Tool call badges */}
      <ToolCallBadge toolCalls={toolCalls} />

      {/* Image viewer */}
      <ImageViewer images={images} />
    </div>
  )
}
