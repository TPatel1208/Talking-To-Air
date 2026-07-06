import ArtifactMessage from './ArtifactMessage'
import ImageViewer from './ImageViewer'
import ToolCallBadge from './ToolCallBadge'

export default function Dashboard({ images, toolCalls, artifacts, accessToken }) {
  const outputCount = (images?.length || 0) + (artifacts?.length || 0)

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
            background: outputCount ? 'var(--teal)' : 'var(--border)',
            transition: 'background 0.3s',
          }}/>
          <span style={{
            fontSize: '11px', fontWeight: '500', color: 'var(--text-muted)',
            letterSpacing: '0.06em', textTransform: 'uppercase',
          }}>
            {outputCount
              ? `${outputCount} output${outputCount > 1 ? 's' : ''}`
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

      {/* Artifact gallery — every artifact type produced this session */}
      {artifacts?.length > 0 && (
        <div style={{ overflow: 'auto', padding: '0 12px', flexShrink: 0, maxHeight: '50%' }}>
          {artifacts.map((artifact, i) => (
            <ArtifactMessage key={artifact.id || i} artifact={artifact} accessToken={accessToken} />
          ))}
        </div>
      )}

      {/* Image viewer */}
      <ImageViewer images={images} />
    </div>
  )
}
