import { superscriptRuns } from '../utils/displayFormat.js'

// Text whose exponents (10¹⁵, cm²) render as <sup> digits in the surrounding
// font. Anything that is not a string renders unchanged.
const supStyle = { fontSize: '0.7em', lineHeight: 0, verticalAlign: 'super' }

export default function SciText({ children }) {
  if (typeof children !== 'string') return children ?? null
  return superscriptRuns(children).map((run, i) => (
    run.sup ? <sup key={i} style={supStyle}>{run.text}</sup> : run.text
  ))
}
