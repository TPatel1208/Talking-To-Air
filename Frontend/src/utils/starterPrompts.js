// T22 story #3/#6: the message a starter chip sends is always its full
// `prompt` — never the short `label` shown on the chip — so clicking one
// is exactly like typing and sending the whole question.
export function starterMessage(starter) {
  return starter.prompt
}
