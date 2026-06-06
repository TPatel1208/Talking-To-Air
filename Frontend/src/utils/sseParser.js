export function createSseParser(onEvent) {
  let buffer = ''
  let eventName = ''
  let eventData = ''
  let lastEventId = ''

  const dispatch = () => {
    if (!eventData) {
      eventName = ''
      return
    }

    if (eventData.endsWith('\n')) {
      eventData = eventData.slice(0, -1)
    }

    onEvent({
      event: eventName || 'message',
      data: eventData,
      id: lastEventId,
    })

    eventName = ''
    eventData = ''
  }

  const parseLine = (line) => {
    if (line === '') {
      dispatch()
      return
    }

    if (line.startsWith(':')) return

    const separatorIndex = line.indexOf(':')
    const field = separatorIndex === -1 ? line : line.slice(0, separatorIndex)
    let value = separatorIndex === -1 ? '' : line.slice(separatorIndex + 1)

    if (value.startsWith(' ')) {
      value = value.slice(1)
    }

    if (field === 'event') {
      eventName = value
    } else if (field === 'data') {
      eventData += `${value}\n`
    } else if (field === 'id' && !value.includes('\u0000')) {
      lastEventId = value
    }
  }

  const feed = (chunk) => {
    buffer += chunk.replace(/\r\n/g, '\n').replace(/\r/g, '\n')

    let lineEndIndex = buffer.indexOf('\n')
    while (lineEndIndex !== -1) {
      const line = buffer.slice(0, lineEndIndex)
      buffer = buffer.slice(lineEndIndex + 1)
      parseLine(line)
      lineEndIndex = buffer.indexOf('\n')
    }
  }

  const end = () => {
    if (buffer) {
      parseLine(buffer)
      buffer = ''
    }
    dispatch()
  }

  return { feed, end }
}
