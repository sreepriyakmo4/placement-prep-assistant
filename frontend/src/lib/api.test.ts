import { describe, it, expect } from 'vitest'
import { parseSSEBuffer } from './api'

describe('parseSSEBuffer', () => {
  it('parses a single complete SSE event with no remainder', () => {
    const buffer = 'data: {"type":"status","message":"Searching..."}\n\n'

    const { events, remainder } = parseSSEBuffer(buffer)

    expect(events).toHaveLength(1)
    expect(events[0]).toEqual({ type: 'status', message: 'Searching...' })
    expect(remainder).toBe('')
  })

  it('parses multiple complete events arriving in one buffer', () => {
    const buffer =
      'data: {"type":"status","message":"Searching..."}\n\n' +
      'data: {"type":"chunk","content":"Hello"}\n\n' +
      'data: {"type":"chunk","content":" world"}\n\n'

    const { events, remainder } = parseSSEBuffer(buffer)

    expect(events).toHaveLength(3)
    expect(events[0].type).toBe('status')
    expect(events[1]).toEqual({ type: 'chunk', content: 'Hello' })
    expect(events[2]).toEqual({ type: 'chunk', content: ' world' })
    expect(remainder).toBe('')
  })

  it('keeps a partial/cut-off frame in the remainder and emits zero events', () => {
    // Simulates a slow network chunk arriving mid-JSON, before the closing
    // "\n\n" that marks the end of an SSE event.
    const buffer = 'data: {"type":"chunk","content":"Hel'

    const { events, remainder } = parseSSEBuffer(buffer)

    expect(events).toHaveLength(0)
    expect(remainder).toBe('data: {"type":"chunk","content":"Hel')
  })

  it('completes a previously partial frame once the rest arrives', () => {
    // First chunk: partial data, no complete event yet.
    const first = parseSSEBuffer('data: {"type":"chunk","content":"Hel')
    expect(first.events).toHaveLength(0)

    // Second chunk arrives and gets appended to the leftover remainder,
    // exactly like queryStream's real loop does with `buffer += ...`.
    const combined = first.remainder + 'lo"}\n\n'
    const second = parseSSEBuffer(combined)

    expect(second.events).toHaveLength(1)
    expect(second.events[0]).toEqual({ type: 'chunk', content: 'Hello' })
    expect(second.remainder).toBe('')
  })

  it('silently skips a malformed JSON chunk without crashing', () => {
    const buffer =
      'data: {this is not valid json}\n\n' +
      'data: {"type":"done","answer":"ok"}\n\n'

    const { events, remainder } = parseSSEBuffer(buffer)

    // The malformed event is dropped; the well-formed one after it still parses.
    expect(events).toHaveLength(1)
    expect(events[0]).toEqual({ type: 'done', answer: 'ok' })
    expect(remainder).toBe('')
  })

  it('ignores lines that are not SSE "data:" frames', () => {
    const buffer = ': this is a comment line, not data\n\ndata: {"type":"status","message":"ok"}\n\n'

    const { events, remainder } = parseSSEBuffer(buffer)

    expect(events).toHaveLength(1)
    expect(events[0]).toEqual({ type: 'status', message: 'ok' })
    expect(remainder).toBe('')
  })

  it('returns no events and empty remainder for an empty buffer', () => {
    const { events, remainder } = parseSSEBuffer('')

    expect(events).toHaveLength(0)
    expect(remainder).toBe('')
  })
})