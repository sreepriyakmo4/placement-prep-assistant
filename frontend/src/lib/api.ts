import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const api = axios.create({ baseURL: BASE_URL })

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(err)
  }
)

// Auth
export const authApi = {
  register: (email: string, password: string) =>
    api.post('/auth/register', { email, password }).then(r => r.data),
  login: (email: string, password: string) =>
    api.post('/auth/login', { email, password }).then(r => r.data),
}

// Documents
export const documentsApi = {
  upload: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return api.post('/documents/upload', form).then(r => r.data)
  },
  list: () => api.get('/documents').then(r => r.data),
  delete: (id: number) => api.delete(`/documents/${id}`).then(r => r.data),
}

// Streamed event shape sent by /chat/query/stream (see backend/app/api/chat.py)
export interface ChatStreamEvent {
  type: 'session' | 'status' | 'chunk' | 'done' | 'error'
  session_id?: number
  message?: string
  content?: string
  answer?: string
  sources?: any[]
  intent?: string
}

// Parses whatever SSE text has arrived so far into complete events + leftover.
// Exported on its own (separate from the network code) so it can be unit
// tested directly with plain strings, without needing to fake a real stream.
export function parseSSEBuffer(buffer: string): { events: ChatStreamEvent[]; remainder: string } {
  const chunks = buffer.split('\n\n')
  const remainder = chunks.pop() ?? ''

  const events: ChatStreamEvent[] = []
  for (const rawEvent of chunks) {
    const line = rawEvent.trim()
    if (!line.startsWith('data:')) continue
    const jsonStr = line.slice(5).trim()
    if (!jsonStr) continue
    try {
      events.push(JSON.parse(jsonStr) as ChatStreamEvent)
    } catch {
      // ignore a partial/malformed chunk rather than killing the stream
    }
  }

  return { events, remainder }
}

// Chat
export const chatApi = {
  // Original non-streaming call — left exactly as before. The /chat/query
  // backend endpoint it hits is also untouched, so nothing that already
  // depends on this keeps working unchanged.
  query: (query: string, session_id?: number) =>
    api.post('/chat/query', { query, session_id }).then(r => r.data),

  // New: streams Server-Sent Events from /chat/query/stream.
  // axios doesn't expose a browser-friendly readable stream for SSE, so this
  // uses the native fetch API directly. onEvent is called once per parsed
  // SSE "data:" line (already JSON.parsed) as soon as it arrives.
  queryStream: async (
    query: string,
    session_id: number | undefined,
    onEvent: (event: ChatStreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<void> => {
    const token = localStorage.getItem('token')

    const res = await fetch(`${BASE_URL}/chat/query/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({ query, session_id }),
      signal,
    })

    if (!res.ok) {
      if (res.status === 401) {
        localStorage.removeItem('token')
        window.location.href = '/login'
      }
      let detail = 'Something went wrong. Please try again.'
      try {
        const data = await res.json()
        detail = data?.detail || detail
      } catch {
        // response wasn't JSON (e.g. empty body) — keep default detail
      }
      throw new Error(detail)
    }
    if (!res.body) {
      throw new Error('Streaming is not supported in this browser.')
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      const { events, remainder } = parseSSEBuffer(buffer)
      buffer = remainder

      for (const event of events) {
        onEvent(event)
      }
    }
  },

  getSessions: () => api.get('/chat/sessions').then(r => r.data),
  getMessages: (session_id: number) =>
    api.get(`/chat/sessions/${session_id}`).then(r => r.data),
  deleteSession: (session_id: number) =>
    api.delete(`/chat/sessions/${session_id}`).then(r => r.data),
}

// Quiz
export const quizApi = {
  generateQuiz: (docId: number) =>
    api.post(`/quiz/generate/${docId}`).then(r => r.data),

  submitQuiz: (docId: number, answers: any[]) =>
    api.post(`/quiz/submit/${docId}`, { answers }).then(r => r.data),

  getQuizHistory: (docId: number) =>
    api.get(`/quiz/history/${docId}`).then(r => r.data),
}