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

// Chat
export const chatApi = {
  query: (query: string, session_id?: number) =>
    api.post('/chat/query', { query, session_id }).then(r => r.data),
  getSessions: () => api.get('/chat/sessions').then(r => r.data),
  getMessages: (session_id: number) =>
    api.get(`/chat/sessions/${session_id}`).then(r => r.data),
  deleteSession: (session_id: number) =>
    api.delete(`/chat/sessions/${session_id}`).then(r => r.data),
}
