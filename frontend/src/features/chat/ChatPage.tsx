import React, { useState, useRef, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { chatApi } from '../../lib/api'
import { useAuth } from '../auth/AuthContext'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  Send, Plus, Trash2, MessageSquare, BookOpen, HelpCircle,
  ClipboardList, Users, ChevronLeft, ChevronRight, LogOut, Loader2,
  FileText, Sparkles, Bot
} from 'lucide-react'
import DocumentsPanel from '../documents/DocumentsPanel'
import { formatDate } from '../../lib/utils'
import { SourceBadge } from './SourceBadge'

interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
  sources?: string
  created_at: string
  // true while this assistant message is still streaming in (no full
  // content yet, or content is actively growing) — used to show the
  // "searching/generating" status text inside the bubble instead of a
  // markdown render of a half-finished string.
  streaming?: boolean
}

interface Session {
  id: number
  title: string
  created_at: string
}

const modeInfo = {
  qa: { label: 'Q&A', icon: HelpCircle, color: 'text-blue-400', bg: 'bg-blue-400/10', border: 'border-blue-400/20' },
  explain: { label: 'Explain', icon: BookOpen, color: 'text-purple-400', bg: 'bg-purple-400/10', border: 'border-purple-400/20' },
  quiz: { label: 'Quiz', icon: ClipboardList, color: 'text-amber-400', bg: 'bg-amber-400/10', border: 'border-amber-400/20' },
  interview: { label: 'Interview', icon: Users, color: 'text-emerald-400', bg: 'bg-emerald-400/10', border: 'border-emerald-400/20' },
}

function ThinkingDots() {
  return (
    <div className="flex items-center gap-1.5 py-1">
      {[0, 1, 2].map(i => (
        <div
          key={i}
          className="w-2 h-2 rounded-full bg-accent"
          style={{ animation: `thinking 1.4s infinite ${i * 0.2}s` }}
        />
      ))}
    </div>
  )
}

export default function ChatPage() {
  const { user, logout } = useAuth()
  const qc = useQueryClient()
  const [currentSessionId, setCurrentSessionId] = useState<number | null>(null)
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [thinking, setThinking] = useState(false)
  // Live status text shown inside the streaming bubble before any tokens
  // have arrived yet, e.g. "🔍 Searching relevant documents..."
  const [statusText, setStatusText] = useState('')
  const [lastIntent, setLastIntent] = useState<string>('qa')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [docsOpen, setDocsOpen] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const { data: sessions = [] } = useQuery<Session[]>({
    queryKey: ['sessions'],
    queryFn: chatApi.getSessions,
  })

  const loadSession = async (id: number) => {
    setCurrentSessionId(id)
    const msgs = await chatApi.getMessages(id)
    setMessages(msgs)
  }

  const deleteSession = useMutation({
    mutationFn: chatApi.deleteSession,
    onSuccess: (_, id) => {
      qc.invalidateQueries({ queryKey: ['sessions'] })
      if (currentSessionId === id) {
        setCurrentSessionId(null)
        setMessages([])
      }
    },
  })

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, thinking])

  const handleSend = async () => {
    const q = input.trim()
    if (!q || thinking) return
    setInput('')

    const tempUserMsg: Message = {
      id: Date.now(),
      role: 'user',
      content: q,
      created_at: new Date().toISOString(),
    }

    // Placeholder assistant bubble that we fill in as chunks stream in,
    // instead of waiting for the whole answer before showing anything.
    const assistantId = Date.now() + 1
    const placeholderAssistantMsg: Message = {
      id: assistantId,
      role: 'assistant',
      content: '',
      created_at: new Date().toISOString(),
      streaming: true,
    }

    setMessages(prev => [...prev, tempUserMsg, placeholderAssistantMsg])
    setThinking(true)
    setStatusText('🔍 Searching relevant documents...')

    let streamedContent = ''
    let resolvedSessionId: number | undefined

    try {
      await chatApi.queryStream(q, currentSessionId || undefined, (event) => {
        if (event.type === 'session' && event.session_id) {
          resolvedSessionId = event.session_id
        } else if (event.type === 'status' && event.message) {
          setStatusText(event.message)
        } else if (event.type === 'chunk' && event.content) {
          streamedContent += event.content
          setMessages(prev => prev.map(m =>
            m.id === assistantId
              ? { ...m, content: streamedContent, streaming: true }
              : m
          ))
        } else if (event.type === 'done') {
          const finalSources = event.sources ? JSON.stringify(event.sources) : undefined
          setLastIntent(event.intent || 'qa')
          setMessages(prev => prev.map(m =>
            m.id === assistantId
              ? { ...m, content: event.answer ?? streamedContent, sources: finalSources, streaming: false }
              : m
          ))
        } else if (event.type === 'error' && event.message) {
          setMessages(prev => prev.map(m =>
            m.id === assistantId
              ? { ...m, content: '⚠️ ' + event.message, streaming: false }
              : m
          ))
        }
      })

      if (!currentSessionId && resolvedSessionId) {
        setCurrentSessionId(resolvedSessionId)
        qc.invalidateQueries({ queryKey: ['sessions'] })
      }
    } catch (e: any) {
      setMessages(prev => prev.map(m =>
        m.id === assistantId
          ? { ...m, content: '⚠️ Error: ' + (e?.message || 'Something went wrong. Please try again.'), streaming: false }
          : m
      ))
    } finally {
      setThinking(false)
      setStatusText('')
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const startNewChat = () => {
    setCurrentSessionId(null)
    setMessages([])
    setLastIntent('qa')
  }

  const modeCfg = modeInfo[lastIntent as keyof typeof modeInfo] || modeInfo.qa
  const ModeIcon = modeCfg.icon

  return (
    <div className="flex h-screen bg-surface text-white overflow-hidden">
      {/* Sidebar */}
      <div className={`flex flex-col bg-surface-1 border-r border-white/5 transition-all duration-300 ${sidebarCollapsed ? 'w-14' : 'w-64'}`}>
        {/* Logo */}
        <div className="flex items-center gap-3 p-4 border-b border-white/5">
          <div className="w-8 h-8 bg-accent rounded-lg flex items-center justify-center flex-shrink-0">
            <BookOpen size={16} className="text-white" />
          </div>
          {!sidebarCollapsed && (
            <span className="font-display font-700 text-white text-base">PlacementAI</span>
          )}
        </div>

        {/* New Chat */}
        <div className="p-3 border-b border-white/5">
          <button
            onClick={startNewChat}
            className={`w-full flex items-center gap-2 bg-accent/10 hover:bg-accent/20 border border-accent/20 rounded-xl transition-all text-accent font-medium text-sm ${sidebarCollapsed ? 'justify-center p-2' : 'px-3 py-2.5'}`}
          >
            <Plus size={16} />
            {!sidebarCollapsed && 'New Chat'}
          </button>
        </div>

        {/* Sessions */}
        <div className="flex-1 overflow-y-auto p-2">
          {!sidebarCollapsed && sessions.map((s) => (
            <div
              key={s.id}
              className={`group flex items-center gap-2 rounded-xl px-3 py-2.5 cursor-pointer transition-all mb-1 ${
                currentSessionId === s.id ? 'bg-accent/15 text-white' : 'text-gray-400 hover:bg-surface-2 hover:text-white'
              }`}
              onClick={() => loadSession(s.id)}
            >
              <MessageSquare size={14} className="flex-shrink-0" />
              <span className="flex-1 text-xs truncate">{s.title}</span>
              <button
                onClick={(e) => { e.stopPropagation(); deleteSession.mutate(s.id) }}
                className="opacity-0 group-hover:opacity-100 text-gray-600 hover:text-red-400 transition-all"
              >
                <Trash2 size={12} />
              </button>
            </div>
          ))}
        </div>

        {/* Docs + User */}
        <div className="p-3 border-t border-white/5 space-y-2">
          <button
            onClick={() => setDocsOpen(true)}
            className={`w-full flex items-center gap-2 text-gray-400 hover:text-white hover:bg-surface-2 rounded-xl transition-all text-sm ${sidebarCollapsed ? 'justify-center p-2' : 'px-3 py-2'}`}
          >
            <FileText size={15} />
            {!sidebarCollapsed && 'Study Materials'}
          </button>
          {!sidebarCollapsed && (
            <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-surface-2">
              <div className="w-6 h-6 bg-accent/20 rounded-full flex items-center justify-center">
                <span className="text-xs text-accent font-bold">{user?.email?.[0].toUpperCase()}</span>
              </div>
              <span className="flex-1 text-xs text-gray-400 truncate">{user?.email}</span>
              <button onClick={logout} className="text-gray-600 hover:text-red-400 transition-colors">
                <LogOut size={13} />
              </button>
            </div>
          )}
        </div>

        {/* Collapse toggle */}
        <button
          onClick={() => setSidebarCollapsed(v => !v)}
          className="p-3 border-t border-white/5 text-gray-600 hover:text-white transition-colors flex justify-center"
        >
          {sidebarCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/5 bg-surface-1/50 backdrop-blur">
          <div className="flex items-center gap-3">
            <h1 className="font-display font-700 text-white text-lg">
              {currentSessionId ? sessions.find(s => s.id === currentSessionId)?.title || 'Chat' : 'New Chat'}
            </h1>
          </div>
          <div className={`inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full border ${modeCfg.bg} ${modeCfg.color} ${modeCfg.border}`}>
            <ModeIcon size={12} />
            {modeCfg.label} Mode
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
          {messages.length === 0 && !thinking && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-16 h-16 bg-accent/10 rounded-2xl flex items-center justify-center mb-4">
                <Sparkles size={28} className="text-accent" />
              </div>
              <h2 className="font-display text-2xl font-700 text-white mb-2">Ready to prepare?</h2>
              <p className="text-gray-500 text-sm max-w-sm mb-8">
                Ask any placement question, request an explanation, generate a quiz, or start a mock interview.
              </p>
              <div className="grid grid-cols-2 gap-3 max-w-sm w-full">
                {[
                  { q: 'What is deadlock?', mode: 'Q&A' },
                  { q: 'Explain binary search with examples', mode: 'Explain' },
                  { q: 'Generate quiz on OS concepts', mode: 'Quiz' },
                  { q: 'Mock interview: data structures', mode: 'Interview' },
                ].map(({ q, mode }) => (
                  <button
                    key={q}
                    onClick={() => { setInput(q); textareaRef.current?.focus() }}
                    className="text-left bg-surface-2 hover:bg-surface-3 border border-white/5 hover:border-accent/20 rounded-xl p-3.5 transition-all group"
                  >
                    <p className="text-xs text-gray-300 group-hover:text-white transition-colors leading-relaxed">{q}</p>
                    <span className="text-xs text-gray-600 mt-1.5 block">{mode}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <div key={msg.id} className={`flex gap-4 animate-slide-up ${msg.role === 'user' ? 'justify-end' : ''}`}>
              {msg.role === 'assistant' && (
                <div className="w-8 h-8 bg-accent rounded-xl flex items-center justify-center flex-shrink-0 mt-0.5">
                  <Bot size={16} className="text-white" />
                </div>
              )}
              <div className={`max-w-3xl ${msg.role === 'user' ? 'order-first' : ''}`}>
                {msg.role === 'user' ? (
                  <div className="bg-accent/15 border border-accent/20 rounded-2xl rounded-tr-sm px-4 py-3 text-sm text-white">
                    {msg.content}
                  </div>
                ) : (
                  <div className="bg-surface-2 border border-white/5 rounded-2xl rounded-tl-sm px-5 py-4">
                    {msg.content ? (
                      <>
                        <div className="markdown-body text-sm text-gray-200 leading-relaxed">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                        </div>
                        {/* A small cursor while still streaming, so it's clear more text is coming */}
                        {msg.streaming && (
                          <span className="inline-block w-1.5 h-4 bg-accent/70 ml-0.5 align-middle animate-pulse" />
                        )}
                        {!msg.streaming && msg.sources && <SourceBadge sources={msg.sources} />}
                      </>
                    ) : (
                      // No tokens yet for this message — show the live status
                      // ("Searching relevant documents..." / "Generating response...")
                      // instead of a blank bubble.
                      <div className="flex items-center gap-2.5 text-sm text-gray-400">
                        <ThinkingDots />
                        {msg.streaming && statusText && <span>{statusText}</span>}
                      </div>
                    )}
                  </div>
                )}
                <p className="text-xs text-gray-700 mt-1.5 px-1">{formatDate(msg.created_at)}</p>
              </div>
              {msg.role === 'user' && (
                <div className="w-8 h-8 bg-surface-3 border border-white/10 rounded-xl flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-xs text-gray-400 font-bold">{user?.email?.[0].toUpperCase()}</span>
                </div>
              )}
            </div>
          ))}

          <div ref={bottomRef} />
        </div>

        {/* Input area */}
        <div className="p-4 border-t border-white/5 bg-surface-1/30 backdrop-blur">
          <div className="max-w-4xl mx-auto">
            <div className="flex gap-3 items-end bg-surface-2 border border-white/10 focus-within:border-accent/40 rounded-2xl px-4 py-3 transition-all">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question, request explanation, quiz, or interview prep..."
                className="flex-1 bg-transparent text-white placeholder-gray-600 text-sm resize-none outline-none leading-relaxed min-h-[40px] max-h-32"
                rows={1}
                style={{ height: 'auto' }}
                onInput={e => {
                  const t = e.target as HTMLTextAreaElement
                  t.style.height = 'auto'
                  t.style.height = Math.min(t.scrollHeight, 128) + 'px'
                }}
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || thinking}
                className="w-9 h-9 flex-shrink-0 bg-accent hover:bg-accent-dim disabled:opacity-30 disabled:cursor-not-allowed rounded-xl flex items-center justify-center transition-all"
              >
                {thinking ? <Loader2 size={16} className="animate-spin text-white" /> : <Send size={15} className="text-white" />}
              </button>
            </div>
            <p className="text-center text-xs text-gray-700 mt-2">
              Shift+Enter for new line · Answers grounded in your uploaded materials
            </p>
          </div>
        </div>
      </div>

      {/* Documents drawer */}
      {docsOpen && (
        <div className="fixed inset-0 z-50 flex" onClick={() => setDocsOpen(false)}>
          <div className="flex-1" />
          <div
            className="w-80 bg-surface-1 border-l border-white/10 h-full shadow-2xl animate-slide-up flex flex-col"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between p-4 border-b border-white/5">
              <span className="text-sm font-semibold text-white">Documents</span>
              <button onClick={() => setDocsOpen(false)} className="text-gray-500 hover:text-white transition-colors">
                <ChevronRight size={18} />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto">
              <DocumentsPanel />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}