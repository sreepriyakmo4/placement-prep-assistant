import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import toast from 'react-hot-toast'
import {
  Brain, Upload, FileText, Trash2, MessageSquare,
  Loader2, CheckCircle, XCircle, Clock, LogOut, ChevronRight, Zap
} from 'lucide-react'
import { documentsApi } from '@/lib/api'
import { useAuthStore } from '@/lib/auth-store'

const STATUS_CONFIG = {
  ready: { icon: CheckCircle, color: 'text-emerald-400', label: 'Ready' },
  processing: { icon: Loader2, color: 'text-amber-400', label: 'Processing', spin: true },
  pending: { icon: Clock, color: 'text-[#9494b8]', label: 'Pending' },
  failed: { icon: XCircle, color: 'text-rose-400', label: 'Failed' },
}

function formatBytes(bytes: number) {
  if (!bytes) return '—'
  const mb = bytes / 1024 / 1024
  return mb > 1 ? `${mb.toFixed(1)} MB` : `${(bytes / 1024).toFixed(0)} KB`
}

export default function DashboardPage() {
  const navigate = useNavigate()
  const { user, clearAuth } = useAuthStore()
  const qc = useQueryClient()
  const fileRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  const { data: docs = [], isLoading } = useQuery({
    queryKey: ['documents'],
    queryFn: () => documentsApi.list().then(r => r.data),
    refetchInterval: 5000, // poll for status updates
  })

  const uploadMutation = useMutation({
    mutationFn: (file: File) => documentsApi.upload(file),
    onSuccess: () => {
      toast.success('PDF uploaded! Processing in background...')
      qc.invalidateQueries({ queryKey: ['documents'] })
    },
    onError: (err: any) => toast.error(err.response?.data?.detail || 'Upload failed'),
  })

  const deleteMutation = useMutation({
    mutationFn: (id: number) => documentsApi.delete(id),
    onSuccess: () => {
      toast.success('Document deleted')
      qc.invalidateQueries({ queryKey: ['documents'] })
    },
  })

  const handleFiles = (files: FileList | null) => {
    if (!files?.length) return
    const file = files[0]
    if (file.type !== 'application/pdf') {
      toast.error('Only PDF files are supported')
      return
    }
    uploadMutation.mutate(file)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    handleFiles(e.dataTransfer.files)
  }

  const readyCount = docs.filter((d: any) => d.status === 'ready').length

  return (
    <div className="min-h-screen bg-grid-pattern">
      {/* Header */}
      <header className="glass border-b border-[rgba(148,148,184,0.15)] sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-[#7c6af7]/20 border border-[#7c6af7]/30 flex items-center justify-center">
              <Brain className="w-4 h-4 text-[#a08fff]" />
            </div>
            <span className="font-semibold text-white">PlacementAI</span>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate('/chat')}
              className="btn-primary flex items-center gap-2"
            >
              <MessageSquare className="w-4 h-4" />
              Start Chat
            </button>
            <span className="text-sm text-[#9494b8] hidden sm:block">
              {user?.full_name || user?.email}
            </span>
            <button
              onClick={() => { clearAuth(); navigate('/login') }}
              className="p-2 rounded-lg text-[#9494b8] hover:text-white hover:bg-[#1a1a47] transition-colors"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-10">
        {/* Stats */}
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mb-8">
          {[
            { label: 'Uploaded PDFs', value: docs.length, icon: FileText, color: 'text-[#a08fff]' },
            { label: 'Ready to Query', value: readyCount, icon: CheckCircle, color: 'text-emerald-400' },
            { label: 'Total Chunks', value: docs.reduce((a: number, d: any) => a + (d.chunk_count || 0), 0), icon: Zap, color: 'text-amber-400' },
          ].map(stat => (
            <div key={stat.label} className="card p-4">
              <div className="flex items-center gap-3">
                <stat.icon className={`w-5 h-5 ${stat.color}`} />
                <div>
                  <div className="text-2xl font-semibold text-white">{stat.value}</div>
                  <div className="text-xs text-[#9494b8]">{stat.label}</div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Upload zone */}
        <div
          className={`border-2 border-dashed rounded-xl p-10 text-center transition-all duration-200 mb-8 cursor-pointer ${
            dragging
              ? 'border-[#7c6af7] bg-[#7c6af7]/10'
              : 'border-[rgba(148,148,184,0.2)] hover:border-[#7c6af7]/50 hover:bg-[#7c6af7]/5'
          }`}
          onClick={() => fileRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          <input
            ref={fileRef}
            type="file"
            accept=".pdf"
            className="hidden"
            onChange={e => handleFiles(e.target.files)}
          />
          {uploadMutation.isPending ? (
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="w-10 h-10 text-[#7c6af7] animate-spin" />
              <p className="text-[#9494b8] text-sm">Uploading PDF...</p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3">
              <div className="w-14 h-14 rounded-2xl bg-[#7c6af7]/10 border border-[#7c6af7]/20 flex items-center justify-center">
                <Upload className="w-6 h-6 text-[#a08fff]" />
              </div>
              <div>
                <p className="text-white font-medium">Drop your PDF here</p>
                <p className="text-[#9494b8] text-sm mt-1">
                  or click to browse • up to {50}MB
                </p>
              </div>
              <p className="text-xs text-[#5555a0]">
                Supports notes, textbooks, previous year questions
              </p>
            </div>
          )}
        </div>

        {/* Documents list */}
        <div>
          <h2 className="text-lg font-semibold text-white mb-4">Study Materials</h2>

          {isLoading ? (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="w-6 h-6 text-[#7c6af7] animate-spin" />
            </div>
          ) : docs.length === 0 ? (
            <div className="card p-12 text-center">
              <FileText className="w-10 h-10 text-[#5555a0] mx-auto mb-3" />
              <p className="text-[#9494b8]">No documents yet</p>
              <p className="text-sm text-[#5555a0] mt-1">Upload a PDF to get started</p>
            </div>
          ) : (
            <div className="space-y-2">
              {docs.map((doc: any) => {
                const cfg = STATUS_CONFIG[doc.status as keyof typeof STATUS_CONFIG] || STATUS_CONFIG.pending
                return (
                  <div key={doc.id} className="card p-4 flex items-center gap-4 group hover:border-[rgba(148,148,184,0.3)] transition-colors">
                    <div className="w-10 h-10 rounded-lg bg-[#7c6af7]/10 flex items-center justify-center flex-shrink-0">
                      <FileText className="w-5 h-5 text-[#a08fff]" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-white font-medium text-sm truncate">{doc.original_filename}</p>
                      <div className="flex items-center gap-3 mt-0.5">
                        <span className="text-xs text-[#5555a0]">{formatBytes(doc.file_size)}</span>
                        {doc.chunk_count > 0 && (
                          <span className="text-xs text-[#5555a0]">{doc.chunk_count} chunks</span>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <div className={`flex items-center gap-1.5 text-xs font-medium ${cfg.color}`}>
                        <cfg.icon className={`w-3.5 h-3.5 ${(cfg as any).spin ? 'animate-spin' : ''}`} />
                        {cfg.label}
                      </div>
                      {doc.status === 'ready' && (
                        <button
                          onClick={() => navigate(`/chat?doc=${doc.id}`)}
                          className="p-1.5 rounded-lg text-[#9494b8] hover:text-[#a08fff] hover:bg-[#7c6af7]/10 transition-colors"
                          title="Chat about this document"
                        >
                          <ChevronRight className="w-4 h-4" />
                        </button>
                      )}
                      <button
                        onClick={() => deleteMutation.mutate(doc.id)}
                        disabled={deleteMutation.isPending}
                        className="p-1.5 rounded-lg text-[#9494b8] hover:text-rose-400 hover:bg-rose-500/10 transition-colors opacity-0 group-hover:opacity-100"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* Go to chat CTA */}
        {readyCount > 0 && (
          <div className="mt-8 card p-6 flex items-center justify-between">
            <div>
              <p className="text-white font-medium">Ready to practice?</p>
              <p className="text-sm text-[#9494b8] mt-0.5">
                {readyCount} document{readyCount !== 1 ? 's' : ''} indexed and ready for Q&A
              </p>
            </div>
            <button onClick={() => navigate('/chat')} className="btn-primary flex items-center gap-2">
              <MessageSquare className="w-4 h-4" />
              Open Chat
            </button>
          </div>
        )}
      </main>
    </div>
  )
}
