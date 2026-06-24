import React, { useCallback, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { documentsApi } from '../../lib/api'
import { FileText, Trash2, CheckCircle, Clock, AlertCircle, Loader2, Plus } from 'lucide-react'
import { formatDate } from '../../lib/utils'

interface Document {
  id: number
  filename: string
  status: 'pending' | 'processing' | 'done' | 'failed'
  uploaded_at: string
}

const statusConfig = {
  done:       { icon: CheckCircle, color: 'text-emerald-400', bg: 'bg-emerald-400/10', label: 'Ready' },
  processing: { icon: Loader2,     color: 'text-amber-400',   bg: 'bg-amber-400/10',   label: 'Processing', spin: true },
  pending:    { icon: Clock,       color: 'text-gray-400',    bg: 'bg-gray-400/10',    label: 'Pending' },
  failed:     { icon: AlertCircle, color: 'text-red-400',     bg: 'bg-red-400/10',     label: 'Failed' },
}

export default function DocumentsPanel() {
  const qc = useQueryClient()
  const [dragOver, setDragOver] = useState(false)
  const [uploading, setUploading] = useState(false)

  const { data: docs = [], isLoading } = useQuery<Document[]>({
    queryKey: ['documents'],
    queryFn: documentsApi.list,
    // Poll every 5s so status updates (pending → processing → done) are visible
    refetchInterval: 5000,
    // Always refetch when the component mounts — this ensures that after login
    // the document list is fetched fresh with the new user's token, not served
    // from a potentially stale cache.
    refetchOnMount: true,
    // Also refetch when the browser tab regains focus
    refetchOnWindowFocus: true,
    // staleTime: 0 means the cache is immediately considered stale,
    // so the next render always triggers a background refetch.
    staleTime: 0,
  })

  const deleteMutation = useMutation({
    mutationFn: documentsApi.delete,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['documents'] }),
  })

  const handleUpload = async (file: File) => {
    if (!file.name.endsWith('.pdf')) return alert('Only PDF files are supported')
    setUploading(true)
    try {
      await documentsApi.upload(file)
      qc.invalidateQueries({ queryKey: ['documents'] })
    } catch (e: any) {
      alert(e.response?.data?.detail || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleUpload(file)
  }, [])

  return (
    <div className="flex flex-col h-full">
      <div className="p-6 border-b border-white/5">
        <h2 className="font-display text-lg font-700 text-white mb-1">Study Materials</h2>
        <p className="text-gray-500 text-xs">Upload PDFs to power your AI assistant</p>
      </div>

      <div className="p-4 flex-1 overflow-y-auto">
        {/* Upload zone */}
        <label
          className={`relative block w-full border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all mb-4 ${
            dragOver ? 'border-accent bg-accent/10' : 'border-white/10 hover:border-accent/40 hover:bg-accent/5'
          }`}
          onDrop={onDrop}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
        >
          <input
            type="file"
            accept=".pdf"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
            disabled={uploading}
          />
          {uploading ? (
            <div className="flex flex-col items-center gap-2">
              <Loader2 size={24} className="text-accent animate-spin" />
              <span className="text-sm text-gray-400">Uploading...</span>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2">
              <div className="w-10 h-10 bg-accent/10 rounded-xl flex items-center justify-center">
                <Plus size={20} className="text-accent" />
              </div>
              <span className="text-sm text-gray-300 font-medium">Drop PDF here</span>
              <span className="text-xs text-gray-600">or click to browse</span>
            </div>
          )}
        </label>

        {/* Document list */}
        {isLoading ? (
          <div className="flex justify-center py-8">
            <Loader2 size={20} className="animate-spin text-gray-500" />
          </div>
        ) : docs.length === 0 ? (
          <div className="text-center py-8 text-gray-600 text-sm">
            No documents yet. Upload a PDF to get started.
          </div>
        ) : (
          <div className="space-y-2">
            {docs.map((doc) => {
              const cfg = statusConfig[doc.status] || statusConfig.pending
              const Icon = cfg.icon
              return (
                <div
                  key={doc.id}
                  className="flex items-start gap-3 bg-surface-2 border border-white/5 rounded-xl p-3 group hover:border-white/10 transition-all"
                >
                  <div className="w-9 h-9 bg-surface-3 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5">
                    <FileText size={16} className="text-accent" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-white font-medium truncate">{doc.filename}</p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full ${cfg.bg} ${cfg.color}`}>
                        <Icon size={10} className={(cfg as any).spin ? 'animate-spin' : ''} />
                        {cfg.label}
                      </span>
                      <span className="text-xs text-gray-600">{formatDate(doc.uploaded_at)}</span>
                    </div>
                  </div>
                  <button
                    onClick={() => deleteMutation.mutate(doc.id)}
                    className="opacity-0 group-hover:opacity-100 text-gray-600 hover:text-red-400 transition-all p-1"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}