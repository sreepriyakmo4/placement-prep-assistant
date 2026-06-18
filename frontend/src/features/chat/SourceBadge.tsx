// Drop-in replacement for the SourceBadge in ChatPage.tsx
// Shows filename, page number, heading, and colour-coded confidence

import { FileText, TrendingUp } from 'lucide-react'

interface Source {
  filename: string
  page_num?: number
  chunk_index: number
  heading?: string
  similarity?: number
  confidence?: string
  content_preview?: string
}

const confidenceColors: Record<string, string> = {
  'Very High': 'text-emerald-400 bg-emerald-400/10 border-emerald-400/20',
  'High':      'text-blue-400   bg-blue-400/10   border-blue-400/20',
  'Moderate':  'text-amber-400  bg-amber-400/10  border-amber-400/20',
  'Low':       'text-gray-400   bg-gray-400/10   border-gray-400/20',
}

export function SourceBadge({ sources }: { sources: string }) {
  let parsed: Source[] = []
  try {
    parsed = JSON.parse(sources)
  } catch {
    return null
  }
  if (!parsed.length) return null

  return (
    <div className="mt-3 pt-3 border-t border-white/5">
      <p className="text-xs text-gray-500 mb-2 font-medium uppercase tracking-wider flex items-center gap-1.5">
        <TrendingUp size={10} />
        Sources from your notes
      </p>
      <div className="flex flex-col gap-2">
        {parsed.map((s, i) => {
          const confStyle = confidenceColors[s.confidence || 'Low'] || confidenceColors['Low']
          return (
            <div
              key={i}
              className="bg-surface-3/60 border border-white/5 rounded-lg px-3 py-2"
            >
              <div className="flex items-center justify-between gap-2 mb-1">
                <span className="inline-flex items-center gap-1.5 text-xs text-gray-300 font-medium">
                  <FileText size={10} className="text-accent flex-shrink-0" />
                  <span className="truncate max-w-[160px]">{s.filename}</span>
                  {s.page_num && (
                    <span className="text-gray-600 flex-shrink-0">p.{s.page_num}</span>
                  )}
                </span>
                {s.confidence && (
                  <span className={`text-xs px-2 py-0.5 rounded-full border flex-shrink-0 ${confStyle}`}>
                    {s.confidence} {s.similarity ? `· ${Math.round(s.similarity * 100)}%` : ''}
                  </span>
                )}
              </div>
              {s.heading && (
                <p className="text-xs text-accent/70 mb-1 truncate">§ {s.heading}</p>
              )}
              {s.content_preview && (
                <p className="text-xs text-gray-600 line-clamp-2 leading-relaxed">
                  {s.content_preview}
                </p>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}