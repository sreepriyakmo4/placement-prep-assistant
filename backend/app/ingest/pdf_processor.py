import re
import fitz  # PyMuPDF
from typing import List, Dict
from app.core.config import settings


def extract_text_from_pdf(file_bytes: bytes) -> List[Dict]:
    """
    Extract text page by page, preserving page numbers and structure.
    Returns list of {page_num, text} dicts instead of one big string.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")  # plain text, preserves newlines
        text = text.strip()
        if text:
            pages.append({"page_num": page_num, "text": text})
    doc.close()
    return pages


def _clean_text(text: str) -> str:
    """Remove excessive whitespace while preserving paragraph breaks."""
    # Normalize multiple blank lines to double newline (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove trailing spaces on each line
    text = re.sub(r'[ \t]+\n', '\n', text)
    # Remove hyphenation at line breaks (common in PDFs)
    text = re.sub(r'-\n(\w)', r'\1', text)
    return text.strip()


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, keeping them intact."""
    # Split on sentence boundaries but keep the delimiter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def split_into_chunks(pages: List[Dict]) -> List[Dict]:
    """
    Semantic chunking strategy:
    1. Respect paragraph boundaries — never split mid-paragraph
    2. Target ~400 tokens (~1600 chars) per chunk for good retrieval
    3. Overlap by carrying the LAST full paragraph of previous chunk
       into the next chunk (context continuity)
    4. Track page number for every chunk (for source display)
    5. Detect headings and always start a new chunk at a heading
    
    Returns list of {content, page_num, chunk_index, heading} dicts.
    """
    TARGET_CHARS = 1400    # ~350 tokens — sweet spot for MiniLM retrieval
    MAX_CHARS = 2000       # hard cap
    OVERLAP_CHARS = 300    # carry last ~300 chars into next chunk

    heading_pattern = re.compile(
        r'^(?:'
        r'\d+[\.\)]\s+[A-Z]'           # "1. Introduction" or "1) Overview"
        r'|[A-Z][A-Z\s]{3,}$'          # "INTRODUCTION" all-caps headings
        r'|(?:Chapter|Unit|Section|Module|Topic)\s+\d+'  # "Chapter 1"
        r'|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:$'  # "Key Concepts:"
        r')',
        re.MULTILINE
    )

    chunks = []
    chunk_index = 0

    for page in pages:
        page_num = page["page_num"]
        text = _clean_text(page["text"])

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        current_chunk_paras = []
        current_len = 0
        current_heading = None
        overlap_text = ""  # last paragraph(s) from previous chunk

        def flush_chunk():
            nonlocal chunk_index, overlap_text
            if not current_chunk_paras:
                return

            content = "\n\n".join(current_chunk_paras)
            if overlap_text:
                content = overlap_text + "\n\n" + content

            content = content.strip()
            if len(content) < 50:  # skip tiny fragments
                return

            chunks.append({
                "content": content,
                "page_num": page_num,
                "chunk_index": chunk_index,
                "heading": current_heading or "",
            })
            chunk_index += 1

            # Overlap: carry last paragraph into next chunk
            if current_chunk_paras:
                last_para = current_chunk_paras[-1]
                overlap_text = last_para if len(last_para) <= OVERLAP_CHARS else last_para[-OVERLAP_CHARS:]
            else:
                overlap_text = ""

        for para in paragraphs:
            is_heading = bool(heading_pattern.match(para)) or (
                len(para) < 80 and para.endswith(':')
            )

            # Start a new chunk at every heading
            if is_heading and current_chunk_paras:
                flush_chunk()
                current_chunk_paras = []
                current_len = 0
                current_heading = para
                continue

            if is_heading:
                current_heading = para
                continue

            para_len = len(para)

            # If adding this paragraph exceeds MAX, flush first
            if current_len + para_len > MAX_CHARS and current_chunk_paras:
                flush_chunk()
                current_chunk_paras = []
                current_len = 0

            current_chunk_paras.append(para)
            current_len += para_len

            # Flush when we hit TARGET naturally at a paragraph boundary
            if current_len >= TARGET_CHARS:
                flush_chunk()
                current_chunk_paras = []
                current_len = 0

        # Flush remaining paragraphs for this page
        if current_chunk_paras:
            flush_chunk()

    return chunks
