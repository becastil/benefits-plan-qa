"""Core pipeline. Builds up over subsequent commits.

This commit: PDF text extraction with page-number preservation and
chunking utilities. No embeddings or LLM calls yet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200


@dataclass
class Chunk:
    text: str
    page: int


def extract_pages(pdf_path: str | Path) -> list[str]:
    """Return a list of page texts, indexed from 0 (page 1 is index 0)."""
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for raw in reader.pages:
        text = raw.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        pages.append(text)
    return pages


def chunk_pages(pages: list[str], chunk_chars: int = CHUNK_CHARS,
                overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Split each page into overlapping chunks while preserving page provenance."""
    chunks: list[Chunk] = []
    for idx, text in enumerate(pages):
        if not text:
            continue
        page_no = idx + 1
        if len(text) <= chunk_chars:
            chunks.append(Chunk(text=text, page=page_no))
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            chunks.append(Chunk(text=text[start:end], page=page_no))
            if end == len(text):
                break
            start = end - overlap
    return chunks
