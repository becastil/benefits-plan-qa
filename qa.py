"""Core pipeline: PDF → page-aware chunks → embeddings → FAISS retrieval → Claude (via OpenRouter) with cited answers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-sonnet-4"
TOP_K = 5
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200

SYSTEM_PROMPT = """You are a careful benefits-plan analyst. You answer questions about
a Summary of Benefits and Coverage (SBC) or similar plan document using ONLY the
excerpts the user provides. Each excerpt is prefixed with [page N].

Rules:
1. Answer in plain English a non-technical HR admin or member would understand.
2. Keep answers short — 1 to 3 sentences. No filler, no hedging, no "I'd be happy to."
3. If the excerpts don't contain enough information, say so explicitly and do not guess.
4. Every answer MUST end with a line in this exact format:
   Sources: page X, page Y
   List only the pages you actually used. Use one page if that's all you cited.
5. Do not invent page numbers. If you can't cite a page, say so and write:
   Sources: (none found in document)"""


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


class PlanIndex:
    """In-memory FAISS index over chunk embeddings, with page metadata kept aligned."""

    def __init__(self, chunks: list[Chunk], model_name: str = EMBED_MODEL_NAME):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        embeddings = self.model.encode(
            [c.text for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        dim = embeddings.shape[1]
        # Inner product on L2-normalized vectors == cosine similarity.
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def retrieve(self, query: str, k: int = TOP_K) -> list[Chunk]:
        q = self.model.encode([query], normalize_embeddings=True,
                              show_progress_bar=False).astype("float32")
        _scores, ids = self.index.search(q, k)
        return [self.chunks[int(i)] for i in ids[0] if i != -1]


def _format_excerpts(chunks: Iterable[Chunk]) -> str:
    return "\n\n---\n\n".join(f"[page {c.page}]\n{c.text}" for c in chunks)


def _openrouter_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def answer(question: str, index: PlanIndex,
           model: str | None = None, k: int = TOP_K) -> str:
    """Retrieve top-k chunks and ask Claude (via OpenRouter) for a cited answer."""
    model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    top_chunks = index.retrieve(question, k=k)
    excerpts = _format_excerpts(top_chunks)
    user_msg = (
        f"Excerpts from the plan document:\n\n{excerpts}\n\n"
        f"Question: {question}"
    )
    client = _openrouter_client()
    response = client.chat.completions.create(
        model=model,
        max_tokens=400,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        extra_headers={
            "HTTP-Referer": "https://github.com/becastil/benefits-plan-qa",
            "X-Title": "benefits-plan-qa",
        },
    )
    return (response.choices[0].message.content or "").strip()


def build_index(pdf_path: str | Path) -> PlanIndex:
    """Convenience: PDF → pages → chunks → embedded FAISS index."""
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages)
    if not chunks:
        raise RuntimeError(f"No extractable text found in {pdf_path}. "
                           "If the PDF is scanned-only, OCR it first.")
    return PlanIndex(chunks)
