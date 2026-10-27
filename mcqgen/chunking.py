"""Simple sentence-aware chunking for source material."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    text: str


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving sentence punctuation."""
    text = text.replace("\x00", " ").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter that does not require a downloaded NLP model."""
    normalized = clean_text(text)
    if not normalized:
        return []
    # Sentence boundaries, plus blank-line boundaries for pasted notes.
    pieces = re.split(r"(?<=[.!?])\s+|\n+", normalized)
    return [piece.strip() for piece in pieces if piece.strip()]


def make_chunks(text: str, max_words: int = 150, overlap_sentences: int = 1) -> list[TextChunk]:
    """Group sentences into bounded chunks and include a small overlap for context."""
    if max_words < 30:
        raise ValueError("max_words must be at least 30")
    sentences = split_sentences(text)
    if not sentences:
        return []

    grouped: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        word_count = len(sentence.split())
        if current and current_words + word_count > max_words:
            grouped.append(" ".join(current))
            keep = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = list(keep)
            current_words = sum(len(part.split()) for part in current)
        current.append(sentence)
        current_words += word_count
        # A single long sentence is still kept intact rather than dropped.
    if current:
        final = " ".join(current)
        if not grouped or final != grouped[-1]:
            grouped.append(final)

    # Remove exact overlap-only repeats while keeping stable IDs.
    unique: list[str] = []
    for chunk in grouped:
        if not unique or chunk != unique[-1]:
            unique.append(chunk)
    return [TextChunk(chunk_id=f"C{i:02d}", text=chunk) for i, chunk in enumerate(unique, start=1)]
