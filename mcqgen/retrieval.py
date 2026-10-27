"""Hybrid lexical retrieval for grounding generated questions in source material.

The default retriever blends TF-IDF cosine similarity with query-token coverage.
It is intentionally local and deterministic: no external API or model download is needed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import TextChunk, make_chunks

_TOKEN = re.compile(r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    tfidf_score: float = 0.0
    lexical_score: float = 0.0
    reason: str = ""


class HybridRetriever:
    """Retrieve relevant passages with a transparent TF-IDF + lexical hybrid score."""

    def __init__(self, text: str, max_words: int = 150) -> None:
        self.chunks: list[TextChunk] = make_chunks(text, max_words=max_words)
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        if self.chunks:
            self.vectorizer = TfidfVectorizer(
                stop_words="english", ngram_range=(1, 2), max_features=12000,
                sublinear_tf=True, norm="l2"
            )
            try:
                self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in self.chunks])
            except ValueError:
                self.vectorizer = None
                self.matrix = None

    @staticmethod
    def _query_terms(query: str) -> set[str]:
        return {
            token.casefold() for token in _TOKEN.findall(query)
            if token.casefold() not in ENGLISH_STOP_WORDS
        }

    def retrieve(self, query: str = "key concepts, definitions, processes, and important facts", k: int = 4) -> list[RetrievedChunk]:
        if not self.chunks:
            return []
        limit = max(1, min(int(k), len(self.chunks)))
        query_terms = self._query_terms(query)
        if self.vectorizer is None or self.matrix is None or not query.strip():
            return [RetrievedChunk(c.chunk_id, c.text, 0.0, 0.0, 0.0, "source order fallback") for c in self.chunks[:limit]]

        tfidf = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        lexical = []
        for chunk in self.chunks:
            terms = self._query_terms(chunk.text)
            lexical.append(len(query_terms & terms) / max(1, len(query_terms)))
        lexical_scores = np.asarray(lexical, dtype=float)
        # TF-IDF handles weighted term overlap; lexical coverage helps short focus queries.
        hybrid = 0.78 * tfidf + 0.22 * lexical_scores
        ranked = sorted(range(len(hybrid)), key=lambda i: (-float(hybrid[i]), i))
        if not ranked or float(hybrid[ranked[0]]) <= 0:
            ranked = list(range(len(self.chunks)))
        chosen = ranked[:limit]
        results: list[RetrievedChunk] = []
        for idx in chosen:
            reasons = []
            if tfidf[idx] > 0:
                reasons.append("term-weight similarity")
            if lexical_scores[idx] > 0:
                reasons.append("query-term coverage")
            results.append(RetrievedChunk(
                self.chunks[idx].chunk_id,
                self.chunks[idx].text,
                round(float(hybrid[idx]), 6),
                round(float(tfidf[idx]), 6),
                round(float(lexical_scores[idx]), 6),
                " + ".join(reasons) if reasons else "fallback: no query overlap",
            ))
        return results


# Backwards-compatible import name used in the first version of StudyForge.
TfidfRetriever = HybridRetriever


class SemanticRetriever:
    """Optional embedding-based retriever. Install requirements-semantic.txt to use it."""

    def __init__(self, text: str, max_words: int = 150, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Semantic retrieval is optional. Install it with: pip install -r requirements-semantic.txt"
            ) from exc
        self.chunks = make_chunks(text, max_words=max_words)
        if not self.chunks:
            self.model = None
            self.embeddings = None
            return
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            [chunk.text for chunk in self.chunks], convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False
        )

    def retrieve(self, query: str = "key concepts, definitions, processes, and important facts", k: int = 4) -> list[RetrievedChunk]:
        if not self.chunks or self.model is None or self.embeddings is None:
            return []
        limit = max(1, min(int(k), len(self.chunks)))
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.embeddings @ query_embedding
        ranked = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:limit]
        return [RetrievedChunk(
            self.chunks[i].chunk_id, self.chunks[i].text, round(float(scores[i]), 6),
            0.0, 0.0, "semantic embedding similarity"
        ) for i in ranked]
