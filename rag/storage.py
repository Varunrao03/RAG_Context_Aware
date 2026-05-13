"""
rag/storage.py
--------------
In-memory vector store — holds chunks and their embeddings, exposes
raw cosine similarity search (no external vector database).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A text fragment with its source identifier and embedding."""
    text: str
    source: str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class SearchResult:
    """A single retrieval result."""
    rank: int
    chunk: Chunk
    score: float


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """
    In-memory vector store backed by a 2-D NumPy matrix.

    All embeddings are assumed to be unit-normed on insertion, so cosine
    similarity reduces to a plain dot product — O(N·D) per query, no index.

    This is intentionally simple and self-contained.  For production scale
    see the migration guide in docs/similarity_and_migration.md.
    """

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._matrix: Optional[np.ndarray] = None  # shape (N, D)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, chunks: List[Chunk]) -> None:
        """
        Add pre-embedded chunks to the store and rebuild the matrix.

        Args:
            chunks: Chunks whose .embedding field is already populated.

        Raises:
            ValueError: If any chunk is missing its embedding.
        """
        for c in chunks:
            if c.embedding is None:
                raise ValueError(
                    f"Chunk from '{c.source}' has no embedding. "
                    "Embed before adding to the store."
                )
        self._chunks.extend(chunks)
        self._rebuild_matrix()
        logger.debug("VectorStore now holds %d chunk(s).", len(self._chunks))

    def clear(self) -> None:
        """Remove all chunks and reset the matrix."""
        self._chunks = []
        self._matrix = None

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(self, query_vec: np.ndarray, top_k: int = 3) -> List[SearchResult]:
        """
        Return the top-K chunks by cosine similarity to query_vec.

        Similarity metric: cosine similarity via dot product on unit-norm vectors.
        See docs/similarity_and_migration.md for the rationale.

        Args:
            query_vec: 1-D unit-norm float32 array.
            top_k:     Number of results to return.

        Returns:
            List[SearchResult] sorted by descending score.

        Raises:
            ValueError: If the store is empty.
            ValueError: If top_k < 1.
        """
        if self._matrix is None or len(self._chunks) == 0:
            raise ValueError(
                "VectorStore is empty. Ingest documents before searching."
            )
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")

        scores: np.ndarray = self._matrix @ query_vec          # (N,)
        k = min(top_k, len(self._chunks))
        ranked = np.argsort(scores)[::-1][:k]

        return [
            SearchResult(rank=i + 1, chunk=self._chunks[idx], score=float(scores[idx]))
            for i, idx in enumerate(ranked)
        ]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def is_empty(self) -> bool:
        return len(self._chunks) == 0

    @property
    def embedding_dim(self) -> Optional[int]:
        """Dimensionality of stored embeddings, or None if store is empty."""
        return self._matrix.shape[1] if self._matrix is not None else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild_matrix(self) -> None:
        """Stack all chunk embeddings into a single (N, D) matrix."""
        self._matrix = np.vstack(
            [c.embedding for c in self._chunks]
        ).astype(np.float32)
