"""
rag/ingestion.py
----------------
IngestionManager — the single class responsible for managing the ingestion
of a text dataset into the vector store.

Responsibilities:
  1. Accept a dataset of 5-10+ technical paragraphs (plain-text strings)
  2. Split each document into overlapping chunks
  3. Embed each chunk via EmbeddingService (TextEmbeddingModel)
  4. Store embedded chunks in VectorStore
  5. Expose the populated VectorStore for retrieval
"""

from __future__ import annotations

import logging
from typing import List, Optional

from rag.embedding import EmbeddingService
from rag.storage import Chunk, VectorStore

logger = logging.getLogger(__name__)


class IngestionManager:
    """
    Manages the full ingestion pipeline for a text dataset.

    Takes raw text documents, chunks them with a sliding window, embeds
    each chunk via TextEmbeddingModel, and loads them into an in-memory
    VectorStore ready for similarity search.

    Args:
        embedding_service: EmbeddingService wrapping TextEmbeddingModel.
                           A default instance is created if not provided.
        chunk_size:        Maximum characters per chunk (default 300).
        chunk_overlap:     Character overlap between consecutive chunks (default 50).
        min_chunk_size:    Minimum characters required to keep a chunk (default 30).

    Example:
        manager = IngestionManager()
        manager.ingest(documents, source_ids)
        store = manager.store   # pass to RawRetriever or ExpandingRetriever
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        min_chunk_size: int = 30,
    ) -> None:
        self._embedder = embedding_service or EmbeddingService()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self._store = VectorStore()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        documents: List[str],
        source_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Ingest a dataset of plain-text documents into the vector store.

        Designed for datasets of 5-10+ technical paragraphs, though there
        is no upper limit — the store is purely in-memory.

        Args:
            documents:  List of raw text strings (one per document/paragraph).
            source_ids: Optional parallel list of document identifiers.
                        Defaults to "doc_0", "doc_1", …

        Raises:
            RuntimeError: If the embedding model fails on any batch.
        """
        if source_ids is None:
            source_ids = [f"doc_{i}" for i in range(len(documents))]

        new_chunks: List[Chunk] = []

        for doc, src in zip(documents, source_ids):
            # Skip empty or whitespace-only documents
            if not doc or not doc.strip():
                logger.warning("Skipping empty document: '%s'", src)
                continue

            chunks = self._split(doc.strip(), src)
            logger.info("'%s' → %d chunk(s)", src, len(chunks))
            new_chunks.extend(chunks)

        if not new_chunks:
            logger.warning("No valid chunks produced from the provided dataset.")
            return

        # Embed all chunks in one batch call to TextEmbeddingModel
        try:
            embeddings = self._embedder.embed([c.text for c in new_chunks])
        except RuntimeError:
            raise

        for chunk, emb in zip(new_chunks, embeddings):
            chunk.embedding = emb

        self._store.add(new_chunks)
        logger.info(
            "Ingestion complete: %d document(s) → %d chunk(s) in VectorStore.",
            len(documents), len(self._store),
        )

    def clear(self) -> None:
        """Remove all ingested data and reset the store."""
        self._store.clear()
        logger.info("IngestionManager: store cleared.")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def store(self) -> VectorStore:
        """The populated VectorStore, ready for similarity search."""
        return self._store

    @property
    def chunk_count(self) -> int:
        """Total number of chunks currently in the store."""
        return len(self._store)

    @property
    def is_empty(self) -> bool:
        return self._store.is_empty

    def summary(self) -> str:
        """Return a one-line summary of the current ingestion state."""
        return (
            f"IngestionManager | chunks: {self.chunk_count} | "
            f"embedding_dim: {self._store.embedding_dim} | "
            f"chunk_size: {self.chunk_size} | overlap: {self.chunk_overlap}"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _split(self, text: str, source: str) -> List[Chunk]:
        """Slide a fixed-size window over text and return Chunk objects."""
        chunks: List[Chunk] = []
        start = 0
        while start < len(text):
            fragment = text[start: start + self.chunk_size].strip()
            if len(fragment) >= self.min_chunk_size:
                chunks.append(Chunk(text=fragment, source=source))
            elif fragment:
                logger.debug(
                    "Discarding short chunk (%d chars) from '%s'.",
                    len(fragment), source,
                )
            start += self.chunk_size - self.chunk_overlap
        return chunks
