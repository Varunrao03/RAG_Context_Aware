"""
rag/embedding.py
----------------
Embedding layer — wraps vertexai.language_models.TextEmbeddingModel.

In production replace the _MockTextEmbeddingModel block with:
    import vertexai
    vertexai.init(project="YOUR_PROJECT", location="us-central1")
    from vertexai.language_models import TextEmbeddingModel
"""

from __future__ import annotations

import logging
from typing import List, Optional, Protocol, runtime_checkable

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vertex AI SDK surface (mocked)
# ---------------------------------------------------------------------------

class _MockTextEmbedding:
    """Mirrors vertexai.language_models.TextEmbedding."""
    def __init__(self, values: List[float]) -> None:
        self.values: List[float] = values


class _MockTextEmbeddingModel:
    """
    Mirrors vertexai.language_models.TextEmbeddingModel.
    Backed by sentence-transformers so vectors are semantically meaningful.

    Real SDK swap-in:
        from vertexai.language_models import TextEmbeddingModel
    """
    _encoder: Optional[SentenceTransformer] = None
    _ENCODER_NAME = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        if _MockTextEmbeddingModel._encoder is None:
            logger.info(
                "[Mock] Loading '%s' to back TextEmbeddingModel('%s')",
                self._ENCODER_NAME, model_name,
            )
            _MockTextEmbeddingModel._encoder = SentenceTransformer(self._ENCODER_NAME)

    @classmethod
    def from_pretrained(cls, model_name: str) -> "_MockTextEmbeddingModel":
        """Mirrors TextEmbeddingModel.from_pretrained(model_name)."""
        logger.debug("[Mock] TextEmbeddingModel.from_pretrained('%s')", model_name)
        return cls(model_name)

    def get_embeddings(self, texts: List[str]) -> List[_MockTextEmbedding]:
        """Mirrors model.get_embeddings(texts) → List[TextEmbedding]."""
        assert self._encoder is not None
        raw = self._encoder.encode(texts, show_progress_bar=False)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalised = raw / norms
        return [_MockTextEmbedding(row.tolist()) for row in normalised]


# Public alias — swap this line for the real import when going live
TextEmbeddingModel = _MockTextEmbeddingModel

# ---------------------------------------------------------------------------
# EmbeddingService
# ---------------------------------------------------------------------------

class EmbeddingService:
    """
    Thin wrapper around TextEmbeddingModel that returns normalised numpy arrays.

    Args:
        model_name: Vertex AI embedding model identifier.
    """

    def __init__(self, model_name: str = "text-embedding-004") -> None:
        self._model_name = model_name
        self._model = TextEmbeddingModel.from_pretrained(model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of strings and return a unit-norm (N, D) float32 array.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim), unit-normed rows.

        Raises:
            ValueError: If texts is empty.
            RuntimeError: If the model fails.
        """
        if not texts:
            raise ValueError("texts must be a non-empty list.")
        try:
            emb_objects = self._model.get_embeddings(texts)
        except Exception as exc:
            raise RuntimeError(
                f"TextEmbeddingModel failed on {len(texts)} text(s): {exc}"
            ) from exc

        matrix = np.array([e.values for e in emb_objects], dtype=np.float32)

        # Ensure unit norm (model mock already normalises, but be defensive)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return matrix / norms

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string and return a 1-D unit-norm float32 array."""
        return self.embed([text])[0]
