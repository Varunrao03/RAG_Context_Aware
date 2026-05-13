"""
tests/test_ingestion.py
-----------------------
Unit tests for rag/ingestion.py — IngestionManager.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from rag.ingestion import IngestionManager
from rag.embedding import EmbeddingService
from rag.storage import VectorStore


# ---------------------------------------------------------------------------
# Sample dataset — 9 technical paragraphs (mirrors the benchmark corpus)
# ---------------------------------------------------------------------------

DATASET = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn "
    "and improve from experience without being explicitly programmed.",
    "Deep learning uses neural networks with many layers to model complex patterns in data, "
    "applied to vision, speech, and natural language tasks.",
    "Natural language processing allows computers to understand, interpret, and generate "
    "human language at scale using statistical and neural methods.",
    "Retrieval-Augmented Generation combines a retrieval system with a language model to "
    "produce grounded, factual responses by anchoring generation in retrieved evidence.",
    "Vector databases store high-dimensional embeddings and support fast approximate "
    "nearest-neighbour search used in semantic search and RAG pipelines.",
    "Transformers rely on self-attention mechanisms to process sequential data in parallel, "
    "forming the backbone of models like BERT, GPT, and T5.",
    "System scalability is achieved through horizontal scaling, load balancing, and "
    "autoscaling strategies that distribute work across multiple compute nodes.",
    "Peak load management uses rate limiting, caching, and queue-based architectures to "
    "maintain low latency under sudden traffic spikes.",
    "Cloud infrastructure enables elastic resource allocation via Kubernetes autoscaling "
    "and CDN caching to handle variable demand efficiently.",
]

SOURCE_IDS = [
    "ml", "dl", "nlp", "rag", "vectordb",
    "transformers", "scalability", "peak_load", "cloud",
]


@pytest.fixture(scope="module")
def embedder() -> EmbeddingService:
    return EmbeddingService()


@pytest.fixture()
def manager(embedder) -> IngestionManager:
    m = IngestionManager(embedding_service=embedder)
    m.ingest(DATASET, SOURCE_IDS)
    return m


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestIngestionManagerConstruction:

    def test_default_construction(self):
        m = IngestionManager()
        assert m.is_empty
        assert m.chunk_count == 0

    def test_custom_chunk_size(self):
        m = IngestionManager(chunk_size=200, chunk_overlap=30)
        assert m.chunk_size == 200
        assert m.chunk_overlap == 30

    def test_store_is_vector_store(self):
        m = IngestionManager()
        assert isinstance(m.store, VectorStore)


# ---------------------------------------------------------------------------
# Ingestion — dataset of 5-10 technical paragraphs
# ---------------------------------------------------------------------------

class TestIngestionManagerIngest:

    def test_ingest_populates_store(self, manager):
        assert not manager.is_empty

    def test_ingest_9_documents_produces_chunks(self, manager):
        # 9 documents, each ~150-200 chars → at least 9 chunks
        assert manager.chunk_count >= 9

    def test_chunk_count_matches_store_length(self, manager):
        assert manager.chunk_count == len(manager.store)

    def test_summary_contains_chunk_count(self, manager):
        s = manager.summary()
        assert str(manager.chunk_count) in s

    def test_embedding_dim_is_set(self, manager):
        assert manager.store.embedding_dim is not None
        assert manager.store.embedding_dim > 0

    def test_ingest_skips_empty_documents(self, embedder):
        m = IngestionManager(embedding_service=embedder)
        valid = "This is a valid technical paragraph with enough characters to survive chunking."
        m.ingest(["", "   ", valid], ["e1", "e2", "valid"])
        assert m.chunk_count > 0

    def test_ingest_all_empty_produces_no_chunks(self, embedder):
        m = IngestionManager(embedding_service=embedder)
        m.ingest(["", "   "], ["e1", "e2"])
        assert m.is_empty

    def test_ingest_default_source_ids(self, embedder):
        m = IngestionManager(embedding_service=embedder)
        doc = "A sufficiently long technical paragraph about machine learning and neural networks."
        m.ingest([doc])  # no source_ids
        assert not m.is_empty

    def test_ingest_is_cumulative(self, embedder):
        m = IngestionManager(embedding_service=embedder)
        doc = "A sufficiently long technical paragraph about vector search and embeddings."
        m.ingest([doc], ["first"])
        count_after_first = m.chunk_count
        m.ingest([doc], ["second"])
        assert m.chunk_count > count_after_first

    def test_clear_resets_store(self, embedder):
        m = IngestionManager(embedding_service=embedder)
        doc = "A sufficiently long technical paragraph about transformers and attention."
        m.ingest([doc], ["d1"])
        m.clear()
        assert m.is_empty
        assert m.chunk_count == 0

    # ── GCP SDK mock test ────────────────────────────────────────────────────

    def test_embedding_service_called_during_ingest(self, embedder):
        """Verify EmbeddingService.embed is called exactly once per ingest call."""
        m = IngestionManager(embedding_service=embedder)
        original = embedder.embed
        calls = []

        def tracking(texts):
            calls.append(len(texts))
            return original(texts)

        embedder.embed = tracking
        try:
            doc = "A sufficiently long technical paragraph about RAG and retrieval systems."
            m.ingest([doc], ["d"])
        finally:
            embedder.embed = original

        assert len(calls) == 1

    def test_text_embedding_model_mock_intercepts_ingest(self):
        """Patch TextEmbeddingModel to verify the GCP SDK surface is called."""
        fake_emb = MagicMock()
        fake_emb.values = [0.1] * 384

        fake_model = MagicMock()
        fake_model.get_embeddings.return_value = [fake_emb]

        with patch(
            "rag.embedding.TextEmbeddingModel.from_pretrained",
            return_value=fake_model,
        ):
            svc = EmbeddingService()
            m = IngestionManager(embedding_service=svc)
            doc = "A sufficiently long technical paragraph about cloud infrastructure scaling."
            m.ingest([doc], ["cloud"])

        fake_model.get_embeddings.assert_called_once()
        assert not m.is_empty


# ---------------------------------------------------------------------------
# Integration — IngestionManager feeds both retrievers
# ---------------------------------------------------------------------------

class TestIngestionManagerIntegration:

    def test_store_from_manager_is_searchable(self, manager):
        """VectorStore produced by IngestionManager supports similarity search."""
        from rag.embedding import EmbeddingService
        svc = EmbeddingService()
        query_vec = svc.embed_one("neural network deep learning")
        results = manager.store.search(query_vec, top_k=3)
        assert len(results) == 3
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_raw_retriever_uses_ingestion_manager(self, embedder):
        """RawRetriever.ingest delegates to IngestionManager internally."""
        from rag.retrieval import RawRetriever
        r = RawRetriever(embedding_service=embedder, top_k=3)
        r.ingest(DATASET, SOURCE_IDS)
        # store property comes from IngestionManager
        assert len(r.store) > 0

    def test_expanding_retriever_uses_ingestion_manager(self, embedder):
        """ExpandingRetriever.ingest delegates to IngestionManager internally."""
        from rag.retrieval import ExpandingRetriever
        e = ExpandingRetriever(embedding_service=embedder, top_k=3)
        e.ingest(DATASET, SOURCE_IDS)
        assert len(e.store) > 0
