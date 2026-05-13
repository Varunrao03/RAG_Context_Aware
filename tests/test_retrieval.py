"""
tests/test_retrieval.py
-----------------------
Integration tests for rag/retrieval.py.

GCP SDK (TextEmbeddingModel + GenerativeModel) is mocked via unittest.mock
so no real Vertex AI credentials are required.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

from rag.retrieval import (
    QueryExpander,
    RawRetriever,
    ExpandingRetriever,
    _MockGenerativeModel,
    GenerativeModel,
)
from rag.embedding import EmbeddingService
from rag.storage import SearchResult


# ---------------------------------------------------------------------------
# Shared corpus fixture
# ---------------------------------------------------------------------------

DOCS = [
    "Machine learning enables systems to learn from data without explicit programming.",
    "Deep learning uses neural networks with many layers to model complex patterns.",
    "Natural language processing allows computers to understand human language.",
    "Retrieval-Augmented Generation combines retrieval with language model generation.",
    "Vector databases store embeddings and support fast similarity search.",
    "Peak load management uses autoscaling and load balancing to handle traffic spikes.",
    "System scalability is achieved through horizontal and vertical scaling strategies.",
]

SOURCES = [
    "ml", "dl", "nlp", "rag", "vectordb", "peak_load", "scalability"
]


@pytest.fixture(scope="module")
def shared_embedder() -> EmbeddingService:
    """One EmbeddingService shared across the module to avoid reloading the model."""
    return EmbeddingService()


@pytest.fixture()
def raw(shared_embedder) -> RawRetriever:
    r = RawRetriever(embedding_service=shared_embedder, top_k=3)
    r.ingest(DOCS, SOURCES)
    return r


@pytest.fixture()
def expanding(shared_embedder) -> ExpandingRetriever:
    e = ExpandingRetriever(embedding_service=shared_embedder, top_k=3)
    e.ingest(DOCS, SOURCES)
    return e


# ---------------------------------------------------------------------------
# QueryExpander tests
# ---------------------------------------------------------------------------

class TestQueryExpander:

    def test_expand_returns_string(self):
        expander = QueryExpander()
        result = expander.expand("machine learning")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_expand_contains_original_query(self):
        expander = QueryExpander()
        query = "peak load"
        result = expander.expand(query)
        assert query.lower() in result.lower()

    def test_expand_enriches_known_keyword(self):
        expander = QueryExpander()
        result = expander.expand("peak load")
        # Mock rules add scalability-related terms
        assert any(kw in result.lower() for kw in ["scalab", "autoscal", "load balanc"])

    def test_expand_empty_query_raises(self):
        expander = QueryExpander()
        with pytest.raises(ValueError, match="empty"):
            expander.expand("")

    def test_expand_whitespace_only_raises(self):
        expander = QueryExpander()
        with pytest.raises(ValueError, match="empty"):
            expander.expand("   ")

    # ── GCP SDK mock test ────────────────────────────────────────────────────
    def test_generative_model_mock_is_called(self):
        """Verify GenerativeModel.generate_content is invoked with the prompt."""
        fake_response = MagicMock()
        fake_response.text = "expanded query text"

        fake_model = MagicMock()
        fake_model.generate_content.return_value = fake_response

        with patch("rag.retrieval.GenerativeModel", return_value=fake_model):
            expander = QueryExpander(model_name="gemini-1.5-flash-001")
            result = expander.expand("test query")

        fake_model.generate_content.assert_called_once()
        assert "test query" in fake_model.generate_content.call_args[0][0]
        assert result == "expanded query text"

    def test_empty_model_response_falls_back_to_original(self):
        """If the model returns empty text, the original query is used."""
        fake_response = MagicMock()
        fake_response.text = "   "

        fake_model = MagicMock()
        fake_model.generate_content.return_value = fake_response

        with patch("rag.retrieval.GenerativeModel", return_value=fake_model):
            expander = QueryExpander()
            result = expander.expand("fallback query")

        assert result == "fallback query"


# ---------------------------------------------------------------------------
# RawRetriever tests
# ---------------------------------------------------------------------------

class TestRawRetriever:

    def test_ingest_populates_store(self, raw):
        assert not raw.store.is_empty

    def test_query_returns_top_k_results(self, raw):
        results = raw.query("machine learning neural network")
        assert len(results) == 3

    def test_query_results_sorted_descending(self, raw):
        results = raw.query("vector similarity search")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_result_type(self, raw):
        results = raw.query("deep learning")
        assert all(isinstance(r, SearchResult) for r in results)

    def test_query_ranks_are_sequential(self, raw):
        results = raw.query("NLP language model")
        assert [r.rank for r in results] == [1, 2, 3]

    def test_query_empty_raises(self, raw):
        with pytest.raises(ValueError, match="empty"):
            raw.query("")

    def test_query_whitespace_raises(self, raw):
        with pytest.raises(ValueError, match="empty"):
            raw.query("   ")

    def test_query_empty_store_raises(self, shared_embedder):
        r = RawRetriever(embedding_service=shared_embedder)
        with pytest.raises(ValueError, match="[Ss]tore is empty"):
            r.query("anything")

    def test_ingest_skips_empty_documents(self, shared_embedder):
        r = RawRetriever(embedding_service=shared_embedder)
        valid_doc = "This is a valid document with enough characters to pass the minimum chunk size threshold."
        r.ingest(["", "   ", valid_doc], ["e1", "e2", "valid"])
        assert len(r.store) > 0

    def test_relevant_query_retrieves_correct_source(self, raw):
        """A query about peak load should surface the peak_load chunk."""
        results = raw.query("autoscaling traffic spikes load balancing")
        top_sources = [r.chunk.source for r in results]
        assert "peak_load" in top_sources

    # ── GCP SDK mock test ────────────────────────────────────────────────────
    def test_embedding_service_mock_intercepts_embed(self, shared_embedder):
        """Patch EmbeddingService.embed to verify it is called during ingest."""
        r = RawRetriever(embedding_service=shared_embedder, top_k=2)

        original_embed = shared_embedder.embed
        call_count = {"n": 0}

        def counting_embed(texts):
            call_count["n"] += 1
            return original_embed(texts)

        shared_embedder.embed = counting_embed
        try:
            r.ingest(
                [
                    "Test document one with sufficient length to pass the minimum chunk size.",
                    "Test document two with sufficient length to pass the minimum chunk size.",
                ],
                ["t1", "t2"],
            )
        finally:
            shared_embedder.embed = original_embed

        assert call_count["n"] >= 1


# ---------------------------------------------------------------------------
# ExpandingRetriever tests
# ---------------------------------------------------------------------------

class TestExpandingRetriever:

    def test_ingest_populates_store(self, expanding):
        assert not expanding.store.is_empty

    def test_query_returns_tuple(self, expanding):
        result = expanding.query("peak load handling")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_query_returns_expanded_string_and_results(self, expanding):
        expanded_q, results = expanding.query("peak load handling")
        assert isinstance(expanded_q, str)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_expanded_query_is_longer_than_original(self, expanding):
        original = "peak load"
        expanded_q, _ = expanding.query(original)
        assert len(expanded_q) > len(original)

    def test_query_results_sorted_descending(self, expanding):
        _, results = expanding.query("vector similarity search")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_empty_raises(self, expanding):
        with pytest.raises(ValueError, match="empty"):
            expanding.query("")

    def test_query_empty_store_raises(self, shared_embedder):
        e = ExpandingRetriever(embedding_service=shared_embedder)
        with pytest.raises(ValueError, match="[Ss]tore is empty"):
            e.query("anything")

    def test_relevant_query_retrieves_correct_source(self, expanding):
        """After expansion, peak load query should still surface peak_load."""
        _, results = expanding.query("how does the system handle peak load")
        top_sources = [r.chunk.source for r in results]
        assert "peak_load" in top_sources or "scalability" in top_sources

    # ── GCP SDK mock test ────────────────────────────────────────────────────
    def test_generative_model_called_during_query(self, shared_embedder):
        """
        Verify GenerativeModel.generate_content is invoked exactly once
        per query call — simulates the real GCP SDK being called.
        """
        fake_response = MagicMock()
        fake_response.text = "expanded peak load scalability autoscaling"

        fake_gen_model = MagicMock()
        fake_gen_model.generate_content.return_value = fake_response

        with patch("rag.retrieval.GenerativeModel", return_value=fake_gen_model):
            e = ExpandingRetriever(embedding_service=shared_embedder, top_k=2)
            e.ingest(["peak load management autoscaling"], ["peak"])
            expanded_q, results = e.query("peak load")

        fake_gen_model.generate_content.assert_called_once()
        assert expanded_q == "expanded peak load scalability autoscaling"

    def test_text_embedding_model_called_for_query(self, shared_embedder):
        """embed_one is called when processing the expanded query."""
        e = ExpandingRetriever(embedding_service=shared_embedder, top_k=2)
        e.ingest(
            ["Some document about vectors and similarity search with enough text to chunk."],
            ["doc"],
        )

        original_embed_one = shared_embedder.embed_one
        calls = []

        def tracking_embed_one(text):
            calls.append(text)
            return original_embed_one(text)

        shared_embedder.embed_one = tracking_embed_one
        try:
            e.query("vector search")
        finally:
            shared_embedder.embed_one = original_embed_one

        assert len(calls) == 1  # exactly one embed_one call per query


# ---------------------------------------------------------------------------
# Strategy A vs Strategy B comparison test
# ---------------------------------------------------------------------------

class TestStrategyComparison:

    def test_both_strategies_return_same_number_of_results(self, raw, expanding):
        query = "how do neural networks learn from data"
        raw_results = raw.query(query)
        _, exp_results = expanding.query(query)
        assert len(raw_results) == len(exp_results) == 3

    def test_expansion_changes_query_text(self, expanding):
        query = "peak load"
        expanded_q, _ = expanding.query(query)
        assert expanded_q != query

    def test_scores_are_floats_in_valid_range(self, raw, expanding):
        query = "retrieval augmented generation"
        raw_results = raw.query(query)
        _, exp_results = expanding.query(query)
        for r in raw_results + exp_results:
            assert -1.0 <= r.score <= 1.0
