"""
tests/test_storage.py
---------------------
Unit tests for rag/storage.py — VectorStore and cosine similarity search.
"""

import numpy as np
import pytest

from rag.storage import Chunk, SearchResult, VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _make_chunk(text: str, source: str, vec: np.ndarray) -> Chunk:
    return Chunk(text=text, source=source, embedding=_unit(vec).astype(np.float32))


# ---------------------------------------------------------------------------
# VectorStore tests
# ---------------------------------------------------------------------------

class TestVectorStore:

    # ── Construction ─────────────────────────────────────────────────────────

    def test_new_store_is_empty(self):
        store = VectorStore()
        assert store.is_empty
        assert len(store) == 0
        assert store.embedding_dim is None

    # ── add() ────────────────────────────────────────────────────────────────

    def test_add_increases_length(self):
        store = VectorStore()
        chunks = [_make_chunk("a", "s1", np.array([1.0, 0.0, 0.0]))]
        store.add(chunks)
        assert len(store) == 1
        assert not store.is_empty

    def test_add_multiple_chunks(self):
        store = VectorStore()
        chunks = [
            _make_chunk("a", "s1", np.array([1.0, 0.0, 0.0])),
            _make_chunk("b", "s2", np.array([0.0, 1.0, 0.0])),
            _make_chunk("c", "s3", np.array([0.0, 0.0, 1.0])),
        ]
        store.add(chunks)
        assert len(store) == 3

    def test_add_chunk_without_embedding_raises(self):
        store = VectorStore()
        bad_chunk = Chunk(text="no embedding", source="src")
        with pytest.raises(ValueError, match="no embedding"):
            store.add([bad_chunk])

    def test_add_is_cumulative(self):
        store = VectorStore()
        store.add([_make_chunk("a", "s1", np.array([1.0, 0.0]))])
        store.add([_make_chunk("b", "s2", np.array([0.0, 1.0]))])
        assert len(store) == 2

    def test_embedding_dim_set_after_add(self):
        store = VectorStore()
        store.add([_make_chunk("x", "s", np.array([1.0, 2.0, 3.0]))])
        assert store.embedding_dim == 3

    # ── clear() ──────────────────────────────────────────────────────────────

    def test_clear_resets_store(self):
        store = VectorStore()
        store.add([_make_chunk("a", "s", np.array([1.0, 0.0]))])
        store.clear()
        assert store.is_empty
        assert store.embedding_dim is None

    # ── search() — basic ─────────────────────────────────────────────────────

    def test_search_returns_top_k_results(self):
        store = VectorStore()
        store.add([
            _make_chunk("a", "s1", np.array([1.0, 0.0, 0.0])),
            _make_chunk("b", "s2", np.array([0.0, 1.0, 0.0])),
            _make_chunk("c", "s3", np.array([0.0, 0.0, 1.0])),
        ])
        results = store.search(_unit(np.array([1.0, 0.0, 0.0])), top_k=2)
        assert len(results) == 2

    def test_search_results_are_sorted_descending(self):
        store = VectorStore()
        store.add([
            _make_chunk("a", "s1", np.array([1.0, 0.0, 0.0])),
            _make_chunk("b", "s2", np.array([0.0, 1.0, 0.0])),
            _make_chunk("c", "s3", np.array([0.5, 0.5, 0.0])),
        ])
        results = store.search(_unit(np.array([1.0, 0.0, 0.0])), top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_rank_field_is_1_indexed(self):
        store = VectorStore()
        store.add([_make_chunk("a", "s", np.array([1.0, 0.0]))])
        results = store.search(_unit(np.array([1.0, 0.0])), top_k=1)
        assert results[0].rank == 1

    def test_search_returns_correct_top_source(self):
        """The most similar chunk should be returned first."""
        store = VectorStore()
        store.add([
            _make_chunk("irrelevant", "wrong", np.array([0.0, 1.0, 0.0])),
            _make_chunk("relevant",   "right", np.array([1.0, 0.0, 0.0])),
        ])
        results = store.search(_unit(np.array([1.0, 0.0, 0.0])), top_k=1)
        assert results[0].chunk.source == "right"

    def test_search_cosine_similarity_value(self):
        """Identical unit vectors → cosine similarity = 1.0."""
        store = VectorStore()
        vec = _unit(np.array([3.0, 4.0, 0.0]))
        store.add([_make_chunk("x", "s", vec)])
        results = store.search(vec, top_k=1)
        assert abs(results[0].score - 1.0) < 1e-5

    def test_search_orthogonal_vectors_score_near_zero(self):
        store = VectorStore()
        store.add([_make_chunk("x", "s", np.array([1.0, 0.0]))])
        results = store.search(_unit(np.array([0.0, 1.0])), top_k=1)
        assert abs(results[0].score) < 1e-5

    # ── search() — edge cases ────────────────────────────────────────────────

    def test_search_empty_store_raises(self):
        store = VectorStore()
        with pytest.raises(ValueError, match="empty"):
            store.search(np.array([1.0, 0.0]), top_k=3)

    def test_search_top_k_zero_raises(self):
        store = VectorStore()
        store.add([_make_chunk("a", "s", np.array([1.0, 0.0]))])
        with pytest.raises(ValueError, match="top_k"):
            store.search(np.array([1.0, 0.0]), top_k=0)

    def test_search_top_k_larger_than_store_returns_all(self):
        store = VectorStore()
        store.add([
            _make_chunk("a", "s1", np.array([1.0, 0.0])),
            _make_chunk("b", "s2", np.array([0.0, 1.0])),
        ])
        results = store.search(_unit(np.array([1.0, 0.0])), top_k=100)
        assert len(results) == 2

    def test_search_result_contains_chunk_and_score(self):
        store = VectorStore()
        store.add([_make_chunk("hello", "src_a", np.array([1.0, 0.0]))])
        results = store.search(_unit(np.array([1.0, 0.0])), top_k=1)
        r = results[0]
        assert isinstance(r, SearchResult)
        assert r.chunk.text == "hello"
        assert r.chunk.source == "src_a"
        assert isinstance(r.score, float)
