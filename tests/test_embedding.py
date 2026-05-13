"""
tests/test_embedding.py
-----------------------
Unit tests for rag/embedding.py.

GCP SDK is mocked via unittest.mock.patch so no real Vertex AI credentials
are needed.  The mock returns deterministic unit-norm vectors.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from rag.embedding import EmbeddingService, _MockTextEmbedding, _MockTextEmbeddingModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def service() -> EmbeddingService:
    """EmbeddingService backed by the built-in mock (no real GCP call)."""
    return EmbeddingService(model_name="text-embedding-004")


# ---------------------------------------------------------------------------
# _MockTextEmbeddingModel unit tests
# ---------------------------------------------------------------------------

class TestMockTextEmbeddingModel:
    def test_from_pretrained_returns_instance(self):
        model = _MockTextEmbeddingModel.from_pretrained("text-embedding-004")
        assert isinstance(model, _MockTextEmbeddingModel)

    def test_get_embeddings_length_matches_input(self):
        model = _MockTextEmbeddingModel.from_pretrained("text-embedding-004")
        texts = ["hello world", "foo bar", "baz"]
        result = model.get_embeddings(texts)
        assert len(result) == len(texts)

    def test_get_embeddings_returns_text_embedding_objects(self):
        model = _MockTextEmbeddingModel.from_pretrained("text-embedding-004")
        result = model.get_embeddings(["test"])
        assert hasattr(result[0], "values")
        assert isinstance(result[0].values, list)

    def test_embeddings_are_unit_norm(self):
        model = _MockTextEmbeddingModel.from_pretrained("text-embedding-004")
        result = model.get_embeddings(["unit norm check"])
        vec = np.array(result[0].values)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_same_text_produces_same_embedding(self):
        model = _MockTextEmbeddingModel.from_pretrained("text-embedding-004")
        r1 = model.get_embeddings(["deterministic"])
        r2 = model.get_embeddings(["deterministic"])
        np.testing.assert_array_almost_equal(r1[0].values, r2[0].values)

    def test_different_texts_produce_different_embeddings(self):
        model = _MockTextEmbeddingModel.from_pretrained("text-embedding-004")
        r = model.get_embeddings(["machine learning", "peak load management"])
        assert r[0].values != r[1].values


# ---------------------------------------------------------------------------
# EmbeddingService unit tests
# ---------------------------------------------------------------------------

class TestEmbeddingService:
    def test_embed_returns_2d_array(self, service):
        result = service.embed(["hello", "world"])
        assert result.ndim == 2
        assert result.shape[0] == 2

    def test_embed_rows_are_unit_norm(self, service):
        result = service.embed(["normalisation check", "another text"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_embed_one_returns_1d_array(self, service):
        result = service.embed_one("single text")
        assert result.ndim == 1

    def test_embed_one_is_unit_norm(self, service):
        result = service.embed_one("unit norm single")
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_embed_empty_list_raises(self, service):
        with pytest.raises(ValueError, match="non-empty"):
            service.embed([])

    def test_embed_dtype_is_float32(self, service):
        result = service.embed(["dtype check"])
        assert result.dtype == np.float32

    def test_model_name_property(self, service):
        assert service.model_name == "text-embedding-004"

    # ── GCP SDK mock test ────────────────────────────────────────────────────
    def test_gcp_sdk_mock_intercepts_get_embeddings(self):
        """
        Verify that patching TextEmbeddingModel.from_pretrained intercepts
        the call — simulates swapping in the real GCP SDK.
        """
        fake_embedding = MagicMock()
        fake_embedding.values = [0.1, 0.2, 0.3]

        fake_model = MagicMock()
        fake_model.get_embeddings.return_value = [fake_embedding]

        with patch(
            "rag.embedding.TextEmbeddingModel.from_pretrained",
            return_value=fake_model,
        ):
            svc = EmbeddingService(model_name="text-embedding-004")
            result = svc.embed(["patched"])

        fake_model.get_embeddings.assert_called_once_with(["patched"])
        assert result.shape == (1, 3)

    def test_model_failure_raises_runtime_error(self):
        """RuntimeError is raised when the underlying model throws."""
        fake_model = MagicMock()
        fake_model.get_embeddings.side_effect = Exception("GCP timeout")

        with patch(
            "rag.embedding.TextEmbeddingModel.from_pretrained",
            return_value=fake_model,
        ):
            svc = EmbeddingService()
            with pytest.raises(RuntimeError, match="GCP timeout"):
                svc.embed(["will fail"])
