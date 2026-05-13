"""
rag/retrieval.py
----------------
Retrieval layer — document ingestion, optional query expansion, and search.

Two public classes:
  RawRetriever        — Strategy A: embed query as-is, search.
  ExpandingRetriever  — Strategy B: expand query via GenerativeModel, then search.
"""

from __future__ import annotations

import logging
import textwrap
from typing import List, Optional

from rag.embedding import EmbeddingService
from rag.ingestion import IngestionManager
from rag.storage import Chunk, SearchResult, VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vertex AI GenerativeModel (mocked)
# ---------------------------------------------------------------------------

class _MockGenerationResponse:
    """Mirrors vertexai.generative_models.GenerationResponse."""
    def __init__(self, text: str) -> None:
        self.text: str = text


class _MockGenerativeModel:
    """
    Mirrors vertexai.generative_models.GenerativeModel.

    Uses rule-based expansion so the mock works offline and produces
    deterministic, semantically richer rewrites.

    Real SDK swap-in:
        from vertexai.generative_models import GenerativeModel
    """

    # Domain-specific expansion vocabulary
    # Each entry maps a keyword to a focused rewrite — not a word dump.
    # The rewrite replaces the original query rather than appending to it,
    # so the embedding stays sharp rather than getting diluted.
    _RULES = {
        "peak load": (
            "handling peak load traffic spikes autoscaling load balancing "
            "horizontal scaling throughput latency capacity planning "
            "rate limiting queue architecture high concurrency"
        ),
        "errors in ai": (
            "reducing hallucinations in large language model outputs "
            "retrieval augmented generation RAG grounding factual accuracy "
            "knowledge base anchoring faithfulness citation verification"
        ),
        "ai-generated": (
            "reducing hallucinations in large language model outputs "
            "retrieval augmented generation RAG grounding factual accuracy "
            "knowledge base anchoring faithfulness citation verification"
        ),
        "language model": (
            "large language model transformer architecture self-attention "
            "BERT GPT T5 sequential text processing parallel computation "
            "NLP natural language understanding generation"
        ),
        "sequential text": (
            "transformer self-attention mechanism parallel sequence processing "
            "BERT GPT positional encoding encoder decoder NLP language model"
        ),
        "machine learning": (
            "machine learning supervised unsupervised model training "
            "gradient descent neural network feature engineering "
            "overfitting regularisation cross-validation"
        ),
        "deep learning": (
            "deep learning convolutional recurrent neural network "
            "backpropagation activation function batch normalisation "
            "dropout GPU training representation learning"
        ),
        "natural language": (
            "natural language processing NLP tokenisation named entity "
            "recognition sentiment analysis language model BERT GPT transformer "
            "text understanding generation"
        ),
        "rag": (
            "retrieval augmented generation vector search embedding "
            "knowledge base grounding hallucination reduction context "
            "injection document retrieval factual accuracy"
        ),
        "vector": (
            "vector embedding similarity search cosine distance "
            "high-dimensional space semantic search nearest neighbour "
            "FAISS Pinecone Weaviate index"
        ),
        "transformer": (
            "transformer architecture self-attention multi-head attention "
            "positional encoding encoder decoder BERT GPT T5 fine-tuning "
            "parallel sequence processing"
        ),
        "scalab": (
            "system scalability horizontal vertical scaling load balancer "
            "autoscaling distributed architecture throughput capacity"
        ),
        "cloud": (
            "cloud infrastructure elastic resource allocation Kubernetes "
            "autoscaling CDN caching containerised workloads pod scaling"
        ),
    }

    # No generic suffix — it dilutes every embedding equally and adds no signal

    _PROMPT_MARKER = "Original query:"

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        logger.debug("[Mock] GenerativeModel('%s') initialised", model_name)

    def generate_content(self, prompt: str) -> _MockGenerationResponse:
        """Mirrors model.generate_content(prompt) → GenerationResponse."""
        original = self._extract_query(prompt)
        expanded = self._expand(original)
        logger.debug(
            "[Mock] GenerativeModel: '%s' → '%s'", original, expanded[:80]
        )
        return _MockGenerationResponse(text=expanded)

    def _extract_query(self, prompt: str) -> str:
        if self._PROMPT_MARKER in prompt:
            after = prompt.split(self._PROMPT_MARKER, 1)[1]
            for line in after.splitlines():
                line = line.strip().strip('"').strip("'")
                if line:
                    return line
        return prompt.strip()

    def _expand(self, query: str) -> str:
        q = query.lower()
        matched = []
        for keyword, expansion in self._RULES.items():
            if keyword in q:
                matched.append(expansion)

        if not matched:
            # No rule matched — return the original query unchanged.
            # Better to leave it focused than add noise.
            return query

        # Build: original intent + matched domain expansions only.
        # No generic suffix — it dilutes the embedding.
        return query + " " + " ".join(matched)


# Public alias — swap for real import when going live
GenerativeModel = _MockGenerativeModel

# ---------------------------------------------------------------------------
# QueryExpander
# ---------------------------------------------------------------------------

class QueryExpander:
    """
    Rewrites a raw user query into an embedding-friendly expanded form
    using vertexai.generative_models.GenerativeModel.
    """

    _PROMPT = textwrap.dedent("""\
        You are a search query optimisation assistant.
        Rewrite and expand the user's query into a richer, embedding-friendly
        format that improves semantic retrieval.

        Guidelines:
        - Preserve the original intent exactly.
        - Add relevant synonyms, related concepts, and technical terminology.
        - Output ONLY the expanded query string.

        Original query: "{query}"

        Expanded query:""")

    def __init__(self, model_name: str = "gemini-1.5-flash-001") -> None:
        self._model = GenerativeModel(model_name)

    def expand(self, query: str) -> str:
        """
        Expand a raw query into an embedding-optimised string.

        Raises:
            ValueError: If query is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")
        response = self._model.generate_content(
            self._PROMPT.format(query=query.strip())
        )
        expanded = response.text.strip()
        return expanded if expanded else query.strip()


# ---------------------------------------------------------------------------
# Shared ingestion helper  (kept for internal use by retrievers)
# ---------------------------------------------------------------------------

def _chunk_text(
    text: str,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
) -> List[Chunk]:
    """Slide a fixed-size window over text and return Chunk objects."""
    chunks: List[Chunk] = []
    start = 0
    while start < len(text):
        fragment = text[start: start + chunk_size].strip()
        if len(fragment) >= min_chunk_size:
            chunks.append(Chunk(text=fragment, source=source))
        elif fragment:
            logger.debug(
                "Discarding short chunk (%d chars) from '%s'.", len(fragment), source
            )
        start += chunk_size - chunk_overlap
    return chunks


# ---------------------------------------------------------------------------
# Strategy A — Raw Vector Retriever
# ---------------------------------------------------------------------------

class RawRetriever:
    """
    Strategy A: embed the query as-is and search by cosine similarity.

    Args:
        embedding_service: EmbeddingService instance.
        top_k:             Number of results to return.
        chunk_size:        Max characters per chunk.
        chunk_overlap:     Overlap between consecutive chunks.
        min_chunk_size:    Minimum characters to keep a chunk.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        top_k: int = 3,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        min_chunk_size: int = 30,
    ) -> None:
        self._embedder = embedding_service or EmbeddingService()
        self._ingestion = IngestionManager(
            embedding_service=self._embedder,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )
        self.top_k = top_k

    def ingest(self, documents: List[str], source_ids: Optional[List[str]] = None) -> None:
        """
        Chunk, embed, and store documents via IngestionManager.

        Args:
            documents:  Plain-text strings.
            source_ids: Parallel identifiers (defaults to doc_0, doc_1, …).
        """
        self._ingestion.ingest(documents, source_ids)

    def query(self, query_text: str) -> List[SearchResult]:
        """
        Embed query_text and return top-K results.

        Raises:
            ValueError: Empty query or empty store.
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query must not be empty.")
        if self._ingestion.is_empty:
            raise ValueError("Store is empty. Ingest documents first.")

        query_vec = self._embedder.embed_one(query_text.strip())
        return self._ingestion.store.search(query_vec, top_k=self.top_k)

    @property
    def store(self) -> VectorStore:
        return self._ingestion.store


# ---------------------------------------------------------------------------
# Strategy B — Expanding Retriever
# ---------------------------------------------------------------------------

class ExpandingRetriever:
    """
    Strategy B: expand the query via GenerativeModel before embedding and search.

    Args:
        embedding_service:     EmbeddingService instance.
        generative_model_name: Vertex AI generative model identifier.
        top_k:                 Number of results to return.
        chunk_size:            Max characters per chunk.
        chunk_overlap:         Overlap between consecutive chunks.
        min_chunk_size:        Minimum characters to keep a chunk.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        generative_model_name: str = "gemini-1.5-flash-001",
        top_k: int = 3,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        min_chunk_size: int = 30,
    ) -> None:
        self._embedder = embedding_service or EmbeddingService()
        self._expander = QueryExpander(generative_model_name)
        self._ingestion = IngestionManager(
            embedding_service=self._embedder,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )
        self.top_k = top_k

    def ingest(self, documents: List[str], source_ids: Optional[List[str]] = None) -> None:
        """Chunk, embed, and store documents via IngestionManager."""
        self._ingestion.ingest(documents, source_ids)

    def query(self, query_text: str) -> tuple[str, List[SearchResult]]:
        """
        Expand query_text, embed the expansion, and return top-K results.

        Returns:
            (expanded_query, List[SearchResult])

        Raises:
            ValueError: Empty query or empty store.
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query must not be empty.")
        if self._ingestion.is_empty:
            raise ValueError("Store is empty. Ingest documents first.")

        expanded = self._expander.expand(query_text.strip())
        query_vec = self._embedder.embed_one(expanded)
        results = self._ingestion.store.search(query_vec, top_k=self.top_k)
        return expanded, results

    @property
    def store(self) -> VectorStore:
        return self._ingestion.store
