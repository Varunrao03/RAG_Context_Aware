"""
rag — modular RAG pipeline package.

Modules
-------
embedding   EmbeddingService wrapping TextEmbeddingModel
ingestion   IngestionManager — dataset ingestion, chunking, embedding
storage     VectorStore with raw cosine similarity search
retrieval   RawRetriever (Strategy A) and ExpandingRetriever (Strategy B)
"""

from rag.embedding import EmbeddingService
from rag.ingestion import IngestionManager
from rag.storage import Chunk, SearchResult, VectorStore
from rag.retrieval import ExpandingRetriever, QueryExpander, RawRetriever

__all__ = [
    "Chunk",
    "EmbeddingService",
    "ExpandingRetriever",
    "IngestionManager",
    "QueryExpander",
    "RawRetriever",
    "SearchResult",
    "VectorStore",
]
